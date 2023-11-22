import torch
import torch.nn as nn
import numpy as np
from .NewModel import NewModel


class gcn_operation(nn.Module):
    def __init__(self, adj_mx, in_dim, out_dim, num_vertices, activation='GLU'):
        super(gcn_operation, self).__init__()
        self.adj_mx = torch.tensor(adj_mx).to(torch.float32)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_vertices = num_vertices
        self.activation = activation
        assert self.activation in {'GLU', 'relu'}
        if self.activation == 'GLU':
            self.FC = nn.Linear(self.in_dim, 2 * self.out_dim, bias=True)
        else:
            self.FC = nn.Linear(self.in_dim, self.out_dim, bias=True)

    def forward(self, x, mask=None):
        adj_mx = self.adj_mx
        if mask is not None:
            adj_mx = adj_mx.to(mask.device) * mask 
        x = torch.einsum('nm, mbc->nbc', adj_mx.to(x.device), x) 
        if self.activation == 'GLU':
            lhs_rhs = self.FC(x) 
            lhs, rhs = torch.split(lhs_rhs, self.out_dim, dim=-1) 
            out = lhs * torch.sigmoid(rhs) 
            del lhs, rhs, lhs_rhs
            return out 
        elif self.activation == 'relu':
            return torch.relu(self.FC(x))  # 3*N, B, Cout

class STSGCM(nn.Module):
    def __init__(self, adj_mx, in_dim, out_dims, num_of_vertices, activation='GLU'):
        super(STSGCM, self).__init__()
        self.adj_mx = adj_mx
        self.in_dim = in_dim
        self.out_dims = out_dims
        self.num_of_vertices = num_of_vertices
        self.activation = activation
        self.gcn_operations = nn.ModuleList()
        self.gcn_operations.append(
            gcn_operation(
                adj_mx=self.adj_mx,
                in_dim=self.in_dim,
                out_dim=self.out_dims[0],
                num_vertices=self.num_of_vertices,
                activation=self.activation
            )
        )

        for i in range(1, len(self.out_dims)):
            self.gcn_operations.append(
                gcn_operation(
                    adj_mx=self.adj_mx,
                    in_dim=self.out_dims[i-1],
                    out_dim=self.out_dims[i],
                    num_vertices=self.num_of_vertices,
                    activation=self.activation
                )
            )

    def forward(self, x, mask=None):
        need_concat = []
        for i in range(len(self.out_dims)):
            x = self.gcn_operations[i](x, mask)
            need_concat.append(x)
        need_concat = [
            torch.unsqueeze(
                h[self.num_of_vertices: 2 * self.num_of_vertices], dim=0
            ) for h in need_concat
        ]
        out = torch.max(torch.cat(need_concat, dim=0), dim=0).values  # (N, B, Cout)
        del need_concat
        return out


class STSGCL(nn.Module):
    def __init__(self,adj_mx,history,num_of_vertices,in_dim,out_dims,strides=4,activation='GLU',temporal_emb=True,spatial_emb=True):
        super(STSGCL, self).__init__()
        self.adj_mx = adj_mx
        self.strides = strides
        self.history = history
        self.in_dim = in_dim
        self.out_dims = out_dims
        self.num_of_vertices = num_of_vertices
        self.activation = activation
        self.temporal_emb = temporal_emb
        self.spatial_emb = spatial_emb
        self.conv1 = nn.Conv1d(self.in_dim, self.out_dims[-1], kernel_size=(1, 2), stride=(1, 1), dilation=(1, 3))
        self.conv2 = nn.Conv1d(self.in_dim, self.out_dims[-1], kernel_size=(1, 2), stride=(1, 1), dilation=(1, 3))
        self.STSGCMS = nn.ModuleList()
        for i in range(self.history - self.strides + 1):
            self.STSGCMS.append(
                STSGCM(
                    adj_mx=self.adj_mx,
                    in_dim=self.in_dim,
                    out_dims=self.out_dims,
                    num_of_vertices=self.num_of_vertices,
                    activation=self.activation
                )
            )
        if self.temporal_emb:
            self.temporal_embedding = nn.Parameter(torch.FloatTensor(1, self.history, 1, self.in_dim))
        if self.spatial_emb:
            self.spatial_embedding = nn.Parameter(torch.FloatTensor(1, 1, self.num_of_vertices, self.in_dim))
        self.reset()

    def reset(self):
        if self.temporal_emb:
            nn.init.xavier_normal_(self.temporal_embedding, gain=0.0003)
        if self.spatial_emb:
            nn.init.xavier_normal_(self.spatial_embedding, gain=0.0003)

    def forward(self, x, mask=None):
        if self.temporal_emb:
            x = x + self.temporal_embedding
        if self.spatial_emb:
            x = x + self.spatial_embedding  
            data_temp = x.permute(0, 3, 2, 1)
        data_left = torch.sigmoid(self.conv1(data_temp))
        data_right = torch.tanh(self.conv2(data_temp)) 
        data_time_axis = data_left * data_right
        data_res = data_time_axis.permute(0, 3, 2, 1)
        need_concat = []
        batch_size = x.shape[0]
        for i in range(self.history - self.strides + 1):
            t = x[:, i: i+self.strides, :, :]
            t = torch.reshape(t, shape=[batch_size, self.strides * self.num_of_vertices, self.in_dim])
            t = self.STSGCMS[i](t.permute(1, 0, 2), mask) 
            t = torch.unsqueeze(t.permute(1, 0, 2), dim=1) 
            need_concat.append(t)
        mid_out = torch.cat(need_concat, dim=1)
        out = mid_out + data_res
        del need_concat, batch_size
 
        return out 


class output_layer(nn.Module):
    def __init__(self, num_of_vertices, history, in_dim, out_dim, 
                 hidden_dim=128, horizon=12):
        super(output_layer, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.history = history
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        self.FC1 = nn.Linear(self.in_dim * self.history, self.hidden_dim, bias=True)
        self.FC2 = nn.Linear(self.hidden_dim, self.horizon * self.out_dim, bias=True)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.permute(0, 2, 1, 3) 
        out1 = torch.relu(self.FC1 (x.reshape(batch_size, self.num_of_vertices, -1)))
        out2 = self.FC2(out1)  
        out2 = out2.reshape(batch_size, self.num_of_vertices, self.horizon, self.out_dim)
        del out1, batch_size
        return out2.permute(0, 2, 1, 3)  



class STFGNN(nn.Module):
    def __init__(self, adj,adj_mx,horizon,hidden_dims,out_layer_dim,activation,temporal_emb,spatial_emb,\
                 first_layer_embedding_size,output_dim,mask,input_dim,save_lane_count_step,**kwargs):
        super(STFGNN, self).__init__()
        self.adj =adj
        self.adj_mx=adj_mx
        self.num_of_vertices = len(adj)
        self.hidden_dims =[[64, 64, 64], [64, 64, 64], [64, 64, 64]]  #hidden_dims
        self.out_layer_dim = out_layer_dim
        self.activation = activation
        self.use_mask = mask
        self.temporal_emb = temporal_emb
        self.spatial_emb = spatial_emb
        self.horizon = horizon 
        self.strides = 4
        self.First_FC = nn.Linear(input_dim, first_layer_embedding_size, bias=True)
        self.STSGCLS = nn.ModuleList()
        self.STSGCLS.append(
            STSGCL(
                adj_mx=self.adj_mx,
                history=save_lane_count_step,
                num_of_vertices=self.num_of_vertices,
                in_dim=first_layer_embedding_size,
                out_dims=self.hidden_dims[0],
                strides=self.strides,
                activation=self.activation,
                temporal_emb=self.temporal_emb,
                spatial_emb=self.spatial_emb
            )
        )
        in_dim = self.hidden_dims[0][-1]
        save_lane_count_step -= (self.strides - 1)
        for idx, hidden_list in enumerate(self.hidden_dims):  
            if idx == 0:
                continue
            self.STSGCLS.append(
                STSGCL(
                    adj_mx=self.adj_mx,
                    history=save_lane_count_step,
                    num_of_vertices=self.num_of_vertices,
                    in_dim=in_dim,
                    out_dims=hidden_list,
                    strides=self.strides,
                    activation=self.activation,
                    temporal_emb=self.temporal_emb,
                    spatial_emb=self.spatial_emb
                )
            )
            save_lane_count_step -= (self.strides - 1)
            in_dim = hidden_list[-1]
        self.predictLayer = nn.ModuleList()
        for t in range(self.horizon):
            self.predictLayer.append(
                output_layer(
                    num_of_vertices=self.num_of_vertices,
                    history=save_lane_count_step,
                    in_dim=in_dim,
                    out_dim = output_dim,
                    hidden_dim=out_layer_dim,
                    horizon=1
                )
            )
        if self.use_mask:
            mask = torch.zeros_like(self.adj)
            mask[self.adj != 0] = self.adj[self.adj != 0]
            self.mask = nn.Parameter(mask)
        else:
            self.mask = None

    def forward(self, x):
        # x=self.scaler.transform(x)
        x=torch.tensor(x).to(torch.float32)
        x = torch.relu(self.First_FC(x)) 
        for model in self.STSGCLS:
            x = model(x, self.mask)
        need_concat = []
        for i in range(self.horizon): 
            out_step = self.predictLayer[i](x) 
            need_concat.append(out_step)
        out = torch.cat(need_concat, dim=1) 
        del need_concat
        return out


class GCNPhasePredict(nn.Module):
    def __init__(self,output_dim,**kwargs):
        super().__init__()
        self.gcn_embedding=STFGNN(output_dim=output_dim,**kwargs)
        self.hidden=output_dim
        self.phase_predict = nn.Sequential(
            nn.Linear(self.hidden, 1),
            nn.Sigmoid()
        )
        self.vol_predict_fc = nn.Linear(self.hidden, 1)


    def gcn_flatten_data(self,states):
        st=[]
        nd=[]
        for state in states:
            st.extend([state["GCNinput"][:,i,...] for i in range(state["GCNinput"].shape[1])])
            nd.append(state["nodes_number"][0])
        nd=list(np.concatenate(nd))
        nd_st=dict(sorted(dict(zip(nd,st)).items()))
        st=np.stack(list(nd_st.values()),axis=1)
        st=np.transpose(st,(0,2,1,3))
        
        return st
        

    def forward(self, states):
        x=self.gcn_flatten_data(states)
        x=self.gcn_embedding(x)
        x=x.squeeze(1) 
        lemb=[torch.stack([x[:,i,:] for i in state["nodes_number"][0]],dim=1) for state in states]
        phase_p = self.phase_predict(x)
        phase_p=phase_p.squeeze(-1) 
        p_p=[torch.stack([phase_p[:,i] for i in state["nodes_number"][0]],dim=1) for state in states]
        
        return lemb, p_p


class GCNVolumePredict(nn.Module):
    def __init__(self, observation, NM_road_predict_hidden, 
                 NM_scale_by_lane_number, **kwargs):
        super().__init__()
        self.observation = observation
        self.hidden = NM_road_predict_hidden
        self.lane_scale = NM_scale_by_lane_number
        self.vol_predict_fc = nn.Linear(self.hidden, 1)

    def forward(self, lane_embedding, phase):
        res=[]
        for i in range(lane_embedding.shape[1]):
            a=lane_embedding[:,i,:].squeeze()
            b=phase[:,i].unsqueeze(-1)
            t=a*b
            res.append(self.vol_predict_fc(t))
        res=torch.stack(res,axis=1)
        return res # [30,12,1]



class LaneEmbedding(nn.Module):
    def __init__(self,  lane_embedding_size):
        super().__init__()
        self.layers = []
        if isinstance(lane_embedding_size, int):
            lane_embedding_size = [lane_embedding_size]
        self.lane_embedding_size = lane_embedding_size
        self.input = 14  # flow, green, predict, predict_valid_mask 10+1+1=12
        last_input = self.input
        for hidden in lane_embedding_size:
            self.layers += [nn.Linear(last_input, hidden), nn.ReLU()]
            last_input = hidden
        self.layers = nn.Sequential(*self.layers)

    def forward(self, states):
        return self.layers(states)



class GCNNewModel(NewModel):  
    def __init__(self, dqn_hidden_size, lane_embedding_size, lane_embedding_instance = None, **kwargs):
        super().__init__(dqn_hidden_size, lane_embedding_size, lane_embedding_instance = None, **kwargs)
        self.weights = [5, 1]
        self.fc = nn.Sequential(nn.Linear(lane_embedding_size * 2 + 1,dqn_hidden_size),nn.ReLU())
        self.gcn_phase_predict = GCNPhasePredict(**kwargs)  
        self.gcn_volume_predict = GCNVolumePredict(**kwargs)
        if lane_embedding_instance is not None:
            self.lane_embedding = lane_embedding_instance
        else:
            self.lane_embedding = LaneEmbedding(lane_embedding_size)


    def forward(self, state):
        flow = torch.tensor(state['GCNflow']).to(torch.float32)
        green = torch.tensor(state['TSgreen']).to(torch.float32).unsqueeze(-1)
        phase = torch.IntTensor(state['TSphase']).to(torch.int64)
        predict = state['predict']
        phase_s = nn.functional.one_hot((phase), num_classes = len(self.phases))
        phase_s = phase_s.float().unsqueeze(-1)  # [BATCH, PHASE, 1]
        x=torch.cat((flow,green,predict),dim=-1) 
        x = self.lane_embedding(x)  # [BATCH, LANE, HIDDEN]

        res = []
        for i, [phase, not_phase] in enumerate(zip(self.phases, self.not_phases)):
            res.append(torch.cat((
                x[:, phase].mean(dim = 1) * self.weights[0],
                x[:, not_phase].mean(dim = 1) * self.weights[1], 
                phase_s[:, i]), dim = -1))
        res = torch.stack(res, dim = 1)  # [BATCH, PHASE, HIDDEN]
        res = self.fc(res)
        
        return res




    



