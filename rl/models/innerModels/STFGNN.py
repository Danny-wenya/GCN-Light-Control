# from _typeshed import Self
import torch
import torch.nn.functional as F
import torch.nn as nn


class Scaler:
    def transform(self, data):
        raise NotImplementedError("Transform not implemented")

    def inverse_transform(self, data):
        raise NotImplementedError("Inverse_transform not implemented")


class StandardScaler(Scaler):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

class gcn_operation(nn.Module):
    def __init__(self, adj, in_dim, out_dim, num_vertices, activation='GLU'):
        """
        图卷积模块
        :param adj: 邻接图
        :param in_dim: 输入维度
        :param out_dim: 输出维度
        :param num_vertices: 节点数量
        :param activation: 激活方式 {'relu', 'GLU'}
        """
        super(gcn_operation, self).__init__()
        self.adj = adj
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
        """
        :param x: (3*N, B, Cin)
        :param mask:(3*N, 3*N)
        :return: (3*N, B, Cout)
        """
        adj = self.adj
        if mask is not None:
            adj = adj.to(mask.device) * mask #不mask

        # 此处将邻接矩阵和输入x做矩阵乘法
        x = torch.einsum('nm, mbc->nbc', adj.to(x.device), x)  # 4*N, B, Cin  爱因斯坦求和约定的操作，可以执行多维线性代数数组操作

        if self.activation == 'GLU':
            lhs_rhs = self.FC(x)  # 4*N, B, 2*Cout  将结果通过一个线性层，将最后一维的长度映射成原来的两倍
            lhs, rhs = torch.split(lhs_rhs, self.out_dim, dim=-1)  # 4*N, B, Cout # 平均分割最后一个维度

            out = lhs * torch.sigmoid(rhs)  # 将两个分割的结果做点乘  实际上这是一个GLU操作
            del lhs, rhs, lhs_rhs

            return out 

        elif self.activation == 'relu':
            return torch.relu(self.FC(x))  # 3*N, B, Cout


class STSGCM(nn.Module):
    def __init__(self, adj, in_dim, out_dims, num_of_vertices, activation='GLU'):
        """
        :param adj: 邻接矩阵
        :param in_dim: 输入维度
        :param out_dims: list 各个图卷积的输出维度
        :param num_of_vertices: 节点数量
        :param activation: 激活方式 {'relu', 'GLU'}
        """
        super(STSGCM, self).__init__()
        self.adj = adj
        self.in_dim = in_dim
        self.out_dims = out_dims
        self.num_of_vertices = num_of_vertices
        self.activation = activation

        self.gcn_operations = nn.ModuleList()

        self.gcn_operations.append(
            gcn_operation(
                adj=self.adj,
                in_dim=self.in_dim,
                out_dim=self.out_dims[0],
                num_vertices=self.num_of_vertices,
                activation=self.activation
            )
        )

        for i in range(1, len(self.out_dims)):
            self.gcn_operations.append(
                gcn_operation(
                    adj=self.adj,
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

        # shape of each element is (1, N, B, Cout)   通过邻接矩阵计算出来的数据维度是（4N, B, Cout），只需要截取代表路口自身数据的临界矩阵。
        need_concat = [
            torch.unsqueeze(
                h[self.num_of_vertices: 2 * self.num_of_vertices], dim=0
            ) for h in need_concat]
        out = torch.max(torch.cat(need_concat, dim=0), dim=0).values  # (N, B, Cout)
        del need_concat

        return out


class STSGCL(nn.Module):
    def __init__(self,
                 adj,
                 history,
                 num_of_vertices,
                 in_dim,
                 out_dims,
                 strides=4,
                 activation='GLU',
                 temporal_emb=True,
                 spatial_emb=True):
        """
        :param adj: 邻接矩阵
        :param history: 输入时间步长
        :param in_dim: 输入维度
        :param out_dims: list 各个图卷积的输出维度
        :param strides: 滑动窗口步长，local时空图使用几个时间步构建的，默认为3
        :param num_of_vertices: 节点数量
        :param activation: 激活方式 {'relu', 'GLU'}
        :param temporal_emb: 加入时间位置嵌入向量
        :param spatial_emb: 加入空间位置嵌入向量
        """
        super(STSGCL, self).__init__()
        self.adj = adj
        self.strides = strides
        self.history = history
        self.in_dim = in_dim
        self.out_dims = out_dims
        self.num_of_vertices = num_of_vertices

        self.activation = activation
        self.temporal_emb = temporal_emb
        self.spatial_emb = spatial_emb


        self.conv1 = nn.Conv2d(self.in_dim, self.out_dims[-1], kernel_size=(1, 2), stride=(1, 1), dilation=(1, 3))
        self.conv2 = nn.Conv2d(self.in_dim, self.out_dims[-1], kernel_size=(1, 2), stride=(1, 1), dilation=(1, 3))


        self.STSGCMS = nn.ModuleList()
        for i in range(self.history - self.strides + 1):
            self.STSGCMS.append(
                STSGCM(
                    adj=self.adj,
                    in_dim=self.in_dim,
                    out_dims=self.out_dims,
                    num_of_vertices=self.num_of_vertices,
                    activation=self.activation
                )
            )

        if self.temporal_emb:
            self.temporal_embedding = nn.Parameter(torch.FloatTensor(1, self.history, 1, self.in_dim))
            # 1, T, 1, Cin

        if self.spatial_emb:
            self.spatial_embedding = nn.Parameter(torch.FloatTensor(1, 1, self.num_of_vertices, self.in_dim))
            # 1, 1, N, Cin

        self.reset()

    def reset(self):
        if self.temporal_emb:
            nn.init.xavier_normal_(self.temporal_embedding, gain=0.0003)

        if self.spatial_emb:
            nn.init.xavier_normal_(self.spatial_embedding, gain=0.0003)

    def forward(self, x, mask=None):
        """
        :param x: B, T, N, Cin
        :param mask: (N, N)
        :return: B, T-3, N, Cout
        """
        if self.temporal_emb:
            x = x + self.temporal_embedding  # self.temporal_embedding维度（1, T, 1, Cin），x维度（B, T, N, Cin）。相当于给x添加了一个可求导的偏置，数据广播后x的B，N两个维度共享偏置，表达了时间信息

        if self.spatial_emb:
            x = x + self.spatial_embedding  # self.spatial_embedding维度（1, 1, N, Cin），x维度（B, T, N, Cin）。相当于给x添加了一个可求导的偏置，数据广播后x的B，T两个维度共享偏置，表达了空间信息


        #############################################
        # shape is (B, C, N, T)
        data_temp = x.permute(0, 3, 2, 1) #将编码维度前置，将时间维度后置，在编码维度进行一维卷积操作，一维卷积对最后一个维度操作，
        data_left = torch.sigmoid(self.conv1(data_temp)) #一维卷积核维度[1,2]，步幅[1,1]，膨胀[1,3]，膨胀后的卷积核为[a，0，0，b]。最后使得输出序列长度-3.
        data_right = torch.tanh(self.conv2(data_temp))  # 将数据做两次一维卷积计算，生成两个新数据，在将两份维度一样的数据相乘（对应位置相乘）
        data_time_axis = data_left * data_right
        data_res = data_time_axis.permute(0, 3, 2, 1)  #次操作的作用相当于将每个路口的每个时间步的数据与其后第3个时间步的数据做了加权聚合，以表示时间序列关系
        # shape is (B, T-3, N, C)   #最后T-3个时间步的数据就能表达T个时间步的数据
        #############################################

        need_concat = []
        batch_size = x.shape[0]

        # 与以上卷积操作类似，再对时间序列每4个数据进行聚合操作，但此次是调用STSGCM进行操作图卷积操作
        for i in range(self.history - self.strides + 1):  #12-4+1,表示对于12个时间步的数据，每4个时间步进行一次聚合计算。
            t = x[:, i: i+self.strides, :, :]  # (B, 4, N, Cin) 截取4个时间步

            t = torch.reshape(t, shape=[batch_size, self.strides * self.num_of_vertices, self.in_dim])
            # (B, 4*N, Cin) # 将两个维度合并，实现数据拉平，第一个维度变为[N,N,N,N]4个N路口的平铺。以进行下一步的adj_mx计算。

            t = self.STSGCMS[i](t.permute(1, 0, 2), mask)  # (4*N, B, Cin) -> (N, B, Cout) # 每4个时间步都有自己独立的STSGCM模型。

            t = torch.unsqueeze(t.permute(1, 0, 2), dim=1)  # (N, B, Cout) -> (B, N, Cout) ->(B, 1, N, Cout) 

            need_concat.append(t)

        mid_out = torch.cat(need_concat, dim=1)  # (B, T-3, N, Cout)
        out = mid_out + data_res #将STSGCM的计算结果与conv1d的计算结果相加（对应位置相加）

        del need_concat, batch_size
 
        return out #得到最后输出结果


class output_layer(nn.Module):
    def __init__(self, num_of_vertices, history, in_dim, out_dim, 
                 hidden_dim=128, horizon=12):
        """
        预测层，注意在作者的实验中是对每一个预测时间step做处理的，也即他会令horizon=1
        :param num_of_vertices:节点数
        :param history:输入时间步长
        :param in_dim: 输入维度
        :param hidden_dim:中间层维度
        :param horizon:预测时间步长
        """
        super(output_layer, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.history = history
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.horizon = horizon

        #print("#####################")
        #print(self.in_dim)
        #print(self.history)
        #print(self.hidden_dim)

        self.FC1 = nn.Linear(self.in_dim * self.history, self.hidden_dim, bias=True)

        #self.FC2 = nn.Linear(self.hidden_dim, self.horizon , bias=True)

        self.FC2 = nn.Linear(self.hidden_dim, self.horizon * self.out_dim, bias=True)
        self.FC3 = nn.Linear(self.hidden_dim, self.horizon * self.out_dim, bias=True)

    def forward(self, x):
        """
        :param x: (B, Tin, N, Cin)
        :return: (B, Tout, N)
        """
        batch_size = x.shape[0]

        x = x.permute(0, 2, 1, 3)  # B, N, Tin, Cin

        out1 = torch.relu(self.FC1 (x.reshape(batch_size, self.num_of_vertices, -1)))
        # (B, N, Tin, Cin) -> (B, N, Tin * Cin) -> (B, N, hidden)

        out2 = self.FC2(out1)  # (B, N, hidden) -> (B, N, horizon * 2)
        out2 = out2.reshape(batch_size, self.num_of_vertices, self.horizon, self.out_dim).permute(0, 2, 1, 3)

        out3 = self.FC3(out1)
        out3 =out3.reshape(batch_size, self.num_of_vertices, self.horizon, self.out_dim).permute(0, 2, 1, 3)
        out3 = torch.cat([torch.softmax(out3[...,i*12:(i+1)*12,:],dim=-2) for i in range(6)],dim=-2)

        del out1, batch_size

        return out2,out3


class STFGNN(nn.Module):
    def __init__(self, window,num_nodes,input_dim,output_dim,hidden_dims,first_layer_embedding_size,
                 out_layer_dim,activation,mask,temporal_emb,spatial_emb,horizon,strides,adj,**kwargs):
        super(STFGNN, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mean=torch.tensor(0.6049921237749322).to(device)
        self.std=torch.tensor(1.7494956921251785).to(device)
        self.scaler =  StandardScaler(mean=self.mean,std=self.std)
        self.window=window
        self.num_nodes=num_nodes
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.hidden_dims=[[16,16,16],[16,16,16],[16,16,16]]
        self.first_layer_embedding_size=first_layer_embedding_size
        self.out_layer_dim=out_layer_dim
        self.activation=activation
        self.mask=mask
        self.temporal_emb=temporal_emb
        self.spatial_emb=spatial_emb
        self.strides=strides
        self.adj=torch.tensor(adj,dtype=torch.float)
        self.horizon=horizon
        self.First_FC = nn.Linear(input_dim, first_layer_embedding_size, bias=True)
        self.STSGCLS = nn.ModuleList()
        self.STSGCLS.append(
            STSGCL(
                adj=self.adj,
                history=self.window,
                num_of_vertices=self.num_nodes,
                in_dim=self.first_layer_embedding_size,
                out_dims=self.hidden_dims[0],
                strides=self.strides,
                activation=self.activation,
                temporal_emb=self.temporal_emb,
                spatial_emb=self.spatial_emb
            )
        )

        in_dim = self.hidden_dims[0][-1]
        self.window -= (self.strides - 1)

        for idx, hidden_list in enumerate(self.hidden_dims):  
            if idx == 0:
                continue
            #print("---------", idx)
            self.STSGCLS.append(
                STSGCL(
                    adj=self.adj,
                    history=self.window,
                    num_of_vertices=self.num_nodes,
                    in_dim=in_dim,
                    out_dims=hidden_list,
                    strides=self.strides,
                    activation=self.activation,
                    temporal_emb=self.temporal_emb,
                    spatial_emb=self.spatial_emb
                )
            )
            self.window -= (self.strides - 1)
            in_dim = hidden_list[-1]

        self.predictLayer = nn.ModuleList()
        #print("***********************")
        #print(history)
        for t in range(self.horizon):
            self.predictLayer.append(
                output_layer(
                    num_of_vertices=self.num_nodes,
                    history=self.window,
                    in_dim=in_dim,
                    out_dim = output_dim,
                    hidden_dim=out_layer_dim,
                    horizon=1
                )
            )

        if self.mask:
            mask = torch.zeros_like(self.adj)
            mask[self.adj != 0] = self.adj[self.adj != 0]
            self.mask = nn.Parameter(mask)
        else:
            self.mask = None

    def forward(self, x):
        # x=self.scaler.transform(x)
        x = torch.relu(self.First_FC(x))  
        for model in self.STSGCLS:
            x = model(x, self.mask)
        need_concat_flow = []
        need_concat_phase = []
        for i in range(self.horizon): 
            out_flow,out_phase = self.predictLayer[i](x)  
            need_concat_flow.append(out_flow)
            need_concat_phase.append(out_phase)
        out1 = torch.cat(need_concat_flow, dim=1)
        # out1=self.scaler.inverse_transform(out1)
        out2=torch.cat(need_concat_phase, dim=1)  
        del need_concat_flow,need_concat_phase

        return out1,out2



