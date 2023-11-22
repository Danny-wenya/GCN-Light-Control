from .AgentBase import *
from .NewModelAgent import NewModelAgent

class GCNNewModelAgent(NewModelAgent):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def gcn_forward(self):  # 在此处的forward直接调用STFGNN的foward，不需要理会原程序的操作。注意设计两个loss。phase_loss和volume_loss
        if isinstance(self.selected_model, torch.nn.ModuleList):
            return self.gcn_f_d
        return self.gcn_f_s
    

    def phase2lane(self,real_p):
        res=[]
        for i in real_p:
            phase=[0 for _ in range(len(self.observation["LaneCount"]))]
            for j in self.observation["TSphase"][i]:
                phase[j]=1
            res.append(torch.tensor(phase))
        res=torch.stack(res,dim=0)
        return res
        

    def gcn_f_s(self, states):
        l_emb, p_p = self.selected_model.inner.gcn_phase_predict(states)
        res = []
        v_p = []
        v_p_realp = []
        for i in self.indices:
            predict=self.selected_model.inner.gcn_volume_predict(l_emb[i], p_p[i])
            v_p.append(predict) 
            states[i]["predict"]=predict
            if 'NextPhase' in states[i]: 
                real_p=self.phase2lane(states[i]["NextPhase"])
                real_predict=self.selected_model.inner.gcn_volume_predict(l_emb[i], real_p)
                v_p_realp.append(real_predict)
                states[i]["predict_realp"]=real_predict
        
        for i in self.indices:
            res.append(self.selected_model.forward(states[i]))
            self.gather_phase_loss(res[-1], p_p[i])
            if 'NextPhase' in states[i] and self.phase_loss_with_replay:
                phase_to_prob = torch.nn.functional.one_hot(torch.IntTensor(states[i]['NextPhase']).to(torch.int64), res[-1].shape[1])
                self.gather_phase_loss(phase_to_prob, p_p[i])
                self.gcn_gather_volume_loss(states[i],'predict_realp')
        return res


    def gcn_f_d(self, states):
        l_emb_0, p_p_0 = self.selected_model[0].inner.gcn_phase_predict(states)
        l_emb_1, p_p_1 = self.selected_model[1].inner.gcn_phase_predict(states)
        assert 'NextInLaneVehicleNumber' not in states[0]
        res=[]
        v_p_0 = []
        for i in self.indices:
            p_0=self.selected_model[0].inner.gcn_volume_predict(l_emb_0[i], p_p_0[i])
            v_p_0.append(p_0)
            states[i]["predict_0"]=p_0
        
        for i in self.indices:
            states[i]['predict'] = states[i]['predict_0']
            forward_res = self.selected_model[0].forward(states[i]) # 预测Q值
            res.append([forward_res])

        v_p_1 = []
        for i in self.indices:
            p_1=self.selected_model[1].inner.gcn_volume_predict(l_emb_1[i], p_p_1[i])
            v_p_1.append(p_1)
            states[i]["predict_1"]=p_1
            
        for num, i in enumerate(self.indices):
            states[i]['predict'] = states[i]['predict_1']
            forward_res = self.selected_model[1].forward(states[i])
            res[num].append(forward_res)
            res[num] = torch.stack(res[num], dim = 1)
            self.gather_phase_loss(forward_res, p_p_1[i])
        return res
    

    def gcn_gather_volume_loss(self, state, predict_key = 'predict'):
        # ri2rl = torch.tensor(road_relation['RoadIn2RoadLink'])
        # real_in = torch.tensor([x[0] >= 0 for x in road_relation['RoadsIn']])
        # ri2rl[~real_in, :] = -1
        # if not self.volume_loss_weight:
        #     return
        # NILVN = 'NextInLaneVehicleNumber'
        # if NILVN not in state or predict_key not in state:
        #     return
        # ri2rl = ri2rl.reshape(-1)
        # mask = ri2rl >= 0

        predict = state[predict_key]
        real = torch.tensor(state["GCNflow"][...,-1]).to(torch.float32).unsqueeze(-1)
        # calc_predict = predict[:, ri2rl[mask]]
        # calc_real = real.reshape(real.shape[0], -1)[:, mask]

        # emb_predict = predict[:, ri2rl[~mask]]
        # emb_real = real.reshape(real.shape[0], -1)[:, ~mask]
        
        # assert calc_predict.requires_grad or calc_predict.sum() == 0
        # assert emb_predict.requires_grad or emb_predict.sum() == 0
        self.volume_loss.append(self.volume_loss_func(predict, real))
        # if (len(emb_real.reshape(-1)) != 0):
        #     self.emb_loss.append(self.volume_loss_func(emb_predict, emb_real))
