from .AgentBase import *
from .DQNAgent import DQNAgent



class NewModelAgent(DQNAgent):
    def __init__(self, NM_lane_embedding_size, NM_phase_loss_weight, 
                 NM_volume_loss_weight, NM_phase_loss_with_replay,
                 out_road_embeddings, road_relation, NM_scale_by_lane_number,metrics,**kwargs):
        self.hidden_size = kwargs['dqn_hidden_size']
        self.lane_embedding_size = NM_lane_embedding_size
        self.phase_loss_weight = NM_phase_loss_weight
        self.volume_loss_weight = NM_volume_loss_weight
        self.lane_scale = NM_scale_by_lane_number
        self.road_relation = road_relation
        self.phase_loss_with_replay = NM_phase_loss_with_replay
        self.metrics=metrics
        
        self.phase_loss = []
        self.volume_loss = []
        self.emb_loss = []
        self.direction_type = 3
        super().__init__(lane_embedding_size = self.lane_embedding_size,metrics=metrics,
                         **kwargs)
        self.phaseid2lanes = self.observation['phaseid2lanes']
        self.phase_loss_func = torch.nn.BCELoss()
        if metrics=='mse':
            self.volume_loss_func = torch.nn.MSELoss()
        elif metrics=='mape':
            self.volume_loss_func = self.MAPE

        self.out_road_embedding = out_road_embeddings[0]
        self.opt = torch.optim.Adam(self.parameters(), self.LR)
        self._init_adj()
    

    def _init_adj(self):
        # 将所有路口的roalinks拉平排列,创建邻接矩阵
        N=12*len(self.road_relation)+1
        adj=torch.zeros([N,N],dtype=torch.float)
        edge_index=dict(zip(range(N),[[72,72,72] for _ in range(N)]))
        for num, rel in enumerate(self.road_relation):
            for ri, ri2rl in zip(rel['RoadsIn'], rel['RoadIn2RoadLink']):
                from_i, from_r = ri
                if from_i>=0:
                    up_rl=self.road_relation[from_i]['RoadOut2RoadLink'][from_r]
                    adj[[num*12+i for i in ri2rl],up_rl]=1
                    for i in ri2rl:
                        edge_index[int(num*12+i)]=up_rl
        self.adj=adj
        self.edge_index=edge_index


    def named_modules(self, memo = None, prefix = ''):
        update = super().named_modules(memo=memo, prefix=prefix)
        for i in update:
            yield i
        try:
            emb = self.out_road_embedding.named_modules(
                memo=memo,
                prefix='.out_road_embedding')
            for i in emb:
                yield i
        except:
            pass

    def embedding_instance(self, instance = None):
        if instance is not None:
            self.model_update.inner.replace_lane_embedding(instance[0])
            self.model_old.inner.replace_lane_embedding(instance[1])
            self.BEST_MODEL.inner.replace_lane_embedding(instance[2])
            self.opt = torch.optim.Adam(self.parameters(), self.LR)
        return [self.model_update.inner.lane_embedding,
                self.model_old.inner.lane_embedding,
                self.BEST_MODEL.inner.lane_embedding]

    def forward(self):
        if isinstance(self.selected_model, torch.nn.ModuleList):
            return [self.state2tensor, self.phase_p_d, self.f_d]
        return [self.state2tensor, self.phase_p_s, self.f_s]

    def state2tensor(self, states):
        res = []
        for state in states:
            res.append(self.model_update.inner.state2tensor(state))
        return res


    def arrange_volume_prediction(self, states, v_p, key = 'predict'):
        """get predict for roadin from calculated roadout or embedding
        """
        v_p_list=[v_p[:,i*12:(i+1)*12].squeeze(-1) for i in range(len(self.road_relation))]
        for num, [state,vp, rel] in enumerate(zip(states, v_p_list,self.road_relation)):
            for ri, ri2rl in zip(rel['RoadsIn'], rel['RoadIn2RoadLink']):
                from_i, from_r = ri
                if from_i < 0:
                    # virtual intersections
                    vid = -1 - from_i
                    emb = self.out_road_embedding
                    emb_res =cuda(emb.forward(torch.tensor([vid]* len(state['Envtime'])), state['Envtime'])).float()
                    vp[:, ri2rl] = emb_res
            state[key] = vp

        return states

    # 进行程序改造。寻找上游节点
    def volume_predict(self,lane_embedding, phase,model_index=0):
        node_fea=[]
        for lane_emb, phase, road_relation in zip(lane_embedding, phase, self.road_relation):
            road2rl = road_relation['RoadOut2RoadLink'] 
            roadwidths = road_relation['RoadOutLanes'] 
            lane_emb = lane_emb * phase  
            
            for rls, rwidth in zip(road2rl, roadwidths):
                assert len(rwidth) == self.direction_type
                if self.lane_scale:
                    lane_emb[:, rls] = lane_emb[:, rls] / (rwidth 
                                        + 1e-6).unsqueeze(0).unsqueeze(-1)
            node_fea.append(lane_emb)
        node_fea = torch.cat(node_fea, dim = 1)  
        batch_size=node_fea.shape[0]
        adj=self.adj.unsqueeze(0).repeat(batch_size,1,1)
        if isinstance(self.selected_model, torch.nn.ModuleList):
            v_p=self.selected_model[model_index].inner.volume_predict(node_fea,adj,self.edge_index) 
        else:
            v_p=self.selected_model.inner.volume_predict(node_fea,adj,self.edge_index) 

        return v_p
       

    def gather_phase_loss(self, res, p_p):
        if not self.phase_loss_weight:
            return
        if not p_p.requires_grad:  # no grad no loss
            return
        _, maxid = res.max(dim = -1)
        laneid = self.phaseid2lanes[maxid].unsqueeze(-1)  # [B, L, 1]
        self.phase_loss.append(self.phase_loss_func(p_p, laneid))

    def gather_volume_loss(self, state, road_relation, predict_key = 'predict'):
        ri2rl = torch.tensor(road_relation['RoadIn2RoadLink'])
        real_in = torch.tensor([x[0] >= 0 for x in road_relation['RoadsIn']])
        ri2rl[~real_in, :] = -1
        if not self.volume_loss_weight:
            return
        NILVN = 'NextInLaneVehicleNumber'
        if NILVN not in state or predict_key not in state:
            return
        ri2rl = ri2rl.reshape(-1)
        mask = ri2rl >= 0
        predict = state[predict_key]
        real = state[NILVN]
        calc_predict = predict[:, ri2rl[mask]]
        calc_real = real.reshape(real.shape[0], -1)[:, mask]
        emb_predict = predict[:, ri2rl[~mask]]
        emb_real = real.reshape(real.shape[0], -1)[:, ~mask]
        assert calc_predict.requires_grad or calc_predict.sum() == 0
        assert emb_predict.requires_grad or emb_predict.sum() == 0
      
        self.volume_loss.append(self.volume_loss_func(calc_predict, calc_real))
        if (len(emb_real.reshape(-1)) != 0):
            self.emb_loss.append(self.volume_loss_func(emb_predict, emb_real))

    def phase_p_s(self, states):
        res = []
        for i in self.indices:
            l_emb, p_p = self.selected_model.inner.phase_predict(
                states[i], 
                self.road_relation[i])
            res.append([states[i], l_emb, p_p])
        return res

    def f_s(self, states):
        states, l_emb, p_p = zip(*states)
        v_p=self.volume_predict(l_emb, p_p)[:,:-1,:]
        states =self.arrange_volume_prediction(states, v_p)
        if 'NextPhase' in states[0]:
            realp = tuple(self.phaseid2lanes[states[i]['NextPhase']].unsqueeze(-1) for i in self.indices)
            v_p_realp=self.volume_predict(l_emb, realp)
            states = self.arrange_volume_prediction(states, 
                                                    v_p_realp, 
                                                    'predict_realp')

        res=[]
        for i in self.indices:
            res.append(self.selected_model.forward(states[i]))
            self.gather_phase_loss(res[-1], p_p[i])
            if 'NextPhase' in states[i] and self.phase_loss_with_replay:
                phase_to_prob = torch.nn.functional.one_hot(
                    states[i]['NextPhase'], 
                    res[-1].shape[1])
                self.gather_phase_loss(phase_to_prob, p_p[i])
            self.gather_volume_loss(states[i], 
                                    self.road_relation[i],
                                    'predict_realp')
        return res

    def phase_p_d(self, states):
        res = []
        for i in self.indices:
            l_emb_0, p_p_0 = self.selected_model[0].inner.phase_predict(
                states[i], self.road_relation[i])
            l_emb_1, p_p_1 = self.selected_model[1].inner.phase_predict(
                states[i], self.road_relation[i])
            res.append([states[i], l_emb_0, p_p_0, l_emb_1, p_p_1])
        return res

    def f_d(self, states):
        res = []
        states, l_emb_0, p_p_0, l_emb_1, p_p_1 = zip(*states)
        assert 'NextInLaneVehicleNumber' not in states[0]

        v_p_0=self.volume_predict(l_emb_0, p_p_0,model_index=0)
        states =self.arrange_volume_prediction(states, v_p_0,'predict_0')
        for i in self.indices:
            states[i]['predict'] = states[i]['predict_0']
            forward_res = self.selected_model[0].forward(states[i])
            res.append([forward_res])

        v_p_1=self.volume_predict(l_emb_1, p_p_1,model_index=1)
        states =self.arrange_volume_prediction(states, v_p_1,'predict_1')
        for num, i in enumerate(self.indices):
            states[i]['predict'] = states[i]['predict_1']
            forward_res = self.selected_model[1].forward(states[i])
            res[num].append(forward_res)
            res[num] = torch.stack(res[num], dim = 1)
            self.gather_phase_loss(forward_res, p_p_1[i])

        return res
    
    def calculate_loss(self, samples, frame, txsw_name, next_action,**kwargs):
        L_reward,mape = super().calculate_loss(samples, frame,txsw_name,self.metrics, **kwargs)
        L_phase = -1  # use MSE, so never be negative
        L_volume = -1
        L_emb = -1
        L=0
        L+=100*L_reward  # 强调Q的计算准确度最重要
        if len(self.phase_loss) > 0:
            L_phase = torch.stack(self.phase_loss).mean()
            L += L_phase
            self.phase_loss = []
        if len(self.emb_loss) > 0:
            self.volume_loss += self.emb_loss
            L_emb = torch.stack(self.emb_loss).mean()
            # print(self.out_road_embedding.embedding.weight.sum(1))
            self.emb_loss = []
        if len(self.volume_loss) > 0:
            L_volume = torch.stack(self.volume_loss).mean()
            L += 5*L_volume
            self.volume_loss = []
        if txsw_name != '':
            if L_phase != -1:
                self.TXSW.add_scalar(txsw_name + '_p', L_phase.item(), frame)
            if L_volume != -1:
                self.TXSW.add_scalar(txsw_name + '_v', L_volume.item(), frame)
            if L_emb != -1:
                self.TXSW.add_scalar(txsw_name + '_v_out', L_emb.item(), frame)
        L_phase = 'NaN' if L_phase == -1 else ('%.5f' % L_phase)
        L_volume = 'NaN' if L_volume == -1 else ('%.5f' % L_volume)
        if self.metrics=='mse':
            log('phase: %9s, volume: %9s, reward: %.5f, total: %.5f, action: %s' \
                % (L_phase, L_volume ,L_reward.item() ,L.item()  ,[i[0] for i in list(next_action[0])]),
                level = 'TRACE')
        elif self.metrics=='mape':
            log('phase loss: %9s, volume loss: %9s%%, reward loss: %.5f%%, action: %s' \
                % (L_phase, L_volume,L_reward.item(),[i[0] for i in list(next_action[0])]),
                level = 'TRACE')
        
        return L
