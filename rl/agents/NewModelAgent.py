from .AgentBase import *
from .DQNAgent import DQNAgent


class NewModelAgent(DQNAgent):
    def __init__(self, NM_lane_embedding_size, NM_phase_loss_weight, 
                 NM_volume_loss_weight, NM_phase_loss_with_replay,
                 out_road_embeddings, road_relation, **kwargs):
        self.hidden_size = kwargs['dqn_hidden_size']
        self.lane_embedding_size = NM_lane_embedding_size
        self.phase_loss_weight = NM_phase_loss_weight
        self.volume_loss_weight = NM_volume_loss_weight
        self.road_relation = road_relation
        self.phase_loss_with_replay = NM_phase_loss_with_replay
        self.phase_loss = []
        self.volume_loss = []
        # self.emb_loss = []
        super().__init__(lane_embedding_size = self.lane_embedding_size,
                         **kwargs)
        self.phase_loss_func = torch.nn.BCELoss()
        self.volume_loss_func = torch.nn.MSELoss()

        # self.out_road_embedding = out_road_embeddings[0]
        self.opt = torch.optim.Adam(self.parameters(), self.LR)

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
            return [self.f_d]
        return [self.f_s]

    def state2tensor(self, states):
        res = []
        for state in states:
            res.append(self.model_update.inner.state2tensor(state))
        return res

    
    def gather_phase_loss(self, res, p_p):
        if not self.phase_loss_weight:
            return
        if not p_p.requires_grad:  # no grad no loss
            return
        _, maxid = res.max(dim = -1)
        laneid=torch.zeros_like(p_p.squeeze(),dtype=torch.float)
        for i,j in enumerate(maxid):
            laneid[i,self.observation['TSphase'][j]]=1
        self.phase_loss.append(self.phase_loss_func(p_p.squeeze(), laneid))

    def gather_volume_loss(self, state, predict_key = 'predict'):
        if not self.volume_loss_weight:
            return
        NILF = 'NextInlaneflow'
        if NILF not in state or predict_key not in state:
            return
        predict = state[predict_key]
        real = state[NILF]
        self.volume_loss.append(self.volume_loss_func(predict, real))

    def state2tensor(self, states):  #在此处处理好STFGNN需要的输入数据
        res = []
        for state in states:
            res.append(self.model_update.inner.state2tensor(state))
        tensor_res=torch.cat([res[i]['Inlaneflow'] for i in self.indices],axis=2)
        return res,tensor_res


    def f_s(self, states):   
        '''在此处接入STFGNN,STFGNN输入数据格式[batch_size,n_steps,n_nodes,n_fea]  
        '''
        states,tensor_states=self.state2tensor(states)
        res = []
        vp,pp=self.selected_model.inner.volume_phase_predict(tensor_states)
        v_p=[vp[...,i*12:(i+1)*12,0].squeeze(1) for i in self.indices]
        p_p=[pp[...,i*12:(i+1)*12,0].squeeze(1) for i in self.indices]

        for i in self.indices:
            states[i]['predict']=v_p[i]
            res.append(self.selected_model.forward(states[i])) # Q值预测，Q值预测使用了当前路口的多个状态       
            self.gather_phase_loss(res[-1], p_p[i])  # 使用Q值计算相位预测损失
            
            if 'NextPhase' in states[i] and self.phase_loss_with_replay: #训练模式
                phase_to_prob = torch.nn.functional.one_hot(
                    states[i]['NextPhase'], 
                    res[-1].shape[1])
                self.gather_phase_loss(phase_to_prob, p_p[i]) # 使用真实值计算相位预测损失 

            self.gather_volume_loss(states[i])  # 计算流量预测损失，流量预测损失按照GCN的计算方式就可以，不必按照他的方式
        return res

    def f_d(self, states):
        res = []
        states,tensor_states=self.state2tensor(states)
        vp0,pp0=self.selected_model[0].inner.volume_phase_predict(tensor_states)
        vp1,pp1=self.selected_model[0].inner.volume_phase_predict(tensor_states)
        v_p0=[vp0[...,i*12:(i+1)*12,0].squeeze(1) for i in self.indices]
        p_p0=[pp0[...,i*12:(i+1)*12,0].squeeze(1) for i in self.indices]
        v_p1=[vp1[...,i*12:(i+1)*12,0].squeeze(1) for i in self.indices]
        p_p1=[pp1[...,i*12:(i+1)*12,0].squeeze(1) for i in self.indices]

        for i in self.indices:
            states[i]['predict_0']=v_p0[i]
            states[i]['predict_1']=v_p1[i]

        for i in self.indices:
            states[i]['predict'] = states[i]['predict_0']
            forward_res = self.selected_model[0].forward(states[i])
            res.append([forward_res])

        for num, i in enumerate(self.indices):
            states[i]['predict'] = states[i]['predict_1']
            forward_res = self.selected_model[1].forward(states[i])
            res[num].append(forward_res)
            res[num] = torch.stack(res[num], dim = -1)
            self.gather_phase_loss(forward_res, p_p1[i])

        return res


    def calculate_loss(self, samples, frame, txsw_name, **kwargs):
        L = super().calculate_loss(samples, frame, txsw_name, **kwargs)
        L_phase = -1  # use MSE, so never be negative
        L_volume = -1
        if len(self.phase_loss) > 0:
            L_phase = torch.stack(self.phase_loss).mean()
            L += L_phase
            self.phase_loss = []
        if len(self.volume_loss) > 0:
            L_volume = torch.stack(self.volume_loss).mean()
            L += L_volume
            self.volume_loss = []
        if txsw_name != '':
            if L_phase != -1:
                self.TXSW.add_scalar(txsw_name + '_p', L_phase.item(), frame)
            if L_volume != -1:
                self.TXSW.add_scalar(txsw_name + '_v', L_volume.item(), frame)
            # if L_emb != -1:
            #     self.TXSW.add_scalar(txsw_name + '_v_out', L_emb.item(), frame)
        L_phase = 'NaN' if L_phase == -1 else ('%.5f' % L_phase)
        L_volume = 'NaN' if L_volume == -1 else ('%.5f' % L_volume)
        log('phase loss: %9s, volume loss: %9s' % (L_phase, L_volume),
            level = 'TRACE')
        return L
