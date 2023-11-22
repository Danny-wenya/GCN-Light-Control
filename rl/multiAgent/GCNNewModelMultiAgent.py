from .MultiAgentBase import *
from .MAL import MAL
from .MALShare import MALShare
from .NewModelMultiAgent import NewModelMultiAgent
from torch import nn
from utils.utils import *


class GCNNewModelMultiAgent(NewModelMultiAgent):
    def __init__(self, action,**kwargs):
        super().__init__(action=action,**kwargs)
        self.eps_flag=False
        self.action_space=action

    
    def forward(self, state,**kwargs):  
        for agent in self.agents:
            agent.forward_model_select(**kwargs) 
        data = state
        funcs = self.agent.gcn_forward() 
        data = funcs(data)   
        return data


    def flatten_one_sample(self,sample_raw):
        sample_raw
        sample=[gcn_input(sample_raw[0])]
        sample=list(zip(*sample))
        sample=list(sample)
        for num in range(len(self.agents)):
            sample[num]=self.flatten_data("dict",sample[num])   # GCNflow [1,1,12,10]
        return sample


    def action(self, states,eps = 0, model_name = 'update', **kwargs):
        if self.eps_flag:
            with torch.no_grad():
                st=self.flatten_one_sample(states)
                state_action = self.forward(state=st, model_name = model_name, **kwargs) 
                res = self.agent.get_action(torch.stack(state_action), eps) 
        else:
            res=[]
            Batch=len(states)
            Agents=len(states[0])
            for i in range(Agents):
                ag=[]
                for _ in range(Batch):
                    ag.append(np.array([np.random.randint(0,self.action_space[i][0])]))  
                ag=np.array(ag)    
                res.append(ag)  
            res=np.array(res)
        res = list(zip(*res))  
        return res
    

    def get_action_in_update_policy(self, samples_raw):
        # samples_raw: [BATCH, SAMPLETYPE=5, AGENT, ...data]
        samples = [x[:-1] + [[x[-1]] * len(self.agents)] for x in samples_raw]
        samples = list(zip(*samples))  # [S_TYPE, BATCH, AGENT, samples...data]
        state, action, reward, next_s, ist = [list(zip(*x)) for x in samples]
        # every data: [AGENT, BATCH, ...data]
        
        state = list(state)
        next_s = list(next_s)
        for num in range(len(self.agents)):
            # flatten inside every agent
            state[num] = self.flatten_data('dict', state[num])
            next_s[num] = self.flatten_data('dict', next_s[num])
            
            state[num]['NextInLaneVehicleNumber'] = \
                next_s[num]['InLaneVehicleNumber'][:, -self.N_STEPS]
            state[num]['NextPhase'] = \
                next_s[num]['TSprevphases'][:, -self.N_STEPS]
        # pdb.set_trace()
        state = self.forward(state,update_policy = 'state')
        next_s = self.forward(next_s,update_policy = 'next_s')
        # state: [AGENT, array(BATCH, ACTION)]

        samples = list(zip(state, action, reward, next_s, ist))
        # samples: [AGENT, S_TYPE, BATCH, ...data]
        return samples
    

    def flatten_data(self,datatype, data):
        if datatype == 'array':
            data = list(zip(*data))
            data = list(map(lambda x: np.stack(x), data))
            return data[0]
        elif datatype == 'dict':
            dic = data
            if len(dic) == 0:
                return {}
            res = {}
            for key in dic[0].keys():
                res[key] = np.stack([x[key] for x in dic])  # GCNflow [1,1,12,10]
            return res
        else:
            raise NotImplementedError('unknown flatten type ' + datatype)
        

    def update_policy(self, samples_raw, frame):
        samples = self.get_action_in_update_policy(samples_raw)
        # gather all sample as one and feed
        sample = [[] for _ in samples[0]]
        for s in samples:
            for sample_one, s_one in zip(sample, s):
                sample_one.extend(s_one)
        sample[0] = torch.stack(sample[0])
        sample[3] = torch.stack(sample[3])
    

        # update only agent
        L = self.agent.calculate_loss(sample, frame, 'loss')
        self._loss_backward(L)
        self.agent.optimizer_step()

    

    

        

    
    



















