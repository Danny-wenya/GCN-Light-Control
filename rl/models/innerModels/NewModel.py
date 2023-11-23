from .InnerModelBase import *
from .STFGNN import STFGNN




class NewModel(InnerModelBase):
    def __init__(self, dqn_hidden_size, lane_embedding_size, observation,**kwargs):
        super().__init__()
        self.weights=[5,2]
        self.hidden = dqn_hidden_size
        self.observation = observation
        self.phases = observation['TSphase']
        self.not_phases=[list(set(range(12))-set(ph)) for ph in self.phases]
        self.fc = nn.Sequential(
            nn.Linear(lane_embedding_size * 2 + 1,
                      dqn_hidden_size),
            nn.ReLU()
        )
        kwargs['adj']=self.observation['adj_mx']
        kwargs['hidden_dims']=[[16,16,16], [16,16,16], [16,16,16]]
        self.volume_phase_predict=STFGNN(**kwargs)
        self.lane_embedding=nn.Sequential(nn.Linear(1+1+1, self.hidden),nn.ReLU())  #flow, green, predict

    @staticmethod
    def default_wrapper(dueling_dqn, **kwargs):
        if dueling_dqn:
            return 'DuelingSplitModel'
        else:
            return 'DQNSplitModel'


    def state2tensor(self,state):
        res = {}
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mean=torch.tensor(0.6049921237749322).to(device)
        TSphase=self.observation['TSphase']
        res['Inlaneflow']=torch.tensor(state['Inlaneflow'],dtype=torch.float)
        TSgreen=torch.zeros_like(res['Inlaneflow'])
        nrl=len(state['TSprevphases'])
        for i,sample in enumerate(state['TSprevphases']):
            for j,k in enumerate(sample):
                if k==-1:
                    idx=[1 for _ in range(nrl)]
                else:
                    idx=TSphase[k]
                TSgreen[i,j,idx]=1
        res['Inlaneflow']=torch.stack([res['Inlaneflow'],TSgreen],axis=-1).to(device)
        res['Onlaneflow']=torch.tensor(state['Onlaneflow'],dtype=torch.float).to(device)
        res['TSgreen']=torch.tensor(state['TSgreen'],dtype=torch.float).to(device)
        res['TSphase']=torch.tensor(state['TSphase']).to(device)
        if 'NextInlaneflow' in state:
            res['NextInlaneflow']=torch.tensor(state['NextInlaneflow'],dtype=torch.float).to(device)

        return res

    
    def forward(self, state):
        flow = state['Onlaneflow']  #换成当前路口的车辆数
        green = state['TSgreen'] 
        phase = state['TSphase']
        predict = state['predict'] 
        phase_s = nn.functional.one_hot((phase), num_classes = len(self.phases))
        phase_s = phase_s.float().unsqueeze(-1)  # [BATCH, PHASE, 1]

        x = torch.stack((flow, green, predict), dim = -1)
        x = self.lane_embedding(x) # [BATCH, LANE, HIDDEN]

        res = []
        for i, [phase, not_phase] in enumerate(zip(self.phases, 
                                                   self.not_phases)):
            res.append(torch.cat((
                x[:, phase].mean(dim = 1) * self.weights[0],
                x[:, not_phase].mean(dim = 1) * self.weights[1], 
                phase_s[:, i]), dim = -1))
        res = torch.stack(res, dim = 1)  # [BATCH, PHASE, HIDDEN]
        res = self.fc(res)

        return res
