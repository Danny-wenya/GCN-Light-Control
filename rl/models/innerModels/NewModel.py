from .InnerModelBase import *
from ..graphnn.graphnn import GCN


"""lane2embedding, predict phase
"""


class PhasePredict(nn.Module):
    def __init__(self, observation, NM_road_predict_hidden, **kwargs):
        super().__init__()
        self.observation = observation
        self.hidden = NM_road_predict_hidden
        self.direction_type = 3

        self.direction_hidden = 2
        self.emb_direction = nn.Embedding(self.direction_type, 
                                          self.direction_hidden)

        self.emb_fc = nn.Linear(2 + self.direction_hidden, self.hidden)

        self.attention = nn.MultiheadAttention(self.hidden, 1)
        self.phase_predict = nn.Sequential(
            nn.Linear(self.hidden, 1),
            nn.Sigmoid()
        )

        self.vol_predict_fc = nn.Linear(self.hidden, self.direction_type)

    def forward(self, states, road_relation):
        lanewidths = road_relation['LaneCount']  # [L]
        flow = states['TSflow']
        # wait = states['TSwait']
        green = states['TSgreen']
        direction = road_relation['RoadLinkDirection']
        direction = self.emb_direction(direction)
        direction = direction.unsqueeze(0).repeat(flow.shape[0], 1, 1)
        x = torch.stack((flow, green), dim = -1)
        x = self.emb_fc(torch.cat((x, direction), dim = -1))  # [B, L, H]
        q = k = v = x.transpose(0, 1)  # [L, B, H]
        x, x_w = self.attention(q, k, v)
        x = x.transpose(0, 1)  # [B, L, H]
        phase_predict = self.phase_predict(x)  # [B, L, 1]

        return x, phase_predict


"""input embedding and phase(predict or real), output every road traffic volume
   of every direction
"""


class VolumePredict(nn.Module):
    def __init__(self, NM_road_predict_hidden, graph_hidden_dim,graph_output_dim,num_heads,num_nodes,**kwargs):
        super().__init__()
        self.hidden = NM_road_predict_hidden
        # self.GAT=GAT(self.hidden, gat_hidden_dim, gat_output_dim,num_nodes, num_heads)
        self.GCN=GCN(self.hidden, graph_hidden_dim, graph_output_dim)

    def forward(self,input_features, adjacency_matrix):
        """calculate predicts for roadout
        """
        input_features=cuda(input_features).float()
        adjacency_matrix=cuda(adjacency_matrix).float()
        v_p=self.GCN(input_features, adjacency_matrix)

        return v_p


class LaneEmbedding(nn.Module):
    def __init__(self, observation, lane_embedding_size):
        super().__init__()
        self.observation = observation
        self.layers = []
        if isinstance(lane_embedding_size, int):
            lane_embedding_size = [lane_embedding_size]
        self.lane_embedding_size = lane_embedding_size
        self.input = 3  # flow, green, predict, predict_valid_mask
        last_input = self.input
        for hidden in lane_embedding_size:
            self.layers += [nn.Linear(last_input, hidden), nn.ReLU()]
            last_input = hidden
        self.layers = nn.Sequential(*self.layers)

    def forward(self, states):
        return self.layers(states)


class NewModel(InnerModelBase):
    def __init__(self, dqn_hidden_size, lane_embedding_size, observation, 
                 lane_embedding_instance = None, **kwargs):
        super().__init__()
        self.hidden = dqn_hidden_size
        self.observation = observation
        self.phases = observation['TSphase']
        self.lane_number = observation['TSflow'][0]
        self.not_phases = []
        for phase in self.phases:
            one = []
            for num in range(self.lane_number):
                if num not in phase:
                    one.append(num)
            self.not_phases.append(one)
        self.weights = [5, 1]

        self.shared_lane = lane_embedding_instance is not None

        self.phase_predict = PhasePredict(observation, **kwargs)
        self.volume_predict = VolumePredict(**kwargs)

        if lane_embedding_instance is not None:
            self.lane_embedding = lane_embedding_instance
        else:
            self.lane_embedding = LaneEmbedding(observation, 
                                                lane_embedding_size)

        self.fc = nn.Sequential(
            nn.Linear(lane_embedding_size * 2 + 1,
                      dqn_hidden_size),
            nn.ReLU()
        )

    @staticmethod
    def default_wrapper(dueling_dqn, **kwargs):
        if dueling_dqn:
            return 'DuelingSplitModel'
        else:
            return 'DQNSplitModel'

    """if share lane embedding module, except it from named modules to avoid 
       multiple optimization
    """
    def named_modules(self, memo = None, prefix = ''):
        for n, p in super().named_modules(memo, prefix):
            if self.shared_lane:
                if n.find(prefix + '.lane_embedding') == 0:
                    continue
            yield n, p

    def replace_lane_embedding(self, lane_embedding):
        self.lane_embedding = lane_embedding
        self.shared_lane = True

    @staticmethod
    def state2tensor(state):
        res = {}
        res['TSflow'] = cuda(torch.tensor(state['TSflow'])).float()
        res['TSgreen'] = cuda(torch.tensor(state['TSgreen'])).float()
        res['TSphase'] = cuda(torch.tensor(state['TSphase'])).long()
        res['Envtime'] = cuda(torch.tensor(state['Envtime'])).float()
        NILVN = 'NextInLaneVehicleNumber'
        if NILVN in state:
            res[NILVN] = cuda(torch.tensor(state[NILVN])).float()
        NP = 'NextPhase'
        if NP in state:
            res[NP] = cuda(torch.tensor(state[NP])).long()
        return res

    def forward(self, state):
        flow = state['TSflow']
        green = state['TSgreen']
        phase = state['TSphase']
        predict = state['predict']
        phase_s = nn.functional.one_hot(phase, num_classes = len(self.phases))
        phase_s = phase_s.float().unsqueeze(-1)  # [BATCH, PHASE, 1]

        x = torch.stack((flow, green, predict), dim = -1)
        x = self.lane_embedding(x)  # [BATCH, LANE, HIDDEN]

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
