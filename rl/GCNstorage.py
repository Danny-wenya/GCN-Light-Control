import numpy as np
import pickle
import torch
import pdb
from collections import deque
from collections import defaultdict
from utils.utils import *



class Scaler:
    """
    归一化接口
    """

    def transform(self, data):
        """
        数据归一化接口

        Args:
            data(np.ndarray): 归一化前的数据

        Returns:
            np.ndarray: 归一化后的数据
        """
        raise NotImplementedError("Transform not implemented")

    def inverse_transform(self, data):
        """
        数据逆归一化接口

        Args:
            data(np.ndarray): 归一化后的数据

        Returns:
            np.ndarray: 归一化前的数据
        """
        raise NotImplementedError("Inverse_transform not implemented")


class NoneScaler(Scaler):
    """
    不归一化
    """

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data


class NormalScaler(Scaler):
    """
    除以最大值归一化
    x = x / x.max
    """

    def __init__(self, maxx):
        self.max = maxx

    def transform(self, data):
        return data / self.max

    def inverse_transform(self, data):
        return data * self.max


class StandardScaler(Scaler):
    """
    Z-score归一化
    x = (x - x.mean) / x.std
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class MinMax01Scaler(Scaler):
    """
    MinMax归一化 结果区间[0, 1]
    x = (x - min) / (max - min)
    """

    def __init__(self, minn, maxx):
        self.min = minn
        self.max = maxx

    def transform(self, data):
        return (data - self.min) / (self.max - self.min)

    def inverse_transform(self, data):
        return data * (self.max - self.min) + self.min


class MinMax11Scaler(Scaler):
    """
    MinMax归一化 结果区间[-1, 1]
    x = (x - min) / (max - min)
    x = x * 2 - 1
    """

    def __init__(self, minn, maxx):
        self.min = minn
        self.max = maxx

    def transform(self, data):
        return ((data - self.min) / (self.max - self.min)) * 2. - 1.

    def inverse_transform(self, data):
        return ((data + 1.) / 2.) * (self.max - self.min) + self.min


class LogScaler(Scaler):
    """
    Log scaler
    x = log(x+eps)
    """

    def __init__(self, eps=0.999):
        self.eps = eps

    def transform(self, data):
        return np.log(data + self.eps)

    def inverse_transform(self, data):
        return np.exp(data) - self.eps

class RolloutStorages:
    """wrapper for several RolloutStorage."""
    def __init__(self, threads, *argv, **kwargs):
        self.rollouts = []
        self.threads = threads
        for _ in range(threads):
            self.rollouts.append(RolloutStorage(*argv, **kwargs))

    def call_rollout_functions(self, name):
        def func(*argv, **kwargs):
            res = []
            splitv = [[] for _ in self.rollouts]
            kwsplits = [{} for _ in self.rollouts]
            for arg in argv:
                alen = None
                try:
                    alen = len(arg)
                except Exception:
                    pass
                assert alen is None or alen == self.threads
                for num, i in enumerate(splitv):
                    if alen is None:
                        i.append(arg)
                    else:
                        i.append(arg[num])
            for key in kwargs:
                arg = kwargs[key]
                klen = None
                try:
                    klen = len(arg)
                except Exception:
                    pass
                assert klen is None or klen == self.threads
                for num, i in enumerate(kwsplits):
                    if alen is None:
                        i[key] = arg
                    else:
                        i[key] = arg[num]
            for rollout, av, kw in zip(self.rollouts, splitv, kwsplits):
                res.append(getattr(rollout, name)(*av, **kw))
            return res
        return func

    def cuda(self):
        self.call_rollout_functions('cuda')
        return self

    def collect_training_data(self, next_v):
        ret = self.call_rollout_functions('collect_training_data')(next_v)
        ret = [x for x in ret if x is not None]
        ret = zip(*ret)
        res = []
        for a in ret:
            if isinstance(a[0], list) or isinstance(a[0], tuple):
                res.append(sum(map(list, a), []))
            elif isinstance(a[0], np.ndarray):
                res.append(np.concatenate(a))
            else:  # tensor
                raise ValueError('error type' + str(a) + str(type(a)))
        if len(res) == 0:
            return None
        return res

    def __getattr__(self, name):
        return self.call_rollout_functions(name)


class RolloutStorage(object):
    """rollout storage for one environment. save multi-agent datas."""
    def __init__(self, num_steps, agent_number, obs_shape, action_space, gamma,
                 recurrent_hidden_state_size = None):

        assert recurrent_hidden_state_size is None  # not supported now
        self.agent_number = agent_number
        self.num_steps = num_steps
        self.GAMMA = gamma
        self.obs = []
        self.actions = np.zeros((num_steps, agent_number, 1), dtype = int)
        self.rewards = np.zeros((num_steps, agent_number, 1), dtype = float)
        self.preds = np.zeros((num_steps + 1, agent_number, 1), dtype = float)
        self.returns = np.zeros((num_steps + 1, agent_number, 1), 
                                dtype = float)
        self.probs = np.zeros((num_steps, agent_number, action_space[0][0]),
                                dtype = float)
        # ist[step] is 1, means obs[step + 1] is not following obs[step]
        self.ist = np.zeros((num_steps,), dtype = bool)
        # self.is_faket = torch.zeros_like(self.ist).bool()

        self.step = 0

    def full(self):
        return self.step == self.num_steps

    def reset(self, init_state):
        self.obs.clear()
        self.actions[:] = 0
        self.rewards[:] = 0
        self.ist[:] = 0
        # self.is_faket[:] = 0
        self.obs.append(init_state)
        self.step = 0

    def insert(self, obs, actions, rewards, ist, preds, probs, is_faket = None):
        #       recurrent_hidden_states, action_log_probs):
        # preds: value prediction 
        assert is_faket is None
        self.obs.append(obs)
        self.actions[self.step] = np.array(actions).copy()
        self.rewards[self.step] = np.array(rewards).copy()
        self.preds[self.step] = np.array(preds).copy()
        self.ist[self.step] = np.array(ist).copy()
        self.probs[self.step] = np.array(probs).copy()
        # self.is_faket[self.step + 1].copy_(torch.tensor(is_faket))

        self.step = (self.step + 1)  # % self.num_steps
        # pdb.set_trace()

    def collect_training_data(self, next_v):
        """return training data. If data is not fully collected, return None.
            else, return corresponding data with input agent and reset
        """
        if not self.full():
            return None
        self.compute_returns(next_v)
        ret_obs = self.obs[:-1]
        next_s = self.obs[1:]
        self.obs = self.obs[-1:]
        self.step = 0
        return (ret_obs, self.actions, self.returns[:-1], self.probs, next_s)

    def after_update(self):
        self.obs = [self.obs[-1]]
        # self.is_faket[0].copy_(self.is_faket[-1])
        self.step = 0

    def compute_returns(self,
                        next_value,
                        use_gae = False,
                        gae_lambda = 1,
                        stop_time_limit = True):
        self.returns[self.step] = next_value
        for step in reversed(range(self.step)):
            self.returns[step] = self.returns[step + 1] * \
                self.GAMMA * (1 - self.ist[step]) + self.rewards[step]


class CityFlowBuffer:
    def __init__(self, maxlen, observation, action):
        ''' use ndarray
        self.state = np.zeros((maxlen, threads, observation), dtype=float)
        self.action = np.zeros((maxlen, threads, action), dtype=float)
        self.reward = np.zeros((maxlen, threads, 1), dtype=float)
        self.next_s = np.zeros((maxlen, threads, observation), dtype=float)
        self.ist = np.zeros((maxlen, threads, 1), dtype=int)
        '''
        self.data = []

    def __len__(self):
        return len(self.data)

    def append(self, data):
        self.data.append(data)

    def __setitem__(self, index, data):
        self.data[index] = data

    def __getitem__(self, index):
        if type(index) == np.ndarray:
            res = []
            for num in index:
                res.append(self.data[num])
            return res
        return self.data[index]


class ReplayBuffer:
    def __init__(self, n_steps, maxlen, batch_size, seed, buffer_instance):
        self.buffer = buffer_instance
        self.maxlen = maxlen
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.random = np.random.RandomState(seed)

        self.position = 0
        self.full = False
        self._reset = False
        self._state_deque = deque(maxlen = n_steps + 1)
        self._action_deque = deque(maxlen = n_steps)
        self._reward_deque = deque(maxlen = n_steps)
        self._ist_deque = deque(maxlen = n_steps)

    def __len__(self):
        return len(self.buffer)

    def cuda(self):
        return self

    def reset(self, state):
        self._reset = True
        self._state_deque.clear()
        self._action_deque.clear()
        self._reward_deque.clear()
        self._ist_deque.clear()
        self._state_deque.append(state)

    def append(self, action, reward, next_s, ist):
        assert(self._reset)
        self._state_deque.append(next_s)
        self._action_deque.append(action)
        self._reward_deque.append(reward)
        self._ist_deque.append(ist)
        if len(self._state_deque) == self.n_steps + 1:
            tot_reward = np.zeros((self.n_steps, *self._reward_deque[0].shape),
                                  dtype = float)
            tot_ist = np.zeros_like(self._ist_deque[0], dtype=int)
            for num, (reward, ist) in enumerate(zip(self._reward_deque, 
                                                    self._ist_deque)):
                tot_reward[num] = reward
                tot_ist += ist
            tot_reward = np.moveaxis(tot_reward, 0, -1)
            tot_ist -= self._ist_deque[-1] 
            data = [[], [], [], [], []]
            now_state_unpack = self._state_deque[0]
            next_state_unpack = self._state_deque[-1]
            for num, can_in in enumerate(tot_ist):
                if can_in == 0:
                    data[0].append(now_state_unpack[num])
                    data[1].append(self._action_deque[0][num])
                    data[2].append(tot_reward[num])
                    data[3].append(next_state_unpack[num])
                    data[4].append(self._ist_deque[-1][num])
            for one_data in zip(*data):
                self._buffer_append(*one_data)

    def _buffer_append(self, state, action, reward, next_s, ist):
        if len(self.buffer) < self.maxlen:
            self.buffer.append([state, action, reward, next_s, ist])
            if len(self.buffer) == self.maxlen:
                self.full = True
        else:
            self.buffer[self.position] = [state, action, reward, next_s, ist]
            self.position = (self.position + 1) % self.maxlen

    def sample(self):
        choice = self.random.choice(self.maxlen, 
                                    self.batch_size, 
                                    replace = False)
        return self.buffer[choice]
    

class GCNReplayBuffer(ReplayBuffer):
    def __init__(self,adj,strides,**kwargs):
        super().__init__(**kwargs)
        self.gcn_raw_data=[]
        self.adj=adj
        self.strides=strides
        self.adj_mx=np.zeros([4*adj.shape[0],4*adj.shape[0]])

    
    def _get_scalar(self, x_train, y_train):
        scaler = NormalScaler(maxx=max(x_train.max(), y_train.max()))
        print('NormalScaler max: ' + str(scaler.max))
        return scaler
    
    def normalize(self,a):
        mu=np.mean(a,axis=1,keepdims=True)
        std=np.std(a,axis=1,keepdims=True)
        return (a-mu)/std

    def compute_dtw(self,a,b,order=1,Ts=12,normal=True):
        if normal:
            a=self.normalize(a) 
            b=self.normalize(b) 
        T0=a.shape[1]
        d=np.reshape(a,[-1,1,T0])-np.reshape(b,[-1,T0,1])  
        d=np.linalg.norm(d,axis=0,ord=order) 
        D=np.zeros([T0,T0]) 
        for i in range(T0): 
            for j in range(max(0,i-Ts),min(T0,i+Ts+1)):
                if (i==0) and (j==0): 
                    D[i,j]=d[i,j]**order 
                    continue
                if (i==0): 
                    D[i,j]=d[i,j]**order+D[i,j-1]
                    continue
                if (j==0):
                    D[i,j]=d[i,j]**order+D[i-1,j]
                    continue
                if (j==i-Ts): 
                    D[i,j]=d[i,j]**order+min(D[i-1,j-1],D[i-1,j])
                    continue
                if (j==i+Ts):
                    D[i,j]=d[i,j]**order+min(D[i-1,j-1],D[i,j-1])
                    continue
                D[i,j]=d[i,j]**order+min(D[i-1,j-1],D[i-1,j],D[i,j-1])
        return D[-1,-1]**(1.0/order)

    def construct_adj_fusion(self, A,A_dtw, steps):
        N = len(self.adj)
        adj = np.zeros([N * steps] * 2) # "steps" = 4 !!!
        for i in range(steps):
            if (i == 1) or (i == 2):
                adj[i * N: (i + 1) * N, i * N: (i + 1) * N] = A
            else:
                adj[i * N: (i + 1) * N, i * N: (i + 1) * N] = A_dtw
        #'''
        for i in range(N):
            for k in range(steps - 1):
                adj[k * N + i, (k + 1) * N + i] = 1
                adj[(k + 1) * N + i, k * N + i] = 1
        #'''
        adj[3 * N: 4 * N, 0:  N] = A_dtw #adj[0 * N : 1 * N, 1 * N : 2 * N]
        adj[0 : N, 3 * N: 4 * N] = A_dtw #adj[0 * N : 1 * N, 1 * N : 2 * N]

        adj[2 * N: 3 * N, 0 : N] = adj[0 * N : 1 * N, 1 * N : 2 * N]
        adj[0 : N, 2 * N: 3 * N] = adj[0 * N : 1 * N, 1 * N : 2 * N]
        adj[1 * N: 2 * N, 3 * N: 4 * N] = adj[0 * N : 1 * N, 1 * N : 2 * N]
        adj[3 * N: 4 * N, 1 * N: 2 * N] = adj[0 * N : 1 * N, 1 * N : 2 * N]

        for i in range(len(adj)):
            adj[i, i] = 1

        return adj


    def _construct_dtw(self): 
        N = len(self.adj) 
        xtr=np.array(self.gcn_raw_data)
        ns=int(xtr.shape[0]/10)*10 #int(xtr.shape[0]/300)*300
        xtr=xtr[0:ns,...]
        xtr=np.reshape(xtr,[-1,10,N])  # 360:每条样本是12s的数据，一个小时3600s有300条数据
        print(np.shape(xtr))
       
        d = np.zeros([N, N])
        for i in range(N):
            for j in range(i+1,N):
                d[i,j]=self.compute_dtw(xtr[:,:,i],xtr[:,:,j])

        print("The calculation of time series is done!")
        dtw = d+ d.T
        n = dtw.shape[0]
        w_adj = np.zeros([n,n])
        adj_percent = 0.01
        top = int(n * adj_percent)
        for i in range(dtw.shape[0]):
            a = dtw[i,:].argsort()[0:top]  
            for j in range(top):
                w_adj[i, a[j]] = 1 

        for i in range(n):
            for j in range(n):
                if (w_adj[i][j] != w_adj[j][i] and w_adj[i][j] ==0):
                    w_adj[i][j] = 1
                if( i==j):
                    w_adj[i][j] = 1

        print("Total route number: ", n)
        print("Sparsity of adj: ", len(w_adj.nonzero()[0])/(n*n))
        print("The weighted matrix of temporal graph is generated!")
        self.dtw = w_adj # 相似矩阵中将每个路口的相似路口置1.

    def _construct_adj(self):
        if len(self.gcn_raw_data)==self.maxlen:
            self._construct_dtw() #得到self.dtw
            adj_mx = self.construct_adj_fusion(self.adj,self.dtw, self.strides)  #self.adj_mx = torch.FloatTensor(self._construct_adj()),这算不算递归？self.strides=4
            print("The shape of localized adjacency matrix: {}".format(
            adj_mx.shape), flush=True)
            self.adj_mx=adj_mx
            self.adjmx_change=True
            with open("./gcn_raw_data.pkl","wb") as f:
                pickle.dump(self.gcn_raw_data,f)
            

    def reset_adj_mx(self):
        self.adj_mx=np.zeros([4*self.adj.shape[0],4*self.adj.shape[0]])
        self.adjmx_change=False

    # def _gcn_sample_normal(self,now_state_unpack):
    #     for _,state in now_state_unpack.items():
    #         state["NormalScaler_max"]=self.saler.max()
    #         state["GCNflow"]=self.saler.transform(np.array(state["GCNflow"])) 

    def _save_gcn_raw_data(self,next_s):
        if len(self.gcn_raw_data)==self.maxlen:
            self.gcn_raw_data=[]
        sample=[]
        for state in next_s: 
            sample.extend([st[-1] for st in state["GCNflow"]])
        self.gcn_raw_data.append(sample)   


    

    def append(self, action, reward, next_s, ist):
        assert(self._reset)
        self._state_deque.append(next_s)
        self._action_deque.append(action)
        self._reward_deque.append(reward)
        self._ist_deque.append(ist)
        self._construct_adj() # 计算adj_mx
        self._save_gcn_raw_data(next_s) # 保存新样本

        if len(self._state_deque) == self.n_steps + 1:
            tot_reward = np.zeros((self.n_steps, *self._reward_deque[0].shape),
                                  dtype = float)
            tot_ist = np.zeros_like(self._ist_deque[0], dtype=int)
            for num, (reward, ist) in enumerate(zip(self._reward_deque, 
                                                    self._ist_deque)):
                tot_reward[num] = reward
                tot_ist += ist
            tot_reward = np.moveaxis(tot_reward, 0, -1)
            tot_ist -= self._ist_deque[-1] 
            data = [[], [], [], [], []]

            now_state_unpack = self._state_deque[0]
            now_state_unpack=[gcn_input(now_state_unpack)]

            next_state_unpack = self._state_deque[-1]  
            next_state_unpack=[gcn_input(next_state_unpack)]
            
            # 组建样本
            for num, can_in in enumerate(tot_ist):
                if can_in == 0:
                    data[0].append(now_state_unpack[num])
                    data[1].append(self._action_deque[0][num])
                    data[2].append(tot_reward[num])
                    data[3].append(next_state_unpack[num])
                    data[4].append(self._ist_deque[-1][num])
           
            for one_data in zip(*data):
                self._buffer_append(*one_data)

         

           
    



