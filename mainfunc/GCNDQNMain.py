from .DQNMain import *
from envs.GCNSubprocVecEnv import getGCNCityFlowEnvVec
from .MainFuncBase import *

from rl.GCNstorage import GCNReplayBuffer, CityFlowBuffer
from rl.models import *
from rl.agents import *
from rl.multiAgent import *
import pickle


class GCNDQNMain(DQNMain):
    def __init__(self, env, agent, multi_agent, model, dueling_dqn, 
                 dqn_initial_eps, dqn_final_eps, dqn_explore_len_frac, 
                 n_frames, dqn_replay_size, dqn_batch_size, render_steps, 
                 model_save_path, n_steps,strides,
                 threads, seed, no_cuda, evaluate_round, evaluate_interval, 
                 development, save_interval,
                 # Specific args
                 log_folder, work_folder, simulate_time, cityflow_config, 
                 cityflow_log, preload_model_file, test_round, 
                 dqn_split_model, clean_logs, 
                 train_cityflow_config, wrapper_model,
                 **kwargs):
        self.SEED = seed
        self.randomstate = np.random.RandomState(seed)
        set_seed(seed)
        cuda(None, not no_cuda)
        self.DEV = development

        self.envs = []
        env = env.lower()
        if env == 'cityflow':
            env_args = {
                'number': threads, 
                'logpath': log_folder,
                'workpath': work_folder, 
                'config': train_cityflow_config, 
                'log': cityflow_log,
            }
            env_args['logpath'] = log_folder + '/train_env/' 
            self.train_env_args = env_args
            self.env = self.get_env(self.train_env_args)
            # one thread, new folder for test
            env_args = env_args.copy()
            env_args['number'] = 1
            env_args['logpath'] = log_folder + '/test_env/' 
            env_args['config'] = cityflow_config
            self.test_env_args = env_args
            self.test_env = self.get_env(self.test_env_args)
        else:
            raise NotImplementedError('envs except cityflow is '
                                      'not implemented')
        log(self.env.observation_space, self.env.action_space, level = 'TRACE')
        self._init_TXSW(**kwargs)

        kwargs_buffer={'n_steps':n_steps,
                 'maxlen':dqn_replay_size, 
                 'batch_size':dqn_batch_size, 
                 'seed':self.randint(), 
                 'buffer_instance':CityFlowBuffer(dqn_replay_size, self.env.observation_space['intersections'],self.env.action_space),
                 'adj':self.env.adj,
                 'strides':strides

                }
        self.replay = cuda(GCNReplayBuffer(**kwargs_buffer))

        if model == 'basic':
            model = 'DQNNaiveFC'
        if model in globals():
            try:
                inner_model = globals()[model]
            except Exception as e:
                raise e
        else:
            raise NotImplementedError('unknown model ' + model)
        if wrapper_model != '':
            if wrapper_model in globals():
                wrapperModel = globals()[wrapper_model]
            else:
                raise NotImplementedError('unknown wrapper model')
        elif 'default_wrapper' in dir(inner_model):
            wrapper_model = inner_model.default_wrapper(
                                dqn_split_model = dqn_split_model, 
                                dueling_dqn = dueling_dqn
                            )
            wrapperModel = globals()[wrapper_model]
        model_args = {
            'observation': self.env.observation_space['intersections'], 
            'adj_mat': self.env.observation_space['adj_mat'],
            'virtual_intersection_out_lines': self.env.observation_space[
                'virtual_intersection_out_lines'],
            'action': self.env.action_space,
            'innerModel': inner_model, 
            'adj':self.env.adj,
            'adj_mx':self.replay.adj_mx,
            'TXSW': self.TXSW,
        }
        if agent.lower() == 'dqn':
            agent = 'DQNAgent'
        if agent in globals():
            Agent = globals()[agent]
        else:
            raise NotImplementedError('unknown agent ' + agent)
        if multi_agent in globals():
            MultiAgent = globals()[multi_agent]
        else:
            raise NotImplementedError('unknown multi-agent ' + multi_agent)
        self.agent = cuda(MultiAgent(n_steps = n_steps, 
                                     AgentClass = Agent,
                                     wrapperModel = wrapperModel,
                                     dueling_dqn = dueling_dqn,
                                     seed = self.randint(), 
                                     **model_args, 
                                     **kwargs))

        self.agent.log_model_structure()


        self.THREADS = threads
        self.N_STEPS = n_steps

        self.test_round = test_round
        self.epoch = 0
        if preload_model_file != '':
            self.model_file = torch.load(preload_model_file,
                                         map_location = 'cpu')
            self.agent.load_state_dict(self.model_file['state_dict'])
            self.epoch = self.model_file['epoch']
            self.env.replay_count(self.model_file['replay_count'])
            self.SIMULATE_TIME = simulate_time
            self.FRAME = self.epoch * self.SIMULATE_TIME

        if test_round > 0:
            return
        

        self.EPS = [dqn_initial_eps, dqn_final_eps, dqn_explore_len_frac]
        self.EPS[2] = (self.EPS[0] - self.EPS[1]) / (n_frames * self.EPS[2])
        self.N_FRAMES = n_frames
        self.RENDER = render_steps

        self.SIMULATE_TIME = simulate_time
        self.TOTAL_EPOCH = n_frames // (simulate_time * threads)
        if n_frames % (simulate_time * threads) != 0:
            self.n_frames = self.TOTAL_EPOCH * simulate_time * threads
            log('n_frames can\'t be divided with simulate_time and threads! '
                'change n_frames to %s.' % self.n_frames, level = 'WARN')
        self.FRAME = self.epoch * self.SIMULATE_TIME * threads

        self.PREVIOUS_REWARD = []
        self.BEST_RESULT = None
        self.REWARD_AVG = 10

        assert save_interval == 0, 'not support save_interval now'
        self.EVAL_ROUND = evaluate_round
        self.EVAL_INTERVAL = evaluate_interval
        if self.EVAL_INTERVAL % threads != 0:
            self.EVAL_INTERVAL = (self.EVAL_INTERVAL // threads + 1) * threads
            log('evaluate_interval can\'t be divided with threads! change '
                'interval to %s.' % self.EVAL_INTERVAL, level = 'WARN')
        self.model_folder = os.path.join(log_folder, model_save_path) + '/'
        if model_save_path == '':
            self.model_folder = ''
            log('model save path not set! will not save model', level = 'WARN')
        else:
            os.makedirs(self.model_folder)

        log('GCNDQNMain init over')


    def get_env(self, args):
        args = args.copy()
        args['logpath'] += str(len(self.envs)) + '/'
        self.envs.append(getGCNCityFlowEnvVec(**args, seed = self.randint()))
        return self.envs[-1]


    def real_sampling(self):
        start_time = time.time()
        replay_full_before = self.replay.full
        state, infos = self.env.reset()  #state [[dict]*6], state["input"]:ndarray[10,12,3]
        self.replay.reset(state[0])
        action = self.agent.action(state,eps = self.eps(), get = True)    # 随机动作
        tot_reward = np.zeros(self.THREADS, dtype=float) # array(1,)
        eval_res = None
        for ite in range(self.SIMULATE_TIME):
            self.FRAME += self.THREADS
            next_s, reward, ist, infos = self.env.step(action)
            tot_reward += np.sum(reward, axis = -1)
            next_action = self.agent.action(next_s,eps = self.eps(), get = True)  

            assert(ist.all() == ist.any())
            self.replay.append(action, reward, next_s[0], ist)
            if self.replay.full and not replay_full_before:
                with open('./buffer_1w.pkl','wb') as f:
                    pickle.dump(self.replay.buffer,f)
                log('replay full, start training')
                replay_full_before = True

            if self.replay.full:
                self.agent.eps_flag=True
                for _ in range(self.THREADS):
                    samples = self.replay.sample()   
                    if self.replay.adjmx_change:
                        self.agent.agent.model_old.inner.adj_mx=self.replay.adj_mx
                        self.agent.agent.model_update.inner.adj_mx=self.replay.adj_mx
                        self.agent.agent.BEST_MODEL.inner.adj_mx=self.replay.adj_mx
                        self.replay.reset_adj_mx()
                    self.agent.update_policy(samples, self.FRAME)  
                    print(f"training..{ite}")
            action = next_action
            state = next_s

            if self.FRAME % self.EVAL_INTERVAL == 0 and self.replay.full:
                log('epoch %4d, frame %7d: start evaluate' 
                    % (self.epoch, self.FRAME))
                eval_res = self.evaluate('update')
                log('evaluate result:', eval_res)
                if not self.BEST_RESULT or eval_res < self.BEST_RESULT:
                    self.BEST_RESULT = eval_res
                    self.agent.update_best()
                    if self.model_folder != '':
                        self.save(self.model_folder + 'best.pt',
                                  eval = eval_res)
                    log('best result updated:', eval_res)

        result = np.stack([x['average_time'] for x in infos]).mean()
        self.PREVIOUS_REWARD.append(result)
        log('epoch %4d, frame %7d: time=%.3f, eps=%.2f, reward=%.6f, '
            'result=%.6f' % (
                self.epoch, 
                self.FRAME, 
                time.time() - start_time, 
                self.eps(), 
                tot_reward.mean(), 
                result))
        self.TXSW.add_scalar('reward', tot_reward.mean(), self.FRAME)
        self.TXSW.add_scalar('result', result, self.FRAME)

        if self.model_folder != '':
            self.save(self.model_folder + '%04d.pt' % self.epoch,
                      result, tot_reward, eval_res)