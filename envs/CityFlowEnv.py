from cityflow import Engine
import numpy as np
import json
import sys
import os

from utils.log import log
from utils.utils import gzip_file


class TrafficSignal:
    """
        Traffic Signal Control
    """
    def __init__(self, ID, eng, yellow_time, roadnet_info):
        self.roadnet_info = roadnet_info
        self.ID = ID
        self.eng = eng
        self.now_idx = 1
        self.old_idx = 1
        self.next_idx = None
        self.now_time = -1
        self.yellow_flag = False
        self.flicker = False
        self.yellow_time = yellow_time
        self.phase_number = len(roadnet_info['phases'])
        self._set_observation_space()
        self.action_space = self.phase_number - 1
        self.set_eng_phase()

    def _set_observation_space(self):
        TSlane=[]
        TSlanelink=[]
        for dic in self.roadnet_info['roadlinks']:
            TSlane.append(set([lk[0] for lk in dic['lanelinks']]))
            TSlanelink.append(dic['lanelinks'])
            
        self.observation_space = {
            'TSwait': [len(self.roadnet_info['roadlinks'])],
            'TSphase': self.roadnet_info['phases'][1:],
            'TSgreen': [len(self.roadnet_info['roadlinks'])],
            'TSlane':TSlane,
            'TSlanelink':TSlanelink,
            'TStime': [1],
            'Envtime': [1],
            'RoadLinkDirection': [x['type'] for x in self.roadnet_info['roadlinks']],
        }



    def set_eng_phase(self):
        self.eng.set_tl_phase(self.ID, self.now_idx)

    def set_signal(self, action, action_pattern):
        if self.yellow_flag:
            # in yellow phase
            if self.now_time >= self.yellow_time:  # yellow time reached
                self.now_idx = self.next_idx
                self.set_eng_phase()
                self.yellow_flag = False
            else:
                pass
        else:
            # determine phase
            if action_pattern == "switch":  # switch by order
                if action == 0:  # keep the phase
                    self.new_idx = self.now_idx
                elif action == 1:  # change to the next phase
                    self.next_idx = self.now_idx + 1
                    if self.next_idx == self.phase_number:
                        self.next_idx = 1
                else:
                    sys.exit("action not recognized\n action must be 0 or 1")

            elif action_pattern == "set":  # set to certain phase
                self.next_idx = action + 1

            # set phase
            if self.now_idx == self.next_idx:  # light phase keeps unchanged
                pass
            else:  # the light phase needs to change
                # change to yellow first, and activate the counter and flag
                self.now_idx = 0
                self.set_eng_phase()
                self.yellow_flag = True

    def step_time(self):
        if self.now_idx != self.old_idx:
            self.now_time = 1
        else:
            self.now_time += 1

    def _get_signal_state(self):
        if self.now_idx == 0:
            return self.next_idx - 1
        return self.now_idx - 1

    def get_state(self):
        TSwait = []
        TSgreen = []
        TSphase = self._get_signal_state()
        TStime = self.now_time
        for roadlink in range(len(self.roadnet_info['roadlinks'])):
            count = self.eng.get_roadlink_vehicle_count(self.ID, roadlink)
            wait_count = self.eng.get_roadlink_waiting_vehicle_count(
                self.ID, roadlink)
            assert count[1] == wait_count[1], f'{count} {wait_count}'
            if count[1] == 0:
                TSwait.append(0)
            else:
                TSwait.append(wait_count[0] / wait_count[1])
            TSgreen.append(0)
        for phase in self.roadnet_info['phases'][self.now_idx]:
            TSgreen[phase] = 1

        envtime = self.eng.get_current_time()
        return {
            'TSwait': TSwait,
            'TSphase': TSphase,
            'TSgreen': TSgreen,
            'TStime': TStime,
            'Envtime': envtime,
        }

    def update_old(self):
        self.old_idx = self.now_idx

class Intersection:
    def __init__(self, ID, config, eng, intersection_names, 
                 virtual_intersection_names):
        self.ID = ID
        self.config = config
        self.eng = eng
        self.intersection_names = intersection_names
        # self.virtual_intersection_names = virtual_intersection_names
        self.save_lane_count_step = 5
        self.gcn_save_step=12
        self.lanename2roaddirection = {}
        self._init_components()
        self._set_observation_space()
        self._set_lane_flow()
        self._set_action_space()

    def _init_components(self):
        self.roadnet_info = self.config['ROADNET_INFO'][self.ID]

        self.TS = TrafficSignal(self.ID, self.eng, self.config['YELLOW_TIME'],
                                self.roadnet_info)
        self.last_reward = 0

        self.saved_phases = np.zeros((self.gcn_save_step,), dtype = int)
        self.saved_phases[:] = -1

        self.lane_vehicle_count_mat = np.zeros((  # this is in count
            self.save_lane_count_step, 
            len(self.roadnet_info['connection']), 3), dtype = int)
        self.lane_vehicle_out_count_mat = np.zeros((
            self.save_lane_count_step, 
            len(self.roadnet_info['connection']), 3), dtype = int)

    def reset(self):
        self._init_components()

    def _get_road_direction_idx(self, roadname, direction):
        for num, i in enumerate(self.roadnet_info['roadlinks']):
            if i['startRoad'] == roadname and i['type'] == direction:
                return num
        raise ValueError(roadname, direction)

    def _set_observation_space(self):
        tot = self.TS.observation_space
        tot['adj_mx']=np.load(self.config['ADJ_MX'])['arr_0']
        tot['TSprevphases'] = [self.save_lane_count_step]
        self.observation_space = tot
        
        
    def _set_lane_flow(self):
        self.pre_sec_vehicle=[[set() for _ in ls] for ls in self.observation_space['TSlane']]
        self.saved_lane_flow=[[[set() for _ in ls] for ls in self.observation_space['TSlane']] for _ in range(self.gcn_save_step)]

    def _set_action_space(self):
        tot = [self.TS.action_space]
        self.action_space = tot

    def set_signal(self, action, action_pattern):
        self.TS.set_signal(action, action_pattern)

    def step_time(self):
        self.TS.step_time()

    def get_state(self):
        res = self.TS.get_state()
        DCphase = []
        self.saved_phases[-1] = res['TSphase']   
        res['TSprevphases'] = list(self.saved_phases)
        res['DCphase'] = DCphase
        res['pressure'] = self._get_pressure_observation()
        res['Inlaneflow']=[[sum([len(s) for s in rl])/len(rl) for rl in step] for step in self.saved_lane_flow]
        res['Onlaneflow']=[len(rl) for rl in self.pre_sec_vehicle]

        return res

    def act(self, actions, pattern):
        if len(actions) == 1:  # input single number, treat as combined action
            arr = []
            action = actions[0]
            for i in self.action_space:
                arr.append(action % i)
                action //= i
            actions = arr
        self.set_signal(actions[0], pattern)
        assert len(actions) == 1

    def get_default_action(self):
        # use single number format as default action
        return [0]

    def _get_pressure_observation(self):
        # count every roadlink pressure and average, Sigma(|P_ri|)
        lane_count_1 = self.eng.get_lane_vehicle_count()
        lane_count_2 = self.eng.get_lane_waiting_vehicle_count()
        lanelinks = [x['lanelinks'] for x in self.roadnet_info['roadlinks']]
        res = []
        for ll in lanelinks:
            start, end = zip(*ll)
            start = set(start)
            end = set(end)
            RA = 0
            RS = 0
            for s in start:
                RA += lane_count_1[s]
            for e in end:
                RS += lane_count_2[e]
            RR = RA - RS
            res.append(RR)
        return np.array(res)

    def _get_average_vehicle_count(self, type, weight):
        res = 0
        cres = 0
        if type.lower() == 'flow':
            target_func = self.eng.get_roadlink_vehicle_count
        elif type.lower() == 'wait':
            target_func = self.eng.get_roadlink_waiting_vehicle_count
        else:
            raise ValueError('unknown type ' + type)
        for roadlink in range(len(self.roadnet_info['roadlinks'])):
            count = target_func(self.ID, roadlink)
            if count[1] == 0:
                pass
            else:
                res += count[0] / count[1]
                cres += 1
        return -res / cres * weight

    def _get_average_pressure(self, type, weight):
        lane_count = self.eng.get_lane_vehicle_count()
        if type.lower() == 'intersection':
            # count intersection as one unit, |Sigma(#in) - Sigma(#out)|
            lanelinks = []
            for roadlink in self.roadnet_info['roadlinks']:
                lanelinks.extend(roadlink['lanelinks'])
            start, end = zip(*lanelinks)
            start = set(start)
            end = set(end)
            res = 0
            for s in start:
                res += lane_count[s]
            for e in end:
                res -= lane_count[e]
            return -abs(res) * weight
        elif type.lower() == 'roadlink':
            # count every roadlink pressure and average, Sigma(|P_ri|)
            lanelinks = [x['lanelinks'] 
                         for x in self.roadnet_info['roadlinks']]
            res = []
            for lanelink in lanelinks:
                start, end = zip(*lanelink)
                start = set(start)
                end = set(end)
                RR = 0
                for s in start:
                    RR += lane_count[s]
                for e in end:
                    RR -= lane_count[e]
                RR = abs(RR)
                res.append(RR)
            return -np.array(res).mean() * weight
        elif type.lower() == 'lanelink':
            # count every lanelink pressure and average
            res = []
            for roadlink in self.roadnet_info['roadlinks']:
                for s, t in roadlink['lanelinks']:
                    res.append(abs(lane_count[s] - lane_count[t]))
            return -np.array(res).mean() * weight
        else:
            raise ValueError('unknown type ' + type)

    def get_reward(self, weight):
        res = 0
        wkeys = list(weight.keys())
        for i in wkeys:
            if not weight[i]:
                del weight[i]
                continue
            weight[i] = float(weight[i])
            if not weight[i]:
                del weight[i]
        if 'AVERAGE_FLOW_VEHICLE_COUNT' in weight:
            res += self._get_average_vehicle_count(
                type = 'flow',
                weight = weight['AVERAGE_FLOW_VEHICLE_COUNT'])
        if 'AVERAGE_WAIT_VEHICLE_COUNT' in weight:
            res += self._get_average_vehicle_count(
                type = 'wait',
                weight = weight['AVERAGE_WAIT_VEHICLE_COUNT'])
        if 'AVERAGE_INTERSECTION_PRESSURE' in weight:
            res += self._get_average_pressure(
                type = 'intersection',
                weight = weight['AVERAGE_INTERSECTION_PRESSURE'])
        if 'AVERAGE_ROADLINK_PRESSURE' in weight:
            res += self._get_average_pressure(
                type = 'roadlink',
                weight = weight['AVERAGE_ROADLINK_PRESSURE'])
        if 'AVERAGE_LANELINK_PRESSURE' in weight:
            res += self._get_average_pressure(
                type = 'lanelink',
                weight = weight['AVERAGE_LANELINK_PRESSURE'])
        if self.config['DELTA_REWARD']:
            res = res - self.last_reward
            self.last_reward += res
            res *= self.config['DELTA_REWARD_MULTIPLIER']
        return res

    def update_old(self):
        self.TS.update_old()


class Flow:
    def __init__(self):
        pass


"""CityFlow Environment, split action into and combine data from Intersections

Args:
    log_path: a folder to save log.
    work_folder: where to read flowFile and roadnetFile.
    config: config file.
    logfile: if set, save cityflow engine log in this file.
    seed: engine seed.
    suffix: whether add suffix to files used or generated in engine. if all 
        files save in same folder, we should add suffix to avoid overwrite.
    predefined_flow: whether use predefined flow file. if set False, will use 
        flow generator to generate flow.
Functions:
    ???
"""


class CityFlowEnv:
    def __init__(self, log_path, work_folder, config, logfile = '', seed = 0, 
                 suffix = True):
        self.log_path = log_path
        self.work_folder = work_folder
        self.config = config
        self.logfile = logfile
        self.seed = seed

        self.env_padding = 'ENV_PADDING' in config and config['ENV_PADDING']

        file_suffix = ''
        if suffix:
            file_suffix = '_' + str(seed)

        self.replay_path = os.path.join(self.log_path,
                                        "replay%s.txt" % file_suffix) + '.%04d'
        self.replay_count = 0
        config_dict = {
            "interval": self.config["INTERVAL"],
            "seed": seed,
            "dir": "",
            "roadnetFile": os.path.join(self.work_folder,
                                        self.config['ROADNET_FILE']),
            "flowFile": os.path.join(self.work_folder,
                                     self.config["FLOW_FILE"]),
            "rlTrafficLight": True,
            "laneChange": True,
            "saveReplay": self.config["SAVEREPLAY"],
            "roadnetLogFile": os.path.join(self.log_path,
                                           "roadnet%s.json" % file_suffix),
            "replayLogFile": self.replay_path % self.replay_count
        }

        if 'linux' not in sys.platform:
            print('[WARN ] not in linux platform, some log from CityFlowEnv '
                  'can\'t be recorded in log file!')

        config_path = os.path.join(log_path, "cityflow_config%s" % file_suffix)
        self.config_path = config_path
        with open(config_path, "w") as f:
            json.dump(config_dict, f)
            self.log("dump cityflow config:", config_path, level = 'TRACE')
        # print(config_path, log)
        if len(logfile) > 0:
            logfile = os.path.join(log_path, '%s%s' % (logfile, file_suffix))
            self.logfile = logfile
        self.init_engine()

        self.list_inter_log = None

        # check min action time
        if self.config["MIN_ACTION_TIME"] <= self.config["YELLOW_TIME"]:
            self.log("MIN_ACTION_TIME should include YELLOW_TIME", 
                     level = "ERROR")
            pass


        roadnet_info_keys = list(self.config['ROADNET_INFO'].keys())
        roadnet_info_keys.sort()
        virtual_inter_keys=None
        self.list_intersection = [Intersection(x, self.config, self.eng, 
                                  roadnet_info_keys, virtual_inter_keys)
                                  for x in roadnet_info_keys]
        self.all_vehicles = set()
        self.wait_ticks = 0
        self._set_observation_space()
        self._set_action_space()


    def log(self, *argv, **kwargs):
        if 'linux' in sys.platform:
            log(*argv, **kwargs)
        else:
            level = 'INFO' if 'level' not in kwargs else kwargs['level']
            if level in ['INFO', 'WARN', 'ERROR']:
                print('[%-5s]' % level, *argv)

    def init_engine(self):
        self.eng = Engine(self.config_path, 4, self.logfile)


    def _set_observation_space_one(self, i):
        observation_space = self.list_intersection[i].observation_space
        return observation_space

    def _set_observation_space(self):
        self.observation_space = []
        for i in range(len(self.list_intersection)):
            self.observation_space.append(self._set_observation_space_one(i))
        """
        for obs in self.observation_space:
            if obs != self.observation_space[0] and self.env_padding and False:
                self.log('observation space not all same, and env_padding is '
                         'set, which is not implemented!', level = 'ERROR')
                raise ValueError
        """
        self.observation_space = {
            'intersections': self.observation_space,
        }


    def _set_action_space_one(self, i):
        return self.list_intersection[i].action_space

    def _set_action_space(self):
        self.action_space = []
        for i in range(len(self.list_intersection)):
            self.action_space.append(self._set_action_space_one(i))


    def reset(self):
        self.eng.reset()
        self.replay_count += 1
        # self.eng.set_random_seed(self.seed + self.replay_count)
        if self.config['SAVEREPLAY']:
            self.eng.set_replay_file(self.replay_path % self.replay_count)
            old_replay = self.replay_path % (self.replay_count - 1)
            gzip_file(old_replay)

        # reset intersections (grid)
        for inter in self.list_intersection:
            inter.reset()

        # get new measurements
        for inter in self.list_intersection:
            inter.step_time()

        self.all_vehicles.clear()
        self.wait_ticks = 0

        state = self._collect_state()
       
        return state, {'average_time': 0.0, 'average_delay': 0.0}
    

    # STFGNN
    def _add_in_lane_flow(self):
        lane_vehicles=self.eng.get_lane_vehicles()
        for inter in self.list_intersection:
            for i,ln in enumerate(inter.observation_space['TSlane']): 
                for j,k in enumerate(ln):
                    inter.saved_lane_flow[i][j][-1]=set(lane_vehicles[k])-inter.pre_sec_vehicle[i][j]
                    inter.pre_sec_vehicle[i][j]=set(lane_vehicles[k])
                 

    def _collect_state(self):
        res = []
        for inter in self.list_intersection:
            res.append(inter.get_state())
        return res

    def _collect_reward(self):
        rew = []
        for inter in self.list_intersection:
            rew.append(inter.get_reward(self.config['REWARD_INFO']))
        return np.array(rew)

    def _is_done(self):
        return self.eng.get_current_time() >= self.config['EPISODE_LEN']

    def _average_time(self):
        return self.eng.get_average_travel_time()

    def _average_delay(self):
        return self.eng.get_average_delay()

    def step(self, action):
        if self.config['ACTION_PATTERN'] == 'switch':
            raise NotImplementedError('ACTION_PATTERN `switch` '
                                      'is not implemented')
        all_reward = np.zeros(len(self.list_intersection), dtype='float')
        self.step_lane_vehicles()
        for _ in range(self.config['MIN_ACTION_TIME']):
            self._add_in_lane_flow()
            self._inner_step(action, self.config['ACTION_PATTERN'])
            state = self._collect_state()
            all_reward += self._collect_reward()
            done = self._is_done()

        all_reward /= self.config['MIN_ACTION_TIME']
        infos = {
            'average_time': self._average_time(), 
            'average_delay': self._average_delay(),
            'current_time': self.eng.get_current_time(),
            'throughput': (len(self.all_vehicles) 
                           - len(self.eng.get_vehicles(True))),
            'average_wait_time': self.wait_ticks / len(self.all_vehicles) \
                                 if len(self.all_vehicles) else -1,
        }

        return state, all_reward, done, infos

    def step_lane_vehicles(self):
        for inter in self.list_intersection:
            inter.saved_lane_flow=[[ln[1:]+[set()] for ln in rl] for rl in inter.saved_lane_flow]
            inter.saved_phases[0:-1]=inter.saved_phases[1:]
            inter.saved_phases[-1]=-1



    def _inner_step(self, actions, pattern):
        for action, inter in zip(actions, self.list_intersection):
            inter.update_old()
            inter.act(action, pattern)
        for i in range(int(1 / self.config['INTERVAL'])):
            # self.flow_generator.check(self.eng.get_current_time())
            self.eng.next_step()  # catch errors and report to above
            if 'NOT_COUNT_VEHICLE_VOLUME' not in self.config or not self.config['NOT_COUNT_VEHICLE_VOLUME']:
                lane_vehicles = self.eng.get_lane_vehicles()
                vehicles = self.eng.get_vehicles(include_waiting = True)
                for v in vehicles:
                    self.all_vehicles.add(v)
                self.wait_ticks += len(vehicles)
                v_speeds = self.eng.get_vehicle_speed()
                for v in v_speeds:
                    if v_speeds[v] > 0.1:
                        self.wait_ticks -= 1
                # self.update_lane_vehicles(lane_vehicles)
        for inter in self.list_intersection:
            inter.step_time()

    def get_default_action(self):
        return [x.get_default_action() for x in self.list_intersection]