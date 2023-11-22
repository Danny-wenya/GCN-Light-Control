from .CityFlowEnv import *
from collections import defaultdict
from cityflow import Engine
import numpy as np
import json
import sys
import os
from utils.log import log
from utils.utils import gzip_file



class GCNIntersection(Intersection):
    def __init__(self, ID, config, eng, intersection_names, 
                 virtual_intersection_names):
        super().__init__(ID, config, eng, intersection_names, virtual_intersection_names)
        self.roadlink_number=None
        self.nodes_number=None
        self.turn=None
        self.nodes_name=None
        self.save_lane_count_step=12
        

    def reset(self):
        self._init_components()

    def _init_components(self):
        self.roadnet_info = self.config['ROADNET_INFO'][self.ID]
        self.phases=self.roadnet_info["phases"]

        self.TS = TrafficSignal(self.ID, self.eng, self.config['YELLOW_TIME'],
                                self.roadnet_info)
        self.last_reward = 0

        self.saved_phases = np.zeros((self.save_lane_count_step,), dtype = int)
        self.saved_phases[:] = -1

        self.lane_vehicle_count_mat = np.zeros((  # this is in count
            self.save_lane_count_step, 
            len(self.roadnet_info['connection']), 3), dtype = int)
        self.lane_vehicle_out_count_mat = np.zeros((
            self.save_lane_count_step, 
            len(self.roadnet_info['connection']), 3), dtype = int)
        


    def get_state(self):
        res = self.TS.get_state()
        DCphase = []
        res['InLaneVehicleNumber'] = self.lane_vehicle_count_mat
        res['OutLaneVehicleNumber'] = self.lane_vehicle_out_count_mat
        self.saved_phases[-1] = res['TSphase']
        res['TSprevphases'] = list(self.saved_phases)
        res['DCphase'] = DCphase
        res['pressure'] = self._get_pressure_observation()
        res['ID']=self.ID
        res['nodes_number']=self.nodes_number
        res['phases']=self.phases
        return res
    

class GCNCityFlowEnv(CityFlowEnv):
    def __init__(self,log_path, work_folder, config, logfile = '', seed = 0, suffix = True):
        self.log_path = log_path
        self.work_folder = work_folder
        self.config = config
        self.logfile = logfile
        self.seed = seed
        self.env_padding = 'ENV_PADDING' in config and config['ENV_PADDING']
        self.save_lane_count_step=12

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
            # raise ValueError

        roadnet_info_keys = list(self.config['ROADNET_INFO'].keys())
        roadnet_info_keys.sort()
        virtual_inters = self.config['VIRTUAL_INTERSECTION_NAMES']
        virtual_inter_keys = list(virtual_inters.keys())
        virtual_inter_keys.sort()
        self.virtual_road_out_lanes = np.zeros((len(virtual_inters), 3), int)
        self.virtual_direction_names = [
            'turn_left',
            'go_straight',
            'turn_right'
        ]
        self.list_intersection = [GCNIntersection(x, self.config, self.eng, 
                                  roadnet_info_keys, virtual_inter_keys)
                                  for x in roadnet_info_keys]
        for inter in self.list_intersection:
            assert self.virtual_direction_names == inter.observation_space[
                'DirectionNames'
            ]
        self.intername2idx = {}
        self.roadname2id = {}
        self.lanename2roaddirection = {}
        self.cycle_total_vehicles=defaultdict(set)
        self.last_in_lane_vehicles = {}
        self.all_vehicles = set()
        self.wait_ticks = 0
        for num, inter in enumerate(self.list_intersection):
            self.intername2idx[inter.ID] = num
            assert roadnet_info_keys[num] == inter.ID
            for rnum, rname in enumerate(inter.roadnames):
                self.roadname2id[rname] = [num, rnum]
        for ID in virtual_inters:
            v_inter = virtual_inters[ID]
            num = virtual_inter_keys.index(ID)
            roadnames = list(v_inter['connection'].keys())
            assert len(roadnames) == 1
            self.roadname2id[roadnames[0]] = [-1 - num, 0]
        for inter in self.list_intersection:
            inter._set_road_links_in(self.roadname2id)
            for rl in inter.roadnet_info['roadlinks']:
                if rl['startRoad'] in self.roadname2id:
                    num, rnum = self.roadname2id[rl['startRoad']]
                    if num < 0:
                        vid = -1 - num
                        typeid = self.virtual_direction_names.index(rl['type'])
                        self.virtual_road_out_lanes[
                            vid, typeid] += rl['lanenumber']
                        continue
                    ninter = self.list_intersection[num]
                    ninter.observation_space['RoadOutLanes'][rnum][
                        ninter.observation_space['DirectionNames'].index(
                            rl['type'])
                    ] += rl['lanenumber']
        # for inter in self.list_intersection:
        #     inter._update_lanename(self.lanename2roaddirection, 
        #                            self.list_intersection,
        #                            self.now_in_lane_vehicles,
        #                            self.last_in_lane_vehicles)
            # self.log(inter.ID, inter.observation_space)
        # self.update_virtual_lanename(virtual_inters, virtual_inter_keys)
        # self.log(self.lanename2roaddirection)

        self.list_inter_log = [[] for i in range(len(self.list_intersection))]

        self._set_adj_mat()
        self._set_observation_space()
        self._set_action_space()

        self.roadnet=json.load(open(self.config["ROADNET_FILE"]))
        self._init_net_lanes()
        self._init_adj()


    def _init_net_lanes(self):
        self.GCN_on_lane_vehichles=defaultdict(float)
        inters=[item["id"] for item in self.roadnet["intersections"]]
        virtual=[item["virtual"] for item in self.roadnet["intersections"]]
        inters_virtual=[value for i,value in enumerate(inters) if virtual[i]==True]
        endRoads=[link["endRoad"] for intersection in self.roadnet["intersections"] for link in intersection["roadLinks"]]
        startRoads=[link["startRoad"] for intersection in self.roadnet["intersections"] for link in intersection["roadLinks"]]
        
        virtual_road=[]
        for road in self.roadnet["roads"]:
            if road["startIntersection"] in inters_virtual:
                if road["id"] in endRoads:
                    print(road["id"])
                if road["id"] in startRoads:
                    virtual_road.append(road["id"])

        iters=self.roadnet["intersections"]
        self.vertices=defaultdict()
        self.vertices_flatten=defaultdict()
        self.on_lane_vehichles=defaultdict()
        self.now_in_lane_vehicles=defaultdict()
        self.GCN_on_lane_vehichles=defaultdict(float)
        num_n=0
        for inter in iters:
            vertices=defaultdict()
            num_r=0
            for link in inter["roadLinks"]:
                rldre=defaultdict(list)
                rldre["lanes"]=set([lanelink["startLaneIndex"] for lanelink in link["laneLinks"]])
                rldre["type"]=link["type"]
                rldre["roadName"]=link["startRoad"]
                rldre["virtual"]=True if link["startRoad"] in virtual_road else False
                rldre["node_number"]=num_n
                rldre["roadlink_number"]=num_r
                num_n+=1
                num_r+=1
                key=link["startRoad"]+"_"+link["type"]
                vertices[key]=rldre
                self.vertices_flatten[key]=rldre
                on_lanes={rldre["roadName"]+"_"+str(i):[0 for _ in range(self.save_lane_count_step)] for i in set(rldre["lanes"])}
                in_lanes={rldre["roadName"]+"_"+str(i):set() for i in set(rldre["lanes"])}
                self.on_lane_vehichles.update(on_lanes)
                self.now_in_lane_vehicles.update(in_lanes)
                self.GCN_on_lane_vehichles.update({key:[0 for _ in range(self.save_lane_count_step)]})
            self.vertices[inter["id"]]=vertices

        self.cycle_total_vehicles=self.now_in_lane_vehicles.copy()


        infos=defaultdict(list)
        for ilst in iters:   
            for rlk in ilst["roadLinks"]:
                lk=defaultdict(list)
                lk["lanes"]=set([lane["endLaneIndex"] for lane in rlk["laneLinks"]])
                lk["roadName"]=rlk["endRoad"]
                lk["startRoad"]=rlk["startRoad"]
                lk["type"]=rlk["type"]
                infos[rlk["endRoad"]].append(lk)

        for _,inter in self.vertices.items():
            for key,vt in inter.items():
                rn=vt["roadName"]
                for lk in infos[rn]:
                    if set(lk["lanes"])|set(vt["lanes"]):
                        node=lk["startRoad"]+"_"+lk["type"]
                        vt["from_vertices"].append(node)
                    else:
                        vt["from_vertices"]=None

    def _init_adj(self):
        self.nodes_name2num=dict()
        self.adj_nodes_names=dict()
        self.adj_nodes_nums=dict()
        for _,inter in self.vertices.items():
            for key,vt in inter.items():
                self.nodes_name2num[key]=vt["node_number"]
                self.adj_nodes_names[key]=vt["from_vertices"]

        for k,v in self.adj_nodes_names.items():
            self.adj_nodes_nums[self.nodes_name2num[k]]=[self.nodes_name2num[i] for i in v]

        N=len(self.adj_nodes_nums)
        self.adj=np.zeros([N,N])
        for i in range(N):
            self.adj[i,self.adj_nodes_nums[i]]=1
        for inter in self.list_intersection:
            vts=self.vertices[inter.ID]
            inter.roadlink_number=[v["roadlink_number"] for _,v in vts.items()]
            inter.nodes_number=[v["node_number"] for _,v in vts.items()]
            inter.turn=[v["type"] for _,v in vts.items()]
            inter.nodes_name=[k for k,_ in vts.items()]

         


    def step_lane_vehicles(self):
        for inter in self.list_intersection:
            inter.lane_vehicle_count_mat[:-1] = inter.lane_vehicle_count_mat[1:]
            inter.lane_vehicle_count_mat[-1] = 0
            inter.lane_vehicle_out_count_mat[:-1] = inter.lane_vehicle_out_count_mat[1:]
            inter.lane_vehicle_out_count_mat[-1] = 0
            inter.saved_phases[:-1] = inter.saved_phases[1:]

        for _,value in self.on_lane_vehichles.items():
            value[:-1],value[-1]=value[1:],0

        for _,value in self.GCN_on_lane_vehichles.items():
            value[:-1],value[-1]=value[1:],0

        for _,value in self.cycle_total_vehicles.items():
            value.clear()

    def temp1(self,):
        for i in self.now_in_lane_vehicles:  
            self.last_in_lane_vehicles[i] = self.now_in_lane_vehicles[i].copy()
            self.now_in_lane_vehicles[i].clear()



    def update_lane_vehicles(self, lane_vehicles):  
        # for i in self.now_in_lane_vehicles:  
        #     self.last_in_lane_vehicles[i] = self.now_in_lane_vehicles[i].copy()
        #     self.now_in_lane_vehicles[i].clear()
        self.temp1()

        for k in self.now_in_lane_vehicles:
            for i in lane_vehicles[k]:
                self.now_in_lane_vehicles[k].add(i)
                self.cycle_total_vehicles[k].add(i)

        for k,v in self.cycle_total_vehicles.items():
            self.on_lane_vehichles[k][-1]=len(v)
            

        for k,v in self.GCN_on_lane_vehichles.items():
            rn=self.vertices_flatten[k]["roadName"]
            ls=self.vertices_flatten[k]["lanes"]
            v[-1]=sum([self.on_lane_vehichles[rn+"_"+str(i)][-1] for i in ls])/len(ls)
           
                       

    def _collect_state(self):
        res = []
        turn_name=["go_straight","turn_left","turn_right"]
        for inter in self.list_intersection:
            state=inter.get_state()
            state["GCNflow"]=[self.GCN_on_lane_vehichles[i] for i in inter.nodes_name]
            state["turn"]=[]
            state["turn"]=[turn_name.index(i)+1 for i in inter.turn]
            res.append(state)
           
        return res


    def record_veh(self,):
            vehicles = self.eng.get_vehicles(include_waiting = True)
            for v in vehicles:
                self.all_vehicles.add(v)
            self.wait_ticks += len(vehicles)
            v_speeds = self.eng.get_vehicle_speed()
            for v in v_speeds:
                if v_speeds[v] > 0.1:
                    self.wait_ticks -= 1

    def temp_inter(self,actions,pattern):
        for action, inter in zip(actions, self.list_intersection):
            inter.update_old()
            inter.act(action, pattern)


    def _inner_step(self, actions, pattern):
        # for action, inter in zip(actions, self.list_intersection):
        #     inter.update_old()
        #     inter.act(action, pattern)

        self.temp_inter(actions,pattern)
        for i in range(int(1 / self.config['INTERVAL'])):
            # self.flow_generator.check(self.eng.get_current_time())
            self.eng.next_step()  # catch errors and report to above
            if 'NOT_COUNT_VEHICLE_VOLUME' not in self.config or not self.config['NOT_COUNT_VEHICLE_VOLUME']:
                lane_vehicles = self.eng.get_lane_vehicles()
                # vehicles = self.eng.get_vehicles(include_waiting = True)
                # for v in vehicles:
                #     self.all_vehicles.add(v)
                # self.wait_ticks += len(vehicles)
                # v_speeds = self.eng.get_vehicle_speed()
                # for v in v_speeds:
                #     if v_speeds[v] > 0.1:
                #         self.wait_ticks -= 1

                self.record_veh()
                self.update_lane_vehicles(lane_vehicles)
        for inter in self.list_intersection:
            inter.step_time()


    def min_step(self,action):
        all_reward = np.zeros(len(self.list_intersection), dtype='float')
        for i in range(self.config['MIN_ACTION_TIME']):
            if i == 0:
                self.step_lane_vehicles()
            self._inner_step(action, self.config['ACTION_PATTERN'])
            state = self._collect_state()
            all_reward += self._collect_reward()
            done = self._is_done()

        all_reward /= self.config['MIN_ACTION_TIME']
        return state,all_reward, done

    def step(self, action):
            if self.config['ACTION_PATTERN'] == 'switch':
                raise NotImplementedError('ACTION_PATTERN `switch` '
                                        'is not implemented')

            # all_reward = np.zeros(len(self.list_intersection), dtype='float')
            # for i in range(self.config['MIN_ACTION_TIME']):
            #     if i == 0:
            #         self.step_lane_vehicles()
            #     self._inner_step(action, self.config['ACTION_PATTERN'])
            #     state = self._collect_state()
            #     all_reward += self._collect_reward()
            #     done = self._is_done()

            # all_reward /= self.config['MIN_ACTION_TIME']

            state,all_reward, done=self.min_step(action)
            infos = {
                'average_time': self._average_time(), 
                'average_delay': self._average_delay(),
                'current_time': self.eng.get_current_time(),
                'throughput': (len(self.all_vehicles) 
                            - len(self.eng.get_vehicles(True))),
                'average_wait_time': self.wait_ticks / len(self.all_vehicles) \
                                    if len(self.all_vehicles) else -1,
            }

            return state,all_reward, done, infos



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

            # self.flow_generator.reset()

            for k in self.now_in_lane_vehicles:
                self.now_in_lane_vehicles[k] = set()
            for k in self.last_in_lane_vehicles:
                self.last_in_lane_vehicles[k] = set()

            self.all_vehicles.clear()
            self.wait_ticks = 0

            state = self._collect_state()
            return state, {'average_time': 0.0, 'average_delay': 0.0}



