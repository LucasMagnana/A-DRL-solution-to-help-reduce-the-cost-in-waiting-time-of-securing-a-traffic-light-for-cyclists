import threading
import torch
import numpy as np
import os
import copy

from DQNAgent import DQNAgent
from PPOAgent import PPOAgent
from TD3Agent import TD3Agent

class Structure:
    def __init__(self, edges, list_edges_name, net, traci, method, test, min_group_size, alpha_bike, use_drl=True, cnn=True, open=True):


        self.module_traci = traci
        
        self.min_group_size = min_group_size
        self.activated = False

        self.net = net  

        self.list_edges_name = list_edges_name

        self.tls = self.net.getEdge("E_EW").getTLS()  

        self.method = method

        self.open = open


        self.next_step_decision = 0

        self.use_drl = use_drl

        self.phases = None

        if(self.use_drl):
            self.max_reward = 1
            self.cnn = cnn
            self.drl_cum_reward = 0
            self.global_drl_cum_reward = 0
            self.drl_mean_reward = -1
            self.drl_decision_made = False
            self.test = test
            if(self.cnn):
                self.ob_shape = (2, 8, int(self.net.getEdge("E_EW").getLength()//5)+2)
            else:
                self.ob_shape = [21]
                self.lanes_capacities = [10, 10]
                self.bike_lanes_capacity = 36

            self.action_space = 4

            if(os.path.exists("files/train/"+self.method+"_trained.n")):
                print("WARNING: Loading model...")
                model_to_load = "files/train/"+self.method+"_trained.n"
            else:
                print("WARNING: Creating a new model...")
                model_to_load = None


            if(self.method == "DQN"):
                self.drl_agent = DQNAgent(self.ob_shape, self.action_space, model_to_load=model_to_load)
            elif(self.method == "3DQN"):
                self.drl_agent = DQNAgent(self.ob_shape, self.action_space, double=True, duelling=True, model_to_load=model_to_load)
            elif(self.method == "TD3"):
                self.drl_agent = TD3Agent(self.ob_shape, 1)
            elif(self.method == "PPO"):
                self.drl_agent = PPOAgent(self.ob_shape, self.action_space, model_to_load=model_to_load)
                self.val = None
                self.action_probs = None


            self.bikes_waiting_time_coeff = 1 #alpha_bike
            self.cars_waiting_time_coeff = 1 #1-alpha_bike


    def create_tls_phases(self):
        if(self.phases == None):
            self.phases = []
            for p in self.module_traci.trafficlight.getAllProgramLogics(self.tls.getID())[0].getPhases():
                self.phases.append(p.state)
            self.actual_phase = 0
            self.phases_correspondance = range(0, 8, 2)
            #self.dict_transition = {"0;1": 1, "0;2": 3, "0;3": 3, "1;2":3, "1;3":3, "2;0": 7, "2;1": 7, "2;3": 5, "3;0": 7, "3;1": 7}
            self.dict_transition = {"0;1": 1, "0;2": 1, "0;3": 1, "1;0": 3, "1;2":3, "1;3":3, "2;0": 5, "2;1": 5, "2;3": 5, "3;0": 7, "3;1": 7, "3;2": 7}


    '''def create_tls_phases(self):
        self.phases = []
        num_lane = 4
        neutral_phase = "r"*3*num_lane*2

        for i in range(num_lane):
            green_phase = neutral_phase
            yellow_phase = neutral_phase

            if(i%2 == 0):
                green_phase = neutral_phase[:i*3] + "GGg" + neutral_phase[i*3+3:12+i*3] + "GGg" + neutral_phase[12+i*3+3:]
            else:
                green_phase = neutral_phase[:i*3] + "gGg" + neutral_phase[i*3+3:12+i*3] + "gGg" + neutral_phase[12+i*3+3:]
           
            yellow_phase = neutral_phase[:i*3] + "yyy" + neutral_phase[i*3+3:12+i*3] + "yyy" + neutral_phase[12+i*3+3:]

            if(self.action_space == 9):
                green_dur = 30
            elif(self.action_space == 4):
                green_dur = 9999

            self.phases.append(self.module_traci.trafficlight.Phase(duration=green_dur, state=green_phase, minDur=green_dur, maxDur=green_dur))
            self.phases.append(self.module_traci.trafficlight.Phase(duration=4, state=yellow_phase, minDur=4, maxDur=4))

        if(self.action_space == 4):
            self.original_phases = copy.deepcopy(self.phases)
            self.actual_phase = 0
            self.phases = [self.original_phases[-1], self.original_phases[-1], self.original_phases[0]]'''

        



    def update_tls_program(self):
        self.module_traci.trafficlight.setRedYellowGreenState(self.tls.getID(), self.actual_phases[0])
        self.time_elapsed_in_chosen_phase = 0   
        if(len(self.actual_phases) > 1):
            self.transition_end = 3
        


    def reset(self, dict_scenario):
        self.dict_scenario = dict_scenario

        self.id_cyclists_crossing_struct = []
        self.id_cyclists_waiting = []
        self.num_cyclists_crossed = 0
        self.num_cyclists_canceled = 0

        self.next_step_decision = 0   
        self.module_traci.trafficlight.setPhase(self.tls.getID(), 0)
        

        if(self.use_drl):
            if(self.drl_agent.num_decisions_made != 0):                
                self.drl_mean_reward = self.global_drl_cum_reward/self.drl_agent.num_decisions_made
            self.drl_cum_reward = 0
            self.drl_decision_made = False
            self.action = None
            self.ob = []
            self.bikes_waiting_time = 0
            self.cars_waiting_time = 0

            self.actual_phases = [self.phases[0]]

            self.actual_phase = 0

            self.update_tls_program()



    def step(self, step, edges):
        self.update_next_step_decision(step)
        
        if(self.use_drl):

            if(self.action_space == 9):
                if(self.module_traci.trafficlight.getPhase(self.tls.getID()) == 0 and not self.drl_decision_made):
                    self.drl_decision_making(step)
                    self.drl_decision_made = True
                elif(self.module_traci.trafficlight.getPhase(self.tls.getID()) != 0 and self.drl_decision_made):
                    self.drl_decision_made = False
            else:
                if(len(self.actual_phases) > 1):
                    if(self.transition_end == 0):
                        self.actual_phases.pop(0)
                        self.update_tls_program()
                    else:
                        self.transition_end -= 1
                elif(self.time_elapsed_in_chosen_phase >= 10):
                    self.drl_decision_making(step)
                else:
                    self.time_elapsed_in_chosen_phase += 1





    def drl_decision_making(self, step, end=False, forced_stop=False):  
        self.ob_prec = self.ob
        if(self.cnn):
            self.ob = self.create_observation_cnn()
        else:
            self.ob = self.create_observation()

        if(len(self.ob_prec) == 0):
            self.calculate_sum_waiting_time()
        else:
            if(self.action != None):
                reward = self.calculate_reward()
                self.global_drl_cum_reward += reward
                if(reward < self.max_reward):
                    if(reward != 0):
                        self.max_reward = reward
                reward /= self.drl_mean_reward  #self.max_reward
                reward = -reward

                if(forced_stop):
                    reward = -10000
                    
                self.drl_cum_reward += reward
                if(not self.test):                   
                    if(self.method == "PPO"):                   
                        self.drl_agent.memorize(self.ob_prec, self.val, self.action_probs, self.action, self.ob, reward, end)  
                    else:
                        self.drl_agent.memorize(self.ob_prec, self.action, self.ob, reward, end)  

            if(self.method == "PPO"):  
                self.action, self.val, self.action_probs = self.drl_agent.act(self.ob)
            else:
                self.action = self.drl_agent.act(self.ob)

            #print("decision", self.actual_phase, "->", self.action)
            if(self.action != self.actual_phase):
                self.actual_phases = []
                key_dict_transition = str(self.actual_phase)+";"+str(self.action)
                
                if(key_dict_transition in self.dict_transition):
                    self.actual_phases.append(self.phases[self.dict_transition[key_dict_transition]])

                self.actual_phases.append(self.phases[self.phases_correspondance[self.action]])

                self.actual_phase = self.action

            self.update_tls_program()


                    





    def create_observation_cnn(self):
        ob_num = []
        ob_speed = []
        for en in self.list_edges_name:
            edge = self.net.getEdge("E_"+en)
            ob_lane_num = np.zeros((2, self.ob_shape[-1]))
            ob_lane_speed = np.zeros((2, self.ob_shape[-1]))
            for vehicle_id in self.module_traci.edge.getLastStepVehicleIDs(edge.getID()):
                index_lane = int(self.module_traci.vehicle.getLaneID(vehicle_id)[-1])
                if("N" in en):
                    pos = abs(self.module_traci.vehicle.getPosition(vehicle_id)[1])
                else:
                    pos = abs(self.module_traci.vehicle.getPosition(vehicle_id)[0])

                index = int(pos//5)
                position_in_grid = 0
                ob_lane_num[index_lane][index] += 1
                ob_lane_speed[index_lane][index] += self.module_traci.vehicle.getSpeed(vehicle_id)

            for i in range(len(ob_lane_num)):
                for j in range(len(ob_lane_num[i])):
                    if(ob_lane_num[i][j] > 1):
                        ob_lane_speed[i][j] /= ob_lane_num[i][j]

                ob_num.append(ob_lane_num[i])
                ob_speed.append(ob_lane_speed[i])

        ob = np.array([ob_num, ob_speed])
        return ob


    def create_observation(self):
        ob = []
        for en in self.list_edges_name:
            edge = self.net.getEdge("E_"+en)
            edge_start_x = edge.getFromNode().getCoord()[0]
            tab_num_vehicles = [0, 0]
            tab_num_stopped_vehicles = [0, 0]
            for vehicle_id in self.module_traci.edge.getLastStepVehicleIDs(edge.getID()):
                num_lane = int(self.module_traci.vehicle.getLaneID(vehicle_id)[-1])
                tab_num_vehicles[num_lane] += 1
                if(self.module_traci.vehicle.getSpeed(vehicle_id)<0.5):
                    tab_num_stopped_vehicles[num_lane] += 1

            ob.append(tab_num_vehicles[0]/self.lanes_capacities[0])
            ob.append(tab_num_stopped_vehicles[0]/self.lanes_capacities[0])
            ob.append(tab_num_vehicles[1]/self.lanes_capacities[1])
            ob.append(tab_num_stopped_vehicles[1]/self.lanes_capacities[1])

        ob.append(self.time_elapsed_in_chosen_phase/30)
        light_phase_encoded = np.zeros(4)
        light_phase_encoded[self.actual_phase] = 1
        
        return np.concatenate((ob, light_phase_encoded))
        


    def calculate_sum_waiting_time(self):
        last_cars_wt = self.cars_waiting_time
        last_bikes_wt = self.bikes_waiting_time
        self.cars_waiting_time = 0
        self.bikes_waiting_time = 0
        for vehicle_type in self.dict_scenario:           
            for vehicle_id in self.dict_scenario[vehicle_type]:
                if(vehicle_type == "cars"):
                    self.cars_waiting_time += self.dict_scenario[vehicle_type][vehicle_id]["waiting_time"]
                else:
                    self.bikes_waiting_time += self.dict_scenario[vehicle_type][vehicle_id]["waiting_time"]

        
        return last_cars_wt, last_bikes_wt

        
    def calculate_reward(self):
        waiting_vehicle_number = 0
        for vehi_id in self.module_traci.vehicle.getIDList():
            if(self.module_traci.vehicle.getSpeed(vehi_id)<0.5):
                waiting_vehicle_number += 1
        return -(waiting_vehicle_number**2)


    '''def calculate_reward(self):
        last_cars_wt, last_bikes_wt = self.calculate_sum_waiting_time()
        diff_cars =  last_cars_wt - self.cars_waiting_time
        diff_bikes = last_bikes_wt - self.bikes_waiting_time
        
        return (self.bikes_waiting_time_coeff*diff_bikes+self.cars_waiting_time_coeff*diff_cars)'''

    def update_next_step_decision(self, step):
        if(step > self.next_step_decision):
            if(self.module_traci.trafficlight.getPhase(self.tls.getID())%2 == 1):
                self.next_step_decision = step + 13

            

    def change_light_phase(self):
        self.module_traci.trafficlight.setPhase(self.tls.getID(), self.module_traci.trafficlight.getPhase(self.tls.getID())+1)

            

    def static_decision_making(self, step):
        if(self.module_traci.trafficlight.getPhase(self.tls.getID()) == 0 and\
        len(set(self.module_traci.edge.getLastStepVehicleIDs(e)) & set(self.id_cyclists_crossing_struct)) >= self.min_group_size):           
            self.change_light_phase()       
        elif(self.module_traci.trafficlight.getPhase(self.tls.getID()) == 2 and\
        len(set(self.module_traci.edge.getLastStepVehicleIDs(e)) & set(self.id_cyclists_crossing_struct)) == 0):
            self.change_light_phase()

    


