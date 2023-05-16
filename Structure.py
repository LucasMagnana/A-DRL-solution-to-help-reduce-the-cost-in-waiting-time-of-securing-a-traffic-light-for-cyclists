import threading
import torch
import numpy as np
import os
import copy

from DQNAgent import DQNAgent
from PPOAgent import PPOAgent
from TD3Agent import TD3Agent

class Structure:
    def __init__(self, edges, net, traci, simu_length, method, test, min_group_size, alpha_bike, use_drl=True, cnn=False, open=True):


        self.module_traci = traci
        
        self.min_group_size = min_group_size
        self.activated = False

        self.net = net  

        self.tls = self.net.getEdge("E0").getTLS()  


        self.simu_length = simu_length

        self.method = method

        self.open = open


        self.next_step_decision = 0

        self.use_drl = use_drl

        self.phases = None

        if(self.use_drl):
            self.max_reward = None
            self.cnn = cnn
            self.drl_cum_reward = 0
            self.drl_decision_made = False
            self.test = test
            if(self.cnn):
                self.ob_shape = (2, 8, int(self.net.getEdge("E0").getLength()//5+2))
            else:
                self.ob_shape = [21]
                self.car_lanes_capacity = 20
                self.bike_lanes_capacity = 30

            self.action_space = 4

            if(os.path.exists("files/train/"+str(alpha_bike)+"/"+self.method+"_trained.n")):
                model_to_load = "files/train/"+str(alpha_bike)+"/"+self.method+"_trained.n"
            else:
                model_to_load = None


            if(self.method == "DQN"):
                self.drl_agent = DQNAgent(self.ob_shape, self.action_space, model_to_load=model_to_load)
            elif(self.method == "3DQN"):
                self.drl_agent = DQNAgent(self.ob_shape, self.action_space, double=True, duelling=True, model_to_load=model_to_load)
            elif(self.method == "TD3"):
                self.drl_agent = TD3Agent(self.ob_shape, 1)
            elif(self.method == "PPO"):
                self.drl_agent = PPOAgent(self.ob_shape, 1, model_to_load=model_to_load, continuous_action_space=True)
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
            self.dict_transition = {"0;1": 1, "0;2": 1, "0;3": 1, "1,0": 3, "1;2":3, "1;3":3, "2;0": 5, "2;1": 5, "2;3": 5, "3;0": 7, "3;1": 7, "3;2": 7}


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
        


    def reset(self, dict_cyclists, dict_scenario):
        self.dict_cyclists = dict_cyclists
        self.dict_scenario = dict_scenario

        self.id_cyclists_crossing_struct = []
        self.id_cyclists_waiting = []
        self.num_cyclists_crossed = 0
        self.num_cyclists_canceled = 0

        self.next_step_decision = 0   
        self.module_traci.trafficlight.setPhase(self.tls.getID(), 0)
        

        if(self.use_drl):
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





    def drl_decision_making(self, step):  
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
                if(self.max_reward == None or reward < self.max_reward):
                    if(reward != 0):
                        self.max_reward = reward
                reward /= self.max_reward
                reward = -reward
                self.drl_cum_reward += reward
                if(not self.test):                   
                    if(self.method == "PPO"):                   
                        self.drl_agent.memorize(self.ob_prec, self.val, self.action_probs, self.action, self.ob, reward, False)  
                    else:
                        self.drl_agent.memorize(self.ob_prec, self.action, self.ob, reward, False)  

            if(self.method == "PPO"):  
                self.action, self.val, self.action_probs = self.drl_agent.act(self.ob)
            else:
                self.action = self.drl_agent.act(self.ob)


            if(self.action != self.actual_phase):
                self.actual_phases = []
                key_dict_transition = str(self.actual_phase)+";"+str(self.action)
                
                if(key_dict_transition in self.dict_transition):
                    self.actual_phases.append(self.phases[self.dict_transition[key_dict_transition]])

                self.actual_phases.append(self.phases[self.phases_correspondance[self.action]])

                self.update_tls_program()
                self.actual_phase = self.action


                    





    def create_observation_cnn(self):
        ob_num = []
        ob_speed = []
        for edge_id in range(4):
            edge = self.net.getEdge("E"+str(edge_id))
            edge_start_x = edge.getFromNode().getCoord()[0]
            ob_lane_num = np.zeros((2, self.ob_shape[-1]))
            ob_lane_speed = np.zeros((2, self.ob_shape[-1]))
            for vehicle_id in self.module_traci.edge.getLastStepVehicleIDs(edge.getID()):
                if("_c" in vehicle_id):
                    index_v = 0
                else:
                    index_v = 1
                index = int(self.module_traci.vehicle.getLaneID(vehicle_id)[-1])
                position_in_grid = int(round(self.module_traci.vehicle.getPosition(vehicle_id)[0]-edge_start_x))//5
                ob_lane_num[index_v][index] += 1
                ob_lane_speed[index_v][index] += self.module_traci.vehicle.getSpeed(vehicle_id)

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
        for edge_id in range(4):
            edge = self.net.getEdge("E"+str(edge_id))
            edge_start_x = edge.getFromNode().getCoord()[0]
            num_bikes = 0
            num_stopped_bikes = 0
            num_cars = 0
            num_stopped_cars = 0
            for vehicle_id in self.module_traci.edge.getLastStepVehicleIDs(edge.getID()):
                if("_c" in vehicle_id):
                    num_cars += 1
                    if(self.module_traci.vehicle.getSpeed(vehicle_id)<0.5):
                        num_stopped_cars += 1
                else:
                    num_bikes += 1
                    if(self.module_traci.vehicle.getSpeed(vehicle_id)<0.5):
                        num_stopped_bikes += 1
            ob.append(num_bikes/self.bike_lanes_capacity)
            ob.append(num_stopped_bikes/self.bike_lanes_capacity)
            ob.append(num_cars/self.car_lanes_capacity)
            ob.append(num_stopped_cars/self.car_lanes_capacity)

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

    


