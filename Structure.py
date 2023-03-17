import threading
import torch
import numpy as np
import os

from DQNAgent import DQNAgent
from PPOAgent import PPOAgent

class Structure:
    def __init__(self, edges, net, traci, simu_length, method, test, min_group_size, alpha_bike, open=True):


        self.module_traci = traci
        
        self.min_group_size = min_group_size
        self.activated = False

        self.net = net  

        self.tls = self.net.getEdge("E1").getTLS()  


        self.simu_length = simu_length

        self.method = method

        if(self.method == "actuated"):
            self.actuated_next_change_step = 5
        self.open = open


        self.next_step_decision = 0

        self.use_drl = "DQN" in self.method or "PPO" in self.method

        if(self.use_drl):
            self.test = test
            self.ob_shape = (2, 4, int(self.net.getEdge("E1").getLength()//5+2))

            if(self.test):
                model_to_load = "files/train/"+str(alpha_bike)+"/"+self.method+"_trained.n"
            else:
                model_to_load = None

            if(self.method == "DQN"):
                self.drl_agent = DQNAgent(self.ob_shape, 2, model_to_load=model_to_load)
            elif(self.method == "3DQN"):
                self.drl_agent = DQNAgent(self.ob_shape, 2, double=True, duelling=True, model_to_load=model_to_load)
            elif(self.method == "PPO"):
                if(os.path.exists("files/train/"+str(alpha_bike)+"/"+self.method+"_trained.n")):
                    model_to_load = "files/train/"+str(alpha_bike)+"/"+self.method+"_trained.n"
                self.drl_agent = PPOAgent(self.ob_shape, 2, model_to_load=model_to_load)
                self.val = None
                self.action_probs = None


            self.bikes_waiting_time_coeff = alpha_bike
            self.cars_waiting_time_coeff = 1-alpha_bike


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
            self.action = -1
            self.ob = []
            self.bikes_waiting_time = 0
            self.cars_waiting_time = 0
            if(self.method == "PPO"):
                self.val = None
                self.action_probs = None
            elif("DQN" in self.method):
                self.next_step_learning = self.drl_agent.hyperParams.LEARNING_STEP
        elif(self.method == "actuated"):
            self.actuated_next_change_step = 5



    def step(self, step, edges):

        self.update_next_step_decision(step)
        
        if(self.use_drl):
            if(self.module_traci.trafficlight.getPhase(self.tls.getID())%2 == 0):
                self.drl_decision_making(step)
            if("DQN" in self.method and step > self.drl_agent.hyperParams.LEARNING_START and not self.test and step>self.next_step_learning):
                self.drl_agent.learn()
                self.next_step_learning += self.drl_agent.hyperParams.LEARNING_STEP
        elif(self.method == "actuated"):
            if(step > self.next_step_decision):
                self.actuated_decision_making(step)





    def drl_decision_making(self, step):       
        self.create_observation()
        self.ob_prec = self.ob
        self.ob = self.create_observation()

        if(len(self.ob_prec) == 0):
            self.calculate_sum_waiting_time()
        else:
            if(not self.test and self.action >= 0):
                reward = self.calculate_reward()
                if("DQN" in self.method):
                    self.drl_agent.memorize(self.ob_prec, self.action, self.ob, reward, False)  
                elif(self.method == "PPO"):                   
                    self.drl_agent.memorize(self.ob_prec, self.val, self.action_probs, self.action, self.ob, reward, False)  

            if("DQN" in self.method):
                self.action = self.drl_agent.act(self.ob)
            elif(self.method == "PPO"):  
                self.action, self.val, self.action_probs = self.drl_agent.act(self.ob)

            if(self.action == 1):
                self.change_light_phase()


    def actuated_decision_making(self, step):

        if(self.actuated_next_change_step < self.next_step_decision):
            self.actuated_next_change_step = step+5
            return 

        detector_distance = 50

        if(self.module_traci.trafficlight.getPhase(self.tls.getID()) == 0):
            green_lanes = ["E1", "E4"]
        elif(self.module_traci.trafficlight.getPhase(self.tls.getID()) == 2):
            green_lanes = ["E2", "E3"]

        min_distance_car = 9999
        min_distance_bike = 9999
        for lane_id in green_lanes:
            edge = self.net.getEdge(lane_id)
            for i in self.module_traci.edge.getLastStepVehicleIDs(lane_id):
                dist = edge.getLength()-self.module_traci.vehicle.getLanePosition(i)
                if("_c" in i and dist < min_distance_car):
                    min_distance_car = dist
                elif("_c" not in i and dist < min_distance_bike):
                    min_distance_bike = dist

        if(min_distance_car < detector_distance or min_distance_bike < detector_distance):
            self.actuated_next_change_step = step + 5

        if(step > self.actuated_next_change_step):
            self.change_light_phase()




    def create_observation(self):
        ob_num = []
        ob_speed = []
        for i in range(4):
            edge = self.net.getEdge("E"+str(i+1))
            edge_start_x = edge.getFromNode().getCoord()[0]
            ob_lane_num = np.zeros(self.ob_shape[-1])
            ob_lane_speed = np.zeros(self.ob_shape[-1])
            for vehicle_id in self.module_traci.edge.getLastStepVehicleIDs(edge.getID()):
                index = int(self.module_traci.vehicle.getLaneID(vehicle_id)[-1])
                position_in_grid = int(round(self.module_traci.vehicle.getPosition(vehicle_id)[0]-edge_start_x))//5
                ob_lane_num[index] += 1
                ob_lane_speed[index] += self.module_traci.vehicle.getSpeed(vehicle_id)

            for j in range(len(ob_lane_num)):
                if(ob_lane_num[j] != 0):
                    ob_lane_speed[j] /= ob_lane_num[j]

            ob_num.append(ob_lane_num)
            ob_speed.append(ob_lane_speed)

        ob = np.array([ob_num, ob_speed])
        return ob
        


    def calculate_sum_waiting_time(self):
        last_cars_wt = self.cars_waiting_time
        last_bikes_wt = self.bikes_waiting_time
        self.cars_waiting_time = 0
        self.bikes_waiting_time = 0
        for vehicle_id in self.module_traci.vehicle.getIDList():
            if("_c" in vehicle_id):
                self.cars_waiting_time += self.dict_scenario["cars"][int(vehicle_id[:-2])]["waiting_time"]
            else:
                self.bikes_waiting_time += self.dict_scenario["bikes"][int(vehicle_id)]["waiting_time"]
        return last_cars_wt, last_bikes_wt

        

    def calculate_reward(self):
        last_cars_wt, last_bikes_wt = self.calculate_sum_waiting_time()
        diff_cars =  last_cars_wt - self.cars_waiting_time
        diff_bikes = last_bikes_wt - self.bikes_waiting_time
        return self.bikes_waiting_time_coeff*diff_bikes+self.cars_waiting_time_coeff*diff_cars

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

    


