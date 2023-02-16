import threading
import torch
import numpy as np

from DQNAgent import DQNAgent

class Structure:
    def __init__(self, start_edge, end_edge, edges, net, traci, config, simu_length, use_drl, actuated, test, min_group_size, open=True):

        for e in edges:
            id = e.getID()
            if(id == start_edge):
                self.start_edge = e
            if(id == end_edge):
                self.end_edge  = e

        self.path = net.getShortestPath(self.start_edge, self.end_edge, vClass='bicycle')[0]
        self.path = [e.getID() for e in self.path]

        self.module_traci = traci
        
        self.min_group_size = min_group_size
        self.activated = False

        self.net = net

        self.config = config


        self.simu_length = simu_length

        self.actuated = actuated
        if(self.actuated):
            self.actuated_next_change_step = 5
        self.open = open


        self.next_step_decision = 0

        self.use_drl = use_drl
        if(self.use_drl):
            self.test = test
            self.width_ob = (2, 2, int(self.start_edge.getLength()//5+2))

            if(self.test):
                actor_to_load = "files/train/config_"+str(self.config)+"/3DQN_trained.n"
            else:
                actor_to_load = None

            self.drl_agent = DQNAgent(self.width_ob, 2, actor_to_load=actor_to_load)

            self.bikes_waiting_time_coeff = 0.3
            self.cars_waiting_time_coeff = 0.7


    def reset(self, dict_cyclists, dict_scenario):
        self.dict_cyclists = dict_cyclists
        self.dict_scenario = dict_scenario

        self.id_cyclists_crossing_struct = []
        self.id_cyclists_waiting = []
        self.num_cyclists_crossed = 0
        self.num_cyclists_canceled = 0

        self.next_step_decision = 0

        e = self.path[0]
        tls = self.net.getEdge(e).getTLS()       
        self.module_traci.trafficlight.setPhase(tls.getID(), 0)

        if(self.use_drl):
            self.next_step_learning = self.drl_agent.hyperParams.LEARNING_STEP
            self.action = -1
            self.ob = []
            self.bikes_waiting_time = 0
            self.cars_waiting_time = 0
        elif(self.actuated):
            self.actuated_next_change_step = 5



    def step(self, step, edges):

        self.update_next_step_decision(step)
        
        if(self.use_drl):
            if(step > self.next_step_decision):
                self.drl_decision_making(step)
            if(step > self.drl_agent.hyperParams.LEARNING_START and not self.test and step>self.next_step_learning):
                self.drl_agent.learn()
                self.next_step_learning += self.drl_agent.hyperParams.LEARNING_STEP
        elif(self.actuated):
            if(step > self.next_step_decision):
                self.actuated_decision_making(step)
        else:            
            for i in self.module_traci.edge.getLastStepVehicleIDs(self.start_edge.getID()):
                if("_c" not in i and i not in self.id_cyclists_waiting and i not in self.id_cyclists_crossing_struct and self.dict_cyclists[i].struct_candidate):
                    if(len(self.id_cyclists_waiting)==0):
                        for j in range(len(self.id_cyclists_crossing_struct)-1, -1, -1):
                            pos = self.module_traci.vehicle.getPosition(self.id_cyclists_crossing_struct[j])
                            if(self.module_traci.vehicle.getDrivingDistance2D(i, pos[0], pos[1])>0 and self.module_traci.vehicle.getDrivingDistance2D(i, pos[0], pos[1])<=5):
                                self.id_cyclists_waiting.append(i)
                                self.dict_cyclists[i].step_cancel_struct_candidature = step+99999
                                break
                        if(i not in self.id_cyclists_waiting and self.module_traci.vehicle.getSpeed(i)<= 3):
                            self.id_cyclists_waiting.append(i)
                            self.dict_cyclists[i].step_cancel_struct_candidature = step+99999
                        
                    else:
                        for j in range(len(self.id_cyclists_waiting)-1, -1, -1):
                            pos = self.module_traci.vehicle.getPosition(self.id_cyclists_waiting[j])
                            if(self.module_traci.vehicle.getDrivingDistance2D(i, pos[0], pos[1])<=3):
                                self.id_cyclists_waiting.append(i)
                                self.dict_cyclists[i].step_cancel_struct_candidature = step+99999
                                break

            if(len(self.id_cyclists_waiting)>=self.min_group_size):
                self.activated = True
                min_max_speed = 100
                for i in self.id_cyclists_waiting:
                    self.dict_cyclists[i].cross_struct()
                    if(self.dict_cyclists[i].max_speed < min_max_speed):
                        min_max_speed = self.dict_cyclists[i].max_speed
                    #print(i, "crossing")
                    self.id_cyclists_crossing_struct.append(i)
                self.id_cyclists_waiting = []

                for i in self.id_cyclists_crossing_struct:
                    self.dict_cyclists[i].set_max_speed(min_max_speed)

                #print("activated at step", step)

            if(len(self.id_cyclists_crossing_struct)>0):

                if(set(self.module_traci.edge.getLastStepVehicleIDs(self.start_edge.getID())) & set(self.id_cyclists_crossing_struct)):
                    for i in self.id_cyclists_waiting:
                        self.id_cyclists_crossing_struct.append(i)
                        self.dict_cyclists[i].cross_struct()
                        self.dict_cyclists[i].set_max_speed(self.dict_cyclists[self.id_cyclists_crossing_struct[0]].max_speed)                  
                        self.id_cyclists_waiting.remove(i)


            for i in self.id_cyclists_waiting:
                if(i not in self.module_traci.edge.getLastStepVehicleIDs(self.start_edge.getID())):
                    self.id_cyclists_waiting.remove(i)


            if(step > self.next_step_decision):
                self.static_decision_making(step)






    def drl_decision_making(self, step):
        e = self.path[0]
        tls = self.net.getEdge(e).getTLS()     
        self.create_observation()
        self.ob_prec = self.ob
        self.ob = self.create_observation()

        if(len(self.ob_prec) == 0):
            self.calculate_sum_waiting_time()
        else:
            if(not self.test and self.action >= 0):
                reward = self.calculate_reward()
                self.drl_agent.memorize(self.ob_prec, self.action, self.ob, reward, False)  
            self.action = self.drl_agent.act(self.ob)
            if(self.action == 1):
                self.change_light_phase()


    def actuated_decision_making(self, step):
        e = self.path[0]
        tls = self.net.getEdge(e).getTLS()  

        detector_distance = 5

        min_distance_car = 9999
        min_distance_bike = 9999
        for i in self.module_traci.edge.getLastStepVehicleIDs(self.start_edge.getID()):
            dist = self.start_edge.getLength()-self.module_traci.vehicle.getLanePosition(i)
            if("_c" in i and dist < min_distance_car):
                min_distance_car = dist
            elif("_c" not in i and dist < min_distance_bike):
                min_distance_bike = dist
        if(self.module_traci.trafficlight.getPhase(tls.getID()) == 0 and min_distance_car < detector_distance or\
        self.module_traci.trafficlight.getPhase(tls.getID()) == 2 and min_distance_bike < detector_distance):
            self.actuated_next_change_step = step + 5

        if(step > self.actuated_next_change_step):
            self.change_light_phase()




    def create_observation(self):
        edge_start_x = self.start_edge.getFromNode().getCoord()[0]
        ob = [[[0]*int(self.width_ob[-1]),[0]*int(self.width_ob[-1])], [[0]*int(self.width_ob[-1]),[0]*int(self.width_ob[-1])]]
        for vehicle_id in self.module_traci.edge.getLastStepVehicleIDs(self.start_edge.getID()):
            index = int(self.module_traci.vehicle.getLaneID(vehicle_id)[-1])
            position_in_grid = int(round(self.module_traci.vehicle.getPosition(vehicle_id)[0]-edge_start_x))//5
            ob[0][index][position_in_grid] += 1
            ob[1][index][position_in_grid] += self.module_traci.vehicle.getSpeed(vehicle_id)
        
        for i in range(len(ob[0])):
            for j in range(len(ob[0][i])):
                if(ob[1][i][j] != 0):
                    ob[1][i][j]/=ob[0][i][j]

        return np.array(ob)
        


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
        e = self.path[0]
        tls = self.net.getEdge(e).getTLS()  
        if(step > self.next_step_decision):
            if(self.module_traci.trafficlight.getPhase(tls.getID()) == 1):
                self.next_step_decision = step + 8
            elif(self.module_traci.trafficlight.getPhase(tls.getID()) == 3):
                self.next_step_decision = step + 54

            

    def change_light_phase(self):
        e = self.path[0]
        tls = self.net.getEdge(e).getTLS()       
        self.module_traci.trafficlight.setPhase(tls.getID(), self.module_traci.trafficlight.getPhase(tls.getID())+1)

            

    def static_decision_making(self, step):
        e = self.path[0]
        tls = self.net.getEdge(e).getTLS()

        if(self.module_traci.trafficlight.getPhase(tls.getID()) == 0 and\
        len(set(self.module_traci.edge.getLastStepVehicleIDs(e)) & set(self.id_cyclists_crossing_struct)) >= self.min_group_size):           
            self.change_light_phase()       
        elif(self.module_traci.trafficlight.getPhase(tls.getID()) == 2 and\
        len(set(self.module_traci.edge.getLastStepVehicleIDs(e)) & set(self.id_cyclists_crossing_struct)) == 0):
            self.change_light_phase()

        

    def check_lights_1_program(self, step):
        for e in self.path:
            tls = self.net.getEdge(e).getTLS()
            if(tls):
                ind = 0
                for l in self.module_traci.trafficlight.getControlledLinks(tls.getID()):
                    if(l[0][0] == e+"_0"):
                        break
                    else:
                        ind += 1
                light_color = self.module_traci.trafficlight.getRedYellowGreenState(tls.getID())[ind]

                if(len(set(self.module_traci.edge.getLastStepVehicleIDs(e)) & set(self.id_cyclists_crossing_struct)) >= self.min_group_size):
                    if(light_color == 'r'):
                        self.module_traci.trafficlight.setPhase(tls.getID(), (self.module_traci.trafficlight.getPhase(tls.getID())+1)%4)
                    elif(light_color == "g" or light_color == "G"):
                        self.module_traci.trafficlight.setPhaseDuration(tls.getID(), 99999)

                elif(not(set(self.module_traci.edge.getLastStepVehicleIDs(e)) & set(self.id_cyclists_crossing_struct))):
                    if(self.module_traci.trafficlight.getNextSwitch(tls.getID())-step > 10000):
                        self.module_traci.trafficlight.setPhase(tls.getID(), (self.module_traci.trafficlight.getPhase(tls.getID())+1)%4)


