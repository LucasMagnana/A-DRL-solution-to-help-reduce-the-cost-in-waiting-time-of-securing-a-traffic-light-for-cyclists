import threading
import torch

from DDQNAgent import DDQNAgent

class Structure:
    def __init__(self, start_edge, end_edge, edges, net, dict_cyclists, traci, config,\
    dict_edges_index=None, open=True, min_group_size=5, batch_size=32, learning=True, use_drl=False):

        for e in edges:
            id = e.getID()
            if(id == start_edge):
                self.start_edge = e
            if(id == end_edge):
                self.end_edge  = e

        self.path = net.getShortestPath(self.start_edge, self.end_edge, vClass='bicycle')[0]
        self.path = [e.getID() for e in self.path]

        self.dict_cyclists = dict_cyclists

        self.module_traci = traci

        self.id_cyclists_crossing_struct = []
        self.id_cyclists_waiting = []

        '''self.model = model
        self.learning = learning
        self.batch_size=batch_size
        if(self.model != None and self.learning):
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            self.loss = torch.nn.BCELoss()'''
        self.dict_edges_index = dict_edges_index
        self.dict_model_input = {}
        self.list_input_to_learn = []
        self.list_target = []
        self.list_loss = []
        
        self.min_group_size = min_group_size
        self.activated = False

        self.net = net

        self.num_cyclists_crossed = 0
        self.num_cyclists_canceled = 0

        self.open = open

        self.config = config

        for e in self.path:
            tls = self.net.getEdge(e).getTLS()
            if(tls):
                self.module_traci.trafficlight.setPhase(tls.getID(), 2)

        self.use_drl = use_drl
        if(self.use_drl):
            tls = self.net.getEdge(self.path[0]).getTLS()
            self.module_traci.trafficlight.setProgramLogic(tls.getID(), self.module_traci.trafficlight.Logic(1, 0, 0, \
            phases=[self.module_traci.trafficlight.Phase(duration=99999, state="rrrrrrrrrrrrGGGrrrrrrrrr", minDur=9999, maxDur=9999)]))

            self.module_traci.trafficlight.setProgram(tls.getID(), 0)
            self.module_traci.trafficlight.setPhase(tls.getID(), 2)

            self.width_ob = self.start_edge.getLength()//5+2
            self.drl_agent = DDQNAgent(self.width_ob, 2)
            self.ob = []

            self.bikes_waiting_time = 0
            self.cars_waiting_time = 0
            self.bikes_waiting_time_coeff = 0.2
            self.cars_waiting_time_coeff = 1-self.bikes_waiting_time_coeff

            self.need_change_tls_program = False
            self.next_step_drl = 0

            self.action = -1



    def step(self, step, edges):
        #print(step, self.id_cyclists_waiting, self.id_cyclists_crossing_struct)

        if(len(self.drl_agent.buffer)>self.drl_agent.batch_size):
            self.drl_agent.learn()

                
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

        if(self.config == 0):
            self.check_lights_1_program(step)
        elif(self.config == 3):
            if(self.use_drl):
                self.drl_decision_making(step)
            else:
                self.static_decision_making()






    def drl_decision_making(self, step):
        e = self.path[0]
        tls = self.net.getEdge(e).getTLS()       
        if(step > self.next_step_drl):
            self.create_observation()
            self.ob_prec = self.ob
            self.ob = self.create_observation()

            if(len(self.ob_prec) == 0):
                self.calculate_sum_waiting_time()
            else:
                if(self.action >= 0):
                    reward = self.calculate_reward()
                    self.drl_agent.memorize(self.ob_prec, self.action, self.ob, reward, False)  
                self.action = self.drl_agent.act(self.ob)
                if(self.action == 1):
                    self.need_change_tls_program = True
                    self.next_step_drl = step+1


        if(self.need_change_tls_program):            
            last_program = self.module_traci.trafficlight.getProgram(tls.getID())
            self.change_light_program()
            if(last_program != self.module_traci.trafficlight.getProgram(tls.getID())):
                self.need_change_tls_program = False
                self.next_step_drl = step+5
            else:
                self.next_step_drl = step+1



    def create_observation(self):
        edge_start_x = self.start_edge.getFromNode().getCoord()[0]
        self.width_ob = self.start_edge.getLength()//5+2
        ob = [[[0]*int(self.width_ob),[0]*int(self.width_ob)], [[0]*int(self.width_ob),[0]*int(self.width_ob)]]
        for vehicle_id in self.module_traci.edge.getLastStepVehicleIDs(self.start_edge.getID()):
            index = int(self.module_traci.vehicle.getLaneID(vehicle_id)[-1])
            position_in_grid = int(round(self.module_traci.vehicle.getPosition(vehicle_id)[0]-edge_start_x))//5
            ob[0][index][position_in_grid] += 1
            ob[1][index][position_in_grid] += self.module_traci.vehicle.getSpeed(vehicle_id)
        
        for i in range(len(ob[0])):
            for j in range(len(ob[0][i])):
                if(ob[1][i][j] != 0):
                    ob[1][i][j]/=ob[0][i][j]

        return ob
        


    def calculate_sum_waiting_time(self):
        last_cars_wt = self.cars_waiting_time
        last_bikes_wt = self.bikes_waiting_time
        self.cars_waiting_time = 0
        self.bikes_waiting_time = 0
        for vehicle_id in self.module_traci.edge.getLastStepVehicleIDs(self.start_edge.getID()):
            if("_c" in vehicle_id):
                self.cars_waiting_time += self.module_traci.vehicle.getAccumulatedWaitingTime(vehicle_id)
            else:
                self.bikes_waiting_time += self.module_traci.vehicle.getAccumulatedWaitingTime(vehicle_id)
        return last_cars_wt, last_bikes_wt

        

    def calculate_reward(self):
        last_cars_wt, last_bikes_wt = self.calculate_sum_waiting_time()
        return (self.cars_waiting_time_coeff*last_cars_wt + self.bikes_waiting_time_coeff*last_bikes_wt)-\
        (self.cars_waiting_time_coeff*self.cars_waiting_time + self.bikes_waiting_time_coeff*self.bikes_waiting_time)
        

    def change_light_program(self):
        e = self.path[0]
        tls = self.net.getEdge(e).getTLS()       
        if(self.module_traci.trafficlight.getProgram(tls.getID()) == "0"):
            if(self.module_traci.trafficlight.getPhase(tls.getID()) == 2):
                self.module_traci.trafficlight.setPhase(tls.getID(), 3)
            elif(self.module_traci.trafficlight.getPhase(tls.getID()) == 0):
                self.module_traci.trafficlight.setProgram(tls.getID(), 1)
        else:
            self.module_traci.trafficlight.setProgram(tls.getID(), 0)
            self.module_traci.trafficlight.setPhase(tls.getID(), 0)

    def static_decision_making(self):
        e = self.path[0]
        tls = self.net.getEdge(e).getTLS()
        if(len(set(self.module_traci.edge.getLastStepVehicleIDs(e)) & set(self.id_cyclists_crossing_struct)) >= self.min_group_size):           
            if(self.module_traci.trafficlight.getProgram(tls.getID()) == "0"):
                if(self.module_traci.trafficlight.getPhase(tls.getID()) == 2):
                    self.module_traci.trafficlight.setPhase(tls.getID(), 3)
                elif(self.module_traci.trafficlight.getPhase(tls.getID()) == 0):
                    self.module_traci.trafficlight.setProgram(tls.getID(), 1)
        
        elif(len(set(self.module_traci.edge.getLastStepVehicleIDs(e)) & set(self.id_cyclists_crossing_struct)) == 0):
            if(self.module_traci.trafficlight.getProgram(tls.getID()) == "1"):
                self.module_traci.trafficlight.setProgram(tls.getID(), 0)
                self.module_traci.trafficlight.setPhase(tls.getID(), 0)

        

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



    '''def learn(self):
        self.optimizer.zero_grad()
        tens_edges_occupation = torch.stack([i[0] for i in self.list_input_to_learn])
        tens_actual_edge = torch.stack([i[1] for i in self.list_input_to_learn])
        tens_target = torch.FloatTensor(self.list_target).unsqueeze(1)
        #out = self.model(tens_edges_occupation, tens_actual_edge)
        out = self.model(tens_actual_edge)
        l = self.loss(out, tens_target)
        self.list_loss.append(l.item())
        l.backward()
        self.optimizer.step()
        self.list_input_to_learn = []
        self.list_target = []     '''

