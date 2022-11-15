import threading
import torch

class Structure:
    def __init__(self, start_edge, end_edge, edges, net, dict_cyclists, traci,\
    dict_edges_index=None, model=None, open=True, min_group_size=5, batch_size=32, learning=True, lr=1e-5):

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

        self.model = model
        self.learning = learning
        self.batch_size=batch_size
        if(self.model != None and self.learning):
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            self.loss = torch.nn.BCELoss()
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


    def step(self, step, edges):
        #print(step, self.id_cyclists_waiting, self.id_cyclists_crossing_struct)

        if(len(self.list_input_to_learn)>=self.batch_size):
            self.learn()


                
        for i in self.module_traci.edge.getLastStepVehicleIDs(self.start_edge.getID()):
            if(i not in self.id_cyclists_waiting and i not in self.id_cyclists_crossing_struct and self.dict_cyclists[i].struct_candidate):
                if(len(self.id_cyclists_waiting)==0):
                    if(self.module_traci.vehicle.getSpeed(i)<= 1):
                        self.id_cyclists_waiting.append(i)
                        self.dict_cyclists[i].step_cancel_struct_candidature = step+99999
                else:
                    for j in range(len(self.id_cyclists_waiting)-1, -1, -1):
                        pos = self.module_traci.vehicle.getPosition(self.id_cyclists_waiting[j])
                        if(self.module_traci.vehicle.getDrivingDistance2D(i, pos[0], pos[1])<2):
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
                self.num_cyclists_crossed += 1
            self.id_cyclists_waiting = []

            for i in self.id_cyclists_crossing_struct:
                self.dict_cyclists[i].set_max_speed(min_max_speed)
        print()
        print([self.module_traci.vehicle.getSpeed(i) for i in self.id_cyclists_crossing_struct])
        print([self.module_traci.vehicle.getMaxSpeed(i) for i in self.id_cyclists_crossing_struct])
        print([self.dict_cyclists[i].max_speed for i in self.id_cyclists_crossing_struct])

            #print("activated at step", step)

        if(len(self.id_cyclists_crossing_struct)>0):

            if(set(self.module_traci.edge.getLastStepVehicleIDs(self.start_edge.getID())) & set(self.id_cyclists_crossing_struct)):
                for i in self.id_cyclists_waiting:
                    self.id_cyclists_crossing_struct.append(i)
                    self.dict_cyclists[i].cross_struct()
                    self.dict_cyclists[i].set_max_speed(self.dict_cyclists[self.id_cyclists_crossing_struct[0]].max_speed)
                    self.id_cyclists_waiting.remove(i)
            
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

                if(set(self.module_traci.edge.getLastStepVehicleIDs(e)) & set(self.id_cyclists_crossing_struct) and\
                len(self.module_traci.edge.getLastStepVehicleIDs(e)) >= self.min_group_size):
                    if(light_color == 'r'):
                        self.module_traci.trafficlight.setPhase(tls.getID(), (self.module_traci.trafficlight.getPhase(tls.getID())+1)%4)
                    elif(light_color == "g" or light_color == "G"):
                        self.module_traci.trafficlight.setPhaseDuration(tls.getID(), 99999)

                elif(not(set(self.module_traci.edge.getLastStepVehicleIDs(e)) & set(self.id_cyclists_crossing_struct))):
                    if(self.module_traci.trafficlight.getNextSwitch(tls.getID()) > 10000):
                        self.module_traci.trafficlight.setPhase(tls.getID(), (self.module_traci.trafficlight.getPhase(tls.getID())+1)%4)



    def learn(self):
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
        self.list_target = []     

