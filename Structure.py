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

        for e in self.path:
            tls = self.net.getEdge(e).getTLS()
            if(tls):
                str_phase = ""
                str_phase_transition = ""
                str_inverse_phase = ""
                for l in self.module_traci.trafficlight.getControlledLinks(tls.getID()):    
                    if(e in l[0][0]):
                        str_phase+= 'G'
                        str_phase_transition+='y'
                        str_inverse_phase+='r'
                    else:
                        str_phase += 'r'
                        str_phase_transition+='G'
                        str_inverse_phase+='y'

                self.module_traci.trafficlight.setProgramLogic(tls.getID(), self.module_traci.trafficlight.Logic(1, 0, 0, \
                    phases=[self.module_traci.trafficlight.Phase(duration=99999, state=str_phase, minDur=9999, maxDur=9999), \
                        self.module_traci.trafficlight.Phase(duration=3, state=str_phase_transition, minDur=3, maxDur=3),
                        self.module_traci.trafficlight.Phase(duration=3, state=str_inverse_phase, minDur=3, maxDur=3)]))
                self.module_traci.trafficlight.setProgram(tls.getID(), 0)


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
                if(set(self.module_traci.edge.getLastStepVehicleIDs(e)) & set(self.id_cyclists_crossing_struct) and\
                len(self.module_traci.edge.getLastStepVehicleIDs(e)) >= self.min_group_size//2):
                    if(self.module_traci.trafficlight.getProgram(tls.getID()) == "0"):
                        self.module_traci.trafficlight.setProgram(tls.getID(), 1)
                elif(not(set(self.module_traci.edge.getLastStepVehicleIDs(e)) & set(self.id_cyclists_crossing_struct))):
                    if(self.module_traci.trafficlight.getProgram(tls.getID()) == "1"):
                        self.module_traci.trafficlight.setProgram(tls.getID(), 0)




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



    def check_for_candidates(self, step, edges, id=None, force_candidature=False):

        #print(force_candidature)


        list_id_candidates = []

        if(self.model != None and self.dict_edges_index != None):
            edges_occupation=[len(self.module_traci.edge.getLastStepVehicleIDs(e.getID())) for e in edges]

        if(id==None):
            cyclists_id_to_browse = self.dict_cyclists
        else:
            cyclists_id_to_browse = [id]
        for i in cyclists_id_to_browse:
            if(i not in self.id_cyclists_waiting and i not in self.id_cyclists_crossing_struct\
            and not self.dict_cyclists[i].struct_crossed and not self.dict_cyclists[i].canceled_candidature):
                if(self.dict_cyclists[i].actual_edge_id[0] != ":"):
                    key_path_to_struct = self.dict_cyclists[i].actual_edge_id+";"+self.start_edge.getID()
                    travel_time_by_struct = self.dict_shortest_path[key_path_to_struct]["length"]/self.dict_cyclists[i].max_speed+\
                    self.dict_shortest_path[key_path_to_struct]["estimated_waiting_time"]
                    travel_time_by_struct += self.dict_cyclists[i].path_from_struct["length"]/self.dict_cyclists[i].max_speed
                    travel_time_by_struct += self.calculate_estimated_waiting_time_without_struct_tls(self.dict_cyclists[i].path_from_struct["path"])
                    step_arriving_by_crossing_struct = step+travel_time_by_struct*self.time_travel_multiplier
                    self.dict_cyclists[i].estimated_time_diff = self.dict_cyclists[i].estimated_finish_step-step_arriving_by_crossing_struct

                    if(self.model != None and self.dict_edges_index != None):
                        tens_edges_occupation = torch.tensor(edges_occupation, dtype=torch.float)
                        tens_actual_edge = torch.tensor([self.dict_edges_index[self.dict_cyclists[i].actual_edge_id],\
                        self.dict_edges_index[self.dict_cyclists[i].original_path["path"][-1]], self.dict_cyclists[i].estimated_time_diff], dtype=torch.float)
                        with torch.no_grad():
                            #out = self.model(tens_edges_occupation, tens_actual_edge)
                            out = self.model(tens_actual_edge)
                        if(out >= 0.5):
                            self.dict_cyclists[i].struct_candidate=True
                        if(self.learning):
                            self.dict_model_input[i] = (tens_edges_occupation, tens_actual_edge)
                    elif(step_arriving_by_crossing_struct<=self.dict_cyclists[i].estimated_finish_step or force_candidature):
                        self.dict_cyclists[i].struct_candidate=True
                elif(i not in self.pending_for_check_candidates):
                    self.pending_for_check_candidates.append(i)
                        
                        
            

        return


    def calculate_estimated_waiting_time_without_struct_tls(self, path):
        red_duration = 0
        total_duration = 0
        num_tls = 0
        for e in path:
            if(e not in self.path):
                tls = self.net.getEdge(e).getTLS()
                if(tls):
                    num_tls+=1                
                    tl_concerned = []
                    i=0
                    for l in self.module_traci.trafficlight.getControlledLinks(tls.getID()):                
                        if(e in l[0][0]):
                            tl_concerned.append(i)
                        i+=1

                    for p in self.module_traci.trafficlight.getAllProgramLogics(tls.getID())[0].getPhases():
                        #print(p, tl_concerned)
                        total_duration += p.minDur
                        if('r' in p.state[tl_concerned[0]:tl_concerned[-1]]):
                            red_duration += p.minDur
        if(total_duration == 0):
            estimated_wait_tls = 0
        else:
            estimated_wait_tls = red_duration/total_duration*red_duration
            
        return estimated_wait_tls



                                

