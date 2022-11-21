import time 

class Cyclist:

    def __init__(self, id, step, path, net, structure, max_speed, traci, sumolib, step_length, struct_candidate=True):
        self.id = id
        self.start_step = step
        self.net=net
        self.structure = structure

        self.module_sumolib = sumolib
        self.module_traci = traci

        self.struct_candidate = struct_candidate

        
        path = [e.getID() for e in path]
        self.original_path = path
        self.module_traci.route.add(str(self.id)+"_sp", path)
        
        d_speed = traci.vehicletype.getMaxSpeed('bicycle')-2
        self.module_traci.vehicle.add(str(self.id), str(self.id)+"_sp", departLane="best", typeID='bicycle', departSpeed=d_speed)
        
        self.set_max_speed(max_speed)
        self.max_speed = self.module_traci.vehicle.getMaxSpeed(str(self.id))

        for i in range(len(self.original_path)):
            if(self.original_path[i] not in self.structure.path):
                self.last_edge_in_struct_id = self.original_path[i-1]
                break

        self.module_traci.vehicle.setActionStepLength(self.id, step_length)
        self.module_traci.vehicle.setTau(self.id, step_length)
        self.module_traci.vehicle.setMinGap(self.id, 0)
        #self.module_traci.vehicle.setSpeedFactor(self.id, 1)


        self.actual_path = self.original_path


        self.actual_edge_id = self.actual_path[0]

        self.alive=True
        self.canceled_candidature = False

        self.step_cancel_struct_candidature = -1

        self.waiting_time = 0
        self.distance_travelled = 0

        self.arrived=False

        self.path_used = []

        self.going_to_struct = False
        self.crossing_struct = False
        self.wanting_to_exit_struct = False
        self.struct_crossed = False





    def step(self, step):

        if(self.id in self.module_traci.vehicle.getIDList()):
            self.actual_edge_id = self.module_traci.vehicle.getRoadID(self.id)
            if(len(self.path_used) == 0 or self.path_used[-1] != self.actual_edge_id):
                self.path_used.append(self.actual_edge_id)
            if(self.module_traci.vehicle.getSpeed(self.id)<0.5):
                self.waiting_time += 1
            self.distance_travelled = self.module_traci.vehicle.getDistance(self.id)


            if('J' in self.actual_edge_id and self.wanting_to_exit_struct):
                self.exit_struct()

            if(self.actual_edge_id==self.original_path[-1]):
                self.arrived = True

            if(self.step_cancel_struct_candidature > 0 and step>=self.step_cancel_struct_candidature and (self.id in self.structure.id_cyclists_waiting or self.actual_path == self.path_to_struct)):
                self.cancel_struct_candidature()

            if(self.actual_edge_id in self.structure.path):
                if(self.actual_path == self.original_path):
                    self.module_traci.vehicle.changeLane(self.id, 0, 1)
                elif(self.actual_edge_id != self.last_edge_in_struct_id):
                    self.module_traci.vehicle.changeLane(self.id, 1, 1) 

            if(self.struct_candidate and not self.going_to_struct and not self.crossing_struct):
                self.go_to_struct()

            if(self.crossing_struct and self.actual_edge_id == self.last_edge_in_struct_id):
                self.wanting_to_exit_struct = True

            if(self.crossing_struct):
                self.module_traci.vehicle.highlight(self.id, color=(0, 0, 255, 255))
            elif(self.struct_crossed):
                self.module_traci.vehicle.highlight(self.id, color=(0, 255, 0, 255))
            elif(self.struct_candidate):
                self.module_traci.vehicle.highlight(self.id)

        elif(self.id in self.module_traci.simulation.getArrivedIDList()):
            self.alive = False
            self.finish_step=step
            if(self.id in self.structure.id_cyclists_crossing_struct):
                self.structure.id_cyclists_crossing_struct.remove(self.id)
            if(self.id in self.structure.id_cyclists_waiting):
                self.structure.id_cyclists_waiting.remove(self.id)


    def calculate_ETA(self, step, path=None):
        if(path == None):
            path = self.actual_path

        waiting_time = self.calculate_estimated_waiting_time(path)
        self.estimated_distance = self.module_sumolib.route.getLength(self.net, path)
        #self.estimated_distance = self.module_traci.vehicle.getDrivingDistance(self.id, self.actual_path[-1].getID(), 0)
        travel_time = self.estimated_distance/self.max_speed
        self.estimated_travel_time=travel_time+waiting_time*1
        return step+self.estimated_travel_time


    def go_to_struct(self):
        #self.module_traci.vehicle.setStop(self.id, self.structure.start_edge.getID(), self.structure.start_edge.getLength()-1)
        self.going_to_struct = True

    def cross_struct(self):
        self.going_to_struct = False
        self.crossing_struct = True
        self.structure.num_cyclists_crossed += 1
        if(self.module_traci.vehicle.isStopped(self.id)):
            self.module_traci.vehicle.resume(self.id)
        if(self.module_traci.vehicle.getNextStops(self.id)):
            self.module_traci.vehicle.setStop(self.id, self.structure.start_edge.getID(), self.structure.start_edge.getLength()-1, duration=0)


    def exit_struct(self):
        self.wanting_to_exit_struct = False
        self.crossing_struct = False
        self.struct_crossed = True
        self.struct_candidate = False
        self.structure.id_cyclists_crossing_struct.remove(self.id)
        self.set_max_speed(self.max_speed)

    def cancel_struct_candidature(self):
        if(self.id in self.structure.id_cyclists_waiting):
            self.structure.id_cyclists_waiting.remove(self.id)
        else:
            print(self.id, "cancelling while not in waiting list")

        if(self.actual_edge_id[0]!=':'):
            try:
                self.original_path = self.dict_shortest_path[self.actual_edge_id+";"+self.original_path["path"][-1]]
            except KeyError:
                print(self.id, "bugged during cancelling candidature, no path found.")
                self.alive = False
                return
            self.module_traci.vehicle.setRoute(self.id, self.original_path["path"])
        else:
            self.module_traci.vehicle.changeTarget(self.id, self.original_path["path"][-1])
            
        self.actual_path = self.original_path
        self.struct_candidate = False
        self.canceled_candidature = True
        
        if(self.module_traci.vehicle.isStopped(self.id)):
            self.module_traci.vehicle.resume(self.id)
        if(self.module_traci.vehicle.getNextStops(self.id)):
            self.module_traci.vehicle.setStop(self.id, self.structure.start_edge.getID(), self.structure.start_edge.getLength()-1, duration=0)
        self.step_cancel_struct_candidature = -1

        self.structure.num_cyclists_canceled += 1



    def set_max_speed(self, max_speed):
        self.module_traci.vehicle.setMaxSpeed(self.id, max_speed)

            
                
