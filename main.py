import os, sys
from random import randint
import numpy as np 
import pickle
import copy
import torch
import argparse

from Cyclist import Cyclist
from Structure import Structure
from graphs import *
from Model import Model


if __name__ == "__main__": 
    arguments = str(sys.argv)
    gui = False
    struct_open = False
    use_drl = False
    test = False
    save_scenario = False
    actuated = False

    poisson_lambda = 0.2
    min_group_size = 5
    config = 3

    if('--gui' in arguments):
        gui = True
    if('--struct-open' in arguments):
        struct_open = True
    if('--drl' in arguments):
        use_drl = True
    if('--test' in arguments):
        test = True
    if('--save-scenario' in arguments):
        save_scenario = True
    if('--actuated' in arguments):
        actuated = True
    
new_scenario = False
if(use_drl or test):
    new_scenario = True


step_length = 0.2
simu_length = 25*3600
speed_threshold = 0.5

if(config == 0):
    car_poisson_lambda = 0.2
    bike_poisson_lambda = 0.2
    evoluting = "group_size"
elif(config == 1):
    car_poisson_lambda = poisson_lambda
    bike_poisson_lambda = 1
    evoluting = "cars"
elif(config == 2):
    car_poisson_lambda = 0.2
    bike_poisson_lambda = poisson_lambda
    evoluting = "bikes"
elif(config == 3):
    car_poisson_lambda = 0.1
    bike_poisson_lambda = 0.2
    evoluting = "cars"
    struct_open = True


if(use_drl or not new_scenario):
    sub_folders = "w_model/"
else:
    sub_folders = "wou_model/"

if(evoluting=="bikes"):
    sub_folders+="config_"+str(config)+"/"+str(car_poisson_lambda)+"/"
    variable_evoluting = poisson_lambda
elif(evoluting=="cars"):
    sub_folders+="config_"+str(config)+"/"+str(bike_poisson_lambda)+"/"
    variable_evoluting = poisson_lambda
else:
    sub_folders+="config_"+str(config)+"/"+str(bike_poisson_lambda)+"/"
    variable_evoluting = min_group_size



if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

sumoBinary = "/usr/bin/sumo"
if(gui):
    sumoBinary += "-gui"
sumoCmd = [sumoBinary, "-c", "sumo_files/sumo_"+str(config)+".sumocfg", "--max-depart-delay", "10", "--extrapolate-departpos", "--quit-on-end", "--waiting-time-memory", '10000', '--start', '--delay', '0', '--step-length', str(step_length), '--no-warnings']



import traci
import sumolib
import traci.constants as tc


def spawn_cyclist(id_cyclist, step, path, net, structure, step_length, max_speed, struct_candidate, dict_bikes):
    if(struct_open or num_cyclists-id_cyclist+max(len(structure.id_cyclists_waiting), len(traci.edge.getLastStepVehicleIDs(structure.start_edge.getID())))<structure.min_group_size):
        struct_candidate = True
    
    c = Cyclist(str(id_cyclist), step, path, net, structure, max_speed, traci, sumolib, step_length, struct_candidate=struct_candidate)
    dict_bikes[str(id_cyclist)]=c


def spawn_car(id_car, step, path, net, dict_cars):
    path = [e.getID() for e in path]
    traci.route.add(str(id_car)+"_c_sp", path)
    traci.vehicle.add(str(id_car)+"_c", str(id_car)+"_c_sp", departLane="best", departPos="last", typeID='car', departSpeed="last")
    dict_cars[str(id_car)]=[]




traci.start(sumoCmd)

net = sumolib.net.readNet("sumo_files/net_"+str(config)+".net.xml")
edges = net.getEdges()


if(new_scenario):
    print("WARNING : Creating a new scenario...")
    bike_poisson_distrib = np.random.poisson(bike_poisson_lambda, simu_length)
    car_poisson_distrib = np.random.poisson(car_poisson_lambda, simu_length)
    num_cyclists = sum(bike_poisson_distrib)
    num_cars = sum(car_poisson_distrib)
else:
    print("WARNING : Loading the scenario...")
    print("files/"+sub_folders)
    for root, dirs, files in os.walk("files/"+sub_folders):
        for name in files:
            if("DQN" in name and ".dict" in name):
                with open("files/"+sub_folders+"/"+name, 'rb') as infile:
                    old_dict_scenario = pickle.load(infile)
                    old_dict_scenario = old_dict_scenario[car_poisson_lambda][0]
                    break

    num_cyclists = len(old_dict_scenario["bikes"])
    num_cars = len(old_dict_scenario["cars"])

dict_scenario={"cars": {}, "bikes": {}}
    
print("num_cyclists: ", num_cyclists, ", num_cars :", num_cars)

dict_edges_index = {}
for i, e in enumerate(edges) :
    dict_edges_index[e.getID()] = i


dict_vehicles = {"bikes": {}, "cars": {}}

structure = Structure("E_start", "E2", edges, net, dict_vehicles["bikes"], traci, config, dict_scenario, simu_length, use_drl, actuated, test, min_group_size)


id_cyclist = 0
id_car = 0
step = 0

while(step<simu_length or len(dict_vehicles["bikes"]) != 0 or len(dict_vehicles["cars"]) != 0):
    if(new_scenario): #new_scenario
        if(step<simu_length):
            for _ in range(bike_poisson_distrib[int(step)]):
                e1 = net.getEdge("E0")
                e2 = net.getEdge("E"+str(randint(3, 9)))
                path = net.getShortestPath(e1, e2, vClass='bicycle')[0]
                max_speed = np.random.normal(15, 3)
                dict_scenario["bikes"][id_cyclist] = {"start_step": step, "start_edge": e1.getID(), "end_edge": e2.getID(), "max_speed": max_speed,
                "distance_travelled": net.getShortestPath(net.getEdge("E_start"), e2, vClass='bicycle', fromPos=0)[1], "waiting_time": 0}
                spawn_cyclist(id_cyclist, step, path, net, structure, step_length, max_speed, False, dict_vehicles["bikes"])
                id_cyclist+=1
            bike_poisson_distrib[int(step)] = 0
        if(step<simu_length):  
            for _ in range(car_poisson_distrib[int(step)]):
                e1 = net.getEdge("E0")
                e2 = net.getEdge("E"+str(randint(3, 9)))
                path = net.getShortestPath(e1, e2, vClass='passenger')[0]
                dict_scenario["cars"][id_car] = {"start_step": step, "start_edge": e1.getID(), "end_edge": e2.getID(),
                "distance_travelled": net.getShortestPath(net.getEdge("E_start"), e2, vClass='passenger', fromPos=0)[1], "waiting_time": 0}
                spawn_car(id_car, step, path, net, dict_vehicles["cars"])
                id_car+=1
                car_poisson_distrib[int(step)] = 0

    else:
        if(id_cyclist<len(old_dict_scenario["bikes"]) and step >= old_dict_scenario["bikes"][id_cyclist]["start_step"]):
            start_edge_id=old_dict_scenario["bikes"][id_cyclist]["start_edge"]
            end_edge_id=old_dict_scenario["bikes"][id_cyclist]["end_edge"]
            e1 = net.getEdge(start_edge_id)
            e2 = net.getEdge(end_edge_id)
            path = net.getShortestPath(e1, e2, vClass='bicycle')[0]
            dict_scenario["bikes"][id_cyclist] = {"start_step": step, "start_edge": e1.getID(), "end_edge": e2.getID(), "max_speed": old_dict_scenario["bikes"][id_cyclist]["max_speed"],
            "distance_travelled": net.getShortestPath(net.getEdge("E_start"), e2, vClass='bicycle', fromPos=0)[1], "waiting_time": 0}
            spawn_cyclist(id_cyclist, step, path, net, structure, step_length, old_dict_scenario["bikes"][id_cyclist]["max_speed"], False, dict_vehicles["bikes"])
            id_cyclist+=1
        if(id_car<len(old_dict_scenario["cars"]) and step >= old_dict_scenario["cars"][id_car]["start_step"]):
            start_edge_id=old_dict_scenario["bikes"][id_cyclist]["start_edge"]
            end_edge_id=old_dict_scenario["bikes"][id_cyclist]["end_edge"]
            e1 = net.getEdge(start_edge_id)
            e2 = net.getEdge(end_edge_id)
            path = net.getShortestPath(e1, e2, vClass='passenger')[0]
            dict_scenario["cars"][id_car] = {"start_step": step, "start_edge": e1.getID(), "end_edge": e2.getID(),
            "distance_travelled": net.getShortestPath(net.getEdge("E_start"), e2, vClass='passenger', fromPos=0)[1], "waiting_time": 0}
            spawn_car(id_car, step, path, net, dict_vehicles["cars"])
            id_car+=1

    traci.simulationStep() 

    for vehicle_type in dict_vehicles:
        for i in copy.deepcopy(list(dict_vehicles[vehicle_type].keys())):
            sumo_id = i
            if(vehicle_type == "cars"):
                sumo_id+="_c"
            if(sumo_id in traci.simulation.getArrivedIDList()):
                dict_scenario[vehicle_type][int(i)]["finish_step"] = step
                del dict_vehicles[vehicle_type][i]
            else:
                try:
                    if(traci.vehicle.getSpeed(sumo_id)< speed_threshold):
                        dict_scenario[vehicle_type][int(i)]["waiting_time"] += 1
                except traci.exceptions.TraCIException:
                    del dict_scenario[vehicle_type][int(i)]
                    del dict_vehicles[vehicle_type][i]



    #(step%1, step%1<=step_length)
    if(structure.open):
        structure.step(step, edges)

    print(f"\rStep {int(step)}: {len(traci.vehicle.getIDList())} vehicles in simu, {id_cyclist} cyclists spawned since start,\
    {id_car} cars spawned since start.", end="")

    step += step_length


if(save_scenario):
    pre_file_name = ""
    if(use_drl):
        n_d = 1
        if(structure.drl_agent.double):
            n_d += 1
        if(structure.drl_agent.duelling):
            n_d += 1
        pre_file_name = str(n_d)+"DQN_"
        
    pre_file_name += evoluting+"_evolv_"

    print("WARNING: Saving scenario...")
    if(not os.path.exists("files/"+sub_folders)):
        os.makedirs("files/"+sub_folders)
    with open("files/"+sub_folders+pre_file_name+"scenarios.dict", 'wb') as outfile:
        pickle.dump({variable_evoluting : [dict_scenario]}, outfile)

if(use_drl and not test):
    torch.save(structure.drl_agent.actor_target.state_dict(), "files/"+sub_folders+"trained_target.n")
    torch.save(structure.drl_agent.actor.state_dict(), "files/"+sub_folders+"trained.n")





traci.close()

if(len(structure.list_input_to_learn)>0):
    structure.learn()
    
print("\ndata number:", len(dict_scenario["bikes"])+len(dict_scenario["cars"]), ",", structure.num_cyclists_crossed, "cyclits used struct, last step:", step)


bikes_data = compute_data(dict_scenario["bikes"])
cars_data = compute_data(dict_scenario["cars"])

print(f"mean cars travel time: {cars_data[0]}, mean cars waiting time: {cars_data[1]}")
print(f"mean bikes travel time: {bikes_data[0]}, mean bikes waiting time: {bikes_data[1]}")