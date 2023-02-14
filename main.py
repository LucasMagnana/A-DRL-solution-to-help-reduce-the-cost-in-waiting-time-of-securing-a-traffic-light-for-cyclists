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


gui = False
struct_open = False
use_drl = False
test = False
save_scenario = False
actuated = False
new_scenario = True

poisson_lambda = 0.2
min_group_size = 5
config = 3

num_simu = 150
simu_length = 1800

if __name__ == "__main__": 
    arguments = str(sys.argv)

    if('--gui' in arguments):
        gui = True
    if('--struct-open' in arguments):
        struct_open = True
    if('--drl' in arguments):
        use_drl = True
    if('--save-scenario' in arguments):
        save_scenario = True
    if('--test' in arguments):
        test = True
        if(save_scenario):
            num_simu = 20
        else:
            num_simu = 2
    if("--load-scenario" in arguments):
        new_scenario = False
    if('--actuated' in arguments):
        actuated = True
    


step_length = 0.2
speed_threshold = 0.5

car_poisson_lambda = 0.1
bike_poisson_lambda = 0.2
evoluting = "cars"
struct_open = True


if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

sumoBinary = "/usr/bin/sumo"
if(gui):
    sumoBinary += "-gui"
sumoCmd = [sumoBinary, "-c", "sumo_files/sumo_"+str(config)+".sumocfg", "--extrapolate-departpos", "--quit-on-end", "--waiting-time-memory", '10000', '--start', '--delay', '0', '--step-length', str(step_length), '--no-warnings']

import traci
import traci.constants as tc
import sumolib



if(test):
    sub_folders = "test/"
else:
    sub_folders = "train/"


net = sumolib.net.readNet("sumo_files/net_"+str(config)+".net.xml")
edges = net.getEdges()

structure = Structure("E_start", "E2", edges, net, traci, config, simu_length, use_drl, actuated, test, min_group_size)

pre_file_name = ""
if(use_drl):
    n_d = 1
    if(structure.drl_agent.double):
        n_d += 1
    if(structure.drl_agent.duelling):
        n_d += 1
    pre_file_name = str(n_d)+"DQN_"
elif(actuated):
    pre_file_name += "actuated_"
    
pre_file_name += evoluting+"_evolv_"

if(evoluting=="bikes"):
    sub_folders+="config_"+str(config)+"/"
    variable_evoluting = poisson_lambda
elif(evoluting=="cars"):
    sub_folders+="config_"+str(config)+"/"
    variable_evoluting = poisson_lambda
else:
    sub_folders+="config_"+str(config)+"/"
    variable_evoluting = min_group_size

start_num_simu = 0
if(not new_scenario):
    with open("files/"+sub_folders+"3DQN_cars_evolv_scenarios.tab", 'rb') as infile:
        tab_dict_old_scenarios = pickle.load(infile)
        num_simu = len(tab_dict_old_scenarios)

    if(os.path.exists("files/"+sub_folders+pre_file_name+"scenarios.tab")):
        with open("files/"+sub_folders+pre_file_name+"scenarios.tab", 'rb') as infile:
            tab_dict_scenarios = pickle.load(infile)
            start_num_simu = len(tab_dict_scenarios)

print("Starting at ", start_num_simu)

for s in range(start_num_simu, num_simu):
    next_step_wt_update = 0

    num_cyclists_real = 0
    num_cars_real = 0

    traci.start(sumoCmd)

    if(new_scenario):
        print("WARNING : Creating a new scenario...")
        bike_poisson_distrib = np.random.poisson(bike_poisson_lambda, simu_length)
        car_poisson_distrib = np.random.poisson(car_poisson_lambda, simu_length)
        num_cyclists = sum(bike_poisson_distrib)
        num_cars = sum(car_poisson_distrib)
    else:
        print("WARNING : Loading the scenario...")
        old_dict_scenario = tab_dict_old_scenarios[s]

        num_cyclists = len(old_dict_scenario["bikes"])
        num_cars = len(old_dict_scenario["cars"])

        max_id_cyclist = max(old_dict_scenario["bikes"].keys())
        max_id_car = max(old_dict_scenario["cars"].keys())

    num_data = num_cyclists + num_cars
        
    print("num_cyclists: ", num_cyclists, ", num_cars :", num_cars, ", num_data :", num_data)


    dict_scenario={"cars": {}, "bikes": {}}

    dict_vehicles = {"bikes": {}, "cars": {}}

    structure.reset(dict_vehicles["bikes"], dict_scenario)


    id_cyclist = 0
    id_car = 0
    step = 0

    continue_simu = True

    while(step<=simu_length):
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
            while(id_cyclist not in old_dict_scenario["bikes"] and id_cyclist <= max_id_cyclist):
                id_cyclist += 1
            if(id_cyclist in old_dict_scenario["bikes"] and step >= old_dict_scenario["bikes"][id_cyclist]["start_step"]):
                start_edge_id=old_dict_scenario["bikes"][id_cyclist]["start_edge"]
                end_edge_id=old_dict_scenario["bikes"][id_cyclist]["end_edge"]
                e1 = net.getEdge(start_edge_id)
                e2 = net.getEdge(end_edge_id)
                path = net.getShortestPath(e1, e2, vClass='bicycle')[0]
                dict_scenario["bikes"][id_cyclist] = {"start_step": step, "start_edge": e1.getID(), "end_edge": e2.getID(), "max_speed": old_dict_scenario["bikes"][id_cyclist]["max_speed"],
                "distance_travelled": net.getShortestPath(net.getEdge("E_start"), e2, vClass='bicycle', fromPos=0)[1], "waiting_time": 0}
                spawn_cyclist(id_cyclist, step, path, net, structure, step_length, old_dict_scenario["bikes"][id_cyclist]["max_speed"], False, dict_vehicles["bikes"])
                id_cyclist+=1

            while(id_car not in old_dict_scenario["cars"] and id_car <= max_id_car):
                id_car += 1
            if(id_car in old_dict_scenario["cars"] and step >= old_dict_scenario["cars"][id_car]["start_step"]):
                start_edge_id=old_dict_scenario["cars"][id_car]["start_edge"]
                end_edge_id=old_dict_scenario["cars"][id_car]["end_edge"]
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
                    if(vehicle_type == "cars"):
                        num_cars_real += 1
                    else:
                        num_cyclists_real += 1
                    dict_scenario[vehicle_type][int(i)]["finish_step"] = step
                    del dict_vehicles[vehicle_type][i]
                else:
                    if(step >= next_step_wt_update):
                        try:
                            if(traci.vehicle.getSpeed(sumo_id)< speed_threshold):
                                dict_scenario[vehicle_type][int(i)]["waiting_time"] += 1
                        except traci.exceptions.TraCIException:
                            del dict_scenario[vehicle_type][int(i)]
                            del dict_vehicles[vehicle_type][i]
                            
        if(step >= next_step_wt_update):
            next_step_wt_update += 1



        #(step%1, step%1<=step_length)
        if(structure.open):
            structure.step(step, edges)

        print(f"\rStep {int(step)}: {len(traci.vehicle.getIDList())} vehicles in simu, {id_cyclist} cyclists spawned since start,\
        {id_car} cars spawned since start.", end="")


        step += step_length

    traci.close()

    '''for vehicle_type in dict_scenario:
        for i in copy.deepcopy(list(dict_scenario[vehicle_type].keys())):
            if("finish_step" not in dict_scenario[vehicle_type][i]):
                del dict_scenario[vehicle_type][i]'''


    print("\ndata number:", num_cars_real+num_cyclists_real, ",", structure.num_cyclists_crossed, "cyclits used struct, last step:", step)



    if(save_scenario):
        print("WARNING: Saving scenario...")
        if(not os.path.exists("files/"+sub_folders)):
            os.makedirs("files/"+sub_folders)

        if(os.path.exists("files/"+sub_folders+pre_file_name+"scenarios.tab")):
            with open("files/"+sub_folders+pre_file_name+"scenarios.tab", 'rb') as infile:
                tab_dict_scenarios = pickle.load(infile)
        else:
            tab_dict_scenarios = []

        tab_dict_scenarios.append(dict_scenario)
        print(len(tab_dict_scenarios))

        with open("files/"+sub_folders+pre_file_name+"scenarios.tab", 'wb') as outfile:
            pickle.dump(tab_dict_scenarios, outfile)


    bikes_data = compute_data(dict_scenario["bikes"])
    cars_data = compute_data(dict_scenario["cars"])

    print(f"mean cars travel time: {cars_data[0]}, mean cars waiting time: {cars_data[1]}")
    print(f"mean bikes travel time: {bikes_data[0]}, mean bikes waiting time: {bikes_data[1]}")

if(use_drl and not test):
    torch.save(structure.drl_agent.actor_target.state_dict(), "files/"+sub_folders+"trained_target.n")
    torch.save(structure.drl_agent.actor.state_dict(), "files/"+sub_folders+"trained.n")