import os, sys
from random import randint
import numpy as np 
import pickle
import copy
import torch
import argparse
import random
import json
from datetime import datetime


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
method = ""
test = False
save_scenario = False
new_scenario = True
real_data = False

poisson_lambda = 0.2
min_group_size = 5
config = 3

num_simu = 200
simu_length = 1800

if __name__ == "__main__": 
    arguments = str(sys.argv)

    if('--gui' in arguments):
        gui = True

    if('--struct-open' in arguments):
        struct_open = True

    if('--dqn' in arguments):
        method = "DQN"
    elif('--3dqn' in arguments):
        method = "3DQN"
    elif("--ppo" in arguments):
        method = "PPO"
    elif("--actuated" in arguments):
        method = "actuated"

    if('--save-scenario' in arguments):
        save_scenario = True

    if("--real-data" in arguments):
        real_data = True

    
    if("--load-scenario" in arguments):
        new_scenario = False

    if('--test' in arguments):
        test = True
        if(new_scenario):
            num_simu = 20
            if(real_data):
                num_simu = 1
                list_bike_poisson_lambdas = []
                list_car_poisson_lambdas = []
                with open("./real_data.json", "rb") as infile:
                    count_data = json.load(infile)

                first_day_number = None
                for data in count_data["data"]["values"]:
                    d = datetime.strptime(data["time"], '%Y-%m-%dT%H:%M:%S.%f%z')
                    if(first_day_number == None):
                        first_day_number = d.day
                    elif(d.day != first_day_number):
                        if(data["id"] == "S-N"):
                            list_bike_poisson_lambdas.append(data["count"]*0.5/simu_length)
                            list_car_poisson_lambdas.append(data["count"]*0.5/simu_length)

                print("WARNING : Creating a new scenario using real data...")
                bike_poisson_distrib = np.empty(0)
                car_poisson_distrib = np.empty(0)
                for i in range(len(list_bike_poisson_lambdas)):
                    car_poisson_lambda = 0.1
                    bike_poisson_lambda = 0.2 #list_bike_poisson_lambdas[i]

                    bike_poisson_distrib = np.concatenate((bike_poisson_distrib, np.random.poisson(bike_poisson_lambda, simu_length)))
                    car_poisson_distrib = np.concatenate((car_poisson_distrib, np.random.poisson(car_poisson_lambda, simu_length)))
    


step_length = 0.2
speed_threshold = 0.5

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

structure = Structure("E_start", "E2", edges, net, traci, config, simu_length, method, test, min_group_size)

pre_file_name = method+"_"

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
    with open("files/"+sub_folders+"actuated_scenarios.tab", 'rb') as infile:
        tab_dict_old_scenarios = pickle.load(infile)
        num_simu = len(tab_dict_old_scenarios)

    if(os.path.exists("files/"+sub_folders+pre_file_name+"scenarios.tab")):
        with open("files/"+sub_folders+pre_file_name+"scenarios.tab", 'rb') as infile:
            tab_dict_scenarios = pickle.load(infile)
            start_num_simu = len(tab_dict_scenarios)


for s in range(start_num_simu, num_simu):

    if(not test and "PPO" in method):
        structure.drl_agent.start_episode()
        if(s != start_num_simu and s%structure.drl_agent.hyperParams.LEARNING_EP == 0):
            structure.drl_agent.learn()

    next_step_wt_update = 0

    num_cyclists_real = 0
    num_cars_real = 0

    traci.start(sumoCmd)

    if(new_scenario):
        if(not real_data):
            print("WARNING : Creating a new scenario...")
            bike_poisson_lambda = 0.2 #random.uniform(0,max(list_bike_poisson_lambdas))
            car_poisson_lambda = 0.1
            
            bike_poisson_distrib = np.random.poisson(bike_poisson_lambda, simu_length)
            car_poisson_distrib = np.random.poisson(car_poisson_lambda, simu_length)

        num_cyclists = sum(bike_poisson_distrib)
        num_cars = sum(car_poisson_distrib)
        simu_length = len(bike_poisson_distrib)
        
    else:
        print("WARNING : Loading the scenario...")
        old_dict_scenario = tab_dict_old_scenarios[s]

        num_cyclists = len(old_dict_scenario["bikes"])
        num_cars = len(old_dict_scenario["cars"])

        if(real_data or num_simu == 1):
            simu_length *= 24

        if(len(old_dict_scenario["bikes"].keys()) == 0):
            max_id_cyclist = 0
        else:
            max_id_cyclist = max(old_dict_scenario["bikes"].keys())

        if(len(old_dict_scenario["cars"].keys()) == 0):
            max_id_car = 0  
        else:
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

    print(simu_length)

    while(step<=simu_length):
        if(new_scenario): #new_scenario
            if(step<simu_length):
                for _ in range(int(bike_poisson_distrib[int(step)])):
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
                for _ in range(int(car_poisson_distrib[int(step)])):
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

    if(not test and "PPO" in method):
        structure.drl_agent.end_episode()

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

    if(not test and s%50 == 0 and ("DQN" in method or "PPO" in method)):
        torch.save(structure.drl_agent.model.state_dict(), "files/"+sub_folders+pre_file_name+"trained.n")
        if("DQN" in method):
            torch.save(structure.drl_agent.model_target.state_dict(), "files/"+sub_folders+pre_file_name+"trained_target.n")

if(not test and ("DQN" in method or "PPO" in method)):
    torch.save(structure.drl_agent.model.state_dict(), "files/"+sub_folders+pre_file_name+"trained.n")
    if("DQN" in method):
        torch.save(structure.drl_agent.model_target.state_dict(), "files/"+sub_folders+pre_file_name+"trained_target.n")
