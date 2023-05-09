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

def spawn_cyclist(id_cyclist, step, path, net, dict_bikes):
    path = [e.getID() for e in path]
    traci.route.add(str(id_cyclist)+"_sp", path)      
    traci.vehicle.add(str(id_cyclist), str(id_cyclist)+"_sp", departLane="0", typeID='bicycle', departSpeed="avg")
    traci.vehicle.changeLane(str(id_cyclist), 0, 99999)
    dict_bikes[str(id_cyclist)]=[]


def spawn_car(id_car, step, path, net, dict_cars):
    path = [e.getID() for e in path]
    traci.route.add(str(id_car)+"_c_sp", path)
    traci.vehicle.add(str(id_car)+"_c", str(id_car)+"_c_sp", departLane="best", typeID='car')
    dict_cars[str(id_car)]=[]


min_group_size = 5

num_simu = 500
simu_length = 3600

save_scenario = True

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()

    parser.add_argument("--gui", action="store_true")

    parser.add_argument("-m", "--method", type=str, default="actuated")

    parser.add_argument("-a", "--alpha", type=float, default=0.5)

    parser.add_argument("--save-scenario", action="store_true")
    parser.add_argument("--load-scenario", action="store_true")

    parser.add_argument("--test", action="store_true")
    parser.add_argument("--real-data", action="store_true")

    args = parser.parse_args()


    if(args.test):
        if(not args.load_scenario):
            num_simu = 20
            if(args.real_data):
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
    


step_length = 1
speed_threshold = 0.5


if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

sumoBinary = "/usr/bin/sumo"
if(args.gui):
    sumoBinary += "-gui"
sumoCmd = [sumoBinary, "-c", "sumo_files/sumo.sumocfg", "--quit-on-end", "--waiting-time-memory", '10000', '--start', '--delay', '1000', '--step-length', str(step_length),\
'--time-to-teleport', '-1', "--no-warnings"]

import traci
import traci.constants as tc
import sumolib



if(args.test):
    sub_folders = "test/"
else:
    sub_folders = "train/"


net = sumolib.net.readNet("sumo_files/net1.net.xml")
edges = net.getEdges()

structure = Structure(edges, net, traci, simu_length, args.method, args.test, min_group_size, args.alpha)

pre_file_name = args.method+"_"


start_num_simu = 0

if(args.load_scenario):
    with open("files/"+sub_folders+"actuated_scenarios.tab", 'rb') as infile:
        tab_dict_old_scenarios = pickle.load(infile)
        num_simu = len(tab_dict_old_scenarios)

    if(os.path.exists("files/"+sub_folders+pre_file_name+"scenarios.tab")):
        with open("files/"+sub_folders+pre_file_name+"scenarios.tab", 'rb') as infile:
            tab_dict_scenarios = pickle.load(infile)
            start_num_simu = len(tab_dict_scenarios)


if(args.method != "actuated"):
    sub_folders += str(args.alpha)+"/"

#for s in range(start_num_simu, num_simu):

while(structure.drl_agent.num_decisions_made < structure.drl_agent.hyperParams.DECISION_COUNT):

    if(not args.test and "DQN" in args.method and structure.drl_agent.num_decisions_made >= structure.drl_agent.hyperParams.DECISION_CT_LEARNING_START):
        structure.drl_agent.learn()

    if(not args.test and "PPO" in args.method):
        structure.drl_agent.start_episode()
        if(s != start_num_simu and s%structure.drl_agent.hyperParams.LEARNING_EP == 0):
            structure.drl_agent.learn()

    next_step_wt_update = 0

    num_cyclists_real = 0
    num_cars_real = 0

    traci.start(sumoCmd)

    structure.create_tls_phases()

    if(not args.load_scenario):
        if(not args.real_data):
            print("WARNING : Creating a new scenario...")
            bike_poisson_lambda = 0 #random.uniform(0,max(list_bike_poisson_lambdas))
            car_poisson_lambda = 1
            
            bike_poisson_distrib = np.random.poisson(bike_poisson_lambda, simu_length)
            car_poisson_distrib = np.random.poisson(car_poisson_lambda, simu_length)

        num_cyclists = 0 #sum(bike_poisson_distrib)
        num_cars = sum(car_poisson_distrib)
        simu_length = len(bike_poisson_distrib)
        
    else:
        print("WARNING : Loading the scenario...")
        old_dict_scenario = tab_dict_old_scenarios[s]

        num_cyclists = len(old_dict_scenario["bikes"])
        num_cars = len(old_dict_scenario["cars"])

        if(args.real_data or num_simu == 1):
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
        if(not args.load_scenario): #new_scenario
            if(step<simu_length):
                for _ in range(int(bike_poisson_distrib[int(step)])):
                    id_start = random.randint(0, 3)
                    id_end = id_start
                    while(id_end == id_start or id_start == 0 and id_end == 3 or id_start == 3 and id_end == 1):
                        id_end = random.randint(0, 3)
                    e1 = net.getEdge("E"+str(id_start))
                    e2 = net.getEdge("-E"+str(id_end))
                    path = net.getShortestPath(e1, e2, vClass='bicycle')[0]
                    max_speed = np.random.normal(15, 3)
                    dict_scenario["bikes"][id_cyclist] = {"start_step": step, "start_edge": e1.getID(), "end_edge": e2.getID(),
                    "distance_travelled": net.getShortestPath(e1, e2, vClass='bicycle', fromPos=0)[1], "waiting_time": 0}
                    spawn_cyclist(id_cyclist, step, path, net, dict_vehicles["bikes"])
                    id_cyclist+=1
                bike_poisson_distrib[int(step)] = 0

                for _ in range(int(car_poisson_distrib[int(step)])):
                    id_start = random.randint(0, 3)
                    id_end = id_start
                    while(id_end == id_start):
                        id_end = random.randint(0, 3)
                    e1 = net.getEdge("E"+str(id_start))
                    e2 = net.getEdge("-E"+str(id_end))
                    path = net.getShortestPath(e1, e2, vClass='passenger')[0]
                    dict_scenario["cars"][id_car] = {"start_step": step, "start_edge": e1.getID(), "end_edge": e2.getID(),
                    "distance_travelled": net.getShortestPath(e1, e2, vClass='passenger', fromPos=0)[1], "waiting_time": 0}
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
                dict_scenario["bikes"][id_cyclist] = {"start_step": step, "start_edge": e1.getID(), "end_edge": e2.getID(),
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
                            #dict_scenario[vehicle_type][int(i)]["waiting_time"] = traci.vehicle.getAccumulatedWaitingTime(sumo_id)
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

    if(not args.test and "PPO" in args.method):
        structure.drl_agent.end_episode()

    '''for vehicle_type in dict_scenario:
        for i in copy.deepcopy(list(dict_scenario[vehicle_type].keys())):
            if("finish_step" not in dict_scenario[vehicle_type][i]):
                del dict_scenario[vehicle_type][i]'''


    print("\ndata number:", num_cars_real+num_cyclists_real, ",", structure.num_cyclists_crossed, "cyclits used struct, last step:", step,\
           "decisions:", structure.drl_agent.num_decisions_made)



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

    print(f"mean cars waiting time: {cars_data[1]/cars_data[2]}")
    print(f"mean bikes waiting time: {bikes_data[1]/bikes_data[2]}")
    print(f"tot waiting time: {bikes_data[1]+cars_data[1]}")

    if("DQN" in args.method or "PPO" in args.method):
        print(f"cumulative reward:", structure.drl_cum_reward)

    if(not args.test and structure.drl_agent.num_decisions_made%10000 == 0 and ("DQN" in args.method or "PPO" in args.method)):
        torch.save(structure.drl_agent.model.state_dict(), "files/"+sub_folders+pre_file_name+"trained.n")
        if("DQN" in args.method):
            torch.save(structure.drl_agent.model_target.state_dict(), "files/"+sub_folders+pre_file_name+"trained_target.n")

if(not args.test and ("DQN" in args.method or "PPO" in args.method)):
    torch.save(structure.drl_agent.model.state_dict(), "files/"+sub_folders+pre_file_name+"trained.n")
    if("DQN" in args.method):
        torch.save(structure.drl_agent.model_target.state_dict(), "files/"+sub_folders+pre_file_name+"trained_target.n")
