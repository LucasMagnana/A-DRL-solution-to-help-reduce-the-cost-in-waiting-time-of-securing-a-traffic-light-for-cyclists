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


def spawn_vehicle(id_start, id_end, list_edges_name, net, dict_scenario, v_type, dict_vehicles, id_vehicle):

    str_id_vehicle = str(id_vehicle)

    if(id_end == None):
        id_end = id_start
        while(id_end == id_start or id_start == "NS" and id_end == "EW" or id_start == "EW" and id_end == "SN" or\
        id_start == "SN" and id_end == "WE" or id_start == "WE" and id_end == "NS"):
            id_end = list_edges_name[random.randint(0, len(list_edges_name)-1)]

        id_start = "E_"+str(id_start)
        id_end = "-E_"+str(id_end)
        
            
    e1 = net.getEdge(id_start)
    e2 = net.getEdge(id_end)

    if(v_type=="bikes"):
        v_class = "bicycle"
        id_path = str_id_vehicle + "_sp"
        id_sumo = str_id_vehicle
        depart_l = "0"
    elif(v_type=="cars"):
        v_class = "passenger"
        id_path = str_id_vehicle + "_c_sp"
        id_sumo = str_id_vehicle + "_c"
        depart_l = "best"

    path = net.getShortestPath(e1, e2, vClass=v_class)[0]
    dict_scenario[v_type][id_vehicle] = {"start_step": step, "start_edge": e1.getID(), "end_edge": e2.getID(),
    "distance_travelled": net.getShortestPath(e1, e2, vClass=v_class, fromPos=0)[1], "waiting_time": 0}

    path = [e.getID() for e in path]
    traci.route.add(id_path, path)      
    traci.vehicle.add(id_sumo, id_path, departLane=depart_l, typeID=v_class, departSpeed="avg")
    if(v_type=="bikes"):
        traci.vehicle.changeLane(id_sumo, 0, 99999)

    dict_vehicles[v_type][str_id_vehicle]=[]

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


def save(tab_dict_scenarios, args, structure, sub_folders, pre_file_name, use_drl):
    print("WARNING: Saving scenario...")
    if(not os.path.exists("files/"+sub_folders)):
        os.makedirs("files/"+sub_folders)

    if(os.path.exists("files/"+sub_folders+pre_file_name+"scenarios.tab")):
        with open("files/"+sub_folders+pre_file_name+"scenarios.tab", 'rb') as infile:
            tab_saved_dict_scenarios = pickle.load(infile)
    else:
        tab_saved_dict_scenarios = []

    tab_saved_dict_scenarios += tab_dict_scenarios
    print(len(tab_saved_dict_scenarios))

    with open("files/"+sub_folders+pre_file_name+"scenarios.tab", 'wb') as outfile:
        pickle.dump(tab_saved_dict_scenarios, outfile)

    if(not args.test and use_drl):
        with open("files/"+sub_folders+pre_file_name+"losses.tab", 'wb') as outfile:
            pickle.dump(structure.drl_agent.tab_losses, outfile)
        torch.save(structure.drl_agent.model.state_dict(), "files/"+sub_folders+pre_file_name+"trained.n")
        if("DQN" in args.method):
            torch.save(structure.drl_agent.model_target.state_dict(), "files/"+sub_folders+pre_file_name+"trained_target.n")


min_group_size = 5

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

    list_edges_name = ["NS", "SN", "EW", "WE"]

    dict_bike_poisson_lambdas = {}
    dict_bike_poisson_distrib = {}
    dict_car_poisson_distrib = {}

    for en in list_edges_name:
        dict_bike_poisson_lambdas[en] = []
        dict_bike_poisson_distrib[en] = []
        dict_car_poisson_distrib[en] = []


    if(args.test and not args.load_scenario):
        num_simu = 20

    if(args.real_data):        
        with open("./real_data.json", "rb") as infile:
            count_data = json.load(infile)

        first_day_number = None
        for data in count_data["data"]["values"]:
            d = datetime.strptime(data["time"], '%Y-%m-%dT%H:%M:%S.%f%z')
            if(first_day_number == None):
                first_day_number = d.day

            if(not args.test or d.day != first_day_number):
                dict_bike_poisson_lambdas[data["id"]].append(data["count"]/simu_length)

        dict_bike_poisson_lambdas["NS"] = copy.deepcopy(dict_bike_poisson_lambdas["EW"])
        dict_bike_poisson_lambdas["SN"] = copy.deepcopy(dict_bike_poisson_lambdas["WE"])
                
        if(args.test):
            num_simu = 1
            print("WARNING : Creating a new scenario using real data...")
            for en in list_edges_name:
                dict_bike_poisson_distrib[en] = np.empty(0)
                dict_car_poisson_distrib[en] = np.empty(0)
                for pl in dict_bike_poisson_lambdas[en]:           
                    bike_poisson_lambda = pl
                    car_poisson_lambda = bike_poisson_lambda

                    dict_bike_poisson_distrib[en] = np.concatenate((dict_bike_poisson_distrib[en], np.random.poisson(bike_poisson_lambda, simu_length)))
                    dict_car_poisson_distrib[en] = np.concatenate((dict_car_poisson_distrib[en], np.random.poisson(car_poisson_lambda, simu_length)))



step_length = 1
speed_threshold = 0.5


if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


use_drl = "DQN" in args.method or "PPO" in args.method


if("actuated" in args.method):
    conf = "sumo_files/sumo_act.sumocfg"
else:
    conf = "sumo_files/sumo.sumocfg"

sumoBinary = "/usr/bin/sumo"
if(args.gui):
    sumoBinary += "-gui"
sumoCmd = [sumoBinary, "-c", conf, "--quit-on-end", "--waiting-time-memory", '10000', '--start', '--delay', '1000', '--step-length', str(step_length),\
'--time-to-teleport', '-1'] #, "--no-warnings"]

import traci
import traci.constants as tc
import sumolib



if(args.test):
    sub_folders = "test/"
else:
    sub_folders = "train/"

if("actuated" in args.method):
    net = sumolib.net.readNet("sumo_files/net_act.net.xml")
else:
    net = sumolib.net.readNet("sumo_files/net.net.xml")
edges = net.getEdges()

structure = Structure(edges, list_edges_name, net, traci, simu_length, args.method, args.test, min_group_size, args.alpha, use_drl=use_drl)

pre_file_name = args.method+"_"

tab_dict_scenarios = []

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

ep = 0

cont = True

while(cont):

    ep += 1

    if(not args.test and "DQN" in args.method and structure.drl_agent.num_decisions_made >= structure.drl_agent.hyperParams.DECISION_CT_LEARNING_START):
        structure.drl_agent.learn()

    if(not args.test and "PPO" in args.method):
        structure.drl_agent.start_episode()
        if((ep-1) != start_num_simu and (ep-1)%structure.drl_agent.hyperParams.LEARNING_EP == 0):
            structure.drl_agent.learn()

    next_step_wt_update = 0

    num_cyclists_real = 0
    num_cars_real = 0

    traci.start(sumoCmd)

    structure.create_tls_phases()

    if(not args.load_scenario):
        print("WARNING : Creating a new scenario...")
        if(args.real_data):
            if(not args.test):
                for en in list_edges_name:
                    r = randint(0, len(dict_bike_poisson_lambdas[en])-1)
                    bike_poisson_lambda = dict_bike_poisson_lambdas[en][r]
                    car_poisson_lambda = bike_poisson_lambda
                    dict_bike_poisson_distrib[en] = np.random.poisson(bike_poisson_lambda, simu_length)
                    dict_car_poisson_distrib[en] = np.random.poisson(car_poisson_lambda, simu_length)
            else:
                simu_length *= 24

        else:
            for en in list_edges_name:
                bike_poisson_lambda = random.uniform(0.05, 0.1)
                car_poisson_lambda = bike_poisson_lambda
                dict_bike_poisson_distrib[en] = np.random.poisson(bike_poisson_lambda, simu_length)
                dict_car_poisson_distrib[en] = np.random.poisson(car_poisson_lambda, simu_length)
    
        num_cyclists = 0
        num_cars = 0

        for en in list_edges_name:
            num_cyclists += sum(dict_bike_poisson_distrib[en])
            num_cars += sum(dict_car_poisson_distrib[en])
        
    else:
        print("WARNING : Loading the scenario...")
        old_dict_scenario = tab_dict_old_scenarios[ep-1]

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

    forced_stop = False

    print(simu_length)

    while(step<=simu_length or len(traci.vehicle.getIDList()) > 0):

        if(step >= simu_length*2):
            forced_stop = True
            break 

        if(not args.load_scenario): #new_scenario
            if(step<simu_length):
                for en in list_edges_name:
                    for _ in range(int(dict_bike_poisson_distrib[en][int(step)])):
                        spawn_vehicle(en, None, list_edges_name, net, dict_scenario, "bikes", dict_vehicles, id_cyclist)
                        id_cyclist+=1
                    dict_bike_poisson_distrib[en][int(step)] = 0

                    for _ in range(int(dict_car_poisson_distrib[en][int(step)])):
                        spawn_vehicle(en, None, list_edges_name, net, dict_scenario, "cars", dict_vehicles, id_car)
                        id_car+=1
                    dict_car_poisson_distrib[en][int(step)] = 0

        else:
            while(id_cyclist not in old_dict_scenario["bikes"] and id_cyclist <= max_id_cyclist):
                id_cyclist += 1
            if(id_cyclist in old_dict_scenario["bikes"] and step >= old_dict_scenario["bikes"][id_cyclist]["start_step"]):
                start_edge_id=old_dict_scenario["bikes"][id_cyclist]["start_edge"]
                end_edge_id=old_dict_scenario["bikes"][id_cyclist]["end_edge"]

                spawn_vehicle(start_edge_id, end_edge_id, list_edges_name, net, dict_scenario, "bikes", dict_vehicles, id_cyclist)
                id_cyclist+=1

            while(id_car not in old_dict_scenario["cars"] and id_car <= max_id_car):
                id_car += 1
            if(id_car in old_dict_scenario["cars"] and step >= old_dict_scenario["cars"][id_car]["start_step"]):
                start_edge_id=old_dict_scenario["cars"][id_car]["start_edge"]
                end_edge_id=old_dict_scenario["cars"][id_car]["end_edge"]

                spawn_vehicle(start_edge_id, end_edge_id, list_edges_name, net, dict_scenario, "cars", dict_vehicles, id_car)
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

    if(use_drl):
        structure.drl_decision_making(step, end=True, forced_stop=forced_stop)

    traci.close()

    if(not args.test and "PPO" in args.method):
        structure.drl_agent.end_episode()

    '''for vehicle_type in dict_scenario:
        for i in copy.deepcopy(list(dict_scenario[vehicle_type].keys())):
            if("finish_step" not in dict_scenario[vehicle_type][i]):
                del dict_scenario[vehicle_type][i]'''


    print("\nep:", ep, "data number:", num_cars_real+num_cyclists_real, ",", structure.num_cyclists_crossed, "cyclits used struct, last step:", step)
    if(use_drl):
        print("num decisions:", structure.drl_agent.num_decisions_made)

    if(args.test):
        if(args.real_data):
            cont = ep < 1
        else:
            cont = ep < 23
    else:
        cont = structure.drl_agent.num_decisions_made < structure.drl_agent.hyperParams.DECISION_COUNT


    if(save_scenario):

        tab_dict_scenarios.append(dict_scenario)

        if(args.test or ep%50 == 1):
            save(tab_dict_scenarios, args, structure, sub_folders, pre_file_name, use_drl)
            tab_dict_scenarios = []


    bikes_data = compute_data(dict_scenario["bikes"])
    cars_data = compute_data(dict_scenario["cars"])

    print(f"mean cars waiting time: {cars_data[1]/cars_data[2]}")
    print(f"mean bikes waiting time: {bikes_data[1]/bikes_data[2]}")
    print(f"tot waiting time: {bikes_data[1]+cars_data[1]}")

    if(use_drl):
        print(f"cumulative reward:", structure.drl_cum_reward)




if(save_scenario):
    save(tab_dict_scenarios, args, structure, sub_folders, pre_file_name, use_drl)