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
    parse = argparse.ArgumentParser()
    parse.add_argument('--learning', type=bool, default=False)
    parse.add_argument('--poisson-lambda', type=float, default=0.2)
    parse.add_argument('--min-group-size', type=int, default=5)
    parse.add_argument('--gui', type=bool, default=False)
    parse.add_argument('--config', type=int, default=3)
    parse.add_argument('--struct-open', type=bool, default=False)
    parse.add_argument('--use-drl', type=bool, default=False)
    parse.add_argument('--test', type=bool, default=False)
    parse.add_argument('--save-scenario', type=bool, default=False)
    
args = parse.parse_args()


learning = True
batch_size = 32
hidden_size_1 = 64
hidden_size_2 = 32
lr=1e-5

step_length = 0.2
simu_length = 25*3600
speed_threshold = 0.5

if(args.config == 0):
    car_poisson_lambda = 0.2
    bike_poisson_lambda = 0.2
    evoluting = "group_size"
elif(args.config == 1):
    car_poisson_lambda = args.poisson_lambda
    bike_poisson_lambda = 1
    evoluting = "cars"
elif(args.config == 2):
    car_poisson_lambda = 0.2
    bike_poisson_lambda = args.poisson_lambda
    evoluting = "bikes"
elif(args.config == 3):
    car_poisson_lambda = 0.2
    bike_poisson_lambda = 0.4
    evoluting = "cars"
    args.struct_open = True


bike_poisson_distrib = np.random.poisson(bike_poisson_lambda, simu_length)
car_poisson_distrib = np.random.poisson(car_poisson_lambda, simu_length)


if(args.use_drl):
    sub_folders = "w_model/"
else:
    sub_folders = "wou_model/"

if(evoluting=="bikes"):
    sub_folders+="config_"+str(args.config)+"/"+str(car_poisson_lambda)+"/"
    variable_evoluting = args.poisson_lambda
elif(evoluting=="cars"):
    sub_folders+="config_"+str(args.config)+"/"+str(bike_poisson_lambda)+"/"
    variable_evoluting = args.poisson_lambda
else:
    sub_folders+="config_"+str(args.config)+"/"+str(bike_poisson_lambda)+"/"
    variable_evoluting = args.min_group_size



if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

sumoBinary = "/usr/bin/sumo"
if(args.gui):
    sumoBinary += "-gui"
sumoCmd = [sumoBinary, "-c", "sumo_files/sumo_"+str(args.config)+".sumocfg", "--extrapolate-departpos", "--quit-on-end", "--waiting-time-memory", '10000', '--start', '--delay', '0', '--step-length', str(step_length), '--no-warnings']



import traci
import sumolib
import traci.constants as tc


def spawn_cyclist(id_cyclist, step, path, net, structure, step_length, max_speed, struct_candidate, args, dict_cyclists):
    if(args.struct_open or num_cyclists-id_cyclist+max(len(structure.id_cyclists_waiting), len(traci.edge.getLastStepVehicleIDs(structure.start_edge.getID())))<structure.min_group_size):
        struct_candidate = True
    
    c = Cyclist(str(id_cyclist), step, path, net, structure, max_speed, traci, sumolib, step_length, struct_candidate=struct_candidate)
    dict_cyclists[str(id_cyclist)]=c


def spawn_car(id_car, step, path, net, dict_cars):
    path = [e.getID() for e in path]
    traci.route.add(str(id_car)+"_c_sp", path)
    traci.vehicle.add(str(id_car)+"_c", str(id_car)+"_c_sp", departLane="best", departPos="last", typeID='car', departSpeed="last")
    dict_cars[str(id_car)]=[]




traci.start(sumoCmd)

net = sumolib.net.readNet("sumo_files/net_"+str(args.config)+".net.xml")
edges = net.getEdges()



print("WARNING : Creating a new scenario...")
dict_scenario={"cars": [], "bikes": []}
num_cyclists = sum(bike_poisson_distrib)
num_cars = sum(car_poisson_distrib)

if(False):
    print("WARNING : Loading the scenario...")
    with open('scenario.dict', 'rb') as infile:
        dict_scenario = pickle.load(infile)
    num_cyclists = len(dict_scenario["bikes"])
    num_cars = len(dict_scenario["cars"])

    
print("num_cyclists: ", num_cyclists, ", num_cars :", num_cars)

dict_edges_index = {}
for i, e in enumerate(edges) :
    dict_edges_index[e.getID()] = i


dict_cyclists= {}
dict_cars = {}
dict_cyclists_arrived = {}

structure = Structure("E_start", "E2", edges, net, dict_cyclists, traci, args.config, dict_scenario, simu_length, dict_edges_index=dict_edges_index,\
open=args.struct_open, min_group_size=args.min_group_size, batch_size=batch_size, learning=args.learning, use_drl=args.use_drl, test=args.test)


id_cyclist = 0
id_car = 0
step = 0

while(step<simu_length or len(dict_cyclists) != 0 or len(dict_cars) != 0):
    if(True): #new_scenario
        if(step<simu_length):
            for _ in range(bike_poisson_distrib[int(step)]):
                e1 = net.getEdge("E0")
                e2 = net.getEdge("E"+str(randint(3, 9)))
                path = net.getShortestPath(e1, e2, vClass='bicycle')[0]
                max_speed = np.random.normal(15, 3)
                dict_scenario["bikes"].append({"start_step": step, "start_edge": e1.getID(), "end_edge": e2.getID(), "max_speed": max_speed,
                "distance_travelled": net.getShortestPath(net.getEdge("E_start"), e2, vClass='bicycle', fromPos=0)[1], "waiting_time": 0})
                spawn_cyclist(id_cyclist, step, path, net, structure, step_length, max_speed, False, args, dict_cyclists)
                id_cyclist+=1
            bike_poisson_distrib[int(step)] = 0
        if(step<simu_length):  
            for _ in range(car_poisson_distrib[int(step)]):
                e1 = net.getEdge("E0")
                e2 = net.getEdge("E"+str(randint(3, 9)))
                path = net.getShortestPath(e1, e2, vClass='passenger')[0]
                dict_scenario["cars"].append({"start_step": step, "start_edge": e1.getID(), "end_edge": e2.getID(),
                "distance_travelled": net.getShortestPath(net.getEdge("E_start"), e2, vClass='passenger', fromPos=0)[1], "waiting_time": 0})
                spawn_car(id_car, step, path, net, dict_cars)
                id_car+=1
                car_poisson_distrib[int(step)] = 0

    else:
        if(id_cyclist<len(dict_scenario["bikes"]) and step >= dict_scenario["bikes"][id_cyclist]["start_step"]):
            e1=dict_scenario["bikes"][id_cyclist]["start_edge"]
            e2=dict_scenario["bikes"][id_cyclist]["end_edge"]
            path = net.getShortestPath(e1, e2, vClass='bicycle')[0]
            spawn_cyclist(id_cyclist, step, path, net, structure, step_length, dict_scenario["bikes"][id_cyclist]["max_speed"], False, args, dict_cyclists)
            id_cyclist+=1
        if(id_car<len(dict_scenario["cars"]) and step >= dict_scenario["cars"][id_car]["start_step"]):
            e1=dict_scenario["cars"][id_car]["start_edge"]
            e2=dict_scenario["cars"][id_car]["end_edge"]
            path = net.getShortestPath(e1, e2, vClass='passenger')[0]
            spawn_car(id_car, step, path, net, dict_cars)
            id_car+=1

    traci.simulationStep() 

    for i in copy.deepcopy(list(dict_cyclists.keys())):
        if(dict_scenario["bikes"][int(i)]["start_step"]<0 and i in traci.vehicle.getIDList() and traci.vehicle.getRoadID(i) == "E_start"):
            dict_scenario["bikes"][int(i)]["start_step"] = step
        dict_cyclists[i].step(step)
        if(not dict_cyclists[i].alive):
            if(dict_cyclists[i].finish_step > 0):
                dict_cyclists_arrived[i] = dict_cyclists[i]
                target = None
                if(i in structure.dict_model_input and target != None):
                    structure.list_input_to_learn.append(structure.dict_model_input[i])
                    structure.list_target.append(target)                  
                    del structure.dict_model_input[i]

                dict_scenario["bikes"][int(dict_cyclists[i].id)]["finish_step"] = step              
            else:
                traci.vehicle.remove(i)
            del dict_cyclists[i]
        else:
            if(traci.vehicle.getSpeed(i)< speed_threshold):
                dict_scenario["bikes"][int(dict_cyclists[i].id)]["waiting_time"] += 1

    for i in copy.deepcopy(list(dict_cars.keys())):
        sumo_id = i+"_c"
        if(dict_scenario["cars"][int(i)]["start_step"]<0 and sumo_id in traci.vehicle.getIDList() and traci.vehicle.getRoadID(sumo_id) == "E_start"):
            dict_scenario["cars"][int(i)]["start_step"] = step
        if(sumo_id in traci.simulation.getArrivedIDList()):
            dict_scenario["cars"][int(i)]["finish_step"] = step
            del dict_cars[i]
        else:
            if(traci.vehicle.getSpeed(sumo_id)< speed_threshold):
                dict_scenario["cars"][int(i)]["waiting_time"] += 1

    #(step%1, step%1<=step_length)
    if(structure.open):
        structure.step(step, edges)

    print(f"\rStep {int(step)}: {len(traci.vehicle.getIDList())} cyclists in simu, {id_cyclist} cyclists spawned since start,\
{structure.num_cyclists_crossed} cyclists crossed the struct.", end="")

    step += step_length


if(args.save_scenario):
    pre_file_name = evoluting+"_evolv_"

    print("WARNING: Saving scenario...")
    if(not os.path.exists("files/"+sub_folders)):
        os.makedirs("files/"+sub_folders)
        with open("files/"+sub_folders+pre_file_name+"scenarios.dict", 'wb') as outfile:
            pickle.dump({variable_evoluting : [dict_scenario]}, outfile)
    elif(os.path.exists("files/"+sub_folders+pre_file_name+"scenarios.dict")):
        with open("files/"+sub_folders+pre_file_name+"scenarios.dict", 'rb') as infile:
            d_scenarios = pickle.load(infile)
        if(variable_evoluting in d_scenarios):
            d_scenarios[variable_evoluting].append(dict_scenario)
        else:
            d_scenarios[variable_evoluting] = [dict_scenario]
        with open("files/"+sub_folders+pre_file_name+"scenarios.dict", 'wb') as outfile:
            pickle.dump(d_scenarios, outfile)
    else:
        with open("files/"+sub_folders+pre_file_name+"scenarios.dict", 'wb') as outfile:
            pickle.dump({variable_evoluting : [dict_scenario]}, outfile)

if(args.use_drl and not args.test):
    torch.save(structure.drl_agent.actor_target.state_dict(), "files/"+sub_folders+"trained_target.n")
    torch.save(structure.drl_agent.actor.state_dict(), "files/"+sub_folders+"trained.n")





traci.close()

if(len(structure.list_input_to_learn)>0):
    structure.learn()
    
print("\ndata number:", len(dict_cyclists_arrived), ",", structure.num_cyclists_crossed, "cyclits used struct, last step:", step)


mean_cars_travel_time = compute_graphs_data_cars(dict_scenario)
mean_cyclists_travel_time = compute_graphs_data_cyclists(dict_scenario)

print(f"mean car travel time: {mean_cars_travel_time}, mean cyclists travel time: {mean_cyclists_travel_time}")