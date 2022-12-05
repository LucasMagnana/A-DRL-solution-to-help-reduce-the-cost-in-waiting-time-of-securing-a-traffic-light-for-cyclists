import os, sys
from random import randint
import numpy as np 
import pickle
import osmnx as ox
import copy
import torch
import argparse

from Cyclist import Cyclist
from Structure import Structure
from graphs import *
from Model import Model


if __name__ == "__main__": 
    parse = argparse.ArgumentParser()
    parse.add_argument('--new-scenario', type=bool, default=False)
    parse.add_argument('--learning', type=bool, default=False)
    parse.add_argument('--bike-poisson-lambda', type=float, default=0.20)
    parse.add_argument('--car-poisson-lambda', type=float, default=0.25)
    parse.add_argument('--min-group-size', type=int, default=5)
    parse.add_argument('--gui', type=bool, default=False)
    parse.add_argument('--config', type=int, default=0)
    
args = parse.parse_args()


use_model = False
save_model = use_model
learning = True
batch_size = 32
hidden_size_1 = 64
hidden_size_2 = 32
lr=1e-5

step_length = 0.2


bike_poisson_distrib = np.random.poisson(args.bike_poisson_lambda, 3600)
car_poisson_distrib = np.random.poisson(args.car_poisson_lambda, 3600)


if(use_model):
    sub_folders = "w_model/"
else:
    sub_folders = "wou_model/"

sub_folders+="config_"+str(args.config)+"/"





if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

sumoBinary = "/usr/bin/sumo"
if(args.gui):
    sumoBinary += "-gui"
sumoCmd = [sumoBinary, "-c", "sumo_files/sumo_"+str(args.config)+".sumocfg", "--quit-on-end", "--waiting-time-memory", '10000', '--start', '--delay', '0', '--step-length', str(step_length), '--no-warnings']


import traci
import sumolib
import traci.constants as tc


def spawn_cyclist(id_cyclist, step, path, net, structure, step_length, max_speed, struct_candidate, args, dict_cyclists):
    if(args.new_scenario or num_cyclists-id_cyclist+max(len(structure.id_cyclists_waiting), len(traci.edge.getLastStepVehicleIDs(structure.start_edge.getID())))<structure.min_group_size):
        struct_candidate = False
    
    c = Cyclist(str(id_cyclist), step, path, net, structure, max_speed, traci, sumolib, step_length, struct_candidate=struct_candidate)
    dict_cyclists[str(id_cyclist)]=c


def spawn_car(id_car, step, path, net, dict_cars):
    path = [e.getID() for e in path]
    traci.route.add(str(id_car)+"_c_sp", path)
    traci.vehicle.add(str(id_car)+"_c", str(id_car)+"_c_sp", departLane="best", typeID='car')
    dict_cars[str(id_car)]=[]



traci.start(sumoCmd)

net = sumolib.net.readNet("sumo_files/net_"+str(args.config)+".net.xml")
edges = net.getEdges()


if(args.new_scenario):
    print("WARNING : Creating a new scenario...")
    dict_scenario={"cars": [], "bikes": []}
    num_cyclists = sum(bike_poisson_distrib)
    num_cars = sum(car_poisson_distrib)
else:
    print("WARNING : Loading the scenario...")
    with open('scenario.dict', 'rb') as infile:
        dict_scenario = pickle.load(infile)
    num_cyclists = len(dict_scenario["bikes"])
    num_cars = len(dict_scenario["cars"])

    
print(num_cyclists, num_cars)

dict_edges_index = {}
for i, e in enumerate(edges) :
    dict_edges_index[e.getID()] = i


if(use_model == True):
    model = Model(len(edges), hidden_size_1, hidden_size_2)
    print("WARNING : Using neural network...", end="")
    if(os.path.exists("files/"+sub_folders+"model.pt")):
        model.load_state_dict(torch.load("files/"+sub_folders+"model.pt"))
        model.eval()
        print("Loading it.", end="")       
    print("")
else:
    model = None

dict_cyclists= {}
dict_cars = {}
dict_cyclists_arrived = {}

structure = Structure("E0", "E2", edges, net, dict_cyclists, traci, dict_edges_index, model,\
open=not args.new_scenario, min_group_size=args.min_group_size, batch_size=batch_size, learning=args.learning, lr=lr)


id_cyclist = 0
id_car = 0
step = 0

while(len(dict_cyclists) != 0 or id_cyclist<num_cyclists or len(dict_cars) != 0 or id_car<num_cars):
    if(args.new_scenario):
        if(id_cyclist<num_cyclists):
            for _ in range(bike_poisson_distrib[int(step)]):
                e1 = net.getEdge("E0")
                e2 = net.getEdge("E3")#+str(randint(4, 9)))
                path = net.getShortestPath(e1, e2, vClass='bicycle')[0]
                max_speed = np.random.normal(15, 3)
                dict_scenario["bikes"].append({"start_step": step, "start_edge": e1, "end_edge": e2, "max_speed": max_speed, "finish_step": -1})
                spawn_cyclist(id_cyclist, step, path, net, structure, step_length, max_speed, args.config==0, args, dict_cyclists)
                id_cyclist+=1
            bike_poisson_distrib[int(step)] = 0
        if(id_car<num_cars):
            for _ in range(car_poisson_distrib[int(step)]):
                e1 = net.getEdge("E0")
                e2 = net.getEdge("E"+str(randint(3, 9)))
                path = net.getShortestPath(e1, e2, vClass='passenger')[0]
                dict_scenario["cars"].append({"start_step": step, "start_edge": e1, "end_edge": e2, "finish_step": -1})
                spawn_car(id_car, step, path, net, dict_cars)
                id_car+=1
            car_poisson_distrib[int(step)] = 0

    else:
        if(id_cyclist<len(dict_scenario["bikes"]) and step >= dict_scenario["bikes"][id_cyclist]["start_step"]):
            e1=dict_scenario["bikes"][id_cyclist]["start_edge"]
            e2=dict_scenario["bikes"][id_cyclist]["end_edge"]
            path = net.getShortestPath(e1, e2, vClass='bicycle')[0]
            spawn_cyclist(id_cyclist, step, path, net, structure, step_length, dict_scenario["bikes"][id_cyclist]["max_speed"], args.config==0, args, dict_cyclists)
            id_cyclist+=1
        if(id_car<len(dict_scenario["cars"]) and step >= dict_scenario["cars"][id_car]["start_step"]):
            e1=dict_scenario["cars"][id_car]["start_edge"]
            e2=dict_scenario["cars"][id_car]["end_edge"]
            path = net.getShortestPath(e1, e2, vClass='passenger')[0]
            spawn_car(id_car, step, path, net, dict_cars)
            id_car+=1

    traci.simulationStep() 

    for i in copy.deepcopy(list(dict_cyclists.keys())):
        dict_cyclists[i].step(step)
        if(not dict_cyclists[i].alive):
            if(dict_cyclists[i].finish_step > 0):
                dict_cyclists_arrived[i] = dict_cyclists[i]
                target = None
                if(i in structure.dict_model_input and target != None):
                    structure.list_input_to_learn.append(structure.dict_model_input[i])
                    structure.list_target.append(target)                  
                    del structure.dict_model_input[i]

                if(args.new_scenario):
                    dict_scenario["bikes"][int(dict_cyclists[i].id)]["finish_step"] = step
                    dict_scenario["bikes"][int(dict_cyclists[i].id)]["waiting_time"] = dict_cyclists[i].waiting_time
                    dict_scenario["bikes"][int(dict_cyclists[i].id)]["distance_travelled"] = dict_cyclists[i].distance_travelled
                    
            else:
                traci.vehicle.remove(i)
            del dict_cyclists[i]

    for i in copy.deepcopy(list(dict_cars.keys())):
        sumo_id = i+"_c"
        if(sumo_id in traci.simulation.getArrivedIDList()):
            dict_scenario["cars"][int(i)]["finish_step"] = step
            del dict_cars[i]

    #(step%1, step%1<=step_length)
    if(structure.open):
        structure.step(step, edges)

    print(f"\rStep {int(step)}: {len(traci.vehicle.getIDList())} cyclists in simu, {id_cyclist} cyclists spawned since start,\
{structure.num_cyclists_crossed} cyclists crossed the struct.", end="")

    step += step_length

if(args.new_scenario):
    print("WARNING: Saving scenario...")
    with open('scenario.dict', 'wb') as outfile:
        pickle.dump(dict_scenario, outfile)

traci.close()

if(len(structure.list_input_to_learn)>0):
    structure.learn()
    
print("\ndata number:", len(dict_cyclists_arrived), ",", structure.num_cyclists_crossed, "cyclits used struct, last step:", step)

if(args.config == 2):

    mean_cars_finish_step = compute_graphs_data_cyclists(dict_scenario)

    if(not os.path.exists("files/"+sub_folders)):
        os.makedirs("files/"+sub_folders)
        tab_time_diff_cars = []
        tab_x_values = []
    else:
        with open('files/'+sub_folders+'time_diff_car.tab', 'rb') as infile:
            tab_time_diff_cars = pickle.load(infile)
        with open('files/'+sub_folders+'x_values.tab', 'rb') as infile:
            tab_x_values = pickle.load(infile)
    
    if(len(tab_x_values) == 0 or tab_x_values[-1] != args.bike_poisson_lambda):
        tab_time_diff_cars.append([mean_cars_finish_step])
        tab_x_values.append(args.bike_poisson_lambda)
    else:
        tab_time_diff_cars[-1].append(mean_cars_finish_step)


    tab_mean_time_diff_cars = []
    tab_max_time_diff_cars = []
    tab_min_time_diff_cars = []

    for i in range(len(tab_x_values)):
        tab_mean_time_diff_cars.append(sum(tab_time_diff_cars[i])/len(tab_time_diff_cars[i]))
        tab_max_time_diff_cars.append(max(tab_time_diff_cars[i]))
        tab_min_time_diff_cars.append(min(tab_time_diff_cars[i]))


    plt.clf()
    plt.plot(tab_x_values, tab_mean_time_diff_cars, label="mean")
    plt.plot(tab_x_values, tab_max_time_diff_cars, label="max")
    plt.plot(tab_x_values, tab_min_time_diff_cars, label="min")
    plt.legend()
    plt.savefig("images/"+sub_folders+"evolution_cars_time_diff.png")



elif(args.config == 1 and args.learning):
    if(not os.path.exists("images/"+sub_folders)):
        os.makedirs("images/"+sub_folders)

    mean_travel_time = compute_graphs_data_cyclists_wout_struct(dict_scenario)

    if(not os.path.exists("files/"+sub_folders)):
        os.makedirs("files/"+sub_folders)
        tab_travel_time = []
        tab_x_values = []
    else:
        with open('files/'+sub_folders+'travel_time.tab', 'rb') as infile:
            tab_travel_time = pickle.load(infile)
        with open('files/'+sub_folders+'x_values.tab', 'rb') as infile:
            tab_x_values = pickle.load(infile)
    
    if(len(tab_x_values) == 0 or tab_x_values[-1] != args.car_poisson_lambda):
        tab_travel_time.append([mean_travel_time])
        tab_x_values.append(args.car_poisson_lambda)
    else:
        tab_travel_time[-1].append(mean_travel_time)



    tab_mean_travel_time = []
    tab_max_travel_time = []
    tab_min_travel_time = []

    for i in range(len(tab_x_values)):
        tab_mean_travel_time.append(sum(tab_travel_time[i])/len(tab_travel_time[i]))
        tab_max_travel_time.append(max(tab_travel_time[i]))
        tab_min_travel_time.append(min(tab_travel_time[i]))


    plt.clf()
    plt.plot(tab_x_values, tab_mean_travel_time, label="mean")
    plt.plot(tab_x_values, tab_max_travel_time, label="max")
    plt.plot(tab_x_values, tab_min_travel_time, label="min")
    plt.legend()
    plt.savefig("images/"+sub_folders+"evolution_travel_time.png") 

    with open('files/'+sub_folders+'travel_time.tab', 'wb') as outfile:
        pickle.dump(tab_travel_time, outfile)
    with open('files/'+sub_folders+'x_values.tab', 'wb') as outfile:
        pickle.dump(tab_x_values, outfile)





elif(args.config == 0 and not args.new_scenario):
    tab_all_diff_arrival_time, tab_diff_finish_step, tab_diff_waiting_time, tab_diff_distance_travelled, tab_num_type_cyclists =\
    compute_graphs_data_cyclists(structure.open, dict_cyclists_arrived, dict_scenario)
    
    if(not os.path.exists("images/"+sub_folders)):
        os.makedirs("images/"+sub_folders)

    plot_and_save_boxplot(tab_all_diff_arrival_time, "time_diff_struct", structure_was_open=structure.open, sub_folders=sub_folders)

    num_diff_finish_step = 0   
    sum_diff_finish_step = 0
    
    for i in range(len(tab_diff_finish_step)-1):
        sum_diff_finish_step += sum(tab_diff_finish_step[i])
        num_diff_finish_step += len(tab_diff_finish_step[i])

    if(num_diff_finish_step == 0):
        mean_diff_finish_step = 0
    else:
        mean_diff_finish_step = sum_diff_finish_step/num_diff_finish_step


    if(len(tab_diff_finish_step[-1]) == 0):
        mean_diff_finish_step_others = 0
    else:
        mean_diff_finish_step_others = sum(tab_diff_finish_step[-1])/len(tab_diff_finish_step[-1])
    
    print("mean finish time diff for users of struct:", mean_diff_finish_step, ", for others:",mean_diff_finish_step_others)


    if(structure.open):
        labels=["Gagnants", "Perdants", "Annulés", "Reste"]

        plot_and_save_boxplot(tab_diff_finish_step, "mean_time_diff", labels=labels, sub_folders=sub_folders)
        plot_and_save_boxplot(tab_diff_waiting_time, "mean_waiting_time", labels=labels, sub_folders=sub_folders)
        plot_and_save_boxplot(tab_diff_distance_travelled, "mean_distance_travelled", labels=labels, sub_folders=sub_folders)

        plot_and_save_bar(tab_num_type_cyclists, "cyclists_type", labels=labels, sub_folders=sub_folders)


        if(args.learning):
            
            if(not os.path.exists("files/"+sub_folders)):
                os.makedirs("files/"+sub_folders)
                tab_perc_cycl = [[], []]
                tab_time_diff = []
                tab_x_values = []
                if(use_model):
                    tab_mean_loss = []
            else:
                with open('files/'+sub_folders+'perc_cycl.tab', 'rb') as infile:
                    tab_perc_cycl = pickle.load(infile)
                with open('files/'+sub_folders+'time_diff.tab', 'rb') as infile:
                    tab_time_diff = pickle.load(infile)
                with open('files/'+sub_folders+'x_values.tab', 'rb') as infile:
                    tab_x_values = pickle.load(infile)
                if(use_model):
                    with open('files/'+sub_folders+'mean_loss.tab', 'rb') as infile:
                        tab_mean_loss = pickle.load(infile)


            if(len(tab_x_values) == 0 or tab_x_values[-1] != args.bike_poisson_lambda):
                tab_perc_cycl[0].append([structure.num_cyclists_crossed/num_cyclists*100])
                tab_perc_cycl[1].append([structure.num_cyclists_canceled/num_cyclists*100])
                tab_time_diff.append([mean_diff_finish_step])
                tab_x_values.append(args.bike_poisson_lambda)
            else:
                tab_perc_cycl[0][-1].append(structure.num_cyclists_crossed/num_cyclists*100)
                tab_perc_cycl[1][-1].append(structure.num_cyclists_canceled/num_cyclists*100)
                tab_time_diff[-1].append(mean_diff_finish_step)


            print(tab_perc_cycl[0], tab_perc_cycl[1], tab_time_diff)

            tab_mean_perc_cycl = []
            tab_max_perc_cycl = []
            tab_min_perc_cycl = []

            tab_mean_time_diff = []
            tab_max_time_diff = []
            tab_min_time_diff = []

            for i in range(len(tab_x_values)):
                tab_mean_perc_cycl.append(sum(tab_perc_cycl[0][i])/len(tab_perc_cycl[0][i]))
                tab_max_perc_cycl.append(max(tab_perc_cycl[0][i]))
                tab_min_perc_cycl.append(min(tab_perc_cycl[0][i]))

                tab_mean_time_diff.append(sum(tab_time_diff[i])/len(tab_time_diff[i]))
                tab_max_time_diff.append(max(tab_time_diff[i]))
                tab_min_time_diff.append(min(tab_time_diff[i]))


            plt.clf()
            plt.plot(tab_x_values, tab_mean_time_diff, label="mean")
            plt.plot(tab_x_values, tab_max_time_diff, label="max")
            plt.plot(tab_x_values, tab_min_time_diff, label="min")
            plt.legend()
            plt.savefig("images/"+sub_folders+"evolution_time_diff.png")

            plt.clf()
            plt.plot(tab_x_values, tab_mean_perc_cycl, label="mean")
            plt.plot(tab_x_values, tab_max_perc_cycl, label="max")
            plt.plot(tab_x_values, tab_min_perc_cycl, label="min")
            plt.legend()
            plt.savefig("images/"+sub_folders+"evolution_percentage_cycl_using_struct.png")

            with open('files/'+sub_folders+'perc_cycl.tab', 'wb') as outfile:
                pickle.dump(tab_perc_cycl, outfile)
            with open('files/'+sub_folders+'time_diff.tab', 'wb') as outfile:
                pickle.dump(tab_time_diff, outfile)
            with open('files/'+sub_folders+'x_values.tab', 'wb') as outfile:
                pickle.dump(tab_x_values, outfile)

            if(use_model and save_model and len(structure.list_loss) != 0):
                print(tab_mean_loss)
                mean_loss = sum(structure.list_loss)/len(structure.list_loss)
                tab_mean_loss.append(mean_loss)
                
                plt.clf()
                plt.plot(tab_mean_loss)
                plt.savefig("images/"+sub_folders+"evolution_mean_loss.png")

                with open('files/'+sub_folders+'mean_loss.tab', 'wb') as outfile:
                    pickle.dump(tab_mean_loss, outfile)


