import os, sys
from random import randint
import numpy as np 
import pickle
import osmnx as ox
import copy
import torch

from Cyclist import Cyclist
from Structure import Structure
from graphs import *
from Model import Model

#46 11 22 went in the first simu but was not worth


min_group_size=5

use_model = False
save_model = use_model
learning = True
batch_size = 32
hidden_size_1 = 64
hidden_size_2 = 32
lr=1e-5

step_length = 0.2

num_cyclists = 1000
max_num_cyclists_same_time = 50

if(use_model):
    sub_folders = "w_model/"
else:
    sub_folders = "wou_model/"





if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

sumoBinary = "/usr/bin/sumo-gui"
sumoCmd = [sumoBinary, "-c", "sumo_files/sumo.sumocfg", "--waiting-time-memory", '10000', '--start', '--delay', '0', '--step-length', str(step_length), '--no-warnings']


import traci
import sumolib
import traci.constants as tc


def spawn_cyclist(id, step, path, net, structure, step_length):
    max_speed = np.random.normal(15, 3)
    c = Cyclist(id, step, path, net, structure, max_speed, traci, sumolib, step_length)
    dict_cyclists[id]=c


traci.start(sumoCmd)

net = sumolib.net.readNet('sumo_files/net.net.xml')
edges = net.getEdges()


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
dict_cyclists_arrived = {}

structure = Structure("E0", "E3", edges, net, dict_cyclists, traci, dict_edges_index, model,\
min_group_size=min_group_size, batch_size=batch_size, learning=learning, lr=lr)


id = 0
step = 0

while(len(dict_cyclists) != 0 or id<=num_cyclists):
    path=None
    if(id<=num_cyclists and randint(0, 100) == 0):
        if(len(dict_cyclists)<max_num_cyclists_same_time):
            e1 = net.getEdge("E0")
            e2 = net.getEdge("E"+str(randint(4, 9)))
            path = net.getShortestPath(e1, e2, vClass='bicycle')[0]
            spawn_cyclist(str(id), step, path, net, structure, step_length)
            id+=1

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
            else:
                traci.vehicle.remove(i)
            del dict_cyclists[i]

    #(step%1, step%1<=step_length)
    if(step%1<=step_length and structure.open):
        structure.step(step, edges)

    print("\rStep {}: {} cyclists in simu, {} cyclists spawned since start."\
    .format(int(step), len(traci.vehicle.getIDList()), id), end="")

    step += step_length

if(new_scenario):
    print("WARNING: Saving scenario...")
    with open('scenario.tab', 'wb') as outfile:
        pickle.dump(tab_scenario, outfile)

traci.close()

if(len(structure.list_input_to_learn)>0):
    structure.learn()
    
print("\ndata number:", len(dict_cyclists_arrived), ",", structure.num_cyclists_crossed, "cyclits used struct, last step:", step)



if(not new_scenario):
    tab_all_diff_arrival_time, tab_diff_finish_step, tab_diff_waiting_time, tab_diff_distance_travelled, tab_num_type_cyclists =\
    compute_graphs_data(structure.open, dict_cyclists_arrived, tab_scenario)
    
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
    
    print("mean finish time diff for users of struct:", mean_diff_finish_step, ", for others:", sum(tab_diff_finish_step[-1])/len(tab_diff_finish_step[-1]))


    if(structure.open):
        labels=["Gagnants", "Perdants", "Annulés", "Reste"]

        plot_and_save_boxplot(tab_diff_finish_step, "mean_time_diff", labels=labels, sub_folders=sub_folders)
        plot_and_save_boxplot(tab_diff_waiting_time, "mean_waiting_time", labels=labels, sub_folders=sub_folders)
        plot_and_save_boxplot(tab_diff_distance_travelled, "mean_distance_travelled", labels=labels, sub_folders=sub_folders)

        plot_and_save_bar(tab_num_type_cyclists, "cyclists_type", labels=labels, sub_folders=sub_folders)


        if(learning):
            
            if(not os.path.exists("files/"+sub_folders)):
                os.makedirs("files/"+sub_folders)
                tab_num_cycl = [[], []]
                tab_time_diff = []
                if(use_model):
                    tab_mean_loss = []
            else:
                with open('files/'+sub_folders+'num_cycl.tab', 'rb') as infile:
                    tab_num_cycl = pickle.load(infile)
                with open('files/'+sub_folders+'time_diff.tab', 'rb') as infile:
                    tab_time_diff = pickle.load(infile)
                if(use_model):
                    with open('files/'+sub_folders+'mean_loss.tab', 'rb') as infile:
                        tab_mean_loss = pickle.load(infile)

            with open('files/'+sub_folders+'timeouts.dict', 'wb') as outfile:
                pickle.dump(dict_timeouts, outfile)

            with open('files/'+sub_folders+'structure_uses.dict', 'wb') as outfile:
                pickle.dump(dict_structure_uses, outfile)

            tab_num_cycl[0].append(structure.num_cyclists_crossed)
            tab_num_cycl[1].append(structure.num_cyclists_canceled)
            tab_time_diff.append(mean_diff_finish_step)

            print(tab_num_cycl[0], tab_num_cycl[1], tab_time_diff)

            plt.clf()
            plt.plot(tab_time_diff)
            plt.savefig("images/"+sub_folders+"evolution_time_diff.png")

            plt.clf()
            plt.plot(tab_num_cycl[0], label="num crossed")
            plt.plot(tab_num_cycl[1], label="num canceled")
            plt.legend()
            plt.savefig("images/"+sub_folders+"evolution_num_cycl_using_struct.png")

            with open('files/'+sub_folders+'num_cycl.tab', 'wb') as outfile:
                pickle.dump(tab_num_cycl, outfile)
            with open('files/'+sub_folders+'time_diff.tab', 'wb') as outfile:
                pickle.dump(tab_time_diff, outfile)

            if(use_model and save_model and len(structure.list_loss) != 0):
                print(tab_mean_loss)
                mean_loss = sum(structure.list_loss)/len(structure.list_loss)
                tab_mean_loss.append(mean_loss)
                
                plt.clf()
                plt.plot(tab_mean_loss)
                plt.savefig("images/"+sub_folders+"evolution_mean_loss.png")

                with open('files/'+sub_folders+'mean_loss.tab', 'wb') as outfile:
                    pickle.dump(tab_mean_loss, outfile)


