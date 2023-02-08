import matplotlib.pyplot as plt
import pickle
import os
from pprint import pprint

def compute_data(dict_scenario):
    if(len(dict_scenario)>0):
        tab_travel_time = [dict_scenario[b]["finish_step"]-dict_scenario[b]["start_step"] for b in dict_scenario]
        tab_waiting_time = [dict_scenario[b]["waiting_time"] for b in dict_scenario]
        
        return sum(tab_travel_time)/len(tab_travel_time), sum(tab_waiting_time)/len(tab_waiting_time)
    else:
        return 0, 0



def plot_and_save_boxplot(data, file_title, labels=None, structure_was_open=None, sub_folders=""):
    plt.clf()
    fig1, ax1 = plt.subplots()
    ax1.set_title('')
    ax1.boxplot(data, labels=labels)
    file_path = "images/"+sub_folders+file_title
    if(structure_was_open != None):
        if(structure_was_open):
            file_path+="_open"
        else:
            file_path+="_close"
    file_path+=".png"
    plt.savefig(file_path)


def plot_and_save_bar(data, file_title, labels=None, sub_folders=""):
    plt.clf()
    fig1, ax1 = plt.subplots()
    ax1.set_title('')
    ax1.bar(range(len(data)), data, tick_label=labels)
    plt.savefig("images/"+sub_folders+file_title+".png")


def plot_data(data, file_title, title, labels, axis_labels, sub_folders=""):
    plt.clf()
    tab_x = range(len(data[0]))
    for i in range(len(labels)):
        plt.plot(tab_x, data[i], label=labels[i])
    plt.xlabel(axis_labels[0])
    plt.ylabel(axis_labels[1])
    plt.title(title)
    plt.legend()
    plt.savefig("images/"+sub_folders+file_title)


if __name__ == "__main__": 

    config = 3
    variable_fixed = 0.2
    drl = True

    if(drl):
        cars_waiting_time = []
        bikes_waiting_time = []
        cars_travel_time = []
        bikes_travel_time = []
        tot_waiting_time = []

        sub_folders = "w_model/"
        sub_folders+="config_"+str(config)+"/"+str(variable_fixed)+"/"
        for root, dirs, files in os.walk("files/"+sub_folders):
            for filename in files:
                if("scenarios" in filename):
                    with open("files/"+sub_folders+filename, 'rb') as infile:
                        tab_scenarios = pickle.load(infile)

                    tab_mean_waiting_time = [[], []]
                    tab_mean_travel_time = [[], []]
                    tab_waiting_time = [[],[]]
                    for num_simu in range(len(tab_scenarios)):
                        vehicle_type_index = 0

                        for vehicle_type in tab_scenarios[num_simu]:
                            tab_graphs_temp = [[], []]
                            for v in tab_scenarios[num_simu][vehicle_type]:
                                vehicle = tab_scenarios[num_simu][vehicle_type][v]
                                tab_graphs_temp[0].append(vehicle["finish_step"]-vehicle["start_step"])
                                tab_graphs_temp[1].append(vehicle["waiting_time"])
                            tab_mean_travel_time[vehicle_type_index].append(sum(tab_graphs_temp[0])/len(tab_graphs_temp[0]))
                            tab_mean_waiting_time[vehicle_type_index].append(sum(tab_graphs_temp[1])/len(tab_graphs_temp[1]))
                            tab_waiting_time[vehicle_type_index].append(sum(tab_graphs_temp[1]))                           
                            vehicle_type_index+=1

                        if(not os.path.exists("images/"+sub_folders)):
                            os.makedirs("images/"+sub_folders)

                        tab_reward = []

                        for i in range(len(tab_waiting_time[0])):
                            tab_reward.append(0.5*tab_waiting_time[0][i]+0.5*tab_waiting_time[1][i])

                    cars_waiting_time.append(tab_mean_waiting_time[0][1:])
                    bikes_waiting_time.append(tab_mean_waiting_time[1][1:])
                    cars_travel_time.append(tab_mean_travel_time[0][1:])
                    bikes_travel_time.append(tab_mean_travel_time[1][1:])
                    tot_waiting_time.append(tab_reward[1:])

            plot_data(cars_travel_time, "cars_evolution_mean_time_travel.png", "Cars mean travel time", ["DQN"], ["Hours", "Travel Time"], sub_folders)
            plot_data(bikes_travel_time, "bikes_evolution_mean_time_travel.png", "Bikes mean travel time", ["DQN"], ["Hours", "Travel Time"], sub_folders)

            plot_data(cars_waiting_time, "cars_evolution_mean_waiting_time.png", "Cars mean waiting time",["DQN"], ["Hours", "Waiting Time"], sub_folders)
            plot_data(bikes_waiting_time, "bikes_evolution_mean_waiting_time.png", "Bikes mean waiting time",["DQN"], ["Hours", "Waiting Time"], sub_folders)

            plot_data(tot_waiting_time, "evolution_mean_waiting_time.png", "Total mean waiting time", ["DQN"], ["Hours", "Waiting Time"], sub_folders)



    else:
        sub_folders = "wou_model/"
        sub_folders+="config_"+str(config)+"/"+str(variable_fixed)+"/"
        for filename in os.listdir("./files/" + sub_folders):
            if("car" in filename):
                evoluting = "cars"
                compute_flows = True
                index_flow = 0
            elif("bike" in filename):
                evoluting = "bikes"
                compute_flows = True
                index_flow = 1
            else:
                evoluting = "group_size"
                compute_flows = False

            name_complement = ""

            if("struct_open" in filename):
                name_complement = "_struct_open"

            with open("files/"+sub_folders+filename, 'rb') as infile:
                tab_scenarios = pickle.load(infile)

            print("Computing graphs for files/"+sub_folders+filename)
            
            dict_graphs = {"x_mean": [], "mean_cars_t_t": [[], [], []], "mean_bikes_t_t": [[],[],[]], "flow": [], "flow_on_speed": []}

            tab_mean_t_t = [[],[]]

            tab_flows = [[],[]]
            tab_flows_on_speed = [[],[]]

            for lam in tab_scenarios:
                dict_graphs["x_mean"].append(lam)
                for i in range(len(tab_scenarios[lam])):
                    mean_cars_t_t, mean_cars_speed = compute_graphs_data_cars(tab_scenarios[lam][i])
                    mean_bikes_t_t, mean_bikes_speed = compute_graphs_data_cyclists(tab_scenarios[lam][i])

                    tab_mean_t_t[0].append(mean_cars_t_t)
                    tab_mean_t_t[1].append(mean_bikes_t_t)
                    
                    tab_flows[0].append(len(tab_scenarios[lam][i]["cars"])/1000)
                    tab_flows[1].append(len(tab_scenarios[lam][i]["bikes"])/1000)

                    tab_flows_on_speed[0].append(tab_flows[0][-1]/mean_cars_speed)
                    tab_flows_on_speed[1].append(tab_flows[1][-1]/mean_bikes_speed)

                dict_graphs["mean_cars_t_t"][0].append(sum(tab_mean_t_t[0])/len(tab_mean_t_t[0]))
                dict_graphs["mean_cars_t_t"][1].append(min(tab_mean_t_t[0]))
                dict_graphs["mean_cars_t_t"][2].append(max(tab_mean_t_t[0]))

                dict_graphs["mean_bikes_t_t"][0].append(sum(tab_mean_t_t[1])/len(tab_mean_t_t[1]))
                dict_graphs["mean_bikes_t_t"][1].append(min(tab_mean_t_t[1]))
                dict_graphs["mean_bikes_t_t"][2].append(max(tab_mean_t_t[1]))

                if(compute_flows):
                    dict_graphs["flow"].append(sum(tab_flows[index_flow])/len(tab_flows[1]))
                    dict_graphs["flow_on_speed"].append(sum(tab_flows_on_speed[index_flow])/len(tab_flows_on_speed[index_flow]))

                tab_mean_t_t = [[],[]]

                tab_flows = [[],[]]
                tab_flows_on_speed = [[],[]]

            if(not os.path.exists("images/"+sub_folders)):
                os.makedirs("images/"+sub_folders)

            plt.clf()
            plt.plot(dict_graphs["x_mean"], dict_graphs["mean_cars_t_t"][0], label="cars")
            plt.fill_between(dict_graphs["x_mean"], dict_graphs["mean_cars_t_t"][1], dict_graphs["mean_cars_t_t"][2], alpha=0.2)
            plt.plot(dict_graphs["x_mean"], dict_graphs["mean_bikes_t_t"][0], label="bikes")
            plt.fill_between(dict_graphs["x_mean"], dict_graphs["mean_bikes_t_t"][1], dict_graphs["mean_bikes_t_t"][2], alpha=0.2, color="orange")
            plt.legend()
            plt.ylabel("Travel Time")

            plt.xlabel("Flow of "+evoluting+" (per step)")
            name_fig = "images/"+sub_folders+evoluting+name_complement+"_evolution_travel_time"

            plt.savefig(name_fig)

            if(compute_flows):
                plt.clf()
                plt.plot(dict_graphs["flow_on_speed"], dict_graphs["flow"], label="bikes")
                plt.xlabel("Flow divided by speed of "+evoluting)
                plt.ylabel("Flow of "+evoluting+" (per step)")
                name_fig = "images/"+sub_folders+evoluting+name_complement+"_evolution_flow_on_speed"
                plt.savefig(name_fig)