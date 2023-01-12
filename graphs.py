import matplotlib.pyplot as plt
import pickle
import os
from pprint import pprint

def compute_graphs_data_cyclists(dict_scenario):
    if(len(dict_scenario["bikes"])>0):
        tab_travel_time = [b["finish_step"]-b["start_step"] for b in dict_scenario["bikes"]]
        tab_speed = [b["distance_travelled"]/(b["finish_step"]-b["start_step"]) for b in dict_scenario["bikes"]]
        tab_distance_travelled = [b["distance_travelled"] for b in dict_scenario["bikes"]]
        
        return sum(tab_travel_time)/len(tab_travel_time), sum(tab_speed)/len(tab_speed) 
    else:
        return 0, 0



def compute_graphs_data_cars(dict_scenario):
    if(len(dict_scenario["cars"])>0):
        tab_travel_time = [c["finish_step"]-c["start_step"] for c in dict_scenario["cars"]]
        tab_speed = [b["distance_travelled"]/(b["finish_step"]-b["start_step"]) for b in dict_scenario["cars"]]
        return sum(tab_travel_time)/len(tab_travel_time), sum(tab_speed)/len(tab_speed) 
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


def plot_and_save_line(data, file_title, labels=None, sub_folders=""):
    plt.clf()
    fig1, ax1 = plt.subplots()
    ax1.set_title('')
    ax1.bar(range(len(data)), data, tick_label=labels)
    plt.savefig("images/"+sub_folders+file_title+".png")


if __name__ == "__main__": 

    config = 3
    variable_fixed = 0.2
    drl = True

    if(drl):
        sub_folders = "w_model/"
        sub_folders+="config_"+str(config)+"/"+str(variable_fixed)+"/"
        for filename in os.listdir("./files/" + sub_folders):
            if("scenarios" in filename):
                with open("files/"+sub_folders+filename, 'rb') as infile:
                    d_scenarios = pickle.load(infile)
                dict_graphs = {}
                for key in d_scenarios:
                    vehicle_type_index = 0
                    tab_mean_waiting_time = [[], []]
                    tab_mean_travel_time = [[], []]
                    tab_cumulative_reward = [0]
                    tab_waiting_time = [[],[]]
                    for vehicle_type in d_scenarios[key][0]:
                        dict_graphs[vehicle_type] = [[],[]]
                        next_step_hour = 0
                        for vehicle in d_scenarios[key][0][vehicle_type]:
                            if(vehicle["start_step"]>=next_step_hour):
                                if(len(dict_graphs[vehicle_type][0])>0):
                                    tab_mean_travel_time[vehicle_type_index].append(sum(dict_graphs[vehicle_type][0][-1])/len(dict_graphs[vehicle_type][0][-1]))
                                    tab_mean_waiting_time[vehicle_type_index].append(sum(dict_graphs[vehicle_type][1][-1])/len(dict_graphs[vehicle_type][1][-1]))
                                    tab_waiting_time[vehicle_type_index].append(sum(dict_graphs[vehicle_type][1][-1]))
                                dict_graphs[vehicle_type][0].append([])
                                dict_graphs[vehicle_type][1].append([])
                                next_step_hour += 3600
                            dict_graphs[vehicle_type][0][-1].append(vehicle["finish_step"]-vehicle["start_step"])
                            dict_graphs[vehicle_type][1][-1].append(vehicle["waiting_time"])
                        vehicle_type_index+=1

                    if(not os.path.exists("images/"+sub_folders)):
                        os.makedirs("images/"+sub_folders)

                    tab_x = range(len(tab_mean_waiting_time[0]))

                    tab_reward = []

                    for i in range(len(tab_waiting_time[0])):
                        tab_reward.append(0.5*tab_waiting_time[0][i]+0.5*tab_waiting_time[1][i])

                    plt.clf()
                    plt.plot(tab_x, tab_mean_travel_time[0], label=list(d_scenarios[key][0].keys())[0])
                    plt.plot(tab_x, tab_mean_travel_time[1], label=list(d_scenarios[key][0].keys())[1])
                    plt.xlabel("Hour")
                    plt.ylabel("Travel Time")
                    plt.legend()
                    plt.savefig("images/"+sub_folders+"evolution_mean_time_travel.png")

                    plt.clf()
                    plt.plot(tab_x, tab_mean_waiting_time[0], label=list(d_scenarios[key][0].keys())[0])
                    plt.plot(tab_x, tab_mean_waiting_time[1], label=list(d_scenarios[key][0].keys())[1])
                    plt.xlabel("Hour")
                    plt.ylabel("Waiting Time")
                    plt.legend()
                    plt.savefig("images/"+sub_folders+"evolution_mean_waiting_time.png")

                    plt.clf()
                    plt.plot(tab_x, tab_reward)
                    plt.xlabel("Hour")
                    plt.ylabel("Waiting time")
                    plt.savefig("images/"+sub_folders+"evolution_waiting_time.png")

                    '''plt.clf()
                    plt.plot(tab_x, tab_cumulative_reward)
                    plt.xlabel("Hour")
                    plt.ylabel("Cumulative Waiting Time")
                    plt.savefig("images/"+sub_folders+"cumulative_waiting_time.png")'''
                    



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
                d_scenarios = pickle.load(infile)

            print("Computing graphs for files/"+sub_folders+filename)
            
            dict_graphs = {"x_mean": [], "mean_cars_t_t": [[], [], []], "mean_bikes_t_t": [[],[],[]], "flow": [], "flow_on_speed": []}

            tab_mean_t_t = [[],[]]

            tab_flows = [[],[]]
            tab_flows_on_speed = [[],[]]

            for lam in d_scenarios:
                dict_graphs["x_mean"].append(lam)
                for i in range(len(d_scenarios[lam])):
                    mean_cars_t_t, mean_cars_speed = compute_graphs_data_cars(d_scenarios[lam][i])
                    mean_bikes_t_t, mean_bikes_speed = compute_graphs_data_cyclists(d_scenarios[lam][i])

                    tab_mean_t_t[0].append(mean_cars_t_t)
                    tab_mean_t_t[1].append(mean_bikes_t_t)
                    
                    tab_flows[0].append(len(d_scenarios[lam][i]["cars"])/1000)
                    tab_flows[1].append(len(d_scenarios[lam][i]["bikes"])/1000)

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