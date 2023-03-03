import matplotlib.pyplot as plt
import pickle
import os
import sys

def compute_data(dict_scenario):
    if(len(dict_scenario)>0):
        tab_travel_time = []
        tab_waiting_time = []
        for v in dict_scenario:
            if("finish_step" in dict_scenario[v]):
                tab_travel_time.append(dict_scenario[v]["finish_step"]-dict_scenario[v]["start_step"])
                tab_waiting_time.append(dict_scenario[v]["waiting_time"])
        if(len(tab_travel_time)>0):
            return sum(tab_travel_time)/len(tab_travel_time), sum(tab_waiting_time)/len(tab_waiting_time)

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
    arguments = str(sys.argv)

    if("--test" in arguments):
        test = True
        sub_folders = "test/"
    else:   
        test = False
        sub_folders = "train/"
    
    cars_waiting_time = []
    bikes_waiting_time = []
    cars_travel_time = []
    bikes_travel_time = []
    tot_waiting_time = []
    estimated_reward = []

    labels = []

    possible_labels = ["actuated", "2DQN", "3DQN", "DQN", "PPO"]

    sub_folders+="config_"+str(config)+"/"
    for root, dirs, files in os.walk("files/"+sub_folders):
        for filename in files:
            if("scenarios" in filename):
                with open("files/"+sub_folders+filename, 'rb') as infile:
                    tab_scenarios = pickle.load(infile)

                if(test and len(tab_scenarios) == 1):
                    cutted_tab_scenarios = [{"bikes": {}, "cars": {}} for _ in range(24)]

                    for vehicule_type in tab_scenarios[0]:
                        for vehicle_id in tab_scenarios[0][vehicule_type]:
                            data = tab_scenarios[0][vehicule_type][vehicle_id]
                            cutted_tab_scenarios[int(data["start_step"]//1800)][vehicule_type][vehicle_id] = data

                    tab_scenarios = cutted_tab_scenarios
                        

                for l in possible_labels:
                    if(l in filename):
                        labels.append(l)
                        break
                '''if("DQN" in labels[-1]):
                    labels.pop()
                    continue'''
                tab_mean_waiting_time = [[], []]
                tab_mean_travel_time = [[], []]
                tab_waiting_time = [[],[]]
                for num_simu in range(len(tab_scenarios)):
                    vehicle_type_index = 0

                    bikes_data = compute_data(tab_scenarios[num_simu]["bikes"])
                    cars_data = compute_data(tab_scenarios[num_simu]["cars"])

                    for vehicle_type in tab_scenarios[num_simu]:
                        tab_graphs_temp = [[], []]
                        for v in tab_scenarios[num_simu][vehicle_type]:
                            vehicle = tab_scenarios[num_simu][vehicle_type][v]
                            if("finish_step" in vehicle):   
                                tab_graphs_temp[0].append(vehicle["finish_step"]-vehicle["start_step"])
                                tab_graphs_temp[1].append(vehicle["waiting_time"])
                        if(len(tab_graphs_temp[0]) == 0):
                            tab_mean_travel_time[vehicle_type_index].append(0)
                        else:                          
                            tab_mean_travel_time[vehicle_type_index].append(sum(tab_graphs_temp[0])/len(tab_graphs_temp[0]))
                        if(len(tab_graphs_temp[1]) == 0):
                            tab_mean_waiting_time[vehicle_type_index].append(0)
                            tab_waiting_time[vehicle_type_index].append(0)                           
                        else:
                            tab_mean_waiting_time[vehicle_type_index].append(sum(tab_graphs_temp[1])/len(tab_graphs_temp[1]))
                            tab_waiting_time[vehicle_type_index].append(sum(tab_graphs_temp[1]))                           
                        vehicle_type_index+=1

                    if(not os.path.exists("images/"+sub_folders)):
                        os.makedirs("images/"+sub_folders)

                    tab_reward = []
                    for i in range(len(tab_waiting_time[0])):
                        tab_reward.append(0.5*tab_waiting_time[0][i]+0.5*tab_waiting_time[1][i])

                    tab_wt = []
                    for i in range(len(tab_waiting_time[0])):
                        tab_wt.append(tab_waiting_time[0][i]+tab_waiting_time[1][i])

                cars_waiting_time.append(tab_mean_waiting_time[0])
                bikes_waiting_time.append(tab_mean_waiting_time[1])
                cars_travel_time.append(tab_mean_travel_time[0])
                bikes_travel_time.append(tab_mean_travel_time[1])
                tot_waiting_time.append(tab_wt)
                estimated_reward.append(tab_reward)


        plot_data(cars_travel_time, "cars_evolution_mean_time_travel.png", "Cars mean travel time", labels, ["Simulations", "Travel Time"], sub_folders)
        plot_data(bikes_travel_time, "bikes_evolution_mean_time_travel.png", "Bikes mean travel time", labels, ["Simulations", "Travel Time"], sub_folders)

        plot_data(cars_waiting_time, "cars_evolution_mean_waiting_time.png", "Cars mean waiting time",labels, ["Simulations", "Waiting Time"], sub_folders)
        plot_data(bikes_waiting_time, "bikes_evolution_mean_waiting_time.png", "Bikes mean waiting time",labels, ["Simulations", "Waiting Time"], sub_folders)

        plot_data(tot_waiting_time, "evolution_mean_waiting_time.png", "Total mean waiting time", labels, ["Simulations", "Waiting Time"], sub_folders)
        plot_data(estimated_reward, "evolution_estimated_reward.png", "Estimated reward", labels, ["Simulations", "Estimated reward"], sub_folders)