import matplotlib.pyplot as plt
import pickle
import os
import sys
import argparse

def compute_data(dict_scenario):
    if(len(dict_scenario)>0):
        tab_travel_time = []
        tab_waiting_time = []
        for v in dict_scenario:
            if("finish_step" in dict_scenario[v]):
                tab_travel_time.append(dict_scenario[v]["finish_step"]-dict_scenario[v]["start_step"])
                tab_waiting_time.append(dict_scenario[v]["waiting_time"])
        if(len(tab_travel_time)>0):
            return sum(tab_travel_time), sum(tab_waiting_time), len(tab_travel_time)
        else:
            return 0, 0, 1

    return 0, 0, 1



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
    plt.legend(loc='upper right')
    plt.savefig("images/"+sub_folders+file_title)


def cut_tab_scenarios(tab_scenarios):
    cutted_tab_scenarios = [{"bikes": {}, "cars": {}} for _ in range(24)]

    for vehicule_type in tab_scenarios[0]:
        for vehicle_id in tab_scenarios[0][vehicule_type]:
            data = tab_scenarios[0][vehicule_type][vehicle_id]
            cutted_tab_scenarios[int(data["start_step"]//3600)][vehicule_type][vehicle_id] = data

    return cutted_tab_scenarios


if __name__ == "__main__": 

    parser = argparse.ArgumentParser()

    parser.add_argument("-a", "--alpha", type=float, default=0.5)
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()

    arguments = str(sys.argv)

    if(args.test):
        sub_folders = "test/"
    else:   
        sub_folders = "train/"
    
    cars_waiting_time = []
    cars_diff_waiting_time = []

    bikes_waiting_time = []
    bikes_diff_waiting_time = []

    cars_travel_time = []
    bikes_travel_time = []
    tot_waiting_time = []
    estimated_reward = []
    tot_diff_waiting_time = []

    labels = []

    possible_labels = ["2DQN", "3DQN", "DQN", "PPO", "static"]

    list_tab_scenarios = []

    if os.path.exists("files/"+sub_folders+"actuated_scenarios.tab"):
        with open("files/"+sub_folders+"actuated_scenarios.tab", 'rb') as infile:
            tab_scenarios_actuated = pickle.load(infile)

        tab_scenarios_actuated = tab_scenarios_actuated[:23]

        if(args.test and len(tab_scenarios_actuated) == 1):
            tab_scenarios_actuated = cut_tab_scenarios(tab_scenarios_actuated)


        list_tab_scenarios.append(tab_scenarios_actuated)
        labels.append("actuated")


    sub_folders += str(args.alpha)+"/"

    for root, dirs, files in os.walk("files/"+sub_folders):
        for filename in files:
            if("losses" in filename):
                with open("files/"+sub_folders+filename, 'rb') as infile:
                    tab_losses = pickle.load(infile)
            elif("scenarios" in filename):
                with open("files/"+sub_folders+filename, 'rb') as infile:
                    tab_scenarios = pickle.load(infile)
            
                if(args.test and len(tab_scenarios) == 1):
                    tab_scenarios = cut_tab_scenarios(tab_scenarios)
                        

                for l in possible_labels:
                    if(l in filename):
                        labels.append(l)
                        break

                list_tab_scenarios.append(tab_scenarios)
                #list_tab_scenarios.append(tab_scenarios[len(tab_scenarios)-len(list_tab_scenarios[0]):])
    


    for tab_scenarios in list_tab_scenarios:
        
        tab_mean_waiting_time = [[], []]
        tab_waiting_time = [[],[]]
        tab_diff_wt = [[], []]
        tab_reward = []
        tab_wt = []
        tab_diff_wt_tot = []
        for num_simu in range(len(tab_scenarios)):
            sum_travel_time_bikes, sum_waiting_time_bikes, num_bikes = compute_data(tab_scenarios[num_simu]["bikes"])
            sum_travel_time_bikes_actuated, sum_waiting_time_bikes_actuated, num_bikes_actuated = compute_data(list_tab_scenarios[0][num_simu]["bikes"])

            mean_waiting_time_bikes_actuated = sum_waiting_time_bikes_actuated/num_bikes_actuated
            mean_waiting_time_bikes = sum_waiting_time_bikes/num_bikes

            tab_mean_waiting_time[0].append(mean_waiting_time_bikes)
            tab_diff_wt[0].append(mean_waiting_time_bikes-mean_waiting_time_bikes_actuated)

            sum_travel_time_cars, sum_waiting_time_cars, num_cars = compute_data(tab_scenarios[num_simu]["cars"])
            sum_travel_time_cars_actuated, sum_waiting_time_cars_actuated, num_cars_actuated = compute_data(list_tab_scenarios[0][num_simu]["cars"])

            mean_waiting_time_cars = sum_waiting_time_cars/num_cars
            mean_waiting_time_cars_actuated = sum_waiting_time_cars_actuated/num_cars_actuated

            tab_mean_waiting_time[1].append(mean_waiting_time_cars)
            tab_diff_wt[1].append(mean_waiting_time_cars-mean_waiting_time_cars_actuated)

            tab_reward.append(-sum_waiting_time_cars-sum_waiting_time_bikes)
            tab_wt.append(mean_waiting_time_cars+mean_waiting_time_bikes)
            tab_diff_wt_tot.append(tab_diff_wt[0][-1]+tab_diff_wt[1][-1])

        bikes_waiting_time.append(tab_mean_waiting_time[0])
        bikes_diff_waiting_time.append(tab_diff_wt[0])
        cars_waiting_time.append(tab_mean_waiting_time[1])
        cars_diff_waiting_time.append(tab_diff_wt[1])
        tot_waiting_time.append(tab_wt)
        estimated_reward.append(tab_reward)
        tot_diff_waiting_time.append(tab_diff_wt_tot)

    if(not os.path.exists("images/"+sub_folders)):
        os.makedirs("images/"+sub_folders)

    if(not args.test):
        plot_data([tab_losses], "evolution_losses.png", "Loss Evolution",labels, ["Simulations", "Loss"], sub_folders)

    plot_data(cars_waiting_time, "cars_evolution_mean_waiting_time.png", "Cars mean waiting time",labels, ["Simulations", "Waiting Time"], sub_folders)
    plot_data(bikes_waiting_time, "bikes_evolution_mean_waiting_time.png", "Bikes mean waiting time",labels, ["Simulations", "Waiting Time"], sub_folders)
    plot_data(bikes_diff_waiting_time, "bikes_diff_mean_waiting_time.png", "Difference of mean waiting time with actuated for bikes", labels, ["Simulations", "Estimated reward"], sub_folders)
    plot_data(cars_diff_waiting_time, "car_diff_mean_waiting_time.png", "Difference of mean waiting time with actuated for cars", labels, ["Simulations", "Estimated reward"], sub_folders)

    plot_data(tot_waiting_time, "evolution_mean_waiting_time.png", "Total mean waiting time", labels, ["Simulations", "Waiting Time"], sub_folders)
    plot_data(estimated_reward, "evolution_estimated_reward.png", "Estimated reward", labels, ["Simulations", "Estimated reward"], sub_folders)
    plot_data(tot_diff_waiting_time, "diff_mean_waiting_time.png", "Difference of mean waiting time with actuated", labels, ["Simulations", "Estimated reward"], sub_folders)