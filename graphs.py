import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
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


def plot_data(data, vehicle_type, y, file_title, sub_folders=""):
    plt.clf()
    fig = sns.lineplot(data[data["Vehicle type"]==vehicle_type], x="x_axis", y=y, hue="Method").get_figure()
    plt.title(y+" of "+vehicle_type)
    plt.savefig("images/"+sub_folders+file_title)


def cut_tab_scenarios(tab_scenarios):
    cutted_tab_scenarios = [{"bikes": {}, "cars": {}} for _ in range(24)]

    for vehicule_type in tab_scenarios:
        for vehicle_id in tab_scenarios[vehicule_type]:
            data = tab_scenarios[vehicule_type][vehicle_id]
            cutted_tab_scenarios[int(data["start_step"]//3600)][vehicule_type][vehicle_id] = data

    return cutted_tab_scenarios


def add_scenario_data_to_df(tab_scenarios, tab_scenarios_actuated, label, x_axis, columns):
    data = pd.DataFrame(columns=columns)
    for num_simu in range(len(tab_scenarios)):
        sum_travel_time_bikes, sum_waiting_time_bikes, num_bikes = compute_data(tab_scenarios[num_simu]["bikes"])
        sum_travel_time_cars, sum_waiting_time_cars, num_cars = compute_data(tab_scenarios[num_simu]["cars"])

        mean_waiting_time_bikes = sum_waiting_time_bikes/num_bikes
        mean_waiting_time_cars = sum_waiting_time_cars/num_cars

        if(len(tab_scenarios_actuated) > 0):
            sum_travel_time_cars_actuated, sum_waiting_time_cars_actuated, num_cars_actuated = compute_data(tab_scenarios_actuated[num_simu]["cars"])
            sum_travel_time_bikes_actuated, sum_waiting_time_bikes_actuated, num_bikes_actuated = compute_data(tab_scenarios_actuated[num_simu]["bikes"])
            
            mean_waiting_time_bikes_actuated = sum_waiting_time_bikes_actuated/num_bikes_actuated
            mean_waiting_time_cars_actuated = sum_waiting_time_cars_actuated/num_cars_actuated
        else:
            mean_waiting_time_bikes_actuated = mean_waiting_time_bikes
            mean_waiting_time_cars_actuated = mean_waiting_time_cars

        if(x_axis == None):
            x = num_simu
        else:
            x = x_axis

        data = pd.concat([data, pd.DataFrame([[label, "bikes", x, mean_waiting_time_bikes, mean_waiting_time_bikes-mean_waiting_time_bikes_actuated, sum_travel_time_bikes]], columns=columns)], ignore_index=True)
        data = pd.concat([data, pd.DataFrame([[label, "cars", x, mean_waiting_time_cars, mean_waiting_time_cars-mean_waiting_time_cars_actuated, sum_travel_time_cars]], columns=columns)], ignore_index=True)
        data = pd.concat([data, pd.DataFrame([[label, "both", x, mean_waiting_time_bikes+mean_waiting_time_cars,\
        (mean_waiting_time_bikes+mean_waiting_time_cars)-(mean_waiting_time_bikes_actuated+mean_waiting_time_cars_actuated), sum_travel_time_bikes+sum_travel_time_cars]], columns=columns)], ignore_index=True)
    
    return data

if __name__ == "__main__": 

    parser = argparse.ArgumentParser()

    parser.add_argument("-a", "--alpha", type=float, default=0.5)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--full-test", action="store_true")

    args = parser.parse_args()

    arguments = str(sys.argv)

    if(args.full_test):
        sub_folders = "full_test/"
        start_coeff_car_lambda = 1
        num_scenario_same_param = 5
        coeff_car_lambda = start_coeff_car_lambda
        
    elif(args.test):
        sub_folders = "test/"
    else:   
        sub_folders = "train/"
    

    labels = {}

    possible_labels = ["2DQN", "3DQN", "DQN", "PPO", "static"]

    list_tab_scenarios_actuated = []
    list_tab_scenarios = []

    columns=["Method", "Vehicle type", "x_axis", "Mean waiting time", "Difference of mean waiting time with actuated", "Sum of waiting times"]
    data = pd.DataFrame(columns=columns)

    if os.path.exists("files/"+sub_folders+"actuated_scenarios.tab"):
        with open("files/"+sub_folders+"actuated_scenarios.tab", 'rb') as infile:
            tab_scenarios_actuated = pickle.load(infile)


        if(args.test and len(tab_scenarios_actuated) == 1 or args.full_test):
            for num_scenario in range(len(tab_scenarios_actuated)):

                tab_actuated = tab_scenarios_actuated[num_scenario]

                if(args.full_test and num_scenario > 0 and num_scenario%num_scenario_same_param == 0):
                    coeff_car_lambda += 0.1

                if(args.full_test):
                    x_axis = coeff_car_lambda
                else:
                    x_axis = None

                tab_actuated = cut_tab_scenarios(tab_actuated)
                
                data = pd.concat([data, add_scenario_data_to_df(tab_actuated, tab_actuated, "actuated", x_axis, columns)], ignore_index=True)
                list_tab_scenarios_actuated.append(tab_actuated)

    if(not args.full_test):
        sub_folders += str(args.alpha)+"/"

    for root, dirs, files in os.walk("files/"+sub_folders):
        for filename in files:
            if("actuated" not in filename):
                if("losses" in filename):
                    with open("files/"+sub_folders+filename, 'rb') as infile:
                        tab_losses = pickle.load(infile)
                elif("scenarios" in filename):
                    with open("files/"+sub_folders+filename, 'rb') as infile:
                        tab_scenarios = pickle.load(infile)
                    for l in possible_labels:
                        if(l in filename):
                            labels[len(list_tab_scenarios)] = l
                            break
                
                    if(args.test and len(tab_scenarios) == 1 or args.full_test):
                        for tab in tab_scenarios:
                            list_tab_scenarios.append(cut_tab_scenarios(tab))

                #list_tab_scenarios.append(tab_scenarios[len(tab_scenarios)-len(list_tab_scenarios[0]):])

    
    i = 0
    label = labels[i]

    for num_scenario in range(len(list_tab_scenarios)):
        tab_scenarios = list_tab_scenarios[num_scenario]
        if(i in labels):
            label = labels[i]
            if(args.full_test):
                coeff_car_lambda = start_coeff_car_lambda

        if(args.full_test and num_scenario > 0 and num_scenario%num_scenario_same_param == 0):
            coeff_car_lambda += 0.1

        if(args.full_test):
            x_axis = coeff_car_lambda
        else:
            x_axis = None

        if(len(list_tab_scenarios_actuated) > 0):
            tab_scenarios_actuated = list_tab_scenarios_actuated[num_scenario]
        else:
            tab_scenarios_actuated = tab_scenarios


        data = pd.concat([data, add_scenario_data_to_df(tab_scenarios, tab_scenarios_actuated, label, x_axis, columns)], ignore_index=True)

        i+=1
    

    print(data)

    if(not os.path.exists("images/"+sub_folders)):
        os.makedirs("images/"+sub_folders)

    if(not args.test and not args.full_test):
        plot_data([tab_losses], "evolution_losses.png", "Loss Evolution",labels, ["Simulations", "Loss"], sub_folders)


    plot_data(data, "cars", "Mean waiting time", "cars_evolution_mean_waiting_time.png", sub_folders)
    plot_data(data, "bikes", "Mean waiting time", "bikes_evolution_mean_waiting_time.png", sub_folders)
    plot_data(data, "bikes" , "Difference of mean waiting time with actuated", "bikes_diff_mean_waiting_time.png", sub_folders)
    plot_data(data, "cars", "Difference of mean waiting time with actuated", "car_diff_mean_waiting_time.png", sub_folders)

    plot_data(data, "both", "Mean waiting time", "evolution_mean_waiting_time.png", sub_folders)
    plot_data(data, "both", "Difference of mean waiting time with actuated", "diff_mean_waiting_time.png", sub_folders)

    plot_data(data, "both", "Sum of waiting times", "sum_waiting_times.png", sub_folders)
