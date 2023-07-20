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


def plot_data(data, vehicle_type, x_axis_label, y, file_title, sub_folders="", estimator="mean", hue="Method", palette=None):
    plt.clf()
    fig = sns.lineplot(data, x=x_axis_label, y=y, hue=hue, estimator=estimator, palette=palette).get_figure()
    plt.title(y+" of "+vehicle_type)
    plt.savefig("images/"+sub_folders+file_title)


def cut_tab_scenarios(tab_scenarios):
    cutted_tab_scenarios = []

    for vehicule_type in tab_scenarios:
        for vehicle_id in tab_scenarios[vehicule_type]:
            data = tab_scenarios[vehicule_type][vehicle_id]
            while(int(data["start_step"]//3600)+1 > len(cutted_tab_scenarios)):
                cutted_tab_scenarios.append({"bikes": {}, "cars": {}})
            cutted_tab_scenarios[int(data["start_step"]//3600)][vehicule_type][vehicle_id] = data

    return cutted_tab_scenarios


def add_scenario_data_to_df(tab_scenarios, tab_scenarios_actuated, label, x_axis, columns):
    data = pd.DataFrame(columns=columns)
    sum_waiting_time_bikes = 0
    sum_waiting_time_cars = 0
    sum_num_bikes = 0
    sum_num_cars = 0
    for num_simu in range(len(tab_scenarios)):
        sum_travel_time_bikes, sum_waiting_time_bikes, num_bikes = compute_data(tab_scenarios[num_simu]["bikes"])
        sum_travel_time_cars, sum_waiting_time_cars, num_cars = compute_data(tab_scenarios[num_simu]["cars"])

        mean_waiting_time_bikes = sum_waiting_time_bikes/num_bikes
        mean_waiting_time_cars = sum_waiting_time_cars/num_cars
        mean_waiting_time_vehicles = (sum_waiting_time_bikes+sum_waiting_time_cars)/(num_bikes+num_cars)

        if(len(tab_scenarios_actuated) == 0):
            mean_waiting_time_bikes_actuated = mean_waiting_time_bikes
            mean_waiting_time_cars_actuated = mean_waiting_time_cars
        elif(num_simu >= len(tab_scenarios_actuated)):
            mean_waiting_time_bikes_actuated = 0
            mean_waiting_time_cars_actuated = 0
        else:
            sum_travel_time_cars_actuated, sum_waiting_time_cars_actuated, num_cars_actuated = compute_data(tab_scenarios_actuated[num_simu]["cars"])
            sum_travel_time_bikes_actuated, sum_waiting_time_bikes_actuated, num_bikes_actuated = compute_data(tab_scenarios_actuated[num_simu]["bikes"])
            
            mean_waiting_time_bikes_actuated = sum_waiting_time_bikes_actuated/num_bikes_actuated
            mean_waiting_time_cars_actuated = sum_waiting_time_cars_actuated/num_cars_actuated

        if(x_axis == None):
            x = num_simu
        else:
            x = x_axis

        data = pd.concat([data, pd.DataFrame([[label, "bikes", x, num_bikes, mean_waiting_time_bikes, mean_waiting_time_bikes-mean_waiting_time_bikes_actuated]], columns=columns)], ignore_index=True)
        data = pd.concat([data, pd.DataFrame([[label, "cars", x, num_cars, mean_waiting_time_cars, mean_waiting_time_cars-mean_waiting_time_cars_actuated]], columns=columns)], ignore_index=True)
        data = pd.concat([data, pd.DataFrame([[label, "vehicles", x, num_bikes+num_cars, mean_waiting_time_vehicles,\
        (mean_waiting_time_bikes+mean_waiting_time_cars)-(mean_waiting_time_bikes_actuated+mean_waiting_time_cars_actuated)]], columns=columns)], ignore_index=True)

        sum_waiting_time_bikes += sum_waiting_time_bikes
        sum_num_bikes += num_bikes

        sum_waiting_time_cars += sum_waiting_time_cars
        sum_num_cars += num_cars


    return data, sum_waiting_time_bikes, sum_waiting_time_cars, sum_num_bikes, sum_num_cars

if __name__ == "__main__": 

    parser = argparse.ArgumentParser()

    parser.add_argument("-a", "--alpha", type=float, default=0.5)
    parser.add_argument("-s", "--slice", type=int, default=-1)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--full-test", action="store_true")

    args = parser.parse_args()

    arguments = str(sys.argv)

    if(args.full_test):
        sub_folders = "full_test/"
        start_variable_evoluting = 1.5
        num_scenario_same_param = 5
        variable_evoluting = start_variable_evoluting
        x_axis_label = "Multiply coefficient lambda bikes"
        
    elif(args.test):
        sub_folders = "test/"
        x_axis_label = "Hours"
    else:   
        sub_folders = "train/"
        x_axis_label = "Simulations"
    

    labels = {}

    possible_labels = ["2DQN", "3DQN", "DQN", "PPO", "40s", "20s", "normal", "actuated_b"]

    list_tab_scenarios_actuated = []
    list_tab_scenarios = []


    columns=["Method", "Vehicle type", x_axis_label, "Number", "Mean waiting time", "Difference of mean waiting time with actuated"]
    data = pd.DataFrame(columns=columns)

    if(args.full_test or not args.test):
        columns_sum=["Method", "Vehicle type", x_axis_label, "Sum of waiting times", "Sum of vehicles"]
        data_sum = pd.DataFrame(columns=columns_sum)

    if os.path.exists("files/"+sub_folders+"actuated_scenarios.tab"):
        with open("files/"+sub_folders+"actuated_scenarios.tab", 'rb') as infile:
            tab_scenarios_actuated = pickle.load(infile)

        if(args.slice != -1):
            tab_scenarios_actuated = tab_scenarios_actuated[:args.slice]

        for num_scenario in range(len(tab_scenarios_actuated)):

            tab_actuated = tab_scenarios_actuated[num_scenario]

            if(args.full_test and num_scenario > 0 and num_scenario%num_scenario_same_param == 0):
                variable_evoluting -= 0.1

            if(args.full_test):
                x_axis = variable_evoluting
            elif(args.test):
                x_axis = None
            else:
                x_axis = num_scenario

            sum_travel_time_bikes, sum_waiting_time_bikes, num_bikes = compute_data(tab_actuated["bikes"])
            sum_travel_time_cars, sum_waiting_time_cars, num_cars = compute_data(tab_actuated["cars"])


            if(args.test and len(tab_scenarios_actuated) == 1 or args.full_test):
                tab_actuated = cut_tab_scenarios(tab_actuated)
            else:
                tab_actuated = [tab_actuated]

            new_data, sum_waiting_time_bikes, sum_waiting_time_cars, sum_num_bikes, sum_num_cars = add_scenario_data_to_df(tab_actuated, tab_actuated, "actuated", x_axis, columns)


            data = pd.concat([data, new_data], ignore_index=True)

            if(args.full_test or not args.test):
                data_sum = pd.concat([data_sum, pd.DataFrame([["actuated", "vehicles", x_axis, sum_waiting_time_bikes+sum_waiting_time_cars, sum_num_bikes+sum_num_cars],\
            ["actuated", "bikes", x_axis, sum_waiting_time_bikes, sum_num_bikes],\
            ["actuated", "cars", x_axis, sum_waiting_time_cars, sum_num_cars]], columns=columns_sum)])
            
            list_tab_scenarios_actuated.append(tab_actuated)


    for root, dirs, files in os.walk("files/"+sub_folders):
        for filename in files:
            if("actuated_b" in filename or "actuated" not in filename):
                if("losses" in filename):
                    with open("files/"+sub_folders+filename, 'rb') as infile:
                        tab_losses = pickle.load(infile)
                elif("scenarios" in filename):
                    with open("files/"+sub_folders+filename, 'rb') as infile:
                        tab_scenarios = pickle.load(infile)
                    if(args.slice != -1):
                        tab_scenarios = tab_scenarios[:args.slice]
                    for l in possible_labels:
                        if(l in filename):
                            labels[len(list_tab_scenarios)] = l
                            break

                    for tab in tab_scenarios:
                        if(args.test and len(tab_scenarios) == 1 or args.full_test):
                            tab = cut_tab_scenarios(tab)   
                        else:
                            tab = [tab]
                        list_tab_scenarios.append(tab)
                #list_tab_scenarios.append(tab_scenarios[len(tab_scenarios)-len(list_tab_scenarios[0]):])

    
    i = 0
    if(len(labels) == 0):
        labels[0] = "actuated"
        list_tab_scenarios = list_tab_scenarios_actuated
    label = labels[i]

    for num_scenario in range(len(list_tab_scenarios)):
        tab_scenarios = list_tab_scenarios[num_scenario]
        if(i in labels):
            label = labels[i]
            if(args.full_test):
                variable_evoluting = start_variable_evoluting

        if(args.full_test and num_scenario > 0 and num_scenario%num_scenario_same_param == 0):
            variable_evoluting -= 0.1

        if(args.full_test):
            x_axis = variable_evoluting
        elif(args.test):
            x_axis = None
        else:
            x_axis = num_scenario

        if(len(list_tab_scenarios_actuated) > 0):
            if(len(labels) == 1):
                num_scenario_actuated = num_scenario
            else:
                num_scenario_actuated = num_scenario%list(labels.keys())[1]
            tab_scenarios_actuated = list_tab_scenarios_actuated[num_scenario_actuated]
        else:
            tab_scenarios_actuated = tab_scenarios

        new_data, sum_waiting_time_bikes, sum_waiting_time_cars, sum_num_bikes, sum_num_cars = add_scenario_data_to_df(tab_scenarios, tab_scenarios_actuated, label, x_axis, columns)
        data = pd.concat([data, new_data], ignore_index=True)

        if(args.full_test or not args.test):
            data_sum = pd.concat([data_sum, pd.DataFrame([[label, "vehicles", x_axis, sum_waiting_time_bikes+sum_waiting_time_cars, sum_num_bikes+sum_num_cars],\
            [label, "bikes", x_axis, sum_waiting_time_bikes, sum_num_bikes],\
            [label, "cars", x_axis, sum_waiting_time_cars, sum_num_cars]], columns=columns_sum)])

        i+=1
    
   

    if(not os.path.exists("images/"+sub_folders)):
        os.makedirs("images/"+sub_folders)

    if(args.test or args.full_test):
        plot_data(data[data["Vehicle type"]=="bikes"], "bikes", x_axis_label, "Difference of mean waiting time with actuated", "bikes_diff_mean_waiting_time.png", sub_folders)
        plot_data(data[data["Vehicle type"]=="cars"], "cars", x_axis_label, "Difference of mean waiting time with actuated", "car_diff_mean_waiting_time.png", sub_folders)
        plot_data(data[data["Vehicle type"]=="vehicles"], "vehicles", x_axis_label, "Difference of mean waiting time with actuated", "diff_mean_waiting_time.png", sub_folders)
        if(args.full_test):
            plot_data(data[(data["Vehicle type"]=="cars")|(data["Vehicle type"]=="bikes")], "vehicles", x_axis_label, "Number", "evolution_number_vehicules.png", sub_folders,\
            hue="Vehicle type", palette=["green", "red"], estimator="sum")
        else:
            plot_data(data[(data["Vehicle type"]=="cars")|(data["Vehicle type"]=="bikes")], "vehicles", x_axis_label, "Number", "evolution_number_vehicules.png", sub_folders,\
            hue="Vehicle type", palette=["green", "red"])
    else:
        plt.clf()
        fig = sns.lineplot(tab_losses).get_figure()
        plt.title("Loss Evolution")
        plt.savefig("images/"+sub_folders+"evolution_losses.png")

    
    plot_data(data[data["Vehicle type"]=="cars"], "cars", x_axis_label, "Mean waiting time", "cars_evolution_mean_waiting_time.png", sub_folders)
    plot_data(data[data["Vehicle type"]=="bikes"], "bikes", x_axis_label, "Mean waiting time", "bikes_evolution_mean_waiting_time.png", sub_folders)
    plot_data(data[data["Vehicle type"]=="vehicles"], "vehicles", x_axis_label, "Mean waiting time", "evolution_mean_waiting_time.png", sub_folders)

    if(args.full_test or not args.test):
        plot_data(data_sum, "vehicles", x_axis_label, "Sum of waiting times", "sum_waiting_times.png", sub_folders)
