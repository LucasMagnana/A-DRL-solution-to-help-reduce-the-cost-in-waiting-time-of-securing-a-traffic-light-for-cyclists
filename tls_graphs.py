import numpy as np
import matplotlib.pyplot as plt
import pickle 
import argparse
import os


if __name__ == "__main__": 

    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--method", type=str, default="actuated")

    args = parser.parse_args()

    sub_folders = "test/"

    with open("files/"+sub_folders+args.method+"_tls_phases.tab", 'rb') as infile:
            phases_history = pickle.load(infile)


    dict_phases = {
        "NS vélos" : [],
        "NS voitures" : [],
        "EO vélos" : [],
        "EO voitures" : []
    }

    bar_labels = dict_phases.keys()
    dict_bar_phases = {
        "orange" : [0, 0, 0, 0],
        "vert" : [0, 0, 0, 0]
    }


    for p in phases_history:
        for key in dict_phases:
            dict_phases[key].append(0)

        if(p == 0):
            dict_phases["NS vélos"][-1] = 1
            dict_bar_phases["vert"][0] += 1
        elif(p == 1):
            dict_phases["NS vélos"][-1] = 2
            dict_bar_phases["orange"][0] += 1
        elif(p == 2):
            dict_phases["NS voitures"][-1] = 1
            dict_bar_phases["vert"][1] += 1
        elif(p == 3):
            dict_phases["NS voitures"][-1] = 2
            dict_bar_phases["orange"][1] += 1
        elif(p == 4):
            dict_phases["EO vélos"][-1] = 1
            dict_bar_phases["vert"][2] += 1
        elif(p == 5):
            dict_phases["EO vélos"][-1] = 2
            dict_bar_phases["orange"][2] += 1
        elif(p == 6):
            dict_phases["EO voitures"][-1] = 1
            dict_bar_phases["vert"][3] += 1
        elif(p == 7):
            dict_phases["EO voitures"][-1] = 2
            dict_bar_phases["orange"][3] += 1



    x = np.arange(len(bar_labels))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for color, values in dict_bar_phases.items():
        if(color == "vert"):
            c = "green"
        elif(color == "orange"):
            c = "orange"
        offset = width * multiplier
        for i in range(len(values)):
            if(values[i] == 21521):
                values[i] -= 1
            elif(values[i] == 6195):
                values[i] += 1
        rects = ax.bar(x + offset, values, width, label=color, color=c)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("Nombre d'étapes")
    ax.set_title("Nombre d'étapes passées en un type de phase ("+args.method+")")
    ax.set_xticks(x + width - 0.125, bar_labels)
    ax.legend(loc='upper left', ncols=3, title="Type de phase")
    ax.set_ylim(0, 30000)

    if(not os.path.exists("images/"+sub_folders)):
        os.makedirs("images/"+sub_folders)

    plt.savefig("images/"+sub_folders+args.method+"_tls_bars.pdf")


    y = np.arange(len(bar_labels))  # the label locations
    height = 0.5  # the width of the bars

    width = 1


    plt.clf()
    fig, ax = plt.subplots(figsize=(12.8,4.8), layout='constrained')

    i = 0

    for key in dict_phases:
        print(key)
        multiplier = 0
        for phase in dict_phases[key][4*3600:5*3600]:
            if(phase == 0):
                c = "red"
            elif(phase == 1):
                c = "green"
            else:
                c = "orange"
            offset = width * multiplier
            rects = ax.barh(i, 1, height=height, left=width*multiplier, color=c)
            multiplier += 1
        i+=1

    ax.set_xlabel('Etapes')
    ax.set_title("Chronologie des phases de 19h à 20h ("+args.method+")")
    ax.set_yticks(y + height-0.5, bar_labels)

    plt.savefig("images/"+sub_folders+args.method+"_tls_hbars.pdf")


