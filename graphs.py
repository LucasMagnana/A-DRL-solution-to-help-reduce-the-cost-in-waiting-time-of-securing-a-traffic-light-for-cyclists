import matplotlib.pyplot as plt

def compute_graphs_data(structure_was_open, dict_cyclists_arrived, tab_scenario):
    tab_diff_finish_step = [[],[],[], []]
    tab_diff_waiting_time = [[],[],[], []]
    tab_diff_distance_travelled = [[],[],[], []]
    tab_num_type_cyclists = [0, 0, 0, 0]

    tab_all_diff_arrival_time=[]

    for i in dict_cyclists_arrived:
        c = dict_cyclists_arrived[i]
        tab_all_diff_arrival_time.append(tab_scenario[int(c.id)]["finish_step"]-c.finish_step)
        if(structure_was_open):
            if(c.canceled_candidature):
                tab_diff_finish_step[2].append(tab_scenario[int(c.id)]["finish_step"]-c.finish_step)
                tab_diff_waiting_time[2].append(tab_scenario[int(c.id)]["waiting_time"]-c.waiting_time)
                tab_diff_distance_travelled[2].append(tab_scenario[int(c.id)]["distance_travelled"]-c.distance_travelled)
                tab_num_type_cyclists[2]+=1
            elif(c.struct_crossed):
                if(c.finish_step>tab_scenario[int(c.id)]["finish_step"]):
                    tab_diff_finish_step[1].append(tab_scenario[int(c.id)]["finish_step"]-c.finish_step)
                    tab_diff_waiting_time[1].append(tab_scenario[int(c.id)]["waiting_time"]-c.waiting_time)
                    tab_diff_distance_travelled[1].append(tab_scenario[int(c.id)]["distance_travelled"]-c.distance_travelled)
                    tab_num_type_cyclists[1]+=1
                elif(c.finish_step<tab_scenario[int(c.id)]["finish_step"]):
                    tab_diff_finish_step[0].append(tab_scenario[int(c.id)]["finish_step"]-c.finish_step)
                    tab_diff_waiting_time[0].append(tab_scenario[int(c.id)]["waiting_time"]-c.waiting_time)
                    tab_diff_distance_travelled[0].append(tab_scenario[int(c.id)]["distance_travelled"]-c.distance_travelled)
                    tab_num_type_cyclists[0]+=1
            else:
                tab_diff_finish_step[3].append(tab_scenario[int(c.id)]["finish_step"]-c.finish_step)
                tab_diff_waiting_time[3].append(tab_scenario[int(c.id)]["waiting_time"]-c.waiting_time)
                tab_diff_distance_travelled[3].append(tab_scenario[int(c.id)]["distance_travelled"]-c.distance_travelled)
                tab_num_type_cyclists[3]+=1

                    


        '''tab_mean_diff_arrival_time = []
        for i in range(len(tab_diff_finish_step)):
            if(len(tab_diff_finish_step[i])==0):
                tab_mean_diff_arrival_time.append(0)
            else:
                tab_mean_diff_arrival_time.append(sum(tab_diff_finish_step[i])/len(tab_diff_finish_step[i]))


        tab_mean_diff_waiting_time = []
        for i in range(len(tab_diff_waiting_time)):
            if(len(tab_diff_waiting_time[i])==0):
                tab_mean_diff_waiting_time.append(0)
            else:
                tab_mean_diff_waiting_time.append(sum(tab_diff_waiting_time[i])/len(tab_diff_waiting_time[i]))

        tab_mean_diff_distance_travelled = []
        for i in range(len(tab_diff_distance_travelled)):
            if(len(tab_diff_distance_travelled[i])==0):
                tab_mean_diff_distance_travelled.append(0)
            else:
                tab_mean_diff_distance_travelled.append(sum(tab_diff_distance_travelled[i])/len(tab_diff_distance_travelled[i]))'''

        
    return tab_all_diff_arrival_time, tab_diff_finish_step, tab_diff_waiting_time, tab_diff_distance_travelled, tab_num_type_cyclists




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