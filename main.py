from alley.run import run_visual
from setting import Setting
from plotting import overall_plot, overall_plot_exh, overall_plot_all
import pandas as pd


def test_alley():
    s = Setting("alley", "ALLEY", runs=10000, expl_scn="alternative")
    s.run()


def test_store():
    s = Setting("store", "STORE", runs=1000, expl_scn="all")
    s.run()



def test_expl():

    s = Setting("alley", "ALLEY", runs=100000, expl_scn="alternative")
    s.run()

    s = Setting("alleyALL", "ALLEY", runs=100000, expl_scn="all")
    s.run()



def set_params(setting):
    #print(self.setting)
    if setting == "ALLEY":
        param_dicts = ([
            #{"steal_threshold": 0.2, "thief_success_rate": 0.4, "drop_rate": 0.3, "obstacle_rate": 0.7},
            #{"steal_threshold": 0.4, "thief_success_rate": 0.4, "drop_rate": 0.3, "obstacle_rate": 0.7},
            #{"steal_threshold": 0.2, "thief_success_rate": 0.7, "drop_rate": 0.4, "obstacle_rate": 0.7},
            #{"steal_threshold": 0.7, "thief_success_rate": 0.1, "drop_rate": 0.8, "obstacle_rate": 0.2},
            #{"steal_threshold": 0.7, "thief_success_rate": 0.1, "drop_rate": 0.4, "obstacle_rate": 0.7},
            #{"steal_threshold": 0.1, "thief_success_rate": 0.7, "drop_rate": 0.1, "obstacle_rate": 0.1}
        ])

        i_loop = [0.1, 0.5, 0.9]
        param_dicts = \
            []
        for i_1 in i_loop:
            for i_2 in i_loop:
                for i_3 in i_loop:
                    for i_4 in i_loop:
                        p_d = {"steal_threshold": i_1, "thief_success_rate": i_2, "drop_rate": i_3, "obstacle_rate": i_4}
                        param_dicts.append(p_d)

        param_dicts= param_dicts[0::10]

    else:
        print("No dicts here")

    print(param_dicts)
    return param_dicts

def test_params(alley_set):
    '''s = Setting("store", "STORE", runs=100, expl_scn="alternative")
    s.run()
    s = Setting("storeALL", "STORE", runs=100, expl_scn="all")
    s.run()'''

    runs = 10000
    for param in set_params("ALLEY"):
        if alley_set == "alley" or alley_set == "both":
            s = Setting(f"results/{param}/alley", "ALLEY", runs=runs, expl_scn="alternative", param_dict=param)
            s.run()
        if alley_set == "alleyALL" or alley_set == "both":
            s = Setting(f"results/{param}/alleyALL", "ALLEY", runs=runs, expl_scn="all", param_dict=param)
            s.run()

def create_param_plot(alley_set):
    # create plot of all networks for all param sets. In this case, for alley
    if alley_set == "alley" or alley_set == "both":
        list_of_dfs = []
        for param in set_params("ALLEY"):
            fp = f"results/{param}/alley/data/allOutcomes_hor.csv"
            df = pd.read_csv(fp)
            short_par = ""
            for k in param.keys():
                short_par += str(k[0:1])+":"+str(param[k])+" "
            df["params"] = short_par
            df['network_y'] = df['network_y'].replace({'CLI': 'FCON', 'VLEK': 'CNL', 'FENTON': 'CWL', 'CON': 'LOG'})
            df["Bayesian Network Construction"] = df["network_y"]

            df["Average Difference"] = df["sum"] / 2
            list_of_dfs.append(df)
        complete_df = pd.concat(list_of_dfs)
        overall_plot(complete_df, "alleycompletebox.png")

    if alley_set == "alleyALL" or alley_set == "both":
        list_of_dfs = []
        for param in set_params("ALLEY"):
            fp = f"results/{param}/alleyALL/data/allOutcomes_hor.csv"
            df = pd.read_csv(fp)
            short_par = ""
            for k in param.keys():
                short_par += str(k[0:1]) + ":" + str(param[k]) + " "
            df["params"] = short_par
            df['network_y'] = df['network_y'].replace({'CLI': 'ACY', 'VLEK': 'CNL', 'FENTON': 'CWL', 'CON': 'LOG'})
            df["Bayesian Network Construction"] = df["network_y"]
            df["Average Difference"] = df["sum"] / 2

            list_of_dfs.append(df)
        complete_df = pd.concat(list_of_dfs)
        overall_plot(complete_df, "alleyALLcompletebox.png")

def create_double_param_plot():
    # create plot of all networks for all param sets. In this case, for alley
        list_of_dfs = []
        for param in set_params("ALLEY"):
            fp = f"results/{param}/alley/data/allOutcomes_hor.csv"
            df = pd.read_csv(fp)
            short_par = ""
            for k in param.keys():
                short_par += str(k[0:2])+":"+str(param[k])+" "
            df["params"] = short_par
            df['network_y'] = df['network_y'].replace({'CLI': 'FCON', 'VLEK': 'CNL', 'FENTON': 'CWL', 'CON': 'LOG'})
            df["Bayesian Network Construction"] = df["network_y"]

            df["Average Difference"] = df["sum"] / 2
            df["Exhaustive"] = "not exhaustive"

            list_of_dfs.append(df)
        alley_complete_df = pd.concat(list_of_dfs)

        list_of_dfs = []
        for param in set_params("ALLEY"):
            fp = f"results/{param}/alleyALL/data/allOutcomes_hor.csv"
            df = pd.read_csv(fp)
            short_par = ""
            for k in param.keys():
                short_par += str(k[0:2]) + ":" + str(param[k]) + " "
            df["params"] = short_par
            df['network_y'] = df['network_y'].replace({'CLI': 'FCON', 'VLEK': 'CNL', 'FENTON': 'CWL', 'CON': 'LOG'})
            df["Bayesian Network Construction"] = df["network_y"]
            df["Average Difference"] = df["sum"] / 2
            df["Exhaustive"] = "exhaustive"

            list_of_dfs.append(df)


        alley_all_complete_df = pd.concat(list_of_dfs)


        overall_plot_exh(alley_complete_df, False, "allyresults.png")
        overall_plot_exh(alley_all_complete_df, True, "allyallresults.png")
        complete_df = pd.concat([alley_complete_df, alley_all_complete_df])
        overall_plot_all(complete_df, "allresults.png")



#
test_params("alley")
create_param_plot("alley")

test_params("alleyALL")
create_param_plot("alleyALL")
create_double_param_plot()

#run_visual()


#test_expl()
'''
#test_setting()
test_alley()
test_store()
#run_visual()'''