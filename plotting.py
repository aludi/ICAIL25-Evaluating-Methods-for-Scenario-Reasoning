from select import select

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import os, shutil
import re




def make_plots(param_dict, fp, scns):
    plot_boxplots(param_dict, fp, scns)
    plot_test(param_dict, fp, scns)

def plot_test(param_dict, fp, scns):
    # delete all pictures from older run
    folder = f"{fp}/img/scenarios/"

    if not os.path.exists(folder):
        os.makedirs(folder)
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    df = pd.read_csv(f"{fp}/data/allOutcomes.csv")
    df['network'] = df['network'].replace(
        {'CLI': 'FCON', 'VLEK': 'CNL', 'FENTON': 'CWL', 'CON': 'LOG'})

    prefix=""
    if "alley" in fp:
        l = ["Obstacle","SamePlace","ObjectGone","Drop","Steal","Intent"]
        x = []
        for s in scns:
            x.append(f"P{s}")
        l.extend(x)
        l.extend(["network"])
        df = df[l]

        d = df[["Obstacle","SamePlace","ObjectGone","Drop","Steal","Intent"]]
        combination = d.drop_duplicates().values.tolist()

        for combinations in combination:
            selected_rows = df[(df['Obstacle'] == combinations[0]) &
                               (df['SamePlace'] == combinations[1]) &
                               (df['ObjectGone'] == combinations[2]) &
                               (df['Drop'] == combinations[3]) &
                               (df['Steal'] == combinations[4]) &
                               (df['Intent'] == combinations[5])]

            comb_string = (f"Obs={combinations[0]}, SamePl={combinations[1]}, Objg={combinations[2]},"
                           f"Drop={combinations[3]}, Steal={combinations[4]}, Intent={combinations[5]}")


            if selected_rows.size > 0:
                # selecting the scenario colums

                scn_cols = []
                for s in scns:
                    scn_cols.append(f"P{s}")
                scn_cols.append("network")
                scn_col_df = selected_rows[scn_cols]
                scn_col_df.reset_index()
                scn_col_df_groundtruth = scn_col_df[scn_col_df["network"] == "frequency"]

                prefix = 0
                for s in scn_cols:
                    if s != "network":
                        #print(scn_col_df_groundtruth[s])
                        if scn_col_df_groundtruth[s].any() > 0:
                            prefix = 1

                selected_rows.set_index('network', inplace=True)
                selected_rows.plot(kind='bar', stacked=True)
                plt.savefig(f"{fp}/img/scenarios/{prefix}posteriors{comb_string}.png")
                plt.close()

            #plt.show()
    elif "store" in fp:
        l = ["Winner","Pos0t1","Pos1t1"]
        x = []
        for s in scns:
            x.append(f"P{s}")
        l.extend(x)
        l.extend(["network"])
        df = df[l]
        d = df[["Winner", "Pos0t1", "Pos1t1"]]
        combination = d.drop_duplicates().values.tolist()
        for combinations in combination:
            selected_rows = df[(df['Winner'] == combinations[0]) &
                               (df['Pos0t1'] == combinations[1]) &
                               (df['Pos1t1'] == combinations[2])]
            #print(selected_rows)

            comb_string = (f"win={combinations[0]}, pos0={combinations[1]}, pos1={combinations[2]}")
            if selected_rows.size > 0:
                #print(selected_rows["network"])
                selected_rows.set_index('network', inplace=True)
                selected_rows.plot(kind='bar', stacked=True)
                plt.savefig(f"{fp}/img/scenarios/{prefix}posteriors{comb_string}.png")
                plt.close()

        pass
    else:
        print("plotting not implemented yet")


def plot_boxplots(param_dict, fp, scns):
    #plt.show()
    print("don't forget the difference is 1 for impossible states...")
    df = pd.read_csv(f"{fp}/data/allOutcomes_hor.csv")
    df['network_y']=df['network_y'].replace({'CLI': 'FCON', 'VLEK': 'CNL', 'FENTON': 'CWL', 'CON': 'LOG'})

    df["Bayesian Network Construction"] = df["network_y"]
    print(df["Bayesian Network Construction"] )
    df["Average Difference"] = df["sum"]/2

    my_colors = ["tab:blue", "tab:orange",
                 "tab:green", "tab:red", "tab:yellow"]
    title = ("Average difference between"
             " frequency of scenario in ground truth and \n"
             " predicted posterior of scenario,"
             " per method for each evidence valuation")

    sns.boxplot(data=df, x="Bayesian Network Construction", y="Average Difference").set_title(title)
    #print(df["difference"])
    print(df)
    plt.savefig(f"{fp}/img/posteriors_dif_box.png")
    #plt.show()
    plt.close()
    #plt.show()
    #plt.close()

def overall_plot(df, name):
    title = "Difference between the ground truth and the networks"
    cpl = sns.catplot(data=df, x="Bayesian Network Construction", y="Average Difference", col="params", kind="box", col_wrap=7, height=4, aspect=.6)
    cpl.fig.subplots_adjust(top=0.9)
    cpl.fig.suptitle(title)
    plt.savefig(name)
    plt.close()

def overall_plot_exh(df, exh, name):
    if exh == False:
        title = ("Difference between ground truth outcome frequency and predicted outcome probability per BN in non-exhaustive setting")
        color = "#1f77b4"
    else:
        title = (
            "Difference between ground truth outcome frequency and predicted outcome probability per BN in exhaustive setting")
        color="#ff7f0e"

    cpl = sns.catplot(data=df, x="Bayesian Network Construction", y="Average Difference",
                      col="params", kind="strip", color=color,
                      alpha=.3, col_wrap=3, height=3, aspect=1)
    cpl.fig.subplots_adjust(top=0.9)
    cpl.fig.suptitle(title, fontsize=10)
    plt.savefig(name)
    plt.show()
    plt.close()

def overall_plot_all(df, name):
    title = "Difference between ground truth outcome frequency and predicted outcome probability per Bayesian network"
    cpl = sns.catplot(data=df, x="Bayesian Network Construction", y="Average Difference", hue="Exhaustive", col="params", kind="strip",
                      dodge="True",alpha=.3, col_wrap=3, height=3, aspect=1)
    cpl.fig.subplots_adjust(top=0.9)
    cpl.fig.suptitle(title, fontsize=10)
    plt.savefig(name)
    plt.show()
    plt.close()


