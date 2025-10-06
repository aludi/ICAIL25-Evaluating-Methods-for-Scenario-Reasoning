
'''
template that contains model, data about the model, BNs,
and images.
Per setting there is a preprocessing step
'''

from alley.alley.model import AlleyModel
from bn_tools import build_bn, inference, power_set_generation
from plotting import make_plots
import pandas as pd
import os
import re


class Setting:
    def __init__(self, fp, setting, runs, expl_scn, param_dict):
        self.events = {}
        self.evidence = {}
        self.file_path = fp
        self.setting = setting
        self.runs = runs
        self.expl_scn = expl_scn
        self.param_dicts = param_dict
        self.build_file_structure()
        self.set_bn_types()

    def build_file_structure(self):
        if not os.path.exists(self.file_path):
            os.makedirs(self.file_path)
        for path in [f"{self.file_path}/data", f"{self.file_path}/bn", f"{self.file_path}/img",
                     f"{self.file_path}/img/vlekinf/"]:
            if not os.path.exists(path):
                os.makedirs(path)


    def set_bn_types(self):
        if self.setting == "ALLEY":
            self.bn_types = ["DEF","CLI", "VLEK", "FENTON", "CON"] #["CLI"] #["DEF", "CON", "MUL", "FENTON", "VLEK","CLI"]
        if self.setting == "STORE":
            self.bn_types = ["DEF", "CON","MUL", "FENTON", "VLEK","CLI"]


    def generate_data(self, param_dict):
        df_list = []
        if "ALL" in self.file_path: # if we are in the "all" case then don't generate data but instead draw it from the original set.
            print("taking data from /alley/data instead")
            org_file_path = self.file_path.replace("ALL", "")
            print(org_file_path)
            df = pd.read_csv(f"{org_file_path}/data/data.csv")
        else:
            for i in range(0, self.runs):
                model = self.run_model(param_dict)
                df = self.preprocess_model(model)
                df_list.append(df)
            df = pd.concat(df_list, ignore_index=True)
        df=self.assign_scenarios(df)
        df.to_csv(f"{self.file_path}/data/data.csv", index=False)

    def assign_scenarios(self, df):
        df['row_tuple'] = df.apply(tuple, axis=1)
        #print(df['row_tuple'])
        #print(df['row_tuple'].unique())
        unique_rows = {val: f'scn{i + 1}' for i, val in enumerate(df['row_tuple'].unique())}  # counting unique scns
        df['scn'] = df['row_tuple'].map(unique_rows)
        df = df.drop(columns=['row_tuple'])
        return df




    def preprocess_model(self, model):
        if self.setting == "ALLEY":
            df_model = model.datacollector.get_model_vars_dataframe()
            df_params = df_model[["steal_threshold", "thief_success_rate", "drop_rate"]]
            df = df_model[["Obstacle", "SamePlace"]].iloc[[2]]
            df_agents = model.datacollector.get_agent_vars_dataframe()
            df_agents = df_agents[df_agents["step"] == 2]

            if len((df_agents[df_agents["role"] == "potentialvictim"]["object"])) > 0:
                if df_agents[df_agents["role"] == "potentialvictim"]["object"].iloc[[0]].item():
                    df["ObjectGone"] = False
                else:
                    df["ObjectGone"] = True
            else:
                df["ObjectGone"] = False

            if len((df_agents[df_agents["role"] == "potentialvictim"][
                "drop"])) > 0:  # bug: drop = True but object gone is false
                if df_agents[df_agents["role"] == "potentialvictim"]["drop"].iloc[[0]].item():
                    df["Drop"] = True
                else:
                    df["Drop"] = False
            else:
                df["Drop"] = False

            if len(df_agents[df_agents["role"] == "thief"]["steal"]) > 0:
                if df_agents[df_agents["role"] == "thief"]["steal"].iloc[[0]].item():
                    df["Steal"] = True
                else:
                    df["Steal"] = False
            else:
                df["Steal"] = False

            if len(df_agents[df_agents["role"] == "thief"]["intent"]) > 0:
                if df_agents[df_agents["role"] == "thief"]["intent"].iloc[[0]].item():
                    df["Intent"] = True
                else:
                    df["Intent"] = False
            else:
                df["Intent"] = False
            return df

        elif self.setting == "STORE":
            df_agents = model.datacollector.get_agent_vars_dataframe()
            final_step = df_agents["step"].max()
            step_before = final_step - 1
            df = df_agents[df_agents["step"] == final_step]["winner"]
            d = df.to_dict()
            if d[(final_step, 0)] == True and d[(final_step, 1)] == True:
                x="both"
            if d[(final_step, 0)] == True and d[(final_step, 1)] == False:
                x=0
            if d[(final_step, 0)] == False and d[(final_step, 1)] == True:
                x=1

            df = df_agents[df_agents["step"] == step_before]["pos"]
            d = df.to_dict()
            (x0, y0) = d[(step_before, 0)]
            (x1,y1) = d[(step_before, 1)]
            d = {"Winner":x, "Pos0t1":f"x{str(x0)}y{str(y0)}", "Pos1t1":f"x{str(x1)}y{str(y1)}"}
            df = pd.DataFrame(d, index=[0])
            return df
        else:
            print("no preprocessing defined for this model")
            exit()


    def get_data(self, type):
        fp = self.file_path.replace("ALL", "")

        df = pd.read_csv(f"{fp}/data/data.csv")
        #if self.setting == "ALLEY":

        df.loc[(~df["scn"].isin(self.scns)), 'scn'] = "leak"

        if type in ["FENTON", "VLEK", "CLI", "CON", "FENTONCLI", "DEF"]:
            df_scenarios = pd.get_dummies(df["scn"], prefix='',
                                          prefix_sep='', dtype=bool)
            #print(df_scenarios)
            df = pd.concat([df, df_scenarios], axis=1)
            #print(df)
            df = df.drop(columns=["scn"])
            if "leak" in df.columns:
                df = df.drop(columns=["leak"])

            df.to_csv(f"{self.file_path}/data/FentonchangedData.csv", index=False)
            path = "FentonchangedData"

        else:
            df.to_csv(f"{self.file_path}/data/changedData.csv", index=False)
            path="changedData"

        return f"{self.file_path}/data/{path}.csv"



    def get_arcs(self, type):
        arcs = {"mandatory":[], "forbidden":[]}
        if self.setting == "ALLEY":
            if type == "MUL":
                arcs["mandatory"] = [("scn", "Intent"), ("scn", "SamePlace"), ("scn", "Steal"), ("scn","ObjectGone"),
                        ("scn", "Obstacle"), ("scn", "Drop"),("Intent", "SamePlace"), ("Intent", "Steal"), ("SamePlace", "Steal"), ("Steal", "ObjectGone"),
                        ("Obstacle", "SamePlace"), ("Drop", "ObjectGone"), ("Drop", "Steal"), ("Drop", "Intent")]

                l = []
                events = ["Intent", "SamePlace", "Steal", "ObjectGone", "Obstacle", "Drop"]
                for e1 in events:
                    for e in events:
                        if (e1, e) not in arcs["mandatory"]:
                            l.append((e1, e))
                arcs["forbidden"] = l

            elif type == "VLEK" or type == "FENTON":
                arcs["mandatory"] = [("Intent", "SamePlace"), ("Intent", "Steal"), ("SamePlace", "Steal"), ("Steal", "ObjectGone"),
                        # scn1
                        ("Obstacle", "SamePlace"),  ("Drop", "ObjectGone"), ("Drop", "Steal"), ("Drop", "Intent")
                        ]
                events = ["Intent", "SamePlace", "Steal", "ObjectGone", "Obstacle", "Drop"]
                l = []
                for scn in self.scns:
                    for e in events:
                        l.append((scn, e))
                arcs["mandatory"].extend(l)
                l = []
                for e1 in events:
                    for e in events:
                        if (e1, e) not in arcs["mandatory"]:
                            l.append((e1, e))
                arcs["forbidden"] = l
                l = []
                for i in range(0, len(self.scns)):
                    for j in range(0, len(self.scns)):
                        l.append((self.scns[i], self.scns[j]))
                arcs["forbidden"].extend(l)

            elif type == "CON":
                arcs["mandatory"] = [("Intent", "SamePlace"), ("Obstacle", "SamePlace"),
                        ("Intent", "Steal"), ("SamePlace", "Steal"),
                        ("Steal", "ObjectGone"), ("Drop", "ObjectGone"), ("Drop", "Intent"),
                        ("Drop", "Steal")]
                l = []
                events = ["Intent", "SamePlace", "Steal", "ObjectGone", "Obstacle", "Drop"]
                for e1 in events:
                    for e in events:
                        if (e1, e) not in arcs["mandatory"]:
                            l.append((e1, e))
                arcs["forbidden"] = l
                l = []
                for scn in self.scns:
                    for e in events:
                        l.append((e, scn))
                arcs["mandatory"].extend(l)
                l = []
                for i in range(0, len(self.scns)):
                    for j in range(0, len(self.scns)):
                        l.append((self.scns[i], self.scns[j]))
                arcs["forbidden"].extend(l)

            elif type == "DEF":
                pass
            elif type == "CLI":
                l = []
                for i in range(0, len(self.scns)):
                    for j in range(i+1, len(self.scns)):
                        l.append((self.scns[i], self.scns[j]))
                arcs["mandatory"] = l

                arcs["mandatory"].extend([("Intent", "SamePlace"), ("Obstacle", "SamePlace"),
                                     ("Intent", "Steal"), ("SamePlace", "Steal"),
                                     ("Steal", "ObjectGone"), ("Drop", "ObjectGone"),
                                     ("Drop", "Steal"), ("Drop", "Intent")]) # ("Drop", "Intent")
                l = []
                events = ["Intent", "SamePlace", "Steal", "ObjectGone", "Obstacle", "Drop"]
                for scn in self.scns:
                    for e in events:
                        l.append((scn, e))
                arcs["mandatory"].extend(l)

                l = []
                for e1 in events:
                    for e in events:
                        if (e1, e) not in arcs["mandatory"]:
                            l.append((e1, e))
                arcs["forbidden"] = l

                '''
                events = ["Intent", "SamePlace", "Steal", "ObjectGone", "Obstacle", "Drop"]
                l = []
                for scn in self.scns:
                    for e in events:
                        l.append((scn, e))
                arcs["mandatory"].extend(l)
                
                l = []
                for i in range(0, len(events)):
                    for j in range(0, len(events)):
                        l.append((events[i], events[j]))
                arcs["forbidden"] = l'''
            elif type == "FENTONCLI":
                l = []
                for i in range(0, len(self.scns)):  # scenarios cannot be connected to each other
                    for j in range(0, len(self.scns)):
                        l.append((self.scns[i], self.scns[j]))
                #print(l)
                arcs["forbidden"] = l
                events = ["Intent", "SamePlace", "Steal", "ObjectGone", "Obstacle", "Drop"]
                l = []
                for scn in self.scns:   # events must be connected to parent scenarios
                    for e in events:
                        l.append((scn, e))
                arcs["mandatory"] = l
                l = []
                for i in range(0, len(events)): # events not connected to each other
                    for j in range(0, len(events)):
                        l.append((events[i], events[j]))
                arcs["forbidden"].extend(l)

            else:
                print("BN ")

        elif self.setting == "STORE":
            if type == "DEF":
                pass

            elif type == "MUL":
                arcs["mandatory"] = [("scn", "Winner"), ("scn", "Pos0t1"), ("scn", "Pos1t1"), ("Pos0t1", "Pos1t1"),
                                     ("Pos0t1", "Winner"), ("Pos1t1", "Winner")]

            elif type == "VLEK" or type == "FENTON":
                arcs["mandatory"] = [("Pos0t1", "Winner"), ("Pos1t1", "Winner"), ("Pos0t1", "Pos1t1")]
                events = ["Winner", "Pos0t1", "Pos1t1"]
                l = []
                for scn in self.scns:
                    for e in events:
                        l.append((scn, e))
                arcs["mandatory"].extend(l)
                l = []
                for i in range(0, len(self.scns)):
                    for j in range(0, len(self.scns)):
                        l.append((self.scns[i], self.scns[j]))
                arcs["forbidden"] = l

            elif type == "CON":
                arcs["mandatory"] = [("Winner", "scn"), ("Pos0t1", "scn"),
                                     ("Pos1t1", "scn"), ("Pos0t1", "Pos1t1"),]

            elif type == "CLI":
                l = []
                for i in range(0, len(self.scns)):
                    for j in range(i+1, len(self.scns)):
                        l.append((self.scns[i], self.scns[j]))
                arcs["mandatory"] = l
                events = ["Winner", "Pos0t1", "Pos1t1"]
                l = []
                for scn in self.scns:
                    for e in events:
                        l.append((scn, e))
                arcs["mandatory"].extend(l)
                l = []
                for i in range(0, len(events)):
                    for j in range(0, len(events)):
                        l.append((events[i], events[j]))
                arcs["forbidden"] = l

            elif type == "FENTONCLI":
                l = []
                for i in range(0, len(self.scns)):  # scenarios cannot be connected to each other
                    for j in range(0, len(self.scns)):
                        l.append((self.scns[i], self.scns[j]))
                #print(l)
                arcs["forbidden"] = l
                events = ["Winner", "Pos0t1", "Pos1t1"]
                l = []
                for scn in self.scns:   # scenarios are connected to events
                    for e in events:
                        l.append((scn, e))
                arcs["mandatory"] = l
                l = []
                for i in range(0, len(events)): # events cannot be connected to each other
                    for j in range(0, len(events)):
                        l.append((events[i], events[j]))
                arcs["forbidden"].extend(l)
            else:
                print("bn type not implemented yet")

        return arcs

    def create_bns(self):
        for bn_type in self.bn_types:
            csv= self.get_data(bn_type)
            arcs = self.get_arcs(bn_type)
            const = bn_type
            name = f"{self.file_path}/bn/{bn_type}"
            scns = self.scns
            print("building ", name)
            build_bn(csv, arcs, const, name, scns)


    def inference(self):
        inference(self.file_path, self.bn_types, self.scns, self.evidence)

    def set_numscn(self):
        fp = self.file_path.replace("ALL", "")
        df = pd.read_csv(f"{fp}/data/data.csv")
        # print("Unique scenarios")
        x = df.value_counts().rename_axis().reset_index(name='probability')
        self.num_scn = len(x)

    def set_rel_events(self):
        fp = self.file_path.replace("ALL", "")
        df = pd.read_csv(f"{fp}/data/data.csv")
        self.events = {}
        for col in df.columns:
            if col != "scn":
                vals = df[col].unique()
                list_vals = []
                for val in list(vals):
                    list_vals.append(str(val))
                self.events[col] = list_vals
                self.evidence[col] = list_vals
        #print("all events")
        #print(self.events)


    def define_explicit_scenarios(self):
        df = pd.read_csv(f"{self.file_path}/data/scnProbs.csv")
        if self.expl_scn == "all": # current situation: all scenarios that are possible are made explicit
            self.scns = list(df["scn"].unique())
        elif self.expl_scn == "alternative":
            # we are selecting two full alternative scenarios
            if self.setting == "ALLEY":
                # making one guilt and one inncent scenario explict.
                guilt_scn = list(df[(df["Obstacle"] == True) & (df["SamePlace"] == True) & (df["ObjectGone"] == True) &
                                    (df["Drop"] == False) & (df["Intent"] == True) & (df["Steal"] == True)]["scn"])[0]# guilt scenario
                print(guilt_scn)
                innocent_scn = list(df[(df["Obstacle"] == True) & (df["SamePlace"] == True) & (df["ObjectGone"] == True) &
                                       (df["Drop"] == True) & (df["Intent"] == True) & (df["Steal"] == False)]["scn"])[0]# innocent scenario
                print(innocent_scn)

            elif self.setting == "STORE":

                guilt_scn = list(df[(df["Winner"] == "1") & (df["Pos0t1"] == "x0y1") &
                                    (df["Pos1t1"] == "x1y1")]["scn"])[0]  # guilt scenario
                print(guilt_scn)
                innocent_scn =list(df[(df["Winner"] == "0") & (df["Pos0t1"] == "x0y0")
                                      & (df["Pos1t1"] == "x2y1")]["scn"])[0]  # innocent scenario
                print(innocent_scn)
            else:
                print("no explicit scenarios implemetned")
                exit()
            self.scns = [guilt_scn, innocent_scn]
        else:
            print("explicit scenarios are not defed")


    def count_scn_num_freq(self):
        fp = self.file_path.replace("ALL", "")
        df = pd.read_csv(f"{fp}/data/data.csv")
        # print("Unique scenarios")
        x = df.value_counts().rename_axis().reset_index(name='probability')
        df_scenarios = pd.get_dummies(x["scn"], prefix='', prefix_sep='')
        # Concatenate the original DataFrame (without 'scn') and the new dummy columns
        x = pd.concat([x, df_scenarios], axis=1)
        x.rename(columns=lambda col: col.replace('scn', 'Pscn') if re.match(r'^scn\d+$', col) else col, inplace=True)

        x["freq"] = x["probability"] /df.shape[0] # the number of rows in the dataframe of "data" # self.runs
        x["network"] = "frequency"
        x.drop(columns=["probability"])
        x.to_csv(f"{self.file_path}/data/scnProbs.csv")


    def initialize_scenario_information(self):
        self.set_numscn()
        self.set_rel_events()
        self.count_scn_num_freq()
        self.define_explicit_scenarios()

    def counting_in_frequency_data(self):
        self.count_frequency_scenarios()

    def count_frequency_scenarios(self):
        fp = self.file_path.replace("ALL", "")
        # scenarios voor volledige scns ipv partial...
        df = pd.read_csv(f"{fp}/data/data.csv")
        df_e = df[list(self.evidence.keys())].drop_duplicates()
        event_sets = power_set_generation(self.evidence) #df_e.to_dict(orient="records")
        #print(event_sets)
        all_scenarios = self.scns # [f"scn{i}" for i in range(1, self.num_scn+1)]
        l = []
        for event_set in event_sets:
            fil = {k: v for k, v in event_set.items() if v != "None"}
            event_set.clear()
            event_set.update(fil)
            query_s = []
            for k in event_set.keys():
                v = event_set[k]
                if self.setting == "STORE":
                    query_s.append(f"{k} == '{v}'")
                elif self.setting == "ALLEY":
                    query_s.append(f"{k} == {v}")
                else:
                    print("selecting evidence per row in count frequency scenario not yet implemented for this setting")
            if query_s != []:
                query_s = " and ".join(query_s)
                d = df.query(query_s)
            else:
                d = df


            d_e = d.value_counts().rename_axis().reset_index(name='probability')
            d_e["probability"] = d_e["probability"]/d.value_counts().sum()
            d_e = d_e[["scn", "probability"]]

            d_e = d_e.set_index('scn').T #d_e.pivot(columns="scn",values="probability").fillna(0)
            d_e = d_e.reindex(columns=all_scenarios, fill_value = 0)

            for ev_key in event_set.keys():
                d_e[ev_key] = event_set[ev_key]
            #print("the events that are specified:, ", d_e)
            l.append(d_e)

        x = pd.concat(l)
        #print(l)
        x = x.reset_index()
        x = x.drop("index", axis=1)
        x = x.fillna("None")
        x.rename(columns=lambda col: col.replace('scn', 'Pscn') if re.match(r'^scn\d+$', col) else col, inplace=True)
        #x["freq"] = x["probability"]
        x["network"] = "frequency"
        #x.drop(columns=["probability"])
        #print("scenario frequencies with possible leak states per evidence set")
        x.to_csv(f"{self.file_path}/data/freqscnProbs.csv")
        #exit()

    def merge_ground_truth_with_data(self):
        df_g = pd.read_csv(f"{self.file_path}/data/freqscnProbs.csv")
        df_d = pd.read_csv(f"{self.file_path}/data/outcomeProbs.csv")
        df = pd.concat([df_g, df_d])
        df.to_csv(f"{self.file_path}/data/allOutcomes.csv")

    def merge_ground_truth_with_data_horizontally(self):
        df_g = pd.read_csv(f"{self.file_path}/data/freqscnProbs.csv")
        df_d = pd.read_csv(f"{self.file_path}/data/outcomeProbs.csv")
        df = pd.merge(df_g, df_d, on=list(self.evidence.keys()))
        df.to_csv(f"{self.file_path}/data/allOutcomes_hor.csv")


    def calculate_difference(self):
        # calculate for each evidence set, the difference between the grequency in the ground truth and the posterior frequency in the BN
        print("here I will calculate the difference between "
              "each measure and the ground truth")
        df = pd.read_csv(f"{self.file_path}/data/allOutcomes_hor.csv")
        dif_cols = []
        for i in range(1, self.num_scn+1):
            try:
                df[f"dif_Pscn{i}"] = abs(df[f"Pscn{i}_x"] - df[f"Pscn{i}_y"])    # Pscn_x = ground truth frequency, Pscn_y probability in given BN
                dif_cols.append(f"dif_Pscn{i}")
            except KeyError: # the keyerror occurs when we attempt to sum a column that does not exist (for example, scn1 if we consider only partial scenarios s2, s5)
                pass
        df["sum"] = df[dif_cols].sum(axis=1)
        df.to_csv(f"{self.file_path}/data/allOutcomes_hor.csv")


    def data_processing(self):
        #self.count_frequency_scenarios()    # counts frequency per scenario per evidence set
        self.merge_ground_truth_with_data()
        self.merge_ground_truth_with_data_horizontally()

    def calculate_stats(self):
        self.calculate_difference()

    def analyse_differences(self):
        df = pd.read_csv(f"{self.file_path}/data/allOutcomes_hor.csv")
        dif_list = []
        for s in self.scns:
            dif_list.append(f"dif_P{s}")
        filtered_df = df[(df[dif_list] > 0.05).any(axis=1)]
        filtered_df.to_csv(f"{self.file_path}/data/badFitCols.csv")



    def run_model(self, param_dict):
        if self.setting == "ALLEY":
            model = AlleyModel(2, 2, 3, param_dict=param_dict)
            for i in range(0, 3):
                model.step()
        elif self.setting == "STORE":
            model = StoreModel(2, 10, 10, param_dict=param_dict)
            for i in range(0, 10):
                if model.running is not False:
                    model.step()
        else:
            print("no model of this setting implemented")
            exit()
        return model

    def run(self):

        print(f"generating data by running model and processing for given param set {self.param_dicts}")
        self.generate_data(self.param_dicts)
        print("initializing scenario information")
        self.initialize_scenario_information()

        print("counting in the frequency data")
        self.counting_in_frequency_data()
        print("creating bns")
        self.create_bns()

        print("performing inference")
        self.inference()


        print("combining dataframes")
        self.data_processing()

        print("calculating differences")
        self.calculate_stats()
        print("analysing differences")
        self.analyse_differences()
        #print("making plots")
        #make_plots(self.param_dicts, self.file_path, self.scns)

