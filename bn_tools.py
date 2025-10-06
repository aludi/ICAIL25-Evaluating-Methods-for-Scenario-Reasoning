from copy import deepcopy

import pandas as pd
import pyagrum as gum
import pyagrum.lib.image as gumimage
import itertools as iter
import copy
import re
import os, shutil


def build_bn(csv, arcs, const, name, scns):
    '''

    :param df: (adapted) dataframe from which we learn params
    :param arcs: arcs that need to be created (specifed)
    :param const: type of constraint node between scenarios. Options: Vlek, Fenton, No
    :param name: file name and structure
    :return: nothing
    '''
    # for the default / machine learning algorithm based on data
    if const == "DEF":
        learner = gum.BNLearner(csv)
        bn = learner.learnBN()
    else:
        learner = gum.BNLearner(csv)
        for (arc_h, arc_t) in arcs["mandatory"]:
            learner.addMandatoryArc(arc_h,arc_t)

        for (arc_h, arc_t) in arcs["forbidden"]:
            learner.addForbiddenArc(arc_h,arc_t)

        #learner.useAprioriSmoothing()  # pyagrum 0.22.5
        learner.useBDeuPrior()  # pyagrum 0.22.5

        learner.useGreedyHillClimbing()
        bn = learner.learnBN()

        if "VLEK" in const:
            bn = add_constraint_node(bn, scns)
        elif "FENTON" in const:
            bn = add_fenton_constraint_node(bn, scns)
        elif "CON" in const:
            bn = change_scn_nodes(bn, scns, csv)

    gum.saveBN(bn, f"{name}_notlogical.net")

    if const in ["VLEK", "FENTON", "CLI"]:
        fix_01_with_scns_as_parents(bn, scns, csv, const)
    print("save bn", bn)
    # this is the BN using 0.999 and 0.001, without converting those values to 0 and 1

    # now finally we convert each entry in the cpt that is bigger/smaller than threshold to be 0 and 1
    # to comply with the methods

    '''for node in bn.nodes():
        # depending on the bn
        eps = 0.05# 0.005

        for i in bn.cpt(node).loopIn():
            if bn.cpt(node).get(i) > 1 - eps:
                bn.cpt(node).set(i, 1)
            elif bn.cpt(node).get(i) < eps:
                bn.cpt(node).set(i, 0)
            else:
                pass'''


    gum.saveBN(bn, f"{name}.net")
    gumimage.exportInference(bn, f"{name}.png")
    convert_networks_to_hugin(f"{name}")


def power_set_generation(valuations):
    combinations = []
    for r in range(1, len(valuations) + 1):
        # Generate all combinations of keys of size r
        for keys in iter.combinations(valuations.keys(), r):
            # Generate the Cartesian product of the values for the selected keys
            for values in iter.product(*(valuations[key] for key in keys)):
                # Create a dictionary for each combination
                combinations.append(dict(zip(keys, values)))
    return combinations

def inference(fp, bns, scns, rel_event_dict):
    fp1 = fp.replace("ALL", "")
    df = pd.read_csv(f"{fp1}/data/data.csv")
    nodes = list(df)


    if "scn" in nodes:
        print("we have one node that contains all the scenarios")
        nodes.remove("scn")
        scns_considered = scns #df["scn"].unique() #only look at scenarios that have been made expkicit

    num_scn = len(scns_considered)


    power_set = power_set_generation(rel_event_dict)
    #print("power set")
    #print(power_set)

    #truth_vals = list(iter.product(states, repeat=len(nodes)))
    l_l = []
    col_names = []
    col_names.extend(nodes)
    scn_post_names = ["network"]
    for scn in scns_considered:
        scn_post_names.append(f"P{scn}")
    col_names.extend(scn_post_names)

    for bn_name in bns:

        print(f"running inference on {bn_name}")
        bn = gum.loadBN(f"{fp}/bn/{bn_name}.net")
        ie = gum.LazyPropagation(bn)

        '''if bn_name == "VLEK":
            constr_l = []
            #print(scns_considered)
            for scn in scns_considered:  # alternative solution with weighted virtual evidence
                other = copy.deepcopy(scns_considered)
                other.remove(scn)
                weight = 1
                for other_scn in other:
                    print(other_scn, bn.cpt(other_scn)[0])
                    weight = weight*bn.cpt(other_scn)[0]
                #print("prob of true scenario: ", scn, bn.cpt(scn)[1])
                constr_l.append((1-bn.cpt(scn)[1]))
            # old solution with equal virtual evidence
            # constr_l = [1] * num_scn
            constr_l.append(0)


            gumimage.exportInference(bn, f"{fp}/img/vlekinf/vlek.png", evs={})
            a = {}
            a["constraint"] = constr_l

            #for scn in scns_considered:
            #    print("prior of scenario ", scn, ie.posterior(scn))

            ie.addEvidence("constraint", constr_l)

            #for scn in scns_considered:
            #    print("after constraint, ",scn, ie.posterior(scn))


            gumimage.exportInference(bn, f"{fp}/img/vlekinf/vlekVE.png", evs=a)

            ie = gum.LazyPropagation(bn)'''

        if bn_name == "FENTON" or bn_name == "VLEK":
            b={}
            gumimage.exportInference(bn, f"{fp}/img/vlekinf/{bn_name}.png", evs={})
            b["constraint"] = 1  # fenton constraint does not work with virtual evidence
            try:
                gumimage.exportInference(bn, f"{fp}/img/vlekinf/{bn_name}VE.png", evs=b)
            except Exception as e:
                print("inconsistent to set evidence in picture")
            ie = gum.LazyPropagation(bn)




        for i in range(0, len(power_set)):
            l = []
            #a = dict(zip(nodes, truth_vals[i]))
            a = power_set[i]
            #print(a)

            val_dict = {}
            for n in nodes:
                if n not in a.keys():
                    val_dict[n] = "None"
                if n in a.keys():
                    val_dict[n] = a[n]

            l.extend([*val_dict.values()])
            l.extend([bn_name])
            try:
                ie.setEvidence(a)
            except Exception as e:
                exit()
            else:
                if 'constraint' in list(bn.names()):
                    if  "FENTON" in bn_name or "VLEK" in bn_name:
                        ie.addEvidence("constraint",1)    # fenton constraint does not work with virtual evidence
                    else:
                        pass
                        #ie.addEvidence("constraint", constr_l)

                outcome_probs = []
                for scn in scns_considered:
                    if "scn" in bn.names():
                        try:    # try to make a prediction for the given set of evidence
                            if ie.evidenceProbability() < 0.000001:
                                # exception for rare evidence
                                scn_posterior = 0
                            else:
                                scn_posterior = ie.posterior("scn")[{"scn": scn}]
                                #print(scn, scn_posterior)


                        except Exception as e:   # evidence can be incompatible = their joint is 0 (pyagrum error), interpreting this as probability of scn = 0
                            scn_posterior = 0
                            #print(scn, scn_posterior, "inconsistent")

                    else:
                        try:
                            if ie.evidenceProbability() < 0.000001:
                                # exception for rare evidence
                                scn_posterior = 0
                            else:
                                scn_posterior = ie.posterior(scn)[{scn: "True"}]

                        except Exception as e:
                            scn_posterior = 0
                    outcome_probs.append(scn_posterior)
            l.extend(outcome_probs)
            l_l.append(l)
    df = pd.DataFrame(l_l, columns=col_names)
    df.to_csv(f"{fp}/data/outcomeProbs.csv")

def change_scn_nodes(bn,scns, csv):
    # extract scenarios from csv
    df = pd.read_csv(csv)
    for s in scns:
        row = df[df[s] == True].iloc[0]
        d = {col:str(row[col]) for col in df.columns if col not in scns}
        I = gum.Instantiation(bn.cpt(s))
        d_t = copy.deepcopy(d)
        d_t[s] = 'True'
        d_f = copy.deepcopy(d)
        d_f[s] = 'False'
        while not I.end():
            i_dict = I.todict(withLabels=True)
            if i_dict == d_t:   # the instantiation is the same as the scenario, and it's true:
                #pass
                bn.cpt(s).set(I, 1)
            elif i_dict == d_f:
                #pass
                bn.cpt(s).set(I,0)
            else:
                if i_dict[s] == "False":
                    bn.cpt(s).set(I,1)
                else:
                    bn.cpt(s).set(I,0)
            I.inc()
    return bn

def fix_01_with_scns_as_parents(bn, scns, csv, type_bn):
    # extract scenarios from csv
    df = pd.read_csv(csv)

    for node in bn.names():
        if node not in ["constraint", "aux"]:
            if node not in scns:
                I = gum.Instantiation(bn.cpt(node))
                while not I.end():
                    i_dict = I.todict(withLabels=True)
                    for s in scns:
                            if i_dict[s] == "True" and bn.cpt(node).get(I) != 0.5:
                                if bn.cpt(node).get(I) > 0.5:
                                    x= 0
                                else:
                                    x=1
                                d_false = copy.deepcopy(i_dict)
                                d_false[node] = "False"
                                J = gum.Instantiation(I)
                                bn.cpt(node).set(J, 1-x)
                                J.fromdict(d_false)
                                bn.cpt(node).set(J, abs(0-x))


                    I.inc()
            else:
                if type_bn in "CLI":
                    I = gum.Instantiation(bn.cpt(node))
                    while not I.end():
                        i_dict = I.todict(withLabels=True)
                        for s in scns:
                            if s in i_dict.keys() and s != node:
                                if i_dict[s] == "True" and bn.cpt(node).get(I) != 0.5:
                                    d_false = copy.deepcopy(i_dict)
                                    d_false[node] = "False"
                                    J = gum.Instantiation(I)
                                    bn.cpt(node).set(J, 0)
                                    J.fromdict(d_false)
                                    bn.cpt(node).set(J, 1)
                        I.inc()


    '''for s in scns:
        row = df[df[s] == True].iloc[0]
        d = {col:str(row[col]) for col in df.columns if col not in scns}
        I = gum.Instantiation(bn.cpt(s))
        d_t = copy.deepcopy(d)
        d_t[s] = 'True'
        d_f = copy.deepcopy(d)
        d_f[s] = 'False'
        while not I.end():
            i_dict = I.todict(withLabels=True)
            if i_dict == d_t:   # the instantiation is the same as the scenario, and it's true:
                #pass
                bn.cpt(s).set(I, 1)
            elif i_dict == d_f:
                #pass
                bn.cpt(s).set(I,0)
            else:
                if i_dict[s] == "False":
                    bn.cpt(s).set(I,1)
                else:
                    bn.cpt(s).set(I,0)
            I.inc()
    return bn'''


def add_constraint_node(bn, expl_scns):
    '''bn.add(gum.LabelizedVariable('constraint', 'constraint', 3))
    bn.addArc('scn1', 'constraint')
    bn.addArc('scn2', 'constraint')
    bn.cpt("constraint")[{'scn1': 0, "scn2": 0}] = [0.5, 0.5, 0]
    bn.cpt("constraint")[{'scn1': 1, "scn2": 0}] = [1, 0, 0]
    bn.cpt("constraint")[{'scn1': 0, "scn2": 1}] = [0, 1, 0]
    bn.cpt("constraint")[{'scn1': 1, "scn2": 1}] = [0, 0, 1]'''

    outcomes = copy.deepcopy(expl_scns)
    scns = copy.deepcopy(expl_scns)

    outcomes.append("NA")
    bn.add(gum.LabelizedVariable('aux', 'aux', outcomes))
    for scn in scns:
        bn.addArc(scn, 'aux')

    truth_table = generate_truth_table(scns)
    labels = assign_labels(truth_table, "VLEK")
    for row, label in zip(truth_table, labels):
        bn.cpt("aux")[row] = label

    bn.add(gum.LabelizedVariable("constraint", 'constraint', ["False", "True"]))
    bn.addArc('aux', 'constraint')
    sum_prob = 0

    for s in scns:
        p = bn.cpt(s)[1]
        bn.cpt("constraint")[{"aux": s}] = [p, 1 - p]
        sum_prob += p
    bn.cpt("constraint")[{'aux': "NA"}] = [1, 0]

    # exception dict 0.1, 0.1, ... 0
    d = {}
    # constraint node cpt like in thesis (both scn false is 50/50)
    '''for scn in expl_scns:
        d[scn] = 0
    l  = [1/len(expl_scns)] * len(expl_scns)    #
    l.append(0)'''

    # case where no exhaustive (new virtual vlek soluition 8 nov)
    # constraint node like in 2014 article / fenton 2011
    '''for scn in expl_scns:
        d[scn] = 0
    l = [0] * len(expl_scns)
    l.append(1)

    bn.cpt("constraint")[d] = l'''
    return bn


def generate_truth_table(scenarios):
    # Generate all combinations of True and False for the given number of scenarios
    truth_table = [dict(zip(scenarios, values)) for values in iter.product([True, False], repeat=len(scenarios))]
    return truth_table


def assign_labels(truth_table, bn_type):
    labels = []

    i = 0 # number of extra states
    if bn_type == "VLEK" or bn_type == "FENTONALL":
        # we only want NA
        i = 1
    else:
        i = 2
    for row in truth_table:

        label = [0] * (len(row) + i)  # Initialize the label list with all 0s, one extra for "leak" and one extra for 'NA'

        true_count = sum(row.values())  # Count the number of True values

        if bn_type == "FENTONLEAK":
            if true_count == 1:
                # If exactly one True value exists, set the corresponding index to 1
                index = list(row.values()).index(True)
                label[index] = 1

            elif true_count == 0:
                # if none of the parents are true, set the leak state to 1
                label[-2] = 1

            else:
                # If more than one True, set the last index (for 'NA') to 1
                label[-1] = 1
        elif bn_type == "VLEK" or bn_type == "FENTONALL":
            if true_count == 1:
                # If exactly one True value exists, set the corresponding index to 1
                index = list(row.values()).index(True)
                label[index] = 1
            else:
                # If none or more than one True, set the last index (for 'NA') to 1
                label[-1] = 1
        else:
            print("bn type not implemented bn_tools 198")
            exit()

        labels.append(label)

    return labels

def add_fenton_constraint_node(bn, expl_scn):
    if len(expl_scn) < 3:
        # we are considering only two alternative scenarios
        l_s = "2alt"
    else:
        l_s = "all"

    outcomes = copy.deepcopy(expl_scn)
    scns = copy.deepcopy(expl_scn)
    if l_s == "2alt":
        outcomes.append("leak")

    outcomes.append("NA")
    bn.add(gum.LabelizedVariable('aux', 'aux', outcomes))
    for scn in scns:
        bn.addArc(scn, 'aux')

    truth_table = generate_truth_table(scns)
    if l_s == "2alt":
        labels = assign_labels(truth_table, "FENTONLEAK")
    else:
        labels = assign_labels(truth_table, "FENTONALL")



    for row, label in zip(truth_table, labels):
        bn.cpt("aux")[row] = label

    bn.add(gum.LabelizedVariable("constraint", 'constraint', ["False", "True"]))
    bn.addArc('aux', 'constraint')
    sum_prob = 0

    for s in expl_scn:
        p =  bn.cpt(s)[1]
        bn.cpt("constraint")[{"aux":s}] = [p, 1-p]
        sum_prob += p

    if l_s == "2alt":
        bn.cpt("constraint")[{"aux":"leak"}] = [sum_prob, 1-sum_prob]

    bn.cpt("constraint")[{'aux': "NA"}] = [1, 0]

    #print(bn.cpt("aux"))
    #print(bn.cpt("constraint"))

    #gum.saveBN(bn, f"alley/bn/FENTONTest.net")
    return bn


def add_quotes(input_string):
    # Remove the parentheses from the string
    elements = input_string.split()

    # Add quotes around each element
    quoted_elements = [f'\"{elem}\"' for elem in elements]

    # Join the quoted elements back into a string with spaces
    output_string = f"({' '.join(quoted_elements)} );\n"

    return output_string

def convert_networks_to_hugin(name):
    #print(name)
    with open(f"{name}.net", 'r') as file:
        lines = file.readlines()

    # Remove the first three lines
    lines[2] = f"name = \"bn\";\n"

    for i, line in enumerate(lines):
        if "states" in line:
            s = line.split("=")[1]
            s_i = s.replace("(", "")
            s_i = s_i.replace(");", "")
            new_s = add_quotes(s_i)
            lines[i] = line.replace(s, new_s)

    with open(f"{name}hugin.net", 'w') as file:
        file.writelines(lines)





