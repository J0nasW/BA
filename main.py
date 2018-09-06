"""
MAIN FILE in the "BA" Neuronal Network Repositoy

CALL BY:    <main.py>

RETURN:     -

INFO:       This Git-Repository holds all the Code written for my Bachelor Thesis at the Chair of Automatic Control at TF Uni Kiel.
            The Programming Language is Python.

            In this Main-Function, modules are being called and reviewed.
"""

from modules import random_search as rs
from modules import random_search_v2 as rs2
from modules import weights_nn as w
from modules import genetic_algorithm as ga
from modules import visiualize as vs
from modules import inspect_nn as ins
from modules import parameters

import easygui as eg

def start():
    global choices

    msg = "Willkommen im TW Circuit20 Si10mulator. Bitte wählen Sie eine Option:"
    title = "TW Circuit Simulator"
    choices = ["Lokale Simulation","Parameter Dump öffnen","Parameter mit Weight Dump öffnen","Dumps inspizieren"]
    choice = eg.choicebox(msg, title, choices)

    return choice

def local_simulation():
    msg = "Wählen Sie eine der unten aufgeführten lokalen Simulationen:"
    title = "TW Circuit Simulator - Local Simulation"
    choices = ["RandomSearch_v2","RandomSearch_v2 mit Weight-Optimization","Genetic Algorithm"]
    choice = eg.choicebox(msg, title, choices)

    if choice == choices[0]:
        local_rs()
    elif choice == choices[1]:
        local_rs_w()
    elif choice == choices[2]:
        local_ga()

def local_rs():
    fieldmsg = "Legen Sie die Dauer der lokalen Simulation fest. Bei keiner Eingabe wird die Simulation nicht durchgeführt."
    fieldtitle = "TW Circuit - RandomSearch_v2"
    fieldNames = ["Dauer der Parametersimulation (in Sek.):","Dauer der Visualisierung"]
    fieldValues = [60,10]
    fieldValues = eg.multenterbox(fieldmsg, fieldtitle, fieldNames)

    date, best_reward_p = rs2.main(int(fieldValues[0]))
    parameter_matrices = parameters.current_dir + "/parameter_dumps/" + date + "_rs2_v2_" + best_reward_p + ".hkl"
    vs.main(parameter_matrices, int(fieldValues[1])) # Callig the VISIUALIZATION Module to show the newly learned paramteter matrices

def local_rs_w():
    fieldmsg = "Legen Sie die Dauer der lokalen Simulation fest. Bei keiner Eingabe wird die Simulation nicht durchgeführt."
    fieldtitle = "TW Circuit - RandomSearch_v2 with Weights"
    fieldNames = ["Dauer der Parametersimulation (in Sek.):","Dauer der Gewichtungsoptimierung: (in Sek.)","Dauer der Visualisierung"]
    fieldValues = [60,60,10]
    fieldValues = eg.multenterbox(fieldmsg, fieldtitle, fieldNames)

    date, best_reward_p = rs2.main(int(fieldValues[0]))
    parameter_matrices = parameters.current_dir + "/parameter_dumps/" + date + "_rs2_v2_" + best_reward_p + ".hkl"
    date, best_reward_w = w.main(int(fieldValues[1]), parameter_matrices, best_reward_p)
    if best_reward_p <= best_reward_w:
        weight_matrices = parameters.current_dir + "/weight_dumps/" + date + "_" + best_reward_w + ".hkl"
        vs.main_with_weights(parameter_matrices, weight_matrices, int(fieldValues[2])) # Callig the VISIUALIZATION Module to show the newly learned paramteter matrices
    else:
        eg.msgbox(msg="Weight Run Failed!")
        vs.main(parameter_matrices, int(fieldValues[2])) # Callig the VISIUALIZATION Module to show the newly learned paramteter matrices

def local_ga():
    fieldmsg = "Legen Sie die Dauer der lokalen Simulation fest. Bei keiner Eingabe wird die Simulation nicht durchgeführt."
    fieldtitle = "TW Circuit - Genetic Algorithm"
    fieldNames = ["Dauer der Parametersimulation (in Sek.):","Plot der Lernkurven erwünscht? <0=NEIN,1=JA>","Dauer der Visualisierung"]
    fieldValues = [60,10]
    fieldValues = eg.multenterbox(fieldmsg, fieldtitle, fieldNames)

    date, best_reward_p = ga.main(int(fieldValues[0]),int(fieldValues[1]))
    parameter_matrices = parameters.current_dir + "/parameter_dumps/" + date + "_ga_" + best_reward_p + ".hkl"
    vs.main(parameter_matrices, int(fieldValues[2])) # Callig the VISIUALIZATION Module to show the newly learned paramteter matrices


def load_parameter():
    msg = "Parameter Dump auswählen oder Best-Score-Visualisierung"
    choices = ["Parameter auswählen","Beste Visualisierug"]
    reply = eg.buttonbox(msg, choices=choices)

    if reply == choices[0]:
        parameter_dir = eg.fileopenbox()
        vis_time = eg.integerbox("Dauer der Simulation (in Sek.):","TW Circuit")
        vs.main(parameter_dir, int(vis_time))
    elif reply == choices[1]:
        parameter_dir = "parameter_dumps/20180817_01-56-01_rs2_v2_131.hkl"
        vis_time = eg.integerbox("Dauer der Simulation (in Sek.):","TW Circuit")
        vs.main(parameter_dir, int(vis_time))

def load_parameter_weights():
    eg.msgbox(msg="Wählen Sie zuerst den Parameter-Dump.")
    parameter_dir = eg.fileopenbox()
    eg.msgbox(msg="Wählen Sie nun den entsprechenden Weight-Dump.")
    weight_dir = eg.fileopenbox()
    vis_time = eg.integerbox("Dauer der Simulation (in Sek.):","TW Circuit")
    vs.main_with_weights(parameter_dir, weight_dir, int(vis_time))

def inspect():

    choices = ["Parameter-Dump", "Weight-Dump"]
    file = eg.buttonbox("Welche Datei wollen Sie inspizieren?", choices=choices)

    if file == choices[0]:
        dir = eg.fileopenbox()
        ins.parameters(dir)
    elif file == choices[1]:
        dir = eg.fileopenbox()
        ins.weights(dir)

def main():
    global parameter_matrices, weight_matrices

    choice = start()

    if choice == choices[0]:
        local_simulation()
    elif choice == choices[1]:
        load_parameter()
    elif choice == choices[2]:
        load_parameter_weights()
    elif choice == choices[3]:
        inspect()

if __name__=="__main__":
    main()
