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
from modules import visiualize as vs
from modules import inspect_nn as ins
from modules import parameters

import easygui as eg

def start():
    global choices

    msg = "Willkommen im TW Circuit Simulator. Bitte wählen Sie eine Option:"
    title = "TW Circuit Simulator"
    choices = ["Lokale Simulation","Parameter Dump öffnen","Parameter mit Weight Dump öffnen","Dumps inspizieren"]
    choice = eg.choicebox(msg, title, choices)

    return choice

def local_simulation():
    fieldmsg = "Legen Sie die Dauer der lokalen Simulation fest. Bei keiner Eingabe wird die Simulation nicht durchgeführt."
    fieldtitle = "TW Circuit - Simulationsdauer"
    fieldNames = ["Dauer der Parametersimulation (in Sek.):","Dauer der Gewichtungsoptimierung: (in Sek.)","Dauer der Visualisierung"]
    fieldValues = [60,60,10]
    fieldValues = eg.multenterbox(fieldmsg, fieldtitle, fieldNames)

    if fieldValues[1] == "":
        date, best_reward_p = rs2.main(int(fieldValues[0]))
    else:
        date, best_reward_p = rs2.main(int(fieldValues[0]))
        parameter_matrices = parameters.current_dir + "/parameter_dumps/" + date + "_rs2_v2_" + best_reward_p + ".hkl"
        date, best_reward_w = w.main(int(fieldValues[1]), parameter_matrices, best_reward_p)
        if best_reward_p <= best_reward_w:
            weight_matrices = parameters.current_dir + "/weight_dumps/" + date + "_" + best_reward_w + ".hkl"
            vs.main_with_weights(parameter_matrices, weight_matrices, int(fieldValues[2])) # Callig the VISIUALIZATION Module to show the newly learned paramteter matrices
        else:
            eg.msgbox(msg="Weight Run Failed!")
            vs.main(parameter_matrices, int(fieldValues[2])) # Callig the VISIUALIZATION Module to show the newly learned paramteter matrices

def load_parameter():
    eg.msgbox(msg="Wählen Sie den Parameter-Dump.")
    parameter_dir = eg.fileopenbox()
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


    vis_runtime = 10 # in sec. - FOR VISIUALIZATION

    sim_time_parameters = 10 # FOR SIMULATION of Parameters
    sim_time_weights = 10 # FOR SIMULATION of additional Weights

    # Simulation time in HOURS ------------------------------
    #sim_time_parameters = sim_time_parameters * 60 * 60
    #sim_time_weights = sim_time_weights * 60 * 60
    #--------------------------------------------------------
    # Simulation time in MINUTES ----------------------------
    #sim_time_parameters = sim_time_parameters * 60
    #sim_time_weights = sim_time_weights * 60
    #--------------------------------------------------------

    # RANDOM SEARCH V2
    #date, best_reward_p = rs2.main(sim_time_parameters)
    #parameter_matrices = parameters.current_dir + "/parameter_dumps/" + date + "_rs2_v2_" + best_reward_p + ".hkl"
    #parameter_matrices = parameters.current_dir + "/parameter_dumps/20180806_03-25-01_rs2_v2_95.hkl"
    #parameter_matrices = parameters.current_dir + "/parameter_dumps/20180806_12-15-01_rs2_v2_48.hkl" #Bisher bester Satz - zum filmen geeignet
    #parameter_matrices = parameters.current_dir + "/parameter_dumps/20180812_03-00-01_rs2_v2_122.hkl" #Bisher hoechster Satz
<<<<<<< HEAD
=======


    # Simulation 15.08.2018 - symmetrical Parameters
    #parameter_matrices = parameters.current_dir + "/parameter_dumps/20180817_01-50-02_rs2_v2_158.hkl"
    #parameter_matrices = parameters.current_dir + "/parameter_dumps/20180817_01-52-01_rs2_v2_182.hkl"
    #parameter_matrices = parameters.current_dir + "/parameter_dumps/20180817_01-54-01_rs2_v2_104.hkl"
    #parameter_matrices = parameters.current_dir + "/parameter_dumps/20180817_01-56-01_rs2_v2_131.hkl" #Bisher hoechster Satz

    #weight_matrices = parameters.current_dir + "/weight_dumps/20180817_13-50-02_200.hkl"
    #weight_matrices = parameters.current_dir + "/weight_dumps/20180817_13-52-01_200.hkl"
    #weight_matrices = parameters.current_dir + "/weight_dumps/20180817_13-54-01_200.hkl"
    #weight_matrices = parameters.current_dir + "/weight_dumps/20180817_13-56-01_200.hkl"

    # Visiualize only the Parameter Simulation:
>>>>>>> symmetrical_parameters
    #vs.main(parameter_matrices, vis_runtime) # Callig the VISIUALIZATION Module to show the newly learned paramteter matrices

    # WEIGHT APPLICATION (RandomSearch)
    #date, best_reward_w = w.main(sim_time_weights, parameter_matrices, best_reward_p)
    #if best_reward_p <= best_reward_w:
    #    weight_matrices = parameters.current_dir + "/weight_dumps/" + date + "_" + best_reward_w + ".hkl"
    #    vs.main_with_weights(parameter_matrices, weight_matrices, vis_runtime) # Callig the VISIUALIZATION Module to show the newly learned paramteter matrices
    #else:
    #    vs.main(parameter_matrices, vis_runtime) # Callig the VISIUALIZATION Module to show the newly learned paramteter matrices

    #weight_matrices = parameters.current_dir + "/weight_dumps/20180806_09-25-01_163.hkl"
    #weight_matrices = parameters.current_dir + "/weight_dumps/20180806_15-15-01_167.hkl" #Bisher bester Satz - zum filmen geeignet
<<<<<<< HEAD
    #weight_matrices = parameters.current_dir + "/weight_dumps/20180812_09-00-01_170.hkl" #Bisher hoechster Satz
    #vs.main_with_weights(parameter_matrices, weight_matrices, vis_runtime) # Callig the VISIUALIZATION Module to show the newly learned paramteter matrices

    # INSPECT FUNCTION------------------------------------------------------------------------------------
    # Parameters of RandomSearch Module:
    #inspect.parameters(parameter_matrices)

    # Parameters of Weight Module:
    #inspect.weights(weight_matrices)
=======
    #weight_matrices = parameters.current_dir + "/weight_dumps/20180812_09-00-01_170.hkl" #Bisher höchster Satz
>>>>>>> symmetrical_parameters

    # Visiualize the Parameter and according Weight Simulation:
    #vs.main_with_weights(parameter_matrices, weight_matrices, vis_runtime) # Callig the VISIUALIZATION Module to show the newly learned paramteter matrices

if __name__=="__main__":
    main()
