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
from modules import inspect_nn as inspect
from modules import parameters

def main():

    global parameter_matrices, weight_matrices

    vis_runtime = 5 # in sec. - FOR VISIUALIZATION

    sim_time_parameters = 2 # FOR SIMULATION of Parameters
    sim_time_weights = 2 # FOR SIMULATION of additional Weights
    # Simulation time in HOURS ----------------------------
    #sim_time_parameters = sim_time_parameters * 60 * 60
    #sim_time_weights = sim_time_weights * 60 * 60
    #------------------------------------------------------
    # Simulation time in MINUTES ----------------------------
    #sim_time_parameters = sim_time_parameters * 60
    #sim_time_weights = sim_time_weights * 60
    #------------------------------------------------------

    # RANDOM SEARCH
    #date, best_reward_s = rs.main(10000) # Calling the RANDOM SEARCH Module to calculate new matrices with x Episodes
    #parameter_matrices = parameters.current_dir + "/parameter_dumps/" + date + "_rs_" + best_reward_s + ".p"
    #vs.main(parameter_matrices) # Callig the VISIUALIZATION Module to show the newly learned paramteter matrices

    # RANDOM SEARCH V2
    #date, best_reward_p = rs2.main(sim_time_parameters)
    #parameter_matrices = parameters.current_dir + "/parameter_dumps/" + date + "_rs2_v2_" + best_reward_p + ".hkl"
    #parameter_matrices = parameters.current_dir + "/parameter_dumps/20180806_03-25-01_rs2_v2_95.hkl"
    parameter_matrices = parameters.current_dir + "/parameter_dumps/20180806_12-15-01_rs2_v2_48.hkl" #Bisher bester Satz - zum filmen geeignet
    #parameter_matrices = parameters.current_dir + "/parameter_dumps/20180812_03-00-01_rs2_v2_122.hkl" #Bisher bester Satz - zum filmen geeignet
    #vs.main(parameter_matrices) # Callig the VISIUALIZATION Module to show the newly learned paramteter matrices

    # WEIGHT APPLICATION (RandomSearch)
    #date, best_reward_w = w.main(sim_time_weights, parameter_matrices, best_reward_p)
    #if best_reward_p <= best_reward_w:
    #    weight_matrices = parameters.current_dir + "/weight_dumps/" + date + "_" + best_reward_w + ".hkl"
    #    vs.main_with_weights(parameter_matrices, weight_matrices, vis_runtime) # Callig the VISIUALIZATION Module to show the newly learned paramteter matrices
    #else:
    #    vs.main(parameter_matrices, vis_runtime) # Callig the VISIUALIZATION Module to show the newly learned paramteter matrices

    #weight_matrices = parameters.current_dir + "/weight_dumps/20180806_09-25-01_163.hkl"
    weight_matrices = parameters.current_dir + "/weight_dumps/20180806_15-15-01_167.hkl" #Bisher bester Satz - zum filmen geeignet
    #weight_matrices = parameters.current_dir + "/weight_dumps/20180812_09-00-01_170.hkl" #Bisher höchster Satz
    vs.main_with_weights(parameter_matrices, weight_matrices, vis_runtime) # Callig the VISIUALIZATION Module to show the newly learned paramteter matrices

    # INSPECT FUNCTION------------------------------------------------------------------------------------
    # Parameters of RandomSearch Module:
    #inspect.parameters(parameter_matrices)

    # Parameters of Weight Module:
    #inspect.weights(weight_matrices)


if __name__=="__main__":
    main()
