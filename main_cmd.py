"""
MAIN FILE in the "BA" Neuronal Network Repositoy - FOR CMD USE WITHOUT GUI AND VISIUALIZATION

CALL BY:    <main_cmd.py>

RETURN:     -

INFO:       This Git-Repository holds all the Code written for my Bachelor Thesis at the Chair of Automatic Control at TF Uni Kiel.
            The Programming Language is Python.

            In this Main-Function, modules are being called and reviewed.
"""

from modules import random_search as rs
from modules import random_search_v2 as rs2
from modules import weights_nn as w
from modules import inspect_nn as ins
from modules import parameters

def main():
    global parameter_matrices, weight_matrices

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
    date, best_reward_p = rs2.main(sim_time_parameters)
    parameter_matrices = parameters.current_dir + "/parameter_dumps/" + date + "_rs2_v2_" + best_reward_p + ".hkl"

    # WEIGHT APPLICATION (RandomSearch)
    date, best_reward_w = w.main(sim_time_weights, parameter_matrices, best_reward_p)


if __name__=="__main__":
    main()
