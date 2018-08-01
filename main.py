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
from modules import inspect

def main():

    runtime = 8 # in sec.

    # RANDOM SEARCH
    #date, best_reward_s = rs.main(10000) # Calling the RANDOM SEARCH Module to calculate new matrices with x Episodes
    #parameter_matrices = "parameter_dumps/" + date + "_rs_" + best_reward_s + ".p"
    #vs.main(parameter_matrices) # Callig the VISIUALIZATION Module to show the newly learned paramteter matrices

    # RANDOM SEARCH V2
<<<<<<< HEAD
    load_parameters = "parameter_dumps/20180731_09-32-23_rs2_34.p"
    #date, best_reward_s = rs2.main(1000)
    #call_matrices = "parameter_dumps/" + date + "_rs2_" + best_reward_s + ".p"
    vs.main(load_parameters) # Callig the VISIUALIZATION Module to show the newly learned paramteter matrices

    # WEIGHT APPLICATION (RandomSearch)
    #load_parameters = "parameter_dumps/20180730_14-29-29_rs2_69.p"

    #date, best_reward_s = w.main(100000, load_parameters)
    #weight_matrices = "weight_dumps/" + date + "_" + best_reward_s + ".p"
    #weight_matrices = "weight_dumps/20180731_10-05-45_120.p"
    #weight_matrices = "weight_dumps/20180731_10-16-46_141.p"
    #vs.main_with_weights(load_parameters, weight_matrices) # Callig the VISIUALIZATION Module to show the newly learned paramteter matrices
=======
    #date, best_reward_s = rs2.main(10000)
    #parameter_matrices = "parameter_dumps/" + date + "_rs2_v2_" + best_reward_s + ".p"
    parameter_matrices = "parameter_dumps/20180731_16-29-19_rs2_v2_78.p" # GUTER SATZ
    #vs.main(parameter_matrices) # Callig the VISIUALIZATION Module to show the newly learned paramteter matrices

    # WEIGHT APPLICATION (RandomSearch)
    #date, best_reward_s = w.main(10000, parameter_matrices)
    #weight_matrices = "weight_dumps/" + date + "_" + best_reward_s + ".p"
    weight_matrices = "weight_dumps/20180731_17-32-58_168.p" # GUTER SATZ
    vs.main_with_weights(parameter_matrices, weight_matrices, runtime) # Callig the VISIUALIZATION Module to show the newly learned paramteter matrices
>>>>>>> NeuralCircuit_v2

    # GENETIC ALGORITHMS
    #...

    #-----------------------------------------------------------------------------------------------------


    # REWARD: 30
    #parameter_matrices = "parameter_dumps/result_matrices.p"

    #REWARD: 32
    #parameter_matrices = "parameter_dumps/20180716_16-53-19_result_matrices_reward_32.p"

    #REWARD: 69
    #parameter_matrices = "parameter_dumps/20180730_14-29-29_rs2_69.p"

    #vs.main(parameter_matrices) # Callig the VISIUALIZATION Module to show the newly learned paramteter matrices

    # INSPECT FUNCTION------------------------------------------------------------------------------------
    # Parameters of RandomSearch Module:
    #inspect.main(parameter_matrices)

    # Parameters of Weight Module:
    #inspect.weights(weight_matrices)


if __name__=="__main__":
    main()
