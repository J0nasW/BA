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

def main():

    runtime = 5 # in sec.

    # RANDOM SEARCH
    #date, best_reward_s = rs.main(10000) # Calling the RANDOM SEARCH Module to calculate new matrices with x Episodes
    #parameter_matrices = "parameter_dumps/" + date + "_rs_" + best_reward_s + ".p"
    #vs.main(parameter_matrices) # Callig the VISIUALIZATION Module to show the newly learned paramteter matrices

    # RANDOM SEARCH V2
    #date, best_reward_s = rs2.main(2000)
    #parameter_matrices = "parameter_dumps/" + date + "_rs2_v2_" + best_reward_s + ".hkl"
    parameter_matrices = "parameter_dumps/20180801_15-04-54_rs2_v2_47.hkl" # GUTER SATZ
    #vs.main(parameter_matrices) # Callig the VISIUALIZATION Module to show the newly learned paramteter matrices

    # WEIGHT APPLICATION (RandomSearch)
    #date, best_reward_s = w.main(2000, parameter_matrices)
    #weight_matrices = "weight_dumps/" + date + "_" + best_reward_s + ".hkl"
    weight_matrices = "weight_dumps/20180801_15-19-16_134.hkl" # GUTER SATZ
    #vs.main_with_weights(parameter_matrices, weight_matrices, runtime) # Callig the VISIUALIZATION Module to show the newly learned paramteter matrices

    # GENETIC ALGORITHMS
    #...
    #-----------------------------------------------------------------------------------------------------

    # INSPECT FUNCTION------------------------------------------------------------------------------------
    # Parameters of RandomSearch Module:
    inspect.main(parameter_matrices)

    # Parameters of Weight Module:
    inspect.weights(weight_matrices)


if __name__=="__main__":
    main()
