"""
MAIN FILE in the "BA" Neuronal Network Repositoy

CALL BY:    <main.py>

RETURN:     -

INFO:       This Git-Repository holds all the Code written for my Bachelor Thesis at the Chair of Automatic Control at TF Uni Kiel.
            The Programming Language is Python.

            In this Main-Function, modules are being called and reviewed.
"""

from modules import random_search as rs
from modules import visiualize as vs
from modules import inspect

def main():
    #date, best_reward_s = rs.main(1000) # Calling the RANDOM SEARCH Module to calculate new matrices with x Episodes
    #best_reward_s = str(int(best_reward_s))

    # Calls the calculated Matrices-----------------------------------------------------------------------

    # RANDOM SEARCH
    #call_matrices = "parameter_dumps/" + date + "_rs_" + best_reward_s + ".p"

    # GENETIC ALGORITHMS
    #...

    #-----------------------------------------------------------------------------------------------------

    # REWARD: 30
    #call_matrices = "result_matrices.p"

    #REWARD: 32
    call_matrices = "parameter_dumps/20180716_16-53-19_result_matrices_reward_32.p"

    vs.main(call_matrices) # Callig the VISIUALIZATION Module to show the newly learned paramteter matrices

    # INSPECT FUNCTION------------------------------------------------------------------------------------
    #inspect.main(call_matrices)



if __name__=="__main__":
    main()
