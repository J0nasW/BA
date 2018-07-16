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

def main():
    #date, best_reward_s = rs.main(1000)
    #best_reward_s = str(int(best_reward_s))

    #Calls the calculated Matrices
    #call_matrices = "parameter_dumps/" + date + "_result_matrices_reward_" + best_reward_s + ".p"

    #REWARD: 30
    #call_matrices = "result_matrices.p"

    #REWARD: 32
    call_matrices = "parameter_dumps/20180716_16-53-19_result_matrices_reward_32.p"

    vs.main(call_matrices)



if __name__=="__main__":
    main()
