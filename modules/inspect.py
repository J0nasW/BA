"""
INSPECT MODULE to show parameter Matrices

CALL BY:    <inspect.py>

RETURN:     Print of stored parameter Matrices

INFO:       -
"""

# Some dependencies
import numpy as np # Maths and stuff
import hickle as hkl


def main(load_matrices):
    result = hkl.load(load_matrices)

    w_in_mat = result[0]
    w_sin_mat = result[1]
    sig_in_mat = result[2]
    sig_sin_mat = result[3]
    w_gap_in_mat = result[4]
    w_gap_sin_mat = result[5]
    C_m_mat = result[6]
    G_leak_mat = result[7]
    U_leak_mat = result[8]

    print ("w_A =", w_in_mat)
    print ("w_A_gap =", w_gap_in_mat)
    print ("w_B =", w_sin_mat)
    print ("w_B_gap = ", w_gap_sin_mat)
    print ("sigma_A =", sig_in_mat)
    print ("sigma_B =", sig_sin_mat)
    print ("C_m = ", C_m_mat)
    print ("G_leak = ", G_leak_mat)
    print ("U_leak = ", U_leak_mat)

def weights(load_matrices):
    result = hkl.load(load_matrices)

    A_rnd = result[0]
    B_rnd = result[1]
    B_gap_rnd = result[2]

    print ("Weights of A =", A_rnd)
    print ("Weights of B =", B_rnd)
    print ("Weights of B (Gap-Junctions) =", B_gap_rnd)
