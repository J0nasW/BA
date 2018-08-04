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

    w_A_rnd = result[0]
    w_B_rnd = result[1]
    w_B_gap_rnd = result[2]
    sig_A_rnd = result[3]
    sig_B_rnd = result[4]
    C_m_rnd = result[5]
    G_leak_rnd = result[6]
    U_leak_rnd = result[7]

    print ("w_A =", w_A_rnd)
    print ("w_B =", w_B_rnd)
    print ("w_B_gap = ", w_B_gap_rnd)
    print ("sigma_A =", sig_A_rnd)
    print ("sigma_B =", sig_B_rnd)
    print ("C_m = ", C_m_rnd)
    print ("G_leak = ", G_leak_rnd)
    print ("U_leak = ", U_leak_rnd)

def weights(load_matrices):
    result = hkl.load(load_matrices)

    A_rnd = result[0]
    B_rnd = result[1]

    print ("Weights of A =", A_rnd)
    print ("Weights of B =", B_rnd)
