"""
INSPECT MODULE to show parameter Matrices

CALL BY:    <inspect.py>

RETURN:     Print of stored parameter Matrices

INFO:       -
"""

# Some dependencies
import numpy as np # Maths and stuff
import cPickle as pickle # Store Data into [.p] Files


def main(load_matrices):
    result = pickle.load( open(load_matrices, "r"))

    w_in_mat = result[0]
    w_sin_mat = result[1]
    sig_in_mat = result[2]
    sig_sin_mat = result[3]
    w_gap_in_mat = result[4]
    w_gap_sin_mat = result[5]
    C_m_mat = result[6]
    G_leak_mat = result[7]
    U_leak_mat = result[8]

    print w_in_mat, w_sin_mat, sig_in_mat, sig_sin_mat, w_gap_in_mat, w_gap_sin_mat, C_m_mat, G_leak_mat, U_leak_mat
