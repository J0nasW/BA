"""
INSPECT MODULE to show parameter Matrices

CALL BY:    <inspect.py>

RETURN:     Print of stored parameter Matrices

INFO:       -
"""

# Some dependencies
import numpy as np # Maths and stuff
import hickle as hkl


def parameters(parameter_matrices):
    result = hkl.load(parameter_matrices)

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

def weights(weight_matrices):
    result = hkl.load(weight_matrices)

    A_rnd = result[0]
    B_rnd = result[1]

    print ("Weights of A =", A_rnd)
    print ("Weights of B =", B_rnd)

def visiualize_inspect(parameter_matrices_1, weight_matrices_1, parameter_matrices_2, weight_matrices_2):
    parameter_matrices_1 = hkl.load(parameter_matrices_1)
    parameter_matrices_2 = hkl.load(parameter_matrices_2)
    weight_matrices_1 = hkl.load(weight_matrices_1)
    weight_matrices_2 = hkl.load(weight_matrices_2)

    w_A_1 = parameter_matrices_1[0]
    w_B_1 = parameter_matrices_1[1]
    w_B_gap_1 = parameter_matrices_1[2]
    sig_A_1 = parameter_matrices_1[3]
    sig_B_1 = parameter_matrices_1[4]
    C_m_1 = parameter_matrices_1[5]
    G_leak_1 = parameter_matrices_1[6]
    U_leak_1 = parameter_matrices_1[7]

    w_A_2 = parameter_matrices_2[0]
    w_B_2 = parameter_matrices_2[1]
    w_B_gap_2 = parameter_matrices_2[2]
    sig_A_2 = parameter_matrices_2[3]
    sig_B_2 = parameter_matrices_2[4]
    C_m_2 = parameter_matrices_2[5]
    G_leak_2 = parameter_matrices_2[6]
    U_leak_2 = parameter_matrices_2[7]

    A_rnd_1 = weight_matrices_1[0]
    B_rnd_1 = weight_matrices_1[1]

    A_rnd_2 = weight_matrices_2[0]
    B_rnd_2 = weight_matrices_2[1]

    print ("Neuron Changes:")
    print ("AVA_old:", Cm_1[0], G_leak_1[0], U_leak_1[0])
    print ("AVA_new:", Cm_2[0], G_leak_2[0], U_leak_2[0])
    print ("AVD_old:", Cm_1[1], G_leak_1[1], U_leak_1[1])
    print ("AVD_new:", Cm_2[1], G_leak_2[1], U_leak_2[1])
    print ("PVC_old:", Cm_1[2], G_leak_1[2], U_leak_1[2])
    print ("PVC_new:", Cm_2[2], G_leak_2[2], U_leak_2[2])
    print ("AVB_old:", Cm_1[3], G_leak_1[3], U_leak_1[3])
    print ("AVB_new:", Cm_2[3], G_leak_2[3], U_leak_2[3])

    print("Synapse Changes:")
    print ("Inter_Synapse_old:", w_A_1, sig_A_1)
    print ("Inter_Synapse_new:", w_A_2, sig_A_2)
    print ("Sensory_Synapse_old:", w_B_1, sig_B_1)
    print ("Sensory_Synapse_new:", w_A_2, sig_A_2)
