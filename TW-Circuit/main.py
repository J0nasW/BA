"""
Neuronal Network of C. Elegans implemented in Python with the LIF-Model
(by JW)

Version: I don't know - Still in work!
"""

# Some dependencies
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from lif import I_syn_calc, U_neuron_calc

"""
Parameters for the neural Network

Motor Neurons: FWD, REV
Sensory Neurons: PVD, PLM, AVM, ALM
Inter Neurons: AVA, AVD, PVC, DVA, AVB
"""

# Treshold
v = -50 #mV

# Variables
mu = -40 #mV - Sigmoid mu
E_ex = 0 #mV
E_in = -90 #mV
Default_U_leak = -70 #mV

# Time Constants:
t0 = t = 0
T = 10
delta_t = 0.1

# Making Contact with Neurons through Synapses----------------------------------------------

# A = Connections between Inter Neurons
A = np.matrix('0 -1 -1 0 -1; 1 0 1 0 1; 1 1 0 0 1; -1 0 -1 0 -1; -1 -1 0 0 0')
A_rnd = np.random.rand(5,5)
A_in = np.multiply(A, A_rnd)

# B = Connections between Sensory Neurons and Inter Neurons
B = np.matrix('1 0 1 1 0; 1 1 0 1 0; 0 0 1 0 1; 0 1 1 0 0')
B_rnd = np.random.rand(4,5)
B_in = np.multiply(B, B_rnd)

#-------------------------------------------------------------------------------------------

# Parameter Matrix--------------------------------------------------------------------------

# For Synapses
w_in_mat = np.multiply(np.ones((5,5)), 2) # w [S] Parameter for Synapses between inter Neurons - Sweep from 0S to 3S
w_sin_mat = np.multiply(np.ones((4,5)), 2) # w [S] Parameter for Synapses between sensory and inter Neurons - Sweep from 0S to 3S

sig_in_mat = np.multiply(np.ones((5,5)), 0.2) # sigma Parameter for Synapses between inter Neurons - Sweep from 0.05 - 0.5
sig_sin_mat = np.multiply(np.ones((4,5)), 0.2) # sigma Parameter for Synapses between sensory and inter Neurons - Sweep from 0.05 - 0.5


# For Neurons
C_m_mat = np.multiply(np.ones((1,5)), 0.1) # C_m [F] Parameter for Neurons - Sweep from 1mF to 1F

G_leak_mat = np.multiply(np.ones((1,5)), 2) # G_leak [S] Parameter for Neurons - Sweep from 50mS to 5S

U_leak_mat = np.multiply(np.ones((1,5)), -70) # U_leak [mV] Parameter for Neurons - Sweep from -90mV to 0mV

#-------------------------------------------------------------------------------------------


# Current Matrix----------------------------------------------------------------------------

# Current Matrix for Symapses between inter Neurons
I_s_inter = np.zeros((5,5))
#I_s_inter = np.matrix('5 8 9 2 12; 9 3 91 94 1; 3 1 44 2 3; 34 12 54 2 3; 3 43 5 23 2')
# Current Matrix for Synapses between sensory and inter Neurons
I_s_sensor = np.zeros((4,5))
#I_s_sensor = np.matrix('5 8 9 2 12; 9 3 91 94 1; 3 1 44 2 3; 34 12 54 2 3')

#-------------------------------------------------------------------------------------------

# Voltage Matrix----------------------------------------------------------------------------

x = [0,0,0,0,0] #AVA, AVD, PVC, DVA, AVB
u = [0,0,0,0] #PVD, PLM, AVM, ALM

#-------------------------------------------------------------------------------------------

# State Matrix------------------------------------------------------------------------------

fire = [0,0,0,0,0] #AVA, AVD, PVC, DVA, AVB

#-------------------------------------------------------------------------------------------

# Initialization--------------------------------------------------------------------

def initialize(Default_U_leak):
    for i in range(0,4):
        x[i] = Default_U_leak

    global u, x_arr, u, u_arr, AVA, AVD, PVC, DVA, AVB, PVD, PLM, AVM, ALM, AVA_spike, AVB_spike

    x_arr = ([])
    u_arr = ([])

    AVA = np.array([])
    AVD = np.array([])
    PVC = np.array([])
    DVA = np.array([])
    AVB = np.array([])

    PVD = np.array([])
    PLM = np.array([])
    AVM = np.array([])
    ALM = np.array([])

    AVA_spike = np.array([])
    AVB_spike = np.array([])


#---------------------------------------------------------------------------------

# Compute Function--------------------------------------------------------------------------

def compute(u, w_in_mat, w_sin_mat, sig_in_mat, sig_sin_mat, C_m_mat, G_leak_mat, U_leak_mat):

    # Compute all Synapse Currents in this network------------------------------------------

    for i in range(0,4):
        for j in range (0,4):
            #Synapse Currents between Interneurons
            if A[i, j] == 1:
                # Excitatory Synapse
                I_s_inter[i, j] = I_syn_calc(x[i], x[j], E_ex, w_in_mat[i, j], sig_in_mat[i, j], mu)
            elif A[i, j] == -1:
                # Inhibitory Synapse
                I_s_inter[i, j] = I_syn_calc(x[i], x[j], E_in, w_in_mat[i, j], sig_in_mat[i, j], mu)
            else:
                I_s_inter[i, j] == 0

    for i in range(0,3):
        for j in range(0,4):
            #Synapse Currents between Sensory and Interneurons
            if B[i, j] == 1:
                # Inhibitory Synapse (can't be Excitatory)
                I_s_sensor[i, j] = I_syn_calc(u[i], u[j], E_in, w_sin_mat[i, j], sig_sin_mat[i, j], mu)
            else:
                I_s_sensor[i, j] == 0

    #---------------------------------------------------------------------------------------

    # Now compute inter Neurons Voltages----------------------------------------------------
    for i in range(0,4):
        I_syn = I_s_inter.sum(axis = 0)
        I_gap = 0
        x[i], fire[i] = U_neuron_calc(x[i], I_syn[i], I_gap, C_m_mat[0,i], G_leak_mat[0,i], U_leak_mat[0,i], v, delta_t)

    #---------------------------------------------------------------------------------------

    return x, u, fire

#-------------------------------------------------------------------------------------------

# Main Function-----------------------------------------------------------------------------

def main():
    global u, x_arr, u, u_arr, AVA, AVD, PVC, DVA, AVB, PVD, PLM, AVM, ALM, AVA_spike, AVB_spike
    initialize(Default_U_leak)

    for t in np.arange(t0,T,delta_t):
        x, u, fire = compute(u, w_in_mat, w_sin_mat, sig_in_mat, sig_sin_mat, C_m_mat, G_leak_mat, U_leak_mat)

        AVA = np.append(AVA, x[0])
        AVD = np.append(AVD, x[1])
        PVC = np.append(PVC, x[2])
        DVA = np.append(DVA, x[3])
        AVB = np.append(AVB, x[4])

        PVD = np.append(PVD, u[0])
        PLM = np.append(PLM, u[1])
        AVM = np.append(AVM, u[2])
        ALM = np.append(ALM, u[3])

        AVA_spike = np.append(AVA_spike, fire[0]) # Reverse lokomotion
        AVB_spike = np.append(AVB_spike, fire[4]) # Forward lokomotion

    print AVA_spike
    print AVB_spike

    plt.suptitle('Leaky-Integrate-and-Fire Neuronal Network', fontsize=16)

    plt.subplot(121)
    plt.title('Sensory Neurons', fontsize=10)
    plt.plot(PVD, '-b', label='PVD', linewidth=1)
    plt.plot(PLM, '-y', label='PLM', linewidth=1)
    plt.plot(AVM, '-g', label='AVM', linewidth=1)
    plt.plot(ALM, '-r', label='ALM', linewidth=1)
    plt.xlabel('t (in s)')
    plt.ylabel('u(t) in [mV]')
    plt.legend(loc='upper left')

    plt.subplot(122)
    plt.title('Inter Neurons', fontsize=10)
    plt.plot(AVA, '-b', label='AVA', linewidth=1)
    plt.plot(AVD, '-y', label='AVD', linewidth=1)
    plt.plot(PVC, '-g', label='PVC', linewidth=1)
    plt.plot(DVA, '-r', label='DVA', linewidth=1)
    plt.plot(AVB, '-k', label='AVB', linewidth=1)
    plt.xlabel('t (in s)')
    plt.ylabel('u(t) in [mV]')
    plt.legend(loc='upper left')

    plt.show()

#-------------------------------------------------------------------------------------------



if __name__=="__main__":
    main()
