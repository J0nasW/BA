"""
Neuronal Network of C. Elegans implemented in Python with the LIF-Model
(by JW)

New Circuit Edition (SIM-CE)

Version: I don't know - Still in work!
"""

# Some dependencies
import numpy as np
import matplotlib.pyplot as plt
import gym

from lif import I_syn_calc, I_gap_calc, U_neuron_calc

"""
Parameters for the neural Network

Motor Neurons: FWD, REV
Sensory Neurons: PVD, PLM, AVM, ALM
Inter Neurons: AVA, AVD, PVC, DVA, AVB
"""

# Treshold
v = -20 #mV

# Variables
mu = -40 #mV - Sigmoid mu
E_ex = 0 #mV
E_in = -70 #mV
Default_U_leak = -70 #mV

# Time Constants:
t0 = t = 0
T = 10
delta_t = 0.01

# Making Contact with Neurons through Synapses and Gap-Junctions----------------------------

# A = Connections between Neurons with excitatory Nature (E = 0mV)
A_in = np.matrix('0 0 0 1 1 0 1 0; 0 0 0 1 1 0 0 0; 0 0 0 1 0 0 0 1; 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0; 0 0 0 1 1 0 1 0; 0 0 0 1 1 1 0 1; 0 0 0 0 0 0 1 0')
A_ex = np.matrix('0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0; 0 0 0 0 1 1 1 1; 0 0 0 1 0 1 0 0; 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0')
A_gap = np.matrix('0 0 0 1 0 1 0 0; 0 0 1 0 0 0 0 0; 0 0 0 0 1 0 0 0; 1 0 1 0 0 0 1 0; 0 0 0 0 0 0 0 0; 1 0 0 0 0 0 1 0; 0 0 0 0 0 1 0 0; 0 0 0 0 0 0 0 0')



#-------------------------------------------------------------------------------------------

# Parameter Matrix--------------------------------------------------------------------------

# Weights (or Number of same Synapses) per Synapse (n)
w_in = np.matrix('0 0 0 25 62.5 0 50 0; 0 0 0 75 12 0 0 0; 0 0 0 100 0 0 0 125; 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0; 0 0 0 37.5 125 0 250 0; 0 0 0 25 50 62 0 25; 0 0 0 0 37.5 0 75 0')
w_ex = np.matrix('0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0; 0 0 0 0 325 100 425 725; 0 0 0 25 0 0 1400 0; 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0')
w_gap = np.matrix('0 0 0 120 0 120 0 0; 0 0 120 0 0 0 0 0; 0 0 0 0 120 0 0 0; 120 0 120 0 0 0 600 0; 0 0 0 0 0 0 0 0; 120 0 0 0 0 0 60 0; 0 0 0 600 0 60 0 0; 0 0 0 0 0 0 0 0')

# For Synapses and Gap-Junctions
G_syn = np.multiply(np.ones((8,8)), 2.3) # G_syn [mS] Parameter for Neurons - Sweep from 0.1 to 1 mS/cm^2

V_range = np.multiply(np.ones((8,8)), 3) # V_range [mV] Parameter for Neurons - Sweep from 3 to 6 mV

V_shift = np.multiply(np.ones((8,8)), -30) # V_shift [mV] Parameter for Neurons - Sweep from -10 to -40mV

#-------------------------------------------------------------------------------------------


# Current Matrix----------------------------------------------------------------------------

# Current Matrix for Symapses between Neurons
I_syn = np.zeros((8,8))

# Current Matrix for Gap-Junctions between Neurons
I_gap = np.zeros((8,8))

#-------------------------------------------------------------------------------------------

# Voltage Matrix----------------------------------------------------------------------------

x = [0, 0, 0, 0, 0, 0, 0, 0] #PLM, ALM, AVM, PVC, AVD, LUA, AVA, AVB

#-------------------------------------------------------------------------------------------

# State Matrix------------------------------------------------------------------------------

fire = [0, 0, 0, 0, 0, 0, 0, 0] #PLM, ALM, AVM, PVC, AVD, LUA, AVA, AVB

#-------------------------------------------------------------------------------------------

# Initialization----------------------------------------------------------------------------

def initialize(Default_U_leak):

    # Initializing Neurons and Sensors------------------------------------------------------
    for i in range(0,8):
        x[i] = Default_U_leak

    global PLM, ALM, AVM, PVC, AVD, LUA, AVA, AVB, AVA_spike, AVB_spike

    PLM = np.array([Default_U_leak])
    ALM = np.array([Default_U_leak])
    AVM = np.array([Default_U_leak])
    PVC = np.array([Default_U_leak])
    AVD = np.array([Default_U_leak])

    LUA = np.array([])
    AVA = np.array([])
    AVB = np.array([])

    AVA_spike = np.array([0])
    AVB_spike = np.array([0])
    #---------------------------------------------------------------------------------------

    # Initializing OpenAI Environments------------------------------------------------------
    #env = gym.make('CartPole-v0')
    #env.reset()
    #---------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------

# Compute Function--------------------------------------------------------------------------

def compute(u, w_in_mat, w_sin_mat, sig_in_mat, sig_sin_mat, C_m_mat, G_leak_mat, U_leak_mat):

    # Compute all Synapse Currents in this network------------------------------------------

    for i in range(0,8):
        for j in range (0,8):
            # Synapse Currents between Interneurons
            if A_ex[i, j] == 1:
                # Excitatory Synapse
                I_syn[i, j] = I_syn_calc(x[i], x[j], E_ex, w_in_mat[i, j], sig_in_mat[i, j], mu)
            elif A[i, j] == o:
                I_syn[i, j] = 0

                #HIER WEITER MACHEN !!!!!!!!

            # Gap-Junction Currents between Interneurons
            if A_gap[i, j] == 1:
                # There is a Gap-Junctions
                I_g_inter[i, j] = I_gap_calc(x[i], x[j], w_gap_in_mat[i, j])
            else:
                I_g_inter[i, j] = 0

    for i in range(0,3):
        for j in range(0,4):
            # Synapse Currents between Sensory and Interneurons
            if B[i, j] == 1:
                # Inhibitory Synapse (can't be Excitatory)
                I_s_sensor[i, j] = I_syn_calc(u[i], u[j], E_in, w_sin_mat[i, j], sig_sin_mat[i, j], mu)
            else:
                I_s_sensor[i, j] = 0

            # Gap-Junction Currents between Sensory and Interneurons
            if B_gap[i, j] == 1:
                # There is a Gap-Junctions
                I_g_sensor[i, j] = I_gap_calc(x[i], x[j], w_gap_sin_mat[i, j])
            else:
                I_g_sensor[i, j] = 0

    #---------------------------------------------------------------------------------------

    # Now compute inter Neurons Voltages----------------------------------------------------
    for i in range(0,5):
        I_syn = I_s_inter.sum(axis = 0) # Creates a 1x5 Array with the Sum of all Columns
        I_gap = I_g_inter.sum(axis = 0) # Creates a 1x5 Array with the Sum of all Columns
        x[i], fire[i] = U_neuron_calc(x[i], I_syn[i], I_gap[i], C_m_mat[0,i], G_leak_mat[0,i], U_leak_mat[0,i], v, delta_t)
        #x[i], fire[i] = U_neuron_calc(x[i], I_syn[i], 0, C_m_mat[0,i], G_leak_mat[0,i], U_leak_mat[0,i], v, delta_t)

    #---------------------------------------------------------------------------------------

    return x, u, fire

#-------------------------------------------------------------------------------------------

# Append Function---------------------------------------------------------------------------

def arr(x, u, fire):

    global AVA, AVD, PVC, DVA, AVB, PVD, PLM, AVM, ALM, AVA_spike, AVB_spike

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

#-------------------------------------------------------------------------------------------

# Plot Function-----------------------------------------------------------------------------

def plot():
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



#-------------------------------------------------------------------------------------------
# Main Function-----------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------

def main():
    global u

    initialize(Default_U_leak) # Initializing all Interneurons with the desired leakage voltage

    for t in np.arange(t0,T,delta_t):
        x, u, fire = compute(u, w_in_mat, w_sin_mat, sig_in_mat, sig_sin_mat, C_m_mat, G_leak_mat, U_leak_mat) # Compute the next Interneuron Voltages along with a possible "fire" Event
        arr(x, u, fire) # Storing Information for graphical analysis
        u[1] = u[1] + 0.05
        u[3] = u[3] + 0.03

    plot() # Plotting everyting using matplotlib

#-------------------------------------------------------------------------------------------




if __name__=="__main__":
    main()
