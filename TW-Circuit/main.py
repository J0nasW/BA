"""
Neuronal Network of C. Elegans implemented in Python with the LIF-Model
(by JW)

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
T = 5
delta_t = 0.001

# Making Contact with Neurons through Synapses and Gap-Junctions----------------------------

# A = Connections between Interneurons through Synapses
A = np.matrix('0 -1 -1 0 -1; 1 0 1 0 1; 1 1 0 0 1; -1 0 -1 0 -1; -1 -1 0 0 0')
A_rnd = np.random.rand(5,5)
A_in = np.multiply(A, A_rnd)

# B = Connections between Sensory- and Interneurons through Synapses
B = np.matrix('1 0 1 1 0; 1 1 0 1 0; 0 0 1 0 1; 0 1 1 0 0')
B_rnd = np.random.rand(4,5)
B_in = np.multiply(B, B_rnd)


# A_gap = Connections between Interneurons through Gap-Junctions
A_gap = np.matrix('0 0 1 0 0; 0 0 0 0 0; 1 0 0 1 0; 0 0 1 0 1; 0 0 0 1 0')
A_gap_rnd = np.random.rand(5,5)
A_gap_in = np.multiply(A_gap, A_gap_rnd)

# B = Connections between Sensory- and Interneurons through Gap-Junctions
B_gap = np.matrix('0 0 0 0 0; 0 0 1 0 0; 0 0 0 0 0; 0 1 0 0 0')
B_gap_rnd = np.random.rand(4,5)
B_gap_in = np.multiply(B_gap, B_gap_rnd)

#-------------------------------------------------------------------------------------------

# Parameter Matrix--------------------------------------------------------------------------

# For Synapses
w_in_mat = np.multiply(np.ones((5,5)), 0.7) # w [S] Parameter for Synapses between inter Neurons - Sweep from 0S to 3S
w_sin_mat = np.multiply(np.ones((4,5)), 0.7) # w [S] Parameter for Synapses between sensory and inter Neurons - Sweep from 0S to 3S

sig_in_mat = np.multiply(np.ones((5,5)), 0.2) # sigma Parameter for Synapses between inter Neurons - Sweep from 0.05 - 0.5
sig_sin_mat = np.multiply(np.ones((4,5)), 0.2) # sigma Parameter for Synapses between sensory and inter Neurons - Sweep from 0.05 - 0.5

# For Gap-Junctions
w_gap_in_mat = np.multiply(np.ones((5,5)), 0.7) # w [S] Parameter for Gap-Junctions between inter Neurons - Sweep from 0S to 3S
w_gap_sin_mat = np.multiply(np.ones((5,5)), 0.7) # w [S] Parameter for Gap-Junctions sensory and inter Neurons - Sweep from 0S to 3S

# For Neurons
C_m_mat = np.multiply(np.ones((1,5)), 1) # C_m [F] Parameter for Neurons - Sweep from 1mF to 1F
#C_m_mat = np.matrix('0.1 1 1 1 0.1')

G_leak_mat = np.multiply(np.ones((1,5)), 1.3) # G_leak [S] Parameter for Neurons - Sweep from 50mS to 5S
#G_leak_mat = np.matrix('4 2.3 2.3 2.3 4')

U_leak_mat = np.multiply(np.ones((1,5)), -70) # U_leak [mV] Parameter for Neurons - Sweep from -90mV to 0mV

#-------------------------------------------------------------------------------------------


# Current Matrix----------------------------------------------------------------------------

# Current Matrix for Symapses between inter Neurons
I_s_inter = np.zeros((5,5))
# Current Matrix for Synapses between sensory and inter Neurons
I_s_sensor = np.zeros((4,5))

# Current Matrix for Gap-Junctions between inter Neurons
I_g_inter = np.zeros((5,5))
# Current Matrix for Gap-Junctions between sensory and inter Neurons
I_g_sensor = np.zeros((4,5))

#-------------------------------------------------------------------------------------------

# Voltage Matrix----------------------------------------------------------------------------

x = [0,0,0,0,0] #AVA, AVD, PVC, DVA, AVB
u = [0,0,0,0] #PVD, PLM, AVM, ALM

#-------------------------------------------------------------------------------------------

# State Matrix------------------------------------------------------------------------------

fire = [0,0,0,0,0] #AVA, AVD, PVC, DVA, AVB

#-------------------------------------------------------------------------------------------

# Initialization----------------------------------------------------------------------------

def initialize(Default_U_leak):

    # Initializing Neurons and Sensors------------------------------------------------------
    for i in range(0,5):
        x[i] = Default_U_leak
    for i in range(0,4):
        u[i] = Default_U_leak

    global AVA, AVD, PVC, DVA, AVB, PVD, PLM, AVM, ALM, AVA_spike, AVB_spike, I_PVC, I_DVA, I_AVD, I_AVA, I_AVB

    AVA = np.array([Default_U_leak])
    AVD = np.array([Default_U_leak])
    PVC = np.array([Default_U_leak])
    DVA = np.array([Default_U_leak])
    AVB = np.array([Default_U_leak])

    PVD = np.array([])
    PLM = np.array([])
    AVM = np.array([])
    ALM = np.array([])

    AVA_spike = np.array([])
    AVB_spike = np.array([])

    I_PVC = np.array([])
    I_DVA = np.array([])
    I_AVD = np.array([])
    I_AVA = np.array([])
    I_AVB = np.array([])
    #---------------------------------------------------------------------------------------

    # Initializing OpenAI Environments------------------------------------------------------
    #env = gym.make('CartPole-v0')
    #env.reset()
    #---------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------

# Compute Function--------------------------------------------------------------------------

def compute(x, u):

    # Compute all Synapse Currents in this network------------------------------------------

    for i in range(0,5):
        for j in range (0,5):
            # Synapse Currents between Interneurons
            if A[i, j] == 1:
                # Excitatory Synapse
                I_s_inter[i, j] = I_syn_calc(x[i], x[j], E_ex, w_in_mat[i, j], sig_in_mat[i, j], mu)
            elif A[i, j] == -1:
                # Inhibitory Synapse
                I_s_inter[i, j] = I_syn_calc(x[i], x[j], E_in, w_in_mat[i, j], sig_in_mat[i, j], mu)
            else:
                I_s_inter[i, j] = 0

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
        I_syn_inter = I_s_inter.sum(axis = 0) # Creates a 1x5 Array with the Sum of all Columns
        I_gap_inter = I_g_inter.sum(axis = 0)
        I_syn_stimuli = I_s_sensor.sum(axis = 0)
        I_gap_stimuli = I_g_sensor.sum(axis = 0)
        x[i], fire[i] = U_neuron_calc(x[i], I_syn_inter[i], I_gap_inter[i], I_syn_stimuli[i], I_gap_stimuli[i], C_m_mat[0,i], G_leak_mat[0,i], U_leak_mat[0,i], v, delta_t)
        #x[i], fire[i] = U_neuron_calc(x[i], I_syn[i], 0, C_m_mat[0,i], G_leak_mat[0,i], U_leak_mat[0,i], v, delta_t)

    #---------------------------------------------------------------------------------------

    I_syn = np.add(I_syn_inter, I_syn_stimuli)
    I_gap = np.add(I_gap_inter, I_gap_stimuli)

    return x, u, fire, I_syn, I_gap

#-------------------------------------------------------------------------------------------

# Append Function---------------------------------------------------------------------------

def arr(x, u, fire, I_all):

    global AVA, AVD, PVC, DVA, AVB, PVD, PLM, AVM, ALM, AVA_spike, AVB_spike, I_PVC, I_DVA, I_AVD, I_AVA, I_AVB

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

    I_AVA = np.append(I_AVA, I_all[0])
    I_AVD = np.append(I_AVD, I_all[1])
    I_PVC = np.append(I_PVC, I_all[2])
    I_DVA = np.append(I_DVA, I_all[3])
    I_AVB = np.append(I_AVB, I_all[4])

#-------------------------------------------------------------------------------------------

# Plot Function-----------------------------------------------------------------------------

def plot():

    plt.figure(1)
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
    plt.xlabel('t in [s]')
    plt.ylabel('u(t) in [mV]')
    plt.legend(loc='upper left')


    plt.figure(2)
    plt.suptitle('Neuron Currents', fontsize=16)

    plt.subplot(321)
    plt.title('PVC', fontsize=10)
    plt.plot(I_PVC, '-r', label='PVC', linewidth=1)
    plt.subplot(322)
    plt.title('DVA', fontsize=10)
    plt.plot(I_DVA, '-r', label='DVA', linewidth=1)
    plt.subplot(323)
    plt.title('AVD', fontsize=10)
    plt.plot(I_AVD, '-r', label='AVD', linewidth=1)
    plt.subplot(324)
    plt.title('AVA', fontsize=10)
    plt.plot(I_AVA, '-r', label='AVA', linewidth=1)
    plt.subplot(325)
    plt.title('AVB', fontsize=10)
    plt.plot(I_AVB, '-r', label='AVB', linewidth=1)
    plt.xlabel('t')
    plt.ylabel('i(t) in [mA]')


    plt.show()

#-------------------------------------------------------------------------------------------



#-------------------------------------------------------------------------------------------
# Main Function-----------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------

def main():
    global x, u

    initialize(Default_U_leak) # Initializing all Interneurons with the desired leakage voltage
    #u = [-20, -40, -40, -20]
    for t in np.arange(t0,T,delta_t):
        x, u, fire, I_syn, I_gap = compute(x, u) # Compute the next Interneuron Voltages along with a possible "fire" Event
        I_all = np.add(I_syn, I_gap)
        arr(x, u, fire, I_all) # Storing Information for graphical analysis
        #u[1] = u[1] + 0.04
        #u[0] = u[0] + 0.05

    print AVA_spike
    print AVB_spike

    plot() # Plotting everyting using matplotlib

#-------------------------------------------------------------------------------------------




if __name__=="__main__":
    main()
