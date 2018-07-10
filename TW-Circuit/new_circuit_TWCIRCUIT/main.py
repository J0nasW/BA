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
T = 10
delta_t = 0.01

# Making Contact with Neurons through Synapses and Gap-Junctions----------------------------

# A = Connections between Neurons through excitatory Synapses
A_ex = np.matrix('0 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0 0; 1 0 0 0 0 0 1 1 1; 0 0 0 0 0 0 0 0 0; 0 0 0 0 1 0 0 1 1; 0 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0 0')
A_ex_rnd = np.random.rand(9,9)
A_ex_w = np.multiply(A_ex, A_ex_rnd)

# A = Connections between Neurons through inhibitory Synapses
A_in = np.matrix('0 0 0 0 1 1 0 1 0; 0 0 0 0 0 1 1 1 0; 0 0 0 0 1 0 0 0 1; 0 0 0 0 1 0 1 0 0; 0 0 0 0 0 0 0 0 0; 0 0 0 0 1 0 0 1 1; 0 0 0 0 0 0 0 0 0; 0 0 0 0 1 0 1 0 1; 0 0 0 0 0 0 1 1 0')
A_in_rnd = np.random.rand(9,9)
A_in_w = np.multiply(A_in, A_in_rnd)

# A = Connections between Neurons through Gap-Junctions
A_gap = np.matrix('0 0 0 0 0 0 0 0 0; 0 0 0 0 1 0 0 0 0; 0 0 0 1 0 0 1 0 0; 0 0 1 0 0 0 0 0 0; 0 1 0 0 0 1 0 1 0; 0 0 0 0 1 0 0 0 1; 0 0 1 0 0 0 0 0 0; 0 0 0 0 1 0 0 0 0; 0 0 0 0 0 1 0 0 0')
A_gap_rnd = np.random.rand(9,9)
A_gap_w = np.multiply(A_gap, A_gap_rnd)

#-------------------------------------------------------------------------------------------

# Parameter Matrix--------------------------------------------------------------------------

# For Synapses
w_in_mat = np.multiply(np.ones((9,9)), 1.7) # w [S] Parameter for Synapses between inter Neurons - Sweep from 0S to 3S

sig_in_mat = np.multiply(np.ones((9,9)), 0.2) # sigma Parameter for Synapses between inter Neurons - Sweep from 0.05 - 0.5
# For Gap-Junctions
w_gap_in_mat = np.multiply(np.ones((9,9)), 1.7) # w [S] Parameter for Gap-Junctions between inter Neurons - Sweep from 0S to 3S

# For Neurons
C_m_mat = np.multiply(np.ones((1,9)), 0.5) # C_m [F] Parameter for Neurons - Sweep from 1mF to 1F

G_leak_mat = np.multiply(np.ones((1,9)), 2.3) # G_leak [S] Parameter for Neurons - Sweep from 50mS to 5S

U_leak_mat = np.multiply(np.ones((1,9)), -70) # U_leak [mV] Parameter for Neurons - Sweep from -90mV to 0mV

#-------------------------------------------------------------------------------------------


# Current Matrix----------------------------------------------------------------------------

# Current Matrix for Symapses between Neurons with inhibitory Synapses
I_in = np.zeros((9,9))

# Current Matrix for Symapses between Neurons with excitatory Synapses
I_ex = np.zeros((9,9))

# Current Matrix for Gap-Junctions between Neurons
I_gap = np.zeros((9,9))

#-------------------------------------------------------------------------------------------

# Voltage Matrix----------------------------------------------------------------------------

x = [0,0,0,0,0] #PVC, DVA, AVD, AVA, AVB
u = [0,0,0,0] #PVD, PLM, AVM, ALM

neurons = np.append(u, x)

#-------------------------------------------------------------------------------------------

# State Matrix------------------------------------------------------------------------------

fire = [0,0,0,0,0,0,0,0,0] #PVD, PLM, AVM, ALM, PVC, DVA, AVD, AVA, AVB

#-------------------------------------------------------------------------------------------

# Initialization----------------------------------------------------------------------------

def initialize(Default_U_leak):

    # Initializing Neurons and Sensors------------------------------------------------------
    for i in range(0,5):
        x[i] = Default_U_leak
    for i in range(0,4):
        u[i] = Default_U_leak

    global PVD, PLM, AVM, ALM, PVC, DVA, AVD, AVA, AVB, AVA_spike, AVB_spike

    PVD = np.array([Default_U_leak])
    PLM = np.array([Default_U_leak])
    AVM = np.array([Default_U_leak])
    ALM = np.array([Default_U_leak])

    PVC = np.array([Default_U_leak])
    DVA = np.array([Default_U_leak])
    AVD = np.array([Default_U_leak])
    AVA = np.array([Default_U_leak])
    AVB = np.array([Default_U_leak])

    AVA_spike = np.array([])
    AVB_spike = np.array([])
    #---------------------------------------------------------------------------------------

    # Initializing OpenAI Environments------------------------------------------------------
    #env = gym.make('CartPole-v0')
    #env.reset()
    #---------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------

# Compute Function--------------------------------------------------------------------------

def compute():

    # Compute all Synapse Currents in this network------------------------------------------

    for i in range(0,8):
        for j in range (0,8):
            # Synapse Currents between Interneurons
            if A_ex[i, j] == 1:
                # Excitatory Synapse
                I_ex[i, j] = I_syn_calc(neurons[i], neurons[j], E_ex, w_in_mat[i, j], sig_in_mat[i, j], mu)
            else:
                I_ex[i, j] = 0

            if A_in[i, j] == 1:
                # Inhibitory Synapse
                I_in[i, j] = I_syn_calc(neurons[i], neurons[j], E_in, w_in_mat[i, j], sig_in_mat[i, j], mu)
            else:
                I_in[i, j] = 0

            # Gap-Junction Currents between Interneurons
            if A_gap[i, j] == 1:
                # There is a Gap-Junctions
                I_gap[i, j] = I_gap_calc(neurons[i], neurons[j], w_gap_in_mat[i, j])
            else:
                I_gap[i, j] = 0

    #---------------------------------------------------------------------------------------

    # Now compute inter Neurons Voltages----------------------------------------------------
    for i in range(0,8):
        I_syn = np.add(I_in, I_ex) # Addition of inhitory and excitatory Synapse-Currents
        I_s = I_syn.sum(axis = 0) # Creates a 1x9 Array with the Sum of all Columns
        I_g = I_gap.sum(axis = 0) # Creates a 1x9 Array with the Sum of all Columns
        neurons[i], fire[i] = U_neuron_calc(neurons[i], I_s[i], I_g[i], C_m_mat[0,i], G_leak_mat[0,i], U_leak_mat[0,i], v, delta_t)

    #---------------------------------------------------------------------------------------

    u = neurons[:4]
    x = neurons[4:]

    print u
    print x

    return x, u, fire

#-------------------------------------------------------------------------------------------

# Append Function---------------------------------------------------------------------------

def arr(x, u, fire):

    global PVD, PLM, AVM, ALM, PVC, DVA, AVD, AVA, AVB, AVA_spike, AVB_spike

    PVD = np.append(PVD, u[0])
    PLM = np.append(PLM, u[1])
    AVM = np.append(AVM, u[2])
    ALM = np.append(ALM, u[3])

    PVC = np.append(PVC, x[0])
    DVA = np.append(DVA, x[1])
    AVD = np.append(AVD, x[2])
    AVA = np.append(AVA, x[3])
    AVB = np.append(AVB, x[4])

    AVA_spike = np.append(AVA_spike, fire[7]) # Reverse lokomotion
    AVB_spike = np.append(AVB_spike, fire[8]) # Forward lokomotion

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
        x, u, fire = compute() # Compute the next Interneuron Voltages along with a possible "fire" Event
        arr(x, u, fire) # Storing Information for graphical analysis

    plot() # Plotting everyting using matplotlib

#-------------------------------------------------------------------------------------------




if __name__=="__main__":
    main()
