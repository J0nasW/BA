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
v = -20 # [mV]

# Variables
E_ex = 0 #mV
E_in = -70 #mV
Default_U_leak = -70 #mV

# Time Constants:
t0 = t = 0
T = 100
delta_t = 1

# Making Contact with Neurons through Synapses and Gap-Junctions----------------------------

# A = Connections between Neurons with excitatory Nature (E = 0mV)
A_in = np.matrix('0 0 0 1 1 0 1 0; 0 0 0 1 1 0 0 0; 0 0 0 1 0 0 0 1; 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0; 0 0 0 1 1 0 1 0; 0 0 0 1 1 1 0 1; 0 0 0 0 0 0 1 0')
A_ex = np.matrix('0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0; 0 0 0 0 1 1 1 1; 0 0 0 1 0 1 0 0; 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0')
A_gap = np.matrix('0 0 0 1 0 1 0 0; 0 0 1 0 0 0 0 0; 0 0 0 0 1 0 0 0; 1 0 1 0 0 0 1 0; 0 0 0 0 0 0 0 0; 1 0 0 0 0 0 1 0; 0 0 0 0 0 1 0 0; 0 0 0 0 0 0 0 0')

#-------------------------------------------------------------------------------------------

# Parameter Matrix--------------------------------------------------------------------------

# Weights (or Number of same Synapses) per Synapse (n) - has to be normalized!
w_in = np.matrix('0 0 0 25 62.5 0 50 0; 0 0 0 75 12 0 0 0; 0 0 0 100 0 0 0 125; 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0; 0 0 0 37.5 125 0 250 0; 0 0 0 25 50 62 0 25; 0 0 0 0 37.5 0 75 0')
w_ex = np.matrix('0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0; 0 0 0 0 325 100 425 725; 0 0 0 25 0 0 1400 0; 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0')
w_gap = np.matrix('0 0 0 120 0 120 0 0; 0 0 120 0 0 0 0 0; 0 0 0 0 120 0 0 0; 120 0 120 0 0 0 600 0; 0 0 0 0 0 0 0 0; 120 0 0 0 0 0 60 0; 0 0 0 600 0 60 0 0; 0 0 0 0 0 0 0 0')

# For Synapses and Gap-Junctions
G_syn = np.multiply(np.ones((8,8)), 2.3) # G_syn [mS] Parameter for Neurons - Sweep from 0.1 to 1 mS/cm^2

G_gap = np.multiply(np.ones((8,8)), 1.02) # G_gap [mS] Parameter for Neurons - Sweep from 0.1 to 1 mS/cm^2

V_range = np.multiply(np.ones((8,8)), 3) # V_range [mV] Parameter for Neurons - Sweep from 3 to 6 mV

V_shift = np.multiply(np.ones((8,8)), -30) # V_shift [mV] Parameter for Neurons - Sweep from -10 to -40mV

# For Neurons
C_m = np.matrix('0.1111 0.1111 0.2 0.075 0.1111 0.0714 0.06667 0.0714')

G_leak = np.multiply(np.ones((1,8)), 0.0525) # G_leak [mS] Parameter for Neurons - Sweep from 0.04 to 1 mS/cm^2

V_leak = np.multiply(np.ones((1,8)), -70) # V_leak [mV] Parameter for Neurons - Sweep from -90 to 0 mV

#-------------------------------------------------------------------------------------------


# Current Matrix----------------------------------------------------------------------------

# Current Matrix for Symapses between Neurons
I_syn_ex = np.zeros((8,8))
I_syn_in = np.zeros((8,8))

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
    LUA = np.array([Default_U_leak])
    AVA = np.array([Default_U_leak])
    AVB = np.array([Default_U_leak])

    AVA_spike = np.array([0])
    AVB_spike = np.array([0])
    #---------------------------------------------------------------------------------------

    # Initializing OpenAI Environments------------------------------------------------------
    #env = gym.make('CartPole-v0')
    #env.reset()
    #---------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------

# Compute Function--------------------------------------------------------------------------

def compute():

    # Compute all Synapse Currents in this network------------------------------------------

    for i in range(0,7):
        for j in range (0,7):
            # Synapse Currents between Interneurons
            if A_ex[i, j] == 1:
                # Excitatory Synapse
                I_syn_ex[i, j] = I_syn_calc(x[i], x[j], E_ex, w_ex[i, j], G_syn[i, j], V_shift[i, j], V_range[i, j])
            else:
                I_syn_ex[i, j] = 0

            if A_in[i, j] == 1:
                # Inhibitory Synapse
                I_syn_in[i, j] = I_syn_calc(x[i], x[j], E_in, w_ex[i, j], G_syn[i, j], V_shift[i, j], V_range[i, j])
            else:
                I_syn_in[i, j] = 0

            # Gap-Junction Currents between Interneurons
            if A_gap[i, j] == 1:
                # There is a Gap-Junctions
                I_gap[i, j] = I_gap_calc(x[i], x[j], w_gap[i, j], G_gap[i, j])
            else:
                I_gap[i, j] = 0

    #---------------------------------------------------------------------------------------

    # Now compute inter Neurons Voltages----------------------------------------------------
    for i in range(0,7):
        I_syn = np.add(I_syn_in, I_syn_ex) # Addition of inhitory and excitatory Synapse-Currents
        I_s = I_syn.sum(axis = 0) # Creates a 1x8 Array with the Sum of all Columns
        I_g = I_gap.sum(axis = 0) # Creates a 1x8 Array with the Sum of all Columns
        x[i], fire[i] = U_neuron_calc(x[i], I_s[i], I_g[i], C_m[0, i], G_leak[0, i], V_leak[0, i], v, delta_t)

    #---------------------------------------------------------------------------------------

    return x, fire

#-------------------------------------------------------------------------------------------

# Append Function---------------------------------------------------------------------------

def arr(x, fire):

    global PLM, ALM, AVM, PVC, AVD, LUA, AVA, AVB, AVA_spike, AVB_spike

    PLM = np.append(PLM, x[0])
    ALM = np.append(ALM, x[1])
    AVM = np.append(AVM, x[2])
    PVC = np.append(PVC, x[3])
    AVD = np.append(AVD, x[4])
    LUA = np.append(LUA, x[5])
    AVA = np.append(AVA, x[6])
    AVB = np.append(AVB, x[7])

    AVA_spike = np.append(AVA_spike, fire[6]) # Reverse lokomotion
    AVB_spike = np.append(AVB_spike, fire[7]) # Forward lokomotion

#-------------------------------------------------------------------------------------------

# Plot Function-----------------------------------------------------------------------------

def plot():
    plt.suptitle('Leaky-Integrate-and-Fire Neuronal Network', fontsize=16)

    plt.subplot(221)
    plt.title('Sensory Neurons', fontsize=10)
    plt.plot(PLM, '-b', label='PLM', linewidth=1)
    plt.plot(ALM, '-y', label='ALM', linewidth=1)
    plt.plot(AVM, '-g', label='AVM', linewidth=1)
    plt.xlabel('t (in s)')
    plt.ylabel('u(t) in [mV]')
    plt.legend(loc='upper left')

    plt.subplot(222)
    plt.title('Inter Neurons', fontsize=10)
    plt.plot(PVC, '-b', label='PVC', linewidth=1)
    plt.plot(AVD, '-y', label='AVD', linewidth=1)
    plt.plot(LUA, '-g', label='LUA', linewidth=1)
    plt.xlabel('t (in s)')
    plt.ylabel('u(t) in [mV]')
    plt.legend(loc='upper left')

    plt.subplot(223)
    plt.title('Fire Neurons', fontsize=10)
    plt.plot(AVA, '-b', label='AVA', linewidth=1)
    plt.plot(AVB, '-y', label='AVB', linewidth=1)
    plt.xlabel('t (in s)')
    plt.ylabel('u(t) in [mV]')
    plt.legend(loc='upper left')

    plt.show()

#-------------------------------------------------------------------------------------------



#-------------------------------------------------------------------------------------------
# Main Function-----------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------

def main():
    initialize(Default_U_leak) # Initializing all Interneurons with the desired leakage voltage

    for t in np.arange(t0,T,delta_t):
        x, fire = compute() # Compute the next Interneuron Voltages along with a possible "fire" Event
        arr(x, fire) # Storing Information for graphical analysis

    plot() # Plotting everyting using matplotlib

#-------------------------------------------------------------------------------------------




if __name__=="__main__":
    main()
