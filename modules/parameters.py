"""
PARAMETERS for the Neural network

IMPORT BY:  <parameters.py>

RETURN:     Several Parameters

INFO:       All Parameters can be changed only here and spread over the whole Project
            Motor Neurons: FWD, REV
            Sensory Neurons: PVD, PLM, AVM, ALM
            Inter Neurons: AVA, AVD, PVC, AVB
"""

import numpy as np

# Neural Network----------------------------------------------------------------------------
# Making Contact with Neurons through Synapses and Gap-Junctions----------------------------

# A = Connections between Interneurons through Synapses
A = np.matrix('0 0 0 -1; 1 0 0 0; 0 0 0 1; -1 0 0 0') # AVA, AVD, PVC, AVB - 4 Synapses (1 = Ex., -1 = Inh.)

# B = Connections between Sensory- and Interneurons through Synapses
B = np.matrix('0 1 3 0; 1 1 3 2; 2 3 1 1; 0 3 1 0') # PVD, PLM, AVM, ALM (1 = Inh., 2 = Ex., 3 = Gap)

#-------------------------------------------------------------------------------------------

# Current Matrix----------------------------------------------------------------------------

# Current Matrix for Symapses between inter Neurons
I_s_inter = np.zeros((4,4))
# Current Matrix for Synapses between sensory and inter Neurons
I_s_sensor = np.zeros((4,4))

# Current Matrix for Gap-Junctions between inter Neurons
I_g_inter = np.zeros((4,4))
# Current Matrix for Gap-Junctions between sensory and inter Neurons
I_g_sensor = np.zeros((4,4))

#-------------------------------------------------------------------------------------------

# Voltage Matrix----------------------------------------------------------------------------

x = [0,0,0,0] #AVA, AVD, PVC, AVB
u = [0,0,0,0] #PVD, PLM, AVM, ALM

#-------------------------------------------------------------------------------------------

# State Matrix------------------------------------------------------------------------------

fire = [0,0,0,0] #AVA, AVD, PVC, AVB

#-------------------------------------------------------------------------------------------



# Substantial Parameters/Constants----------------------------------------------------------

# Treshold
v = -20 #mV

# Variables
mu = -40 #mV - Sigmoid mu
E_ex = 0 #mV
E_in = -70 #mV
Default_U_leak = -70 #mV

# Time Constants:
t0 = t = 0
T = 2
delta_t = 0.01

#--------------------------------------------------------------------------------------------
