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

# -  1 = Inhibitory  -  2 = Excitatory  -  3 = Gap-Junction  -

# A = Connections between Interneurons through Synapses
A = np.matrix('0 0 0 1; 2 0 2 0; 0 2 0 2; 1 0 0 0') # AVA, AVD, PVC, AVB
nbr_of_inter_synapses = 6
A_all = nbr_of_inter_synapses

# B = Connections between Sensory- and Interneurons through Synapses
B = np.matrix('0 1 1 0; 1 1 3 0; 0 3 1 1; 0 1 1 0') # PVD, PLM, AVM, ALM
nbr_of_sensor_synapses = 8
nbr_of_gap_junctions = 2
B_all = nbr_of_sensor_synapses + nbr_of_gap_junctions

nbr_of_sensor_neurons = 4
nbr_of_inter_neurons = 4

#-------------------------------------------------------------------------------------------

# Current Matrix----------------------------------------------------------------------------

# Current Matrix for Symapses between inter Neurons
I_s_inter = np.zeros((nbr_of_inter_neurons,nbr_of_inter_neurons))
# Current Matrix for Synapses between sensory and inter Neurons
I_s_sensor = np.zeros((nbr_of_sensor_neurons,nbr_of_inter_neurons))

# Current Matrix for Gap-Junctions between inter Neurons
I_g_inter = np.zeros((nbr_of_inter_neurons,nbr_of_inter_neurons))
# Current Matrix for Gap-Junctions between sensory and inter Neurons
I_g_sensor = np.zeros((nbr_of_sensor_neurons,nbr_of_inter_neurons))

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
