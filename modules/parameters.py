"""
PARAMETERS for the Neural network

IMPORT BY:  <parameters.py>

RETURN:     Several Parameters

INFO:       All Parameters can be changed only here and spread over the whole Project.
            Only the Dimensions and Connections of Matrices A and B
            as well as some substantial Parameters can be changed.
            Everything else will be calculated.
            Motor Neurons: FWD, REV
            Sensory Neurons: PVD, PLM, AVM, ALM
            Inter Neurons: AVA, AVD, PVC, AVB
"""

import os
import numpy as np

# Neural Network----------------------------------------------------------------------------
# Making Contact with Neurons through Synapses and Gap-Junctions----------------------------

# -  1 = Inhibitory  -  2 = Excitatory  -  3 = Gap-Junction  -

# Interneuron Matrix A:
#
#   ->  AVA  AVD  PVC  AVB
# AVA |  0    0    1    1  |
# AVD |  2    0    2    0  |
# PVC |  0    2    0    2  |
# AVB |  1    1    0    0  |

# A = Connections between Interneurons through Synapses
A = np.matrix('0 0 1 1; 2 0 2 0; 0 2 0 2; 1 1 0 0') # AVA, AVD, PVC, AVB

# Sensorneuron Matrix B:
#
#   ->  AVA  AVD  PVC  AVB
# PVD |  0    1    1    0  |
# PLM |  1    1    3    0  |
# AVM |  0    3    1    1  |
# ALM |  0    1    1    0  |

# B = Connections between Sensory- and Interneurons through Synapses
B = np.matrix('0 1 1 0; 1 1 3 0; 0 3 1 1; 0 1 1 0') # PVD, PLM, AVM, ALM

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

#-------------------------------------------------------------------------------------------

# Sone Calculations based on A and B -------------------------------------------------------

nbr_of_inter_synapses = np.count_nonzero(A)
nbr_of_inter_synapses_symm = int(np.count_nonzero(A) / 2)
nbr_of_sensor_synapses = (B == 1).sum()
nbr_of_sensor_synapses_symm = int((B == 1).sum() / 2)
nbr_of_gap_junctions = (B == 3).sum()
nbr_of_gap_junctions_symm = int((B == 3).sum() / 2)
A_all = np.count_nonzero(A)
A_all_symm = int(np.count_nonzero(A) / 2)
B_all = np.count_nonzero(B)
B_all_symm = int(np.count_nonzero(B) / 2)

nbr_of_sensor_neurons = np.shape(B)[1]
nbr_of_sensor_neurons_symm = int(np.shape(B)[1] / 2)
nbr_of_inter_neurons = np.shape(A)[1]
nbr_of_inter_neurons_symm = int(np.shape(A)[1] / 2)

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

# Misc------------------------------------------------------------------------------

current_dir = os.getcwd()

#-------------------------------------------------------------------------------------------
