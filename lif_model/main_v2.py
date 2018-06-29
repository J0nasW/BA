from __future__ import division # Because otherwise, "/" WILL NOT WORK!

"""
Neuronal Network of C. Elegans implemented in Python with the LIF-Model
(by JW)

Version 2 - New Formula in RK4
"""

# Some dependencies
import numpy as np
import matplotlib.pyplot as plt

from LIFModel import lif_sim_v2_rk4

"""
Parameters for the neural Network

Motor Neurons: FWD, REV
Sensory Neurons: PVD, PLM, AVM, ALM
Inter Neurons: AVA, AVD, PVC, DVA, AVB
"""

u_pre = 20 #mV
u_post = -70 #mV

# Treshold
v = -55 #mV

# Variables
C_m = 1 #mF - Sweep from 1mF to 1F
G_leak = 1148.5 #mS - Sweep from 50mS to 5S
U_leak = -90 #mV - Sweep from -90mV to 0mV
w = 1000 #mS - Sweep from 0S to 3S
sig = 0.05
mu = -40 #mV
E_ex = 0 #mV
E_in = -90 #mV

# Time Constants:
t0 = 0
T = 0.01
delta_t = 0.0001


def main():
    u_rk4_ex_1 = lif_sim_v2_rk4(u_pre, u_post, C_m, G_leak, U_leak, w, sig, mu, E_ex, t0, T, delta_t, v)

    u_rk4_in_2 = lif_sim_v2_rk4(u_pre, u_post, C_m, G_leak, U_leak, w, sig, mu, E_in, t0, T, delta_t, v)

    print len(u_rk4_ex_1)
    print len(u_rk4_in_2)

    u_diff = np.subtract(u_rk4_ex_1, np.reshape(u_rk4_in_2, 100))

    plt.suptitle('"Leaky-Integrate-and-Fire Model"', fontsize=16)

    plt.plot(u_rk4_ex_1, '-r', label='Excitatory Neuron 1', linewidth=1)
    plt.plot(u_rk4_in_2, '-b', label='Inhitory Neuron 2', linewidth=1)
    plt.plot(u_diff, '-g', label='Diff', linewidth=0.5)

    plt.xlabel('t (in ms)')
    plt.ylabel('u(t) in [mV]')
    plt.legend(loc='upper left')
    plt.legend

    plt.show()

if __name__== "__main__":
	main()
