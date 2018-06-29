"""
Neuronal Network of C. Elegans implemented in Python with the LIF-Model
(by JW)

Version: I don't know - Still in work!
"""

# Some dependencies
import numpy as np
import matplotlib.pyplot as plt

from lif import lif_sim

"""
Parameters for the neural Network

Motor Neurons: FWD, REV
Sensory Neurons: PVD, PLM, AVM, ALM
Inter Neurons: AVA, AVD, PVC, DVA, AVB
"""

# Treshold
v = -50 #mV

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
t0 = t = 0
T = 10
delta_t = 1

Sensor_read = [1, 2, 8, 13, 21, 26, 32, 36, 43, 56] #in mV
u_post = U_leak
u_plot = np.array([-90])

def main():
    global u_plot
    global u_post
    for i in range (0, 10):
        u_1 = lif_sim(20, -70, C_m, G_leak, U_leak, w, sig, mu, E_ex, t0, T, delta_t, v)
        print u_1
        u_plot = np.append(u_plot, u_1)


    plt.suptitle('"Leaky-Integrate-and-Fire Model"', fontsize=16)

    plt.plot(u_plot, '-r', label='Excitatory Neuron', linewidth=1)
    plt.xlabel('t (in s)')
    plt.ylabel('u(t) in [mV]')
    plt.legend(loc='upper left')
    plt.legend

    plt.show()

if __name__=="__main__":
    main()
