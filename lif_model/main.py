from __future__ import division # Because otherwise, "/" WILL NOT WORK!

"""
Neuronal Network of C. Elegans implemented in Python with the LIF-Model
(by JW)
"""

# Some dependencies
import numpy as np
import matplotlib.pyplot as plt

from LIFModel import lif_sim_rk2, lif_sim_rk4, lif_sim_euler, lif_sim_analytical

"""
Parameters for the neural Network

Motor Neurons: FWD, REV
Sensory Neurons: PVD, PLM, AVM, ALM
Inter Neurons: AVA, AVD, PVC, DVA, AVB
"""

R_PVD = 10 # Original: 9,4 GOhm - 9400000000
C_PVD = 0.2 # Original: 16 pF - 0.000000000016
tau_PVD = R_PVD * C_PVD

# Treshold
v = 4

# Some other details
u = 0
u_rest = 0
I_0 = 0.5  # 35mA - Constant input current

# Time Constants:
t0 = 0
T = 10
delta_t = 0.001

# Counter Values
t = 0
i = 0


def main():
    u_PVD = lif_sim_analytical(i, t, t0, delta_t, T, u, u_rest, v, R_PVD, C_PVD, tau_PVD, I_0)
    u_euler = lif_sim_euler(i, t, t0, delta_t, T, u, u_rest, v, R_PVD, C_PVD, tau_PVD, I_0)
    u_rk2_PVD = lif_sim_rk2(i, t, t0, delta_t, T, u, u_rest, v, R_PVD, C_PVD, tau_PVD, I_0)
    u_rk4_PVD = lif_sim_rk4(i, t, t0, delta_t, T, u, u_rest, v, R_PVD, C_PVD, tau_PVD, I_0)

    plt.suptitle('Simple "Leaky-Integrate-and-Fire Model"', fontsize=16)
    plt.plot(u_PVD, '-b', label='Analytisches Signal', linewidth=1.5)
    plt.plot(u_euler, '-y', label='Euler-Verfahren', linewidth=0.5)
    plt.plot(u_rk2_PVD, '-g', label='Runge-Kutta 2.Ord.', linewidth=1)
    plt.plot(u_rk4_PVD, '-r', label='Runge-Kutta 4.Ord.', linewidth=1)
    plt.xlabel('t (in ms)')
    plt.ylabel('u(t)')
    plt.legend(loc='upper left')
    plt.legend

    plt.show()

if __name__== "__main__":
	main()
