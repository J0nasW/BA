from __future__ import division # Because otherwise, "/" WILL NOT WORK!

"""
Leaky Integrate and Fire - Simulation implemented in Python
(by JW)
"""

# Some dependencies
import numpy as np


# Synapse Current computation
def I_syn_calc(u_pre, u_post, E, w, sig, mu):

    I_s = (w / (1 + np.exp(sig * (u_pre + mu)))) * (E - u_post)

    return I_s

# Neuron Voltage computation
def U_neuron_calc(u_syn, I_syn, I_gap, C_m, G_leak, U_leak, v, delta_t):

    fire = 0

    if u_syn <= v:
        k1 = ((G_leak * (U_leak - u_syn) + (I_syn + I_gap)) / C_m) * delta_t

        k2 = ((G_leak * (U_leak - (u_syn + 1/2 * k1)) + (I_syn + I_gap)) / C_m) * delta_t

        k3 = ((G_leak * (U_leak - (u_syn + 1/2 * k2)) + (I_syn + I_gap)) / C_m) * delta_t

        k4 = ((G_leak * (U_leak - (u_syn + k3)) + (I_syn + I_gap)) / C_m) * delta_t

        u_syn = u_syn + 1/6 * (k1 + 2 * k2 + 2 * k3 + k4)

    else:
        u_syn = U_leak
        fire = 1

    return u_syn, fire

'''
# LIF Simulation implemented with 4th order Runge-Kutta
def lif_sim(u_pre, u_post, C_m, G_leak, U_leak, w, sig, mu, E, t0, T, delta_t, v):
    fire = 0

    if u_post <= v:
		k1 = ((G_leak * (U_leak - u_post) + w / (1 + np.exp(sig * (u_pre + mu))) * (E - u_post)) / C_m )* delta_t

		k2 = ((G_leak * (U_leak - (u_post + 1/2 * k1)) + w / (1 + np.exp(sig * (u_pre + mu))) * (E - (u_post + 1/2 * k1))) / C_m )* delta_t

		k3 = ((G_leak * (U_leak - (u_post + 1/2 * k2)) + w / (1 + np.exp(sig * (u_pre + mu))) * (E - (u_post + 1/2 * k2))) / C_m )* delta_t

		k4 = ((G_leak * (U_leak - (u_post + k3)) + w / (1 + np.exp(sig * (u_pre + mu))) * (E - (u_post + k3))) / C_m )* delta_t

		u_post = u_post + 1/6 * (k1 + 2 * k2 + 2 * k3 + k4)

    else:
        u_post = U_leak
        fire = 1

    return u_post, fire
'''
