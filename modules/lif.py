from __future__ import division # Because otherwise, "/" WILL NOT WORK!
"""
CALCULATION MODULE for LEAKY INTEGRATE AND FIRE

CALL BY:    <lif.py>

RETURN:     Desired Parameters

INFO:       Contains all mathematically important calculation functions
            Solving the differential equations by Runge Kutta 4th order
"""

# Some dependencies
import numpy as np


# Synapse Current computation
def I_syn_calc(u_pre, u_post, E, w, sig, mu):

    I_s = (w / (1 + np.exp(sig * (u_pre + mu)))) * (E - u_post)

    return I_s

def I_gap_calc(u_pre, u_post, w_gap):

    I_g = w_gap * (u_post - u_pre)

    return I_g

# Neuron Voltage computation
def U_neuron_calc(u_syn, I_syn_inter, I_gap_inter, I_syn_stimuli, I_gap_stimuli, C_m, G_leak, U_leak, v, delta_t):

    fire = 0

    if u_syn <= v:
        k1 = ((G_leak * (U_leak - u_syn) + (I_syn_inter + I_gap_inter + I_syn_stimuli + I_gap_stimuli)) / C_m) * delta_t

        k2 = ((G_leak * (U_leak - (u_syn + 1/2 * k1)) + (I_syn_inter + I_gap_inter + I_syn_stimuli + I_gap_stimuli)) / C_m) * delta_t

        k3 = ((G_leak * (U_leak - (u_syn + 1/2 * k2)) + (I_syn_inter + I_gap_inter + I_syn_stimuli + I_gap_stimuli)) / C_m) * delta_t

        k4 = ((G_leak * (U_leak - (u_syn + k3)) + (I_syn_inter + I_gap_inter + I_syn_stimuli + I_gap_stimuli)) / C_m) * delta_t

        u_syn = u_syn + 1/6 * (k1 + 2 * k2 + 2 * k3 + k4)

    else:
        u_syn = U_leak
        fire = 1

    return u_syn, fire
