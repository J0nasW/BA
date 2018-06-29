from __future__ import division # Because otherwise, "/" WILL NOT WORK!

"""
Leaky Integrate and Fire - Simulation implemented in Python
(by JW)
"""

# Some dependencies
import numpy as np


# LIF Simulation implemented with 4th order Runge-Kutta
def lif_sim(u_pre, u_post, C_m, G_leak, U_leak, w, sig, mu, E, t0, T, delta_t, v):

    if u_post <= v:
		k1 = ((G_leak * (U_leak - u_post) + w / (1 + np.exp(sig * (u_pre + mu))) * (E - u_post)) / C_m )* delta_t

		k2 = ((G_leak * (U_leak - (u_post + 1/2 * k1)) + w / (1 + np.exp(sig * (u_pre + mu))) * (E - (u_post + 1/2 * k1))) / C_m )* delta_t

		k3 = ((G_leak * (U_leak - (u_post + 1/2 * k2)) + w / (1 + np.exp(sig * (u_pre + mu))) * (E - (u_post + 1/2 * k2))) / C_m )* delta_t

		k4 = ((G_leak * (U_leak - (u_post + k3)) + w / (1 + np.exp(sig * (u_pre + mu))) * (E - (u_post + k3))) / C_m )* delta_t

		u_post = u_post + 1/6 * (k1 + 2 * k2 + 2 * k3 + k4)

    else:
        u_post = U_leak

    return u_post
