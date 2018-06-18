from __future__ import division # Because otherwise, "/" WILL NOT WORK!

"""
Leaky Integrate and Fire - Simulation implemented in Python
(by JW)
"""

# Some dependencies
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot, animation, rcParams


# Initial Formula of the LIF Model to calculate the time course of the membrane potential
# u = u_rest + (R * I_0 * (1 - exp( -t/tau_m)))
# Analytical Method
def lif_sim_analytical(i, t, t0, delta_t, T, u, u_rest, v, R, C, tau_m, I_0):
	u_analytical = np.array([])
	j = np.arange(t0, T, delta_t)
	for t in np.arange(t0, T, delta_t):
		if u <= v:
			u = u_rest + (R * I_0 * (1 - np.exp(-(j[i] / tau_m))))
			u_analytical = np.append(u_analytical, u)
			i += 1
		else:
			i = 0
			u = u_rest + (R * I_0 * (1 - np.exp(-(j[i] / tau_m))))
			u_analytical = np.append(u_analytical, u)
			i += 1
	return u_analytical

# Initial Formula of the LIF Model to calculate the time course of the membrane potential
# u = u_rest + (R * I_0 * (1 - exp( -t/tau_m)))
# Euler-Forward Method

def lif_sim_euler(i, t, t0, delta_t, T, u, u_rest, v, R, C, tau_m, I_0):
	u_euler = np.array([0])
	for i in np.arange(t0, T, delta_t):
		if u <= v:
			u = u - 1/tau_m * ((u - u_rest)-(R * I_0)) * delta_t
			u_euler = np.append(u_euler, u)
		else:
			u = 0
			u_euler = np.append(u_euler, u)
			u = u - 1/tau_m * ((u - u_rest)-(R * I_0)) * delta_t
			u_euler = np.append(u_euler, u)

	return u_euler

def lif_sim_rk2(i, t, t0, delta_t, T, u, u_rest, v, R, C, tau_m, I_0):
	u_rk2 = np.array([0])
	for i in np.arange(t0, T, delta_t):
		if u <= v:
			u_trial = u - 1/tau_m * ((u - u_rest)-(R * I_0)) * delta_t
			u = u + 1/2 * (-1/tau_m * ((u - u_rest)-(R * I_0)) - 1/tau_m * (((u - 1/tau_m * ((u - u_rest) - R * I_0)) - u_rest) - R * I_0)) * delta_t
			u_rk2 = np.append(u_rk2, u)
		else:
			u = 0
			u_rk2 = np.append(u_rk2, u)
			u_trial = u - 1/tau_m * ((u - u_rest)-(R * I_0)) * delta_t
			u = u + 1/2 * (-1/tau_m * ((u - u_rest)-(R * I_0)) - 1/tau_m * (((u - 1/tau_m * ((u - u_rest) - R * I_0)) - u_rest) - R * I_0)) * delta_t
			u_rk2 = np.append(u_rk2, u)

	return u_rk2

def lif_sim_rk4(i, t, t0, delta_t, T, u, u_rest, v, R, C, tau_m, I_0):
	u_rk4 = np.array([0])
	for i in np.arange(t0, T, delta_t):
		if u <= v:
			k1 = - 1/tau_m * ((u - u_rest)-(R * I_0)) * delta_t
			k2 = - 1/tau_m * (((u + 1/2 * k1) - u_rest)-(R * I_0)) * delta_t
			k3 = - 1/tau_m * (((u + 1/2 * k2) - u_rest)-(R * I_0)) * delta_t
			k4 = - 1/tau_m * (((u + k3) - u_rest)-(R * I_0)) * delta_t
			u = u + 1/6 * (k1 + 2 * k2 + 2 * k3 + k4)
			u_rk4 = np.append(u_rk4, u)
		else:
			u = 0
			u_rk4 = np.append(u_rk4, u)
			k1 = - 1/tau_m * ((u - u_rest)-(R * I_0)) * delta_t
			k2 = - 1/tau_m * (((u + 1/2 * k1) - u_rest)-(R * I_0)) * delta_t
			k3 = - 1/tau_m * (((u + 1/2 * k2) - u_rest)-(R * I_0)) * delta_t
			k4 = - 1/tau_m * (((u + k3) - u_rest)-(R * I_0)) * delta_t
			u = u + 1/6 * (k1 + 2 * k2 + 2 * k3 + k4)
			u_rk4 = np.append(u_rk4, u)

	return u_rk4
