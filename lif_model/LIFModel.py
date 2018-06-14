from __future__ import division # Because otherwise, "/" WILL NOT WORK!

"""
Leaky Integrate and Fire - Simulation implemented in Python
(by JW)
"""

# Some dependencies
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot, animation, rcParams


"""
Parameters for the LIF Simulation
"""
# SIMULATION PARAMETERS LIF
# Made to variate

# Cell membrane acts like a capacitor in parallel with a resistor
R = 2 #mOhm
C = 20 #mF
tau_m = R * C

# Integrate-and-fire neuron driven by constant input current I0
I_0 = 1.5 #mA

# Setting up the initial State with t=0 and u_rest=0 (resting Voltage of the cell-membrane)
t = 0 #s
u_rest = 0 #mV

#Treshold v
v = 2.5

#Time Variables
"""
t0 = 0
T = 200
delta_t = 1
N = T / delta_t
"""

"""
Actual LIF-Simulation with imported Parameters
"""
# My own parameters
# from parameters import *

# Initialization
u = 0
t = 0
i = 0



# Initial Formula of the LIF Model to calculate the time course of the membrane potential
# u = u_rest + (R * I_0 * (1 - exp( -t/tau_m)))
# Analytical Method
def lif_sim_analytical(i, t, t0, T, delta_t, N, u, u_rest, v, R, C, tau_m, I_0):
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
			print len(j)
	return u_analytical

# Initial Formula of the LIF Model to calculate the time course of the membrane potential
# u = u_rest + (R * I_0 * (1 - exp( -t/tau_m)))
# Euler-Forward Method

def lif_sim_euler(i, t, t0, delta_t, T, N, u, u_rest, v, R, C, tau_m, I_0):
	u_euler = np.array([0])

	for i in np.arange(0, T, delta_t):
		if u <= v:
			u = u - 1/tau_m * ((u - u_rest)-(R * I_0)) * delta_t
			u_euler = np.append(u_euler, u)
			i += 1
		else:
			u = 0
			u_euler = np.append(u_euler, u)
			u = u - 1/tau_m * ((u - u_rest)-(R * I_0)) * delta_t
			u_euler = np.append(u_euler, u)
			i += 1

	return u_euler

def lif_sim_rk2(i, t, t0, delta_t, T, N, u, u_rest, v, R, C, tau_m, I_0):
	u_rk2 = np.array([0])

	for i in np.arange(t0, T, delta_t):
		if u <= v:
			u_trial = u - 1/tau_m * ((u - u_rest)-(R * I_0)) * delta_t
			u = u + 1/2 * (-1/tau_m * ((u - u_rest)-(R * I_0)) - 1/tau_m * (((u - 1/tau_m * ((u - u_rest) - R * I_0)) - u_rest) - R * I_0)) * delta_t
			u_rk2 = np.append(u_rk2, u)
			i += 1
		else:
			u = 0
			u_rk2 = np.append(u_rk2, u)
			u_trial = u - 1/tau_m * ((u - u_rest)-(R * I_0)) * delta_t
			u = u + 1/2 * (-1/tau_m * ((u - u_rest)-(R * I_0)) - 1/tau_m * (((u - 1/tau_m * ((u - u_rest) - R * I_0)) - u_rest) - R * I_0)) * delta_t
			u_rk2 = np.append(u_rk2, u)
			i += 1

	return u_rk2

def lif_sim_rk4(i, t, t0, delta_t, T, N, u, u_rest, v, R, C, tau_m, I_0):
	u_rk4 = np.array([0])

	for i in np.arange(t0, T, delta_t):
		if u <= v:
			k1 = - 1/tau_m * ((u - u_rest)-(R * I_0)) * delta_t
			k2 = - 1/tau_m * (((u + 1/2 * k1) - u_rest)-(R * I_0)) * delta_t
			k3 = - 1/tau_m * (((u + 1/2 * k2) - u_rest)-(R * I_0)) * delta_t
			k4 = - 1/tau_m * (((u + k3) - u_rest)-(R * I_0)) * delta_t
			u = u + 1/6 * (k1 + 2 * k2 + 2 * k3 + k4)
			u_rk4 = np.append(u_rk4, u)
			i += 1
		else:
			u = 0
			u_rk4 = np.append(u_rk4, u)
			k1 = - 1/tau_m * ((u - u_rest)-(R * I_0)) * delta_t
			k2 = - 1/tau_m * (((u + 1/2 * k1) - u_rest)-(R * I_0)) * delta_t
			k3 = - 1/tau_m * (((u + 1/2 * k2) - u_rest)-(R * I_0)) * delta_t
			k4 = - 1/tau_m * (((u + k3) - u_rest)-(R * I_0)) * delta_t
			u = u + 1/6 * (k1 + 2 * k2 + 2 * k3 + k4)
			u_rk4 = np.append(u_rk4, u)
			i += 1

	return u_rk4


"""
Main Python-Function with plotting logic
"""

def main():
	u_analytical = lif_sim_analytical(i, t, t0, delta_t, N, u, u_rest, v, R, C, tau_m, I_0)
	u_euler = lif_sim_euler(i, t, t0, delta_t, N, u, u_rest, v, R, C, tau_m, I_0)
	u_rk2 = lif_sim_rk2(i, t, t0, delta_t, N, u, u_rest, v, R, C, tau_m, I_0)
	u_rk4 = lif_sim_rk4(i, t, t0, delta_t, N, u, u_rest, v, R, C, tau_m, I_0)
	# PLOTTING
	plt.suptitle('Simple "Leaky-Integrate-and-Fire Model"', fontsize=16)

	plt.plot(u_analytical, '-b', label='Analytisches Signal', linewidth=1.0)
	plt.plot(u_euler, '-r', label='Euler-Verfahren', linewidth=0.5)
	plt.plot(u_rk2, '-g', label='Runge-Kutta 2.Ord.', linewidth=1.0)
	plt.plot(u_rk4, '-y', label='Runge-Kutta 4.Ord.', linewidth=1.0)

	plt.xlabel('t (Time)')
	plt.ylabel('delta u(t)/v')
	plt.legend(loc='upper left')
	plt.legend

	return plt.show()


if __name__== "__main__":
	main()
