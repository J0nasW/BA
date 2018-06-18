from __future__ import division # Because otherwise, "/" WILL NOT WORK!

"""
Neuronal Network of C. Elegans implemented in Python with the LIF-Model
(by JW)
"""

# Some dependencies
import numpy as np
import matplotlib.pyplot as plt

# Some Parameters
R = 2 # Original: 9,4 GOhm - 9400000000
C = 20 # Original: 16 pF - 0.000000000016
tau_m = R * C

# Treshold
v = 2.5

# Some other details
u = 0
u_rest = 0
I_0 = 1.5  # 35mA - Constant input current

# Time Constants:
t0 = 0
T = 200
delta_t = 0.1
N = T / delta_t

# Counter Values
t = 0
i = 0


def main():
    SignalIn = np.random.randint(2, size=int(N))

    u_array, actor_state = calc(i, SignalIn, t, t0, delta_t, T, u, u_rest, v, R, C, tau_m, I_0)

    print actor_state
    plt.suptitle('Substantial Neuronal Network Model"', fontsize=16)
    plt.subplot(2,1,1)
    plt.plot(u_array, '-b', label='Fire Signal', linewidth=1.5)
    plt.subplot(2,1,2)
    plt.plot(actor_state, '-y', label='Aktor EIN/AUS', linewidth=0.5)
    plt.xlabel('t')
    plt.ylabel('u(t)')
    plt.legend(loc='upper left')
    plt.legend

    plt.show()


def calc(i, SignalIn, t, t0, delta_t, T, u, u_rest, v, R, C, tau_m, I_0):
    u_array = np.array([])
    aktor = np.array([])
    j = 0
    for i in np.arange(t0, T, delta_t):
        if u < v:
            if SignalIn[j] == 1:
                k1 = - 1/tau_m * ((u - u_rest)-(R * I_0)) * delta_t
                k2 = - 1/tau_m * (((u + 1/2 * k1) - u_rest)-(R * I_0)) * delta_t
                k3 = - 1/tau_m * (((u + 1/2 * k2) - u_rest)-(R * I_0)) * delta_t
                k4 = - 1/tau_m * (((u + k3) - u_rest)-(R * I_0)) * delta_t
                u = u + 1/6 * (k1 + 2 * k2 + 2 * k3 + k4)
                u_array = np.append(u_array, u)
                aktor = np.append(aktor, 1)
            else:
                aktor = np.append(aktor, 0)
        else:
            fire(i)
            if SignalIn[j] == 1:
                u = 0
                u_array = np.append(u_array, u)
                k1 = - 1/tau_m * ((u - u_rest)-(R * I_0)) * delta_t
                k2 = - 1/tau_m * (((u + 1/2 * k1) - u_rest)-(R * I_0)) * delta_t
                k3 = - 1/tau_m * (((u + 1/2 * k2) - u_rest)-(R * I_0)) * delta_t
                k4 = - 1/tau_m * (((u + k3) - u_rest)-(R * I_0)) * delta_t
                u = u + 1/6 * (k1 + 2 * k2 + 2 * k3 + k4)
                u_array = np.append(u_array, u)
                aktor = np.append(aktor, 1)
            else:
                aktor = np.append(aktor, 0)
        j += 1

    return u_array, aktor

def fire(time):
    print "FIRE! At", time, "s"

if __name__== "__main__":
	main()
