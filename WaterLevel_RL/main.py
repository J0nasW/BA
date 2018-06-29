from __future__ import division # Because otherwise, "/" WILL NOT WORK!

"""
Neuronal Network with Reinforcement Learning to control Level of a Water Tank
(by JW)
"""

import numpy as np
import matplotlib.pyplot as plt

"""
Parameters for the System

WaterLevel - Water Level of the Tank
u_rest - Resting voltage of the membrane of the Cell
u_spike - Spike Voltage, when the Treshold of -55mV is reached
Reward - RL-Rewards
"""

WaterLevel = 4 # 0-10
LevelHistory = np.array([])
u_rest = -70 #mV
u_spike = 20 #mV
u_th = -55 #mV
u_IN = -70 #mV - Initial Voltage of our Interneuron
u_History = np.array([])

t = 0
i = 0
voltage = 0

Reward = {0:0, 1:0, 2:36, 3:72, 4: 90, 5:100, 6:90, 7:72, 8: 36, 9:0, 10:0} #Sensor:Reward

def main():
    global WaterLevel
    global LevelHistory
    global u_History
    global u_IN
    for i in range(0,10):
        new_WaterLevel, u_IN = Control(WaterLevel, u_IN, u_spike, u_th, u_rest)
        LevelHistory = np.append(LevelHistory, new_WaterLevel)
        u_History = np.append(u_History, u_IN)
        WaterLevel = new_WaterLevel - 1

    print u_History

    plt.suptitle('Water Level over Time', fontsize=16)
    plt.plot(LevelHistory, '-b', label='Wasserlevel', linewidth=1.5)
    plt.xlabel('t (in ms)')
    plt.ylabel('Level')
    plt.legend(loc='upper left')
    plt.legend

    plt.show()


def Control(lvl, u_IN, u_spike, u_th, u_rest):
    if lvl < 5:
        u_IN += u_spike # Excitatory
    elif lvl > 5:
        u_IN -= u_spike # Inhitory
    else:
        print "We're all right!"

    if u_IN >= u_th:
        lvl += 1 # pump
        u_IN = u_rest
        print "FIRE!"

    return lvl, u_IN

if __name__== "__main__":
	main()
