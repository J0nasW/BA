"""
VISIUALIZATION MODULE loading Parameter Matrices

CALL BY:    <visiualize.py>

RETURN:     Environment simulation (animated) & Plots

INFO:       This Module can load a specific File Dump (cPickle) and visiualize the containig matrices onto a OpenAI Gym Environment
"""

# Some dependencies
import numpy as np
import matplotlib.pyplot as plt
import gym
import cPickle as pickle

from lif import I_syn_calc, I_gap_calc, U_neuron_calc
from parameters import *
from random_search_v2 import compute as compute_v2
from random_search_v2 import observe
from weights_nn import compute as compute_with_weights

# Initializing OpenAI Environments------------------------------------------------------
env = gym.make('CartPole-v0')
env.reset()
env_vis = []
#---------------------------------------------------------------------------------------


# Initialization----------------------------------------------------------------------------

def initialize(Default_U_leak):

    # Initializing Neurons and Sensors------------------------------------------------------
    for i in range(0,4):
        x[i] = Default_U_leak
    for i in range(0,4):
        u[i] = Default_U_leak

    global AVA, AVD, PVC, AVB, PVD, PLM, AVM, ALM, AVA_spike, AVD_spike, PVC_spike, AVB_spike, I_PVC, I_AVD, I_AVA, I_AVB, actions_arr, angles_arr, totalreward, done, info

    AVA = np.array([Default_U_leak])
    AVD = np.array([Default_U_leak])
    PVC = np.array([Default_U_leak])
    AVB = np.array([Default_U_leak])

    PVD = np.array([])
    PLM = np.array([])
    AVM = np.array([])
    ALM = np.array([])

    AVA_spike = np.array([])
    AVD_spike = np.array([])
    PVC_spike = np.array([])
    AVB_spike = np.array([])

    I_PVC = np.array([])
    I_AVD = np.array([])
    I_AVA = np.array([])
    I_AVB = np.array([])

    actions_arr = np.array([])
    angles_arr = np.array([])
    #---------------------------------------------------------------------------------------

    totalreward = 0
    done = 0
    info = 0

#-------------------------------------------------------------------------------------------

# Append Function---------------------------------------------------------------------------

def arr(x, u, fire, I_all):

    global AVA, AVD, PVC, DVA, AVB, PVD, PLM, AVM, ALM, AVA_spike, AVD_spike, PVC_spike, AVB_spike, I_PVC, I_AVD, I_AVA, I_AVB
    AVA = np.append(AVA, x[0])
    AVD = np.append(AVD, x[1])
    PVC = np.append(PVC, x[2])
    AVB = np.append(AVB, x[3])

    PVD = np.append(PVD, u[0])
    PLM = np.append(PLM, u[1])
    AVM = np.append(AVM, u[2])
    ALM = np.append(ALM, u[3])

    AVA_spike = np.append(AVA_spike, fire[0]) # Reverse lokomotion
    AVD_spike = np.append(AVD_spike, fire[1]) # Reverse lokomotion
    PVC_spike = np.append(PVC_spike, fire[2]) # Reverse lokomotion
    AVB_spike = np.append(AVB_spike, fire[3]) # Forward lokomotion

    I_AVA = np.append(I_AVA, I_all[0])
    I_AVD = np.append(I_AVD, I_all[1])
    I_PVC = np.append(I_PVC, I_all[2])
    I_AVB = np.append(I_AVB, I_all[3])

#-------------------------------------------------------------------------------------------

# Plot Function-----------------------------------------------------------------------------

def plot():

    plt.figure(1)
    plt.suptitle('Leaky-Integrate-and-Fire Neuronal Network', fontsize=16)

    plt.subplot(121)
    plt.title('Sensory Neurons', fontsize=10)
    plt.plot(PLM, '-y', label='PLM (Phi)', linewidth=1)
    plt.plot(AVM, '-g', label='AVM (-Phi)', linewidth=1)
    plt.plot(ALM, '-r', label='ALM (x)', linewidth=1)
    plt.plot(PVD, '-b', label='PVD (-x)', linewidth=1)
    plt.xlabel('t (in s)')
    plt.ylabel('u(t) in [mV]')
    plt.legend(loc='upper left')

    plt.subplot(122)
    plt.title('Inter Neurons', fontsize=10)
    plt.plot(AVA, '-b', label='AVA (REV)', linewidth=0.3)
    plt.plot(AVD, '-y', label='AVD (REV)', linewidth=1)
    plt.plot(PVC, '-g', label='PVC (FWD)', linewidth=1)
    plt.plot(AVB, '-k', label='AVB (FWD)', linewidth=0.3)
    plt.xlabel('t (in s)')
    plt.ylabel('u(t) in [mV]')
    plt.legend(loc='upper left')


    plt.figure(2)
    plt.suptitle('Neuron Currents', fontsize=16)

    plt.subplot(221)
    plt.title('PVC', fontsize=10)
    plt.plot(I_PVC, '-r', label='PVC', linewidth=1)
    plt.subplot(222)
    plt.title('AVD', fontsize=10)
    plt.plot(I_AVD, '-r', label='AVD', linewidth=1)
    plt.subplot(223)
    plt.title('AVA', fontsize=10)
    plt.plot(I_AVA, '-r', label='AVA', linewidth=0.5)
    plt.xlabel('t')
    plt.ylabel('i(t) in [mA]')
    plt.subplot(224)
    plt.title('AVB', fontsize=10)
    plt.plot(I_AVB, '-r', label='AVB', linewidth=0.5)

    plt.figure(3)
    plt.suptitle('Action and Angle of this Simulation', fontsize=16)
    plt.plot(actions_arr, '-r', label='Actions [LEFT/RIGHT]', linewidth=1)
    plt.plot(angles_arr, '-b', label='Angles [deg]', linewidth=1)
    plt.xlabel('t')
    plt.ylabel('Action / Angle in Deg')
    plt.legend(loc='upper left')


    plt.show()

#-------------------------------------------------------------------------------------------

def import_parameters(parameter_matrices):
    result_parameters = pickle.load( open(parameter_matrices, "r"))

    w_A_rnd = result_parameters[0]
    w_B_rnd = result_parameters[1]
    w_B_gap_rnd = result_parameters[2]
    sig_A_rnd = result_parameters[3]
    sig_B_rnd = result_parameters[4]
    C_m_rnd = result_parameters[5]
    G_leak_rnd = result_parameters[6]
    U_leak_rnd = result_parameters[7]

    return w_A_rnd, w_B_rnd, w_B_gap_rnd, sig_A_rnd, sig_B_rnd, C_m_rnd, G_leak_rnd, U_leak_rnd

def import_weights(load_weights):

    result_weights = pickle.load( open(load_weights, "r"))

    A_rnd = result_weights[0]
    B_rnd = result_weights[1]

    return A_rnd, B_rnd

# OpenAI Gym--------------------------------------------------------------------------------

def run_episode(env, fire):

    global observation, reward, done, info, totalreward, action, env_vis, uncertain, actions_arr, angles_arr

    env_vis.append(env.render(mode = 'rgb_array'))

    # - action = 0 LEFT  - action = 1 RIGHT

    if fire[0] == 1: # AVA (REV) is firing
        action = 0
        observation, reward, done, info = env.step(action)
        #print 'LEFT'
    elif fire[3] == 1: # AVB (FWD) is firing
        action = 1
        observation, reward, done, info = env.step(action)
        #print 'RIGHT'
    else:
        uncertain +=1
        observation, reward, done, info = env.step(action)

    totalreward += reward
    angle = observe(observation)

    if done:
        action = 0

    actions_arr = np.append(actions_arr, action)
    angles_arr = np.append(angles_arr, angle)

    return totalreward, done, uncertain

def env_render(env_vis):
    plt.figure()
    plot  =  plt.imshow(env_vis[0])
    plt.axis('off')

def animate(i):
    plot.set_data(env_vis[i])
    anim  =  anm.FuncAnimation(plt.gcf(), animate, frames=len(env_vis), interval=20, repeat=True, repeat_delay=20)
    display(display_animation(anim,  default_mode='loop'))


#-------------------------------------------------------------------------------------------
# Main Function-----------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------

def main(parameter_matrices):
    global x, u, env, action, uncertain

    observation = env.reset()
    action = 0
    actions = 0
    episodes = 0
    uncertain = 0

    initialize(Default_U_leak) # Initializing all Interneurons with the desired leakage voltage
    #u = [-20, -40, -40, -20]

    w_A_rnd, w_B_rnd, w_B_gap_rnd, sig_A_rnd, sig_B_rnd, C_m_rnd, G_leak_rnd, U_leak_rnd = import_parameters(parameter_matrices)

    for t in np.arange(t0,T,delta_t):
        x, u, fire, I_syn, I_gap = compute_v2(x, u, w_A_rnd, w_B_rnd, w_B_gap_rnd, sig_A_rnd, sig_B_rnd, C_m_rnd, G_leak_rnd, U_leak_rnd) # Compute the next Interneuron Voltages along with a possible "fire" Event
        I_all = np.add(I_syn, I_gap)
        arr(x, u, fire, I_all) # Storing Information for graphical analysis


        # OpenAI GYM PART----------------------------------

        totalreward, done, uncertain = run_episode(env, fire)

        if done:
            env.reset()
            episodes = episodes + 1

    print "Did",episodes,"Episodes and was",uncertain,"out of",len(actions_arr),"times uncertain!"
    env_render(env_vis)

    plot() # Plotting everyting using matplotlib

#-------------------------------------------------------------------------------------------

def main_with_weights(load_parameters, load_weights, runtime):
    global x, u, env, action, uncertain

    observation = env.reset()
    action = 0
    actions = 0
    episodes = 0
    uncertain = 0

    initialize(Default_U_leak) # Initializing all Interneurons with the desired leakage voltage
    #u = [-20, -40, -40, -20]

    w_A_rnd, w_B_rnd, w_B_gap_rnd, sig_A_rnd, sig_B_rnd, C_m_rnd, G_leak_rnd, U_leak_rnd = import_parameters(load_parameters)
    A_rnd, B_rnd = import_weights(load_weights)

    for t in np.arange(t0,runtime,delta_t):
        x, u, fire, I_syn, I_gap = compute_with_weights(x, u, w_A_rnd, w_B_rnd, w_B_gap_rnd, sig_A_rnd, sig_B_rnd, C_m_rnd, G_leak_rnd, U_leak_rnd, A_rnd, B_rnd) # Compute the next Interneuron Voltages along with a possible "fire" Event
        I_all = np.add(I_syn, I_gap)
        arr(x, u, fire, I_all) # Storing Information for graphical analysis


        # OpenAI GYM PART----------------------------------

        totalreward, done, uncertain = run_episode(env, fire)

        if done:
            env.reset()
            episodes = episodes + 1

    print "Did",episodes,"Episodes and was",uncertain,"out of",len(actions_arr),"times uncertain!"
    env_render(env_vis)

    plot() # Plotting everyting using matplotlib


if __name__=="__main__":
    main()
