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

'''
IMPORTANT VALUES - CHANGE THEM:
'''
load_matrices = "result_matrices.p"
#load_matrices = "parameter_dumps/20180716_14-36-43_result_matrices_reward_27.p"


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

    global AVA, AVD, PVC, AVB, PVD, PLM, AVM, ALM, AVA_spike, AVD_spike, PVC_spike, AVB_spike, I_PVC, I_AVD, I_AVA, I_AVB, totalreward, done, info

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
    #---------------------------------------------------------------------------------------

    totalreward = 0
    done = 0
    info = 0

#-------------------------------------------------------------------------------------------

# Compute Function--------------------------------------------------------------------------

def compute(x, u, w_in_mat, w_sin_mat, sig_in_mat, sig_sin_mat, w_gap_in_mat, w_gap_sin_mat, C_m_mat, G_leak_mat, U_leak_mat):

    # Compute all Synapse Currents in this network------------------------------------------

    for i in range(0,4):
        for j in range (0,4):
            # Synapse Currents between Interneurons
            if A[i, j] == 1:
                # Excitatory Synapse
                I_s_inter[i, j] = I_syn_calc(x[i], x[j], E_ex, w_in_mat[i, j], sig_in_mat[i, j], mu)
            elif A[i, j] == -1:
                # Inhibitory Synapse
                I_s_inter[i, j] = I_syn_calc(x[i], x[j], E_in, w_in_mat[i, j], sig_in_mat[i, j], mu)
            else:
                I_s_inter[i, j] = 0

            # Gap-Junction Currents between Interneurons
            if A_gap[i, j] == 1:
                # There is a Gap-Junctions
                I_g_inter[i, j] = I_gap_calc(x[i], x[j], w_gap_in_mat[i, j])
            else:
                I_g_inter[i, j] = 0

    for i in range(0,4):
        for j in range(0,4):
            # Synapse Currents between Sensory and Interneurons
            if B[i, j] == 1:
                # Inhibitory Synapse (can't be Excitatory)
                I_s_sensor[i, j] = I_syn_calc(u[i], u[j], E_in, w_sin_mat[i, j], sig_sin_mat[i, j], mu)
            else:
                I_s_sensor[i, j] = 0

            # Gap-Junction Currents between Sensory and Interneurons
            if B_gap[i, j] == 1:
                # There is a Gap-Junctions
                I_g_sensor[i, j] = I_gap_calc(x[i], x[j], w_gap_sin_mat[i, j])
            else:
                I_g_sensor[i, j] = 0

    #---------------------------------------------------------------------------------------

    # Now compute inter Neurons Voltages----------------------------------------------------
    for i in range(0,4):
        I_syn_inter = I_s_inter.sum(axis = 0) # Creates a 1x5 Array with the Sum of all Columns
        I_gap_inter = I_g_inter.sum(axis = 0)
        I_syn_stimuli = I_s_sensor.sum(axis = 0)
        I_gap_stimuli = I_g_sensor.sum(axis = 0)
        x[i], fire[i] = U_neuron_calc(x[i], I_syn_inter[i], I_gap_inter[i], I_syn_stimuli[i], I_gap_stimuli[i], C_m_mat[0,i], G_leak_mat[0,i], U_leak_mat[0,i], v, delta_t)
        #x[i], fire[i] = U_neuron_calc(x[i], I_syn[i], 0, C_m_mat[0,i], G_leak_mat[0,i], U_leak_mat[0,i], v, delta_t)

    #---------------------------------------------------------------------------------------

    I_syn = np.add(I_syn_inter, I_syn_stimuli)
    I_gap = np.add(I_gap_inter, I_gap_stimuli)

    return x, u, fire, I_syn, I_gap

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
    plt.plot(PVD, '-b', label='PVD', linewidth=1)
    plt.plot(PLM, '-y', label='PLM', linewidth=1)
    plt.plot(AVM, '-g', label='AVM', linewidth=1)
    plt.plot(ALM, '-r', label='ALM', linewidth=1)
    plt.xlabel('t (in s)')
    plt.ylabel('u(t) in [mV]')
    plt.legend(loc='upper left')

    plt.subplot(122)
    plt.title('Inter Neurons', fontsize=10)
    plt.plot(AVA, '-b', label='AVA', linewidth=1)
    plt.plot(AVD, '-y', label='AVD', linewidth=1)
    plt.plot(PVC, '-g', label='PVC', linewidth=0.5)
    plt.plot(AVB, '-k', label='AVB', linewidth=1)
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
    plt.plot(I_AVA, '-r', label='AVA', linewidth=1)
    plt.xlabel('t')
    plt.ylabel('i(t) in [mA]')
    plt.subplot(224)
    plt.title('AVB', fontsize=10)
    plt.plot(I_AVB, '-r', label='AVB', linewidth=1)


    plt.show()

#-------------------------------------------------------------------------------------------

def import_matrices():
    result = pickle.load( open(load_matrices, "rb"))

    w_in_mat = result[0]
    w_sin_mat = result[1]
    sig_in_mat = result[2]
    sig_sin_mat = result[3]
    w_gap_in_mat = result[4]
    w_gap_sin_mat = result[5]
    C_m_mat = result[6]
    G_leak_mat = result[7]
    U_leak_mat = result[8]

    return w_in_mat, w_sin_mat, sig_in_mat, sig_sin_mat, w_gap_in_mat, w_gap_sin_mat, C_m_mat, G_leak_mat, U_leak_mat


# OpenAI Gym--------------------------------------------------------------------------------

def run_episode(env, fire):

    global observation, reward, done, info, totalreward, env_vis, action

    env_vis.append(env.render(mode = 'rgb_array'))

    if fire[0] == 1:
        action = 0
        observation, reward, done, info = env.step(action)
        totalreward += reward
        print 'RIGHT'
    elif fire[3] == 1:
        action = 1
        observation, reward, done, info = env.step(action)
        totalreward += reward
        print 'LEFT'
    else:
        print 'Im not sure :( Going ',action
        #action = 0
        #action = np.random.randint(0,1)
        observation, reward, done, info = env.step(action)
        totalreward += reward


    return observation, totalreward, done, info

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

def main():
    global x, u, env, action

    observation = env.reset()
    action = 0
    episodes = 0

    initialize(Default_U_leak) # Initializing all Interneurons with the desired leakage voltage
    #u = [-20, -40, -40, -20]

    w_in_mat, w_sin_mat, sig_in_mat, sig_sin_mat, w_gap_in_mat, w_gap_sin_mat, C_m_mat, G_leak_mat, U_leak_mat = import_matrices()

    for t in np.arange(t0,T,delta_t):
        x, u, fire, I_syn, I_gap = compute(x, u, w_in_mat, w_sin_mat, sig_in_mat, sig_sin_mat, w_gap_in_mat, w_gap_sin_mat, C_m_mat, G_leak_mat, U_leak_mat) # Compute the next Interneuron Voltages along with a possible "fire" Event
        I_all = np.add(I_syn, I_gap)
        arr(x, u, fire, I_all) # Storing Information for graphical analysis


        # OpenAI GYM PART----------------------------------

        # Make a Step

        observation, totalreward, done, info = run_episode(env, fire)
        angle = (observation[2] * 360) / (2 * np.pi)
        velocity = observation[3]
        cart_pos = observation[0]

        # Adapt, learn, overcome
        if angle >= 0:
            u[1] = -70 + (50/12) * angle # AVD
            u[2] = -70
        else:
            u[2] = -70 + (50/12) * angle # PVC
            u[1] = -70

        if cart_pos >= 0:
            u[3] = -70 + (50/2.4) * cart_pos # ALM
            u[0] = -70
        else:
            u[0] = -70 + (50/2.4) * cart_pos # PVD
            u[3] = -70

        '''
        if velocity >= 0:
            u[3] = -70 + (50/5) * velocity
            u[0] = -70
        else:
            u[0] = -70 + (50/5) * velocity
            u[3] = -70
        '''

        if done:
            env.reset()
            episodes = episodes + 1

    print "Did",episodes,"Episodes!"
    env_render(env_vis)

    plot() # Plotting everyting using matplotlib

#-------------------------------------------------------------------------------------------


if __name__=="__main__":
    main()
