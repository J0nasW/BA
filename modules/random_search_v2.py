"""
RL MODULE WITH RANDOM SEARCH

CALL BY:    <random_search_v2.py>

RETURN:     Parameter Matrices for the inverted Pendulum Problem
            Stores Data of best Parameters in '<date>_rs2_v2_<reward>.hkl'

INFO:       V2 with improved loading times and better simulation performance
"""

# Some dependencies
import os

import numpy as np # Maths and stuff
import gym.spaces # Simulating the Environments
import hickle as hkl # Performance Dumping in HDF5 Format <.hkl>
import time # For Runtime Evaluations
import datetime # For Datestamp on stored files

from .lif import I_syn_calc, I_gap_calc, U_neuron_calc
from .parameters import *

# Initialization----------------------------------------------------------------------------

def initialize(Default_U_leak):
    global totalreward, done, info

    # Initializing Neurons and Sensors------------------------------------------------------
    for i in range(0,4):
        x[i] = Default_U_leak
    for i in range(0,4):
        u[i] = Default_U_leak

    #OpenAI Gym Parameters-----------------------------------------------------------------

    totalreward = 0
    done = 0
    info = 0

#-------------------------------------------------------------------------------------------

# Random Function---------------------------------------------------------------------------

def random_parameters():

    # Initialize random parameters for our Neurons and Synapses according to the current Network

    # For Synapses
    w_A_rnd = np.random.uniform(low = 0.5, high = 3, size = (1,nbr_of_inter_synapses))
    w_B_rnd = np.random.uniform(low = 0.5, high = 3, size = (1,nbr_of_sensor_synapses))
    w_B_gap_rnd = np.random.uniform(low = 0, high = 3, size = (1,nbr_of_gap_junctions))

    sig_A_rnd = np.random.uniform(low = 0.05, high = 0.5, size = (1,nbr_of_inter_synapses))
    sig_B_rnd = np.random.uniform(low = 0.05, high = 0.5, size = (1,nbr_of_sensor_synapses))

    # For Neurons
    C_m_rnd = np.random.uniform(low = 0.01, high = 0.77, size = (1,nbr_of_inter_neurons))
    G_leak_rnd = np.random.uniform(low = 0.1, high = 2.5, size = (1,nbr_of_inter_neurons))
    U_leak_rnd = np.random.uniform(low = -70, high = -60, size = (1,nbr_of_inter_neurons))

    return w_A_rnd, w_B_rnd, w_B_gap_rnd, sig_A_rnd, sig_B_rnd, C_m_rnd, G_leak_rnd, U_leak_rnd

#-------------------------------------------------------------------------------------------

# Compute Function--------------------------------------------------------------------------

def compute(x, u, w_A_rnd, w_B_rnd, w_B_gap_rnd, sig_A_rnd, sig_B_rnd, C_m_rnd, G_leak_rnd, U_leak_rnd):

    # Some helpint Counter-Variables
    k = 0
    l = 0
    m = 0

    # Compute all Synapse Currents in this network------------------------------------------
    for i in range(0,4):
        for j in range (0,4):
            # Synapse Currents between Interneurons
            if A[i, j] == 1:
                # Excitatory Synapse
                I_s_inter[i, j] = I_syn_calc(x[i], x[j], E_in, w_A_rnd[0, k], sig_A_rnd[0, k], mu)
                k += 1
            elif A[i, j] == 2:
                # Inhibitory Synapse
                I_s_inter[i, j] = I_syn_calc(x[i], x[j], E_ex, w_A_rnd[0, k], sig_A_rnd[0, k], mu)
                k += 1
            else:
                # No Connection here.
                I_s_inter[i, j] = 0


            # Synapse Currents between Sensory and Interneurons
            if B[i, j] == 1:
                # Inhibitory Synapse (can't be Excitatory)
                I_s_sensor[i, j] = I_syn_calc(u[i], u[j], E_in, w_B_rnd[0, l], sig_B_rnd[0, l], mu)
                l += 1
            elif B[i, j] == 2:
                I_s_sensor[i, j] = I_syn_calc(u[i], u[j], E_ex, w_B_rnd[0, l], sig_B_rnd[0, l], mu)
                l += 1
            elif B[i, j] == 3:
                # Gap Junction
                I_g_sensor[i, j] = I_gap_calc(u[i], x[j], w_B_gap_rnd[0, m])
                m += 1
            else:
                # No Connection here.
                I_s_sensor[i, j] = 0
                I_g_sensor[i, j] = 0

    #---------------------------------------------------------------------------------------

    # Compute inter Neurons Voltages----------------------------------------------------
    for i in range(0,4):
        I_syn_inter = I_s_inter.sum(axis = 0) # Creates a 1x5 Array with the Sum of all Columns
        I_gap_inter = I_g_inter.sum(axis = 0)
        I_syn_stimuli = I_s_sensor.sum(axis = 0)
        I_gap_stimuli = I_g_sensor.sum(axis = 0)
        x[i], fire[i] = U_neuron_calc(x[i], I_syn_inter[i], I_gap_inter[i], I_syn_stimuli[i], I_gap_stimuli[i], C_m_rnd[0,i], G_leak_rnd[0,i], U_leak_rnd[0,i], v, delta_t)

    #---------------------------------------------------------------------------------------

    I_syn = np.add(I_syn_inter, I_syn_stimuli)
    I_gap = np.add(I_gap_inter, I_gap_stimuli)

    return x, u, fire, I_syn, I_gap

#-------------------------------------------------------------------------------------------

# OpenAI Gym--------------------------------------------------------------------------------

def run_episode(env, w_A_rnd, w_B_rnd, w_B_gap_rnd, sig_A_rnd, sig_B_rnd, C_m_rnd, G_leak_rnd, U_leak_rnd):
    global x, u, fire, I_syn, I_gap, action

    observation = env.reset()
    totalreward = 0

    for t in np.arange(t0,T,delta_t): # RUNNING THE EPISODE - Trynig to get 200 Steps in this Episode

        # Compute the next Interneuron Voltages along with a possible "fire" Event - Now new with random parameter matrices
        x, u, fire, I_syn, I_gap = compute(x, u, w_A_rnd, w_B_rnd, w_B_gap_rnd, sig_A_rnd, sig_B_rnd, C_m_rnd, G_leak_rnd, U_leak_rnd)

        # Decide for an action and making a Step
        if fire[0] == 1: # Sensory Neuron AVA is firing - resulting in a REVERSE Action (0 - LEFT)
            action = 0
            observation, reward, done, info = env.step(action)
        elif fire[3] == 1: # Sensory Neuron AVB is firing - resulting in a FORWARD Action (1 - RIGHT)
            action = 1
            observation, reward, done, info = env.step(action)
        else:
            # If computing twice in a Simulation, this step is obsolete (No uncertainty)
            observation, reward, done, info = env.step(action) # Have to use the action from the past time step - OpenAI Gym does not provide a "Do nothing"-Action

        totalreward += reward # Counting the total reward.
        observe(observation) # Trace the observation from our simulation back into the sensory neurons
        if done:
            # If reached a total Reward of 200
            break

    return totalreward

def observe(observation):
    global u

    # Interpreter: Calculating some of the observation values into valid data
    cart_pos = observation[0] # [-2.4 2.4]
    cart_vel = observation[1] # [-inf, inf]
    angle = (observation[2] * 360) / (2 * np.pi) # in degrees [-12deg, 12deg] (for Simulations)
    angle_velocity = observation[3] # [-inf, inf]

    # Adapt, learn, overcome---------------------------------------------------------------------------

    # Setting the Angle of the Pole to Sensory Neurons PLM (Phi+) and AVM (Phi-)
    if angle > 0:
        u[1] = Default_U_leak + ((v-Default_U_leak)/15) * (np.sqrt(np.absolute(angle))*5) # PLM
        u[2] = Default_U_leak
    elif angle == 0:
        u[1] = u[2] = Default_U_leak
    else:
        u[2] = Default_U_leak + ((v-Default_U_leak)/15) * (np.sqrt(np.absolute(angle))*5) # AVM
        u[1] = Default_U_leak
    
    # Setting the Cart Position to Sensory Neurons ALM (pos. movement) and PVD (neg. movement)
    if cart_pos > 0:
        u[3] = Default_U_leak + ((v-Default_U_leak)/0.8) * (np.absolute(cart_pos)*7) # ALM
        u[0] = Default_U_leak
    elif cart_pos == 0:
        u[0] = u[3] = Default_U_leak
    else:
        u[0] = Default_U_leak + ((v-Default_U_leak)/0.8) * (np.absolute(cart_pos)*7) # PVD
        u[3] = Default_U_leak
    '''
    # Setting the Anglespeed of the Pole to Sensory Neurons ALM (Phi.+) and PVD (Phi.-)
    if angle_velocity >= 0:
        u[0] = Default_U_leak + ((v-Default_U_leak)/1.05) * (np.absolute(angle_velocity)) # ALM
        u[3] = Default_U_leak
    elif cart_pos == 0:
        u[0] = u[3] = Default_U_leak
    else:
        u[3] = Default_U_leak + ((v-Default_U_leak)/1.05) * (np.absolute(angle_velocity)) # PVD
        u[0] = Default_U_leak
    '''
    return angle


#------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------
# Main Function-----------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------

def main(sim_time):
    global x, u, env, action

    start_time = time.time() # for Runtime reasons

    # Initialize the Environment and some vital Parameters
    action = 0
    episodes = 0
    best_reward = 0
    env = gym.make('CartPole-v0')

    while True:
        initialize(Default_U_leak) # Initializing all Sensory- and Interneurons with the desired leakage voltage [-70mV]
        episodes += 1 # Episode Counter
        w_A_rnd, w_B_rnd, w_B_gap_rnd, sig_A_rnd, sig_B_rnd, C_m_rnd, G_leak_rnd, U_leak_rnd = random_parameters() # Make some new random parameter Matrices
        reward = run_episode(env, w_A_rnd, w_B_rnd, w_B_gap_rnd, sig_A_rnd, sig_B_rnd, C_m_rnd, G_leak_rnd, U_leak_rnd)
        if reward > best_reward:
            # Set current reward as new reward
            best_reward = reward
            # Save Results of the Run with the best reward
            Result = [w_A_rnd, w_B_rnd, w_B_gap_rnd, sig_A_rnd, sig_B_rnd, C_m_rnd, G_leak_rnd, U_leak_rnd]
        #print 'Episode',episodes,'mit Reward',reward,'.'
        if (time.time() - start_time) >= sim_time:
            # End Simulation-Run, if given Simulation time is elapsed.
            break

    # Prepare some Information to dump..
    date = datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S")
    best_reward_s = str(int(best_reward))
    episodes = str(int(episodes))
    elapsed_time = str((time.time() - start_time))

    hkl.dump(Result, (current_dir + "/parameter_dumps/" + date + "_rs2_v2_" + best_reward_s + ".hkl"), mode='w')
    # Information Text File
    file = open((current_dir + "/information/" + date + "_parameter_run_" + best_reward_s + ".txt"), "w")
    file.write(("Parameter run from " + date + " with Reward " + best_reward_s + " and " + episodes + " Episodes.\nSimulation Runtime was " + elapsed_time + "."))
    file.close()

    # Console Prints
    print ('The best Reward was:',best_reward)

    print("--- %s seconds ---" % (time.time() - start_time))
    return date, best_reward_s

#-------------------------------------------------------------------------------------------


if __name__=="__main__":
    main()
