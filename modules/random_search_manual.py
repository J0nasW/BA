"""
RL MODULE WITH RANDOM SEARCH

CALL BY:    <random_search.py>

RETURN:     Parameter Matrices for the inverted Pendulum Problem
            Stores Data of best Parameters in '<date>_result_matrices_reward_<reward>.p'

INFO:       Manual Mode with main Method included
"""

# Some dependencies
import numpy as np # Maths and stuff
import gym.spaces # Simulating the Environments
import cPickle as pickle # Store Data into [.p] Files
import datetime # For Datestamp on stored files

from lif import I_syn_calc, I_gap_calc, U_neuron_calc
from parameters import *

'''
IMPORTANT VALUES - CHANGE THEM:
'''

simulations = 10000 # Number of Simulations - the more the better (and longer in terms of processing)

# Making Contact with Neurons through Synapses and Gap-Junctions----------------------------

# A = Connections between Interneurons through Synapses
A = np.matrix('0 0 -1 -1; 1 0 1 1; 1 1 0 1; -1 -1 0 0')
#A_rnd = np.random.rand(4,4)
#A_in = np.multiply(A, A_rnd)

# B = Connections between Sensory- and Interneurons through Synapses
B = np.matrix('0 1 1 0; 1 1 0 0; 0 0 1 1; 0 1 1 0')
#B_rnd = np.random.rand(4,4)
#B_in = np.multiply(B, B_rnd)


# A_gap = Connections between Interneurons through Gap-Junctions
A_gap = np.matrix('0 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 0')
#A_gap_rnd = np.random.rand(4,4)
#A_gap_in = np.multiply(A_gap, A_gap_rnd)

# B = Connections between Sensory- and Interneurons through Gap-Junctions
B_gap = np.matrix('0 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 0')
#B_gap_rnd = np.random.rand(4,4)
#B_gap_in = np.multiply(B_gap, B_gap_rnd)

#-------------------------------------------------------------------------------------------

# Current Matrix----------------------------------------------------------------------------

# Current Matrix for Symapses between inter Neurons
I_s_inter = np.zeros((4,4))
# Current Matrix for Synapses between sensory and inter Neurons
I_s_sensor = np.zeros((4,4))

# Current Matrix for Gap-Junctions between inter Neurons
I_g_inter = np.zeros((4,4))
# Current Matrix for Gap-Junctions between sensory and inter Neurons
I_g_sensor = np.zeros((4,4))

#-------------------------------------------------------------------------------------------

# Voltage Matrix----------------------------------------------------------------------------

x = [0,0,0,0] #AVA, AVD, PVC, AVB
u = [0,0,0,0] #PVD, PLM, AVM, ALM

#-------------------------------------------------------------------------------------------

# State Matrix------------------------------------------------------------------------------

fire = [0,0,0,0] #AVA, AVD, PVC, AVB

#-------------------------------------------------------------------------------------------

# Initialization----------------------------------------------------------------------------

def initialize(Default_U_leak):
    global totalreward, done, info

    # Initializing Neurons and Sensors------------------------------------------------------
    for i in range(0,4):
        x[i] = Default_U_leak
    for i in range(0,4):
        u[i] = Default_U_leak

    #OpenAI Gym Parameters------------------------------------------------------------------------------

    totalreward = 0
    done = 0
    info = 0

#-------------------------------------------------------------------------------------------

# Random Function---------------------------------------------------------------------------

def random_parameters():

    # Initialize random parameters for our Neurons and Synapses

    # For Synapses
    w_in_mat_rnd = np.random.uniform(low = 0, high = 3, size = (4,4))
    w_sin_mat_rnd = np.random.uniform(low = 0, high = 3, size = (4,4))

    sig_in_mat_rnd = np.random.uniform(low = 0.05, high = 0.5, size = (4,4))
    sig_sin_mat_rnd = np.random.uniform(low = 0.05, high = 0.5, size = (4,4))

    # For Gap-Junctions
    w_gap_in_mat_rnd = np.random.uniform(low = 0, high = 3, size = (4,4))
    w_gap_sin_mat_rnd = np.random.uniform(low = 0, high = 3, size = (4,4))

    # For Neurons
    C_m_mat_rnd = np.random.uniform(low = 0.001, high = 1, size = (1,4))
    G_leak_mat_rnd = np.random.uniform(low = 0.05, high = 5, size = (1,4))
    U_leak_mat_rnd = np.random.uniform(low = -80, high = -60, size = (1,4))

    return w_in_mat_rnd, w_sin_mat_rnd, sig_in_mat_rnd, sig_sin_mat_rnd, w_gap_in_mat_rnd, w_gap_sin_mat_rnd, C_m_mat_rnd, G_leak_mat_rnd, U_leak_mat_rnd

#-------------------------------------------------------------------------------------------

# Compute Function--------------------------------------------------------------------------

def compute(x, u, w_in_mat, w_sin_mat, w_gap_in_mat, w_gap_sin_mat, sig_in_mat, sig_sin_mat, C_m_mat, G_leak_mat, U_leak_mat):

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

    #---------------------------------------------------------------------------------------

    I_syn = np.add(I_syn_inter, I_syn_stimuli)
    I_gap = np.add(I_gap_inter, I_gap_stimuli)

    return x, u, fire, I_syn, I_gap

#-------------------------------------------------------------------------------------------

# OpenAI Gym--------------------------------------------------------------------------------

def run_episode(env, w_in_mat, w_sin_mat, sig_in_mat, sig_sin_mat, w_gap_in_mat, w_gap_sin_mat, C_m_mat, G_leak_mat, U_leak_mat):
    global x, u, fire, I_syn, I_gap, action

    observation = env.reset()
    totalreward = 0

    for t in np.arange(t0,T,delta_t): # RUNNING THE EPISODE - Trynig to get 200 Steps in this Episode

        # Compute the next Interneuron Voltages along with a possible "fire" Event - Now new with random parameter matrices
        x, u, fire, I_syn, I_gap = compute(x, u, w_in_mat, w_sin_mat, sig_in_mat, sig_sin_mat, w_gap_in_mat, w_gap_sin_mat, C_m_mat, G_leak_mat, U_leak_mat)

        # Decide for an action and making a Step
        if fire[0] == 1: # Sensory Neuron AVA is firing - resulting in a REVERSE Action (0)
            action = 0
            observation, reward, done, info = env.step(action)
            totalreward += reward
            #print 'RIGHT'
        elif fire[3] == 1: # Sensory Neuron AVB is firing - resulting in a FORWARD Action (1)
            action = 1
            observation, reward, done, info = env.step(action)
            totalreward += reward
            #print 'LEFT'
        else:
            #print 'Im not sure :( Going ',action
            #action = np.random.randint(0,1) # Tried a random approach - didn't seem to work
            observation, reward, done, info = env.step(action) # Have to use the action from the past time step - OpenAI Gym does not provide a "Do nothing"-Action
            totalreward += reward
        observe(observation)
        if done:
            break

    return totalreward

def observe(observation):
    global u

    cart_pos = observation[0] # [-2.4 2.4]
    #cart_vel = observation[1]
    angle = (observation[2] * 360) / (2 * np.pi) # in degrees [-12deg 12deg] (for Simulations)
    #angle_velocity = observation[3]

    # Adapt, learn, overcome-----------------------------------------------------------------------------------------

    # Setting the Angle of the Pole to Sensory Neurons PLM (Phi+) and AVM (Phi-)
    if angle > 0:
        u[1] = -70 + (50/12) * angle # PLM
        u[2] = -70
    elif angle == 0:
        u[1] = u[2] = -70
    else:
        u[2] = -70 + (50/12) * angle # AVM
        u[1] = -70

    # Setting the Cart Position to Sensory Neurons ALM (pos. movement) and PVD (neg. movement)
    if cart_pos > 0:
        u[3] = -70 + (50/2.4) * cart_pos # ALM
        u[0] = -70
    elif cart_pos == 0:
        u[0] = u[3] = -70
    else:
        u[0] = -70 + (50/2.4) * cart_pos # PVD
        u[3] = -70

    '''
    # Setting the Anglespeed of the Pole to Sensory Neurons ALM (Phi.+) and PVD (Phi.-)
    if angle_velocity >= 0:
        u[3] = -70 + (50/5) * angle_velocity # ALM
        u[0] = -70
    elif cart_pos == 0:
        u[0] = u[3] = -70
    else:
        u[0] = -70 + (50/5) * angle_velocity # PVD
        u[3] = -70
    '''

#------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------
# Main Function-----------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------

def main():
    global x, u, env, action

    env_vis = []
    action = 0
    episodes = 0
    best_reward = 0
    env = gym.make('CartPole-v0')

    for _ in xrange(simulations):

        initialize(Default_U_leak) # Initializing all Sensory- and Interneurons with the desired leakage voltage [-70mV]
        episodes += 1 # Episode Counter
        w_in_mat, w_sin_mat, sig_in_mat, sig_sin_mat, w_gap_in_mat, w_gap_sin_mat, C_m_mat, G_leak_mat, U_leak_mat = random_parameters() # Make some new random parameter Matrices
        reward = run_episode(env, w_in_mat, w_sin_mat, sig_in_mat, sig_sin_mat, w_gap_in_mat, w_gap_sin_mat, C_m_mat, G_leak_mat, U_leak_mat)
        if reward > best_reward:
            # Set current reward as new reward
            best_reward = reward
            # Save Results of the Run with the best reward
            Result = [w_in_mat, w_sin_mat, sig_in_mat, sig_sin_mat, w_gap_in_mat, w_gap_sin_mat, C_m_mat, G_leak_mat, U_leak_mat]
            # Solved the Simulation
            if reward == 200:
                break
        #print 'Episode',episodes,'mit Reward',reward,'.'

    print 'The best Reward was:',best_reward
    if best_reward == 200:
        print 'I SOLVED IT!'

    date = datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S")
    best_reward_s = str(int(best_reward))
    pickle.dump(Result, open(("parameter_dumps/" + date + "_result_matrices_reward_" + best_reward_s + ".p"), "wb"))

#-------------------------------------------------------------------------------------------


if __name__=="__main__":
    main()
