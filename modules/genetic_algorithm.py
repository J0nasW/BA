"""
RL MODULE WITH GENETIC ALGORITHMS

CALL BY:    <genetic_algorithm.py>

RETURN:     Parameter Matrices for the inverted Pendulum Problem
            Stores Data of best Parameters in '<date>_ga_<reward>.hkl'

INFO:       Still under development
"""

# Some dependencies
import os

import numpy as np # Maths and stuff
import gym.spaces # Simulating the Environments
import hickle as hkl # Performance Dumping in HDF5 Format <.hkl>
import time # For Runtime Evaluations
import datetime # For Datestamp on stored files
import matplotlib.pyplot as plt

from .lif import I_syn_calc, I_gap_calc, U_neuron_calc
from .parameters import *

def initialize_limits():
    # Some important Parameters for the genetic Algorithms--------------------------------------
    # Initial Limits of the Parameters

    w_low = 0
    w_high = 3
    w_limit = np.array([w_low, w_high])

    sig_low = 0.05
    sig_high = 0.5
    sig_limit = np.array([sig_low, sig_high])

    C_m_low = 0.01
    C_m_high = 1
    C_m_limit = np.array([C_m_low, C_m_high])

    G_leak_low = 0.1
    G_leak_high = 2.5
    G_leak_limit = np.array([G_leak_low, G_leak_high])

    U_leak_low = -70
    U_leak_high = -50
    U_leak_limit = np.array([U_leak_low, U_leak_high])

    return w_limit, sig_limit, C_m_limit, G_leak_limit, U_leak_limit


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

# Clear Function---------------------------------------------------------------------------

def clear():
    # Reset Generation-Episode-Counter
    gen_episodes = 0
    gen_reward_arr = []
    gen_selection_reward_arr = []
    # Reset Parameter Tensors
    dict_syn = {}
    id_syn = 0
    dict_neuro = {}
    id_neuro = 0
    # Reset Parameter Arrays
    w_A_sel = w_B_sel = w_B_gap_sel = w_sel = sig_A_sel = sig_B_sel = sig_sel = C_m_sel = G_leak_sel = U_leak_sel = np.array([])

    return gen_episodes, gen_reward_arr, gen_selection_reward_arr, dict_syn, id_syn, dict_neuro, id_neuro, w_A_sel, w_B_sel, w_B_gap_sel, w_sel, sig_A_sel, sig_B_sel, sig_sel, C_m_sel, G_leak_sel, U_leak_sel

#-------------------------------------------------------------------------------------------

# Random Function---------------------------------------------------------------------------

def random_parameters(w_limit, sig_limit, C_m_limit, G_leak_limit, U_leak_limit):

    # Initialize random parameters for our Neurons and Synapses according to the current Network

    # For Synapses
    w_A_rnd = np.squeeze(np.random.uniform(low = w_limit[0], high = w_limit[1], size = (1,nbr_of_inter_synapses)))
    w_B_rnd = np.squeeze(np.random.uniform(low = w_limit[0], high = w_limit[1], size = (1,nbr_of_sensor_synapses)))
    w_B_gap_rnd = np.squeeze(np.random.uniform(low = w_limit[0], high = w_limit[1], size = (1,nbr_of_gap_junctions)))

    sig_A_rnd = np.squeeze(np.random.uniform(low = sig_limit[0], high = sig_limit[1], size = (1,nbr_of_inter_synapses)))
    sig_B_rnd = np.squeeze(np.random.uniform(low = sig_limit[0], high = sig_limit[1], size = (1,nbr_of_sensor_synapses)))

    # For Neurons
    C_m_rnd = np.squeeze(np.random.uniform(low = C_m_limit[0], high = C_m_limit[1], size = (1,nbr_of_inter_neurons)))
    G_leak_rnd = np.squeeze(np.random.uniform(low = G_leak_limit[0], high = G_leak_limit[1], size = (1,nbr_of_inter_neurons)))
    U_leak_rnd = np.squeeze(np.random.uniform(low = U_leak_limit[0], high = U_leak_limit[1], size = (1,nbr_of_inter_neurons)))

    return w_A_rnd, w_B_rnd, w_B_gap_rnd, sig_A_rnd, sig_B_rnd, C_m_rnd, G_leak_rnd, U_leak_rnd

def random_parameters_symm(w_limit, sig_limit, C_m_limit, G_leak_limit, U_leak_limit):

    # Initialize symmetrical random parameters for our Neurons and Synapses according to the current Network

    # For Synapses
    w_A_rnd = np.squeeze(np.random.uniform(low = w_limit[0], high = w_limit[1], size = (1,nbr_of_inter_synapses_symm)))
    w_A_rnd = np.append(w_A_rnd, np.flip(w_A_rnd))
    w_B_rnd = np.squeeze(np.random.uniform(low = w_limit[0], high = w_limit[1], size = (1,nbr_of_sensor_synapses_symm)))
    w_B_rnd = np.append(w_B_rnd, np.flip(w_B_rnd))
    w_B_gap_rnd = np.squeeze(np.random.uniform(low = w_limit[0], high = w_limit[1], size = (1,nbr_of_gap_junctions_symm)))
    w_B_gap_rnd = np.append(w_B_gap_rnd, np.flip(w_B_gap_rnd))

    sig_A_rnd = np.squeeze(np.random.uniform(low = sig_limit[0], high = sig_limit[1], size = (1,nbr_of_inter_synapses_symm)))
    sig_A_rnd = np.append(sig_A_rnd, np.flip(sig_A_rnd))
    sig_B_rnd = np.squeeze(np.random.uniform(low = sig_limit[0], high = sig_limit[1], size = (1,nbr_of_sensor_synapses_symm)))
    sig_B_rnd = np.append(sig_B_rnd, np.flip(sig_B_rnd))

    # For Neurons
    C_m_rnd = np.squeeze(np.random.uniform(low = C_m_limit[0], high = C_m_limit[1], size = (1,nbr_of_inter_neurons_symm)))
    C_m_rnd = np.append(C_m_rnd, np.flip(C_m_rnd))
    G_leak_rnd = np.squeeze(np.random.uniform(low = G_leak_limit[0], high = G_leak_limit[1], size = (1,nbr_of_inter_neurons_symm)))
    G_leak_rnd = np.append(G_leak_rnd, np.flip(G_leak_rnd))
    U_leak_rnd = np.squeeze(np.random.uniform(low = U_leak_limit[0], high = U_leak_limit[1], size = (1,nbr_of_inter_neurons_symm)))
    U_leak_rnd = np.append(U_leak_rnd, np.flip(U_leak_rnd))


    return w_A_rnd, w_B_rnd, w_B_gap_rnd, sig_A_rnd, sig_B_rnd, C_m_rnd, G_leak_rnd, U_leak_rnd

#-------------------------------------------------------------------------------------------

# Plot Function-----------------------------------------------------------------------------

def plot(w_limit_arr_low, w_limit_arr_high, sig_limit_arr_low, sig_limit_arr_high, C_m_limit_arr_low, C_m_limit_arr_high, G_leak_limit_arr_low, G_leak_limit_arr_high, U_leak_limit_arr_low, U_leak_limit_arr_high, reward_arr_plot, best_reward_arr_plot):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=20)

    plt.figure(1)
    #plt.suptitle('TW Circuit Simulator - Genetic Algorithm - SYNAPSE', fontsize=16)

    plt.subplot(121)
    plt.title('$\omega_{limit}$', fontsize=22)
    plt.plot(w_limit_arr_low, '-b', label='$\omega_{low}$', linewidth=1)
    plt.plot(w_limit_arr_high, '-g', label='$\omega_{high}$', linewidth=1)
    plt.xlabel('Generation')
    plt.ylabel('$S$')
    plt.legend(loc='upper left')

    plt.subplot(122)
    plt.title('$\sigma_{limit}$', fontsize=22)
    plt.plot(sig_limit_arr_low, '-b', label='$\sigma_{low}$', linewidth=1)
    plt.plot(sig_limit_arr_high, '-g', label='$\sigma_{high}$', linewidth=1)
    plt.xlabel('Generation')
    plt.legend(loc='upper left')

    plt.figure(2)
    #plt.suptitle('TW Circuit Simulator - Genetic Algorithm - NEURON', fontsize=16)

    plt.subplot(131)
    plt.title('$C_{m\_limit}$', fontsize=22)
    plt.plot(C_m_limit_arr_low, '-b', label='$C_{m\_low}$', linewidth=1)
    plt.plot(C_m_limit_arr_high, '-g', label='$C_{m\_high}$', linewidth=1)
    plt.xlabel('Generation')
    plt.ylabel('$mF$')
    plt.legend(loc='upper left')

    plt.subplot(132)
    plt.title('$G_{leak\_limit}$', fontsize=22)
    plt.plot(G_leak_limit_arr_low, '-b', label='$G_{leak\_low}$', linewidth=1)
    plt.plot(G_leak_limit_arr_high, '-g', label='$G_{leak\_high}$', linewidth=1)
    plt.xlabel('Generation')
    plt.ylabel('$S$')
    plt.legend(loc='upper left')

    plt.subplot(133)
    plt.title('$U_{leak\_limit}$', fontsize=22)
    plt.plot(U_leak_limit_arr_low, '-b', label='$U_{leak\_low}$', linewidth=1)
    plt.plot(U_leak_limit_arr_high, '-g', label='$U_{leak\_high}$', linewidth=1)
    plt.xlabel('Generation')
    plt.ylabel('$mV$')
    plt.legend(loc='upper left')

    '''
    plt.figure(3)
    plt.suptitle('TW Circuit Simulator - GENETICAL ALGORITHM - REWARD', fontsize=16)

    plt.subplot(121)
    plt.title('Best Reward of Selection', fontsize=10)
    plt.plot(best_reward_arr_plot, '-r', label='Reward', linewidth=1)
    plt.xlabel('Selection')
    plt.ylabel('Reward')
    plt.legend(loc='upper left')

    plt.subplot(122)
    plt.title('Best Reward of every EPISODE', fontsize=10)
    plt.plot(reward_arr_plot, '-b', label='Reward', linewidth=1)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend(loc='upper left')
    '''


    plt.show()

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
                I_s_inter[i, j] = I_syn_calc(x[i], x[j], E_in, w_A_rnd[k], sig_A_rnd[k], mu)
                k += 1
            elif A[i, j] == 2:
                # Inhibitory Synapse
                I_s_inter[i, j] = I_syn_calc(x[i], x[j], E_ex, w_A_rnd[k], sig_A_rnd[k], mu)
                k += 1
            else:
                # No Connection here.
                I_s_inter[i, j] = 0


            # Synapse Currents between Sensory and Interneurons
            if B[i, j] == 1:
                # Inhibitory Synapse (can't be Excitatory)
                I_s_sensor[i, j] = I_syn_calc(u[i], u[j], E_in, w_B_rnd[l], sig_B_rnd[l], mu)
                l += 1
            elif B[i, j] == 2:
                I_s_sensor[i, j] = I_syn_calc(u[i], u[j], E_ex, w_B_rnd[l], sig_B_rnd[l], mu)
                l += 1
            elif B[i, j] == 3:
                # Gap Junction
                I_g_sensor[i, j] = I_gap_calc(u[i], x[j], w_B_gap_rnd[m])
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
        x[i], fire[i] = U_neuron_calc(x[i], I_syn_inter[i], I_gap_inter[i], I_syn_stimuli[i], I_gap_stimuli[i], C_m_rnd[i], G_leak_rnd[i], U_leak_rnd[i], v, delta_t)

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
        u[1] = Default_U_leak + ((v-Default_U_leak)/max_angle) * np.absolute(angle) # PLM
        u[2] = Default_U_leak
    elif angle == 0:
        u[1] = u[2] = Default_U_leak
    else:
        u[2] = Default_U_leak + ((v-Default_U_leak)/max_angle) * np.absolute(angle) # AVM
        u[1] = Default_U_leak

    if SecondObservation == "cart":
        # Setting the Cart Position to Sensory Neurons ALM (pos. movement) and PVD (neg. movement)
        if cart_pos > 0:
            u[3] = Default_U_leak + ((v-Default_U_leak)/max_cart_pos) * np.absolute(cart_pos) # ALM
            u[0] = Default_U_leak
        elif cart_pos == 0:
            u[0] = u[3] = Default_U_leak
        else:
            u[0] = Default_U_leak + ((v-Default_U_leak)/max_cart_pos) * np.absolute(cart_pos) # PVD
            u[3] = Default_U_leak
    elif SecondObservation == "angle":
        # Setting the Anglespeed of the Pole to Sensory Neurons ALM (Phi.+) and PVD (Phi.-)
        if angle_velocity >= 0:
            u[0] = Default_U_leak + ((v-Default_U_leak)/max_angle_velocity) * np.absolute(angle_velocity) # ALM
            u[3] = Default_U_leak
        elif cart_pos == 0:
            u[0] = u[3] = Default_U_leak
        else:
            u[3] = Default_U_leak + ((v-Default_U_leak)/max_angle_velocity) * np.absolute(angle_velocity) # PVD
            u[0] = Default_U_leak


    return angle


#------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------
# Main Function-----------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------

def main(sim_time,plot_choice):
    global x, u, env, action

    start_time = time.time() # for Runtime reasons

    # Initialize the Environment and some vital Parameters
    action = 0
    episodes = 0
    gen_episodes = 0
    best_reward = 0
    env = gym.make('CartPole-v0')

    # Initialize Genetic Arrays
    current_parameters_syn = np.array([])
    current_parameters_neuro = np.array([])

    # Initialize Parameter Tensors (as Python Dictionaries)
    dict_syn = {}
    id_syn = 0
    dict_neuro = {}
    id_neuro = 0

    #gen_parameter_mat_syn = np.empty((0,5), int)
    #gen_parameter_mat_neuro = np.empty((0,3), int)

    gen_reward_arr = np.array([])
    gen_selection_reward_arr = np.array([])
    gen_selection_parameter_mat_syn = np.empty((0,5), int)
    gen_selection_parameter_mat_neuro = np.empty((0,3), int)
    w_A_sel = w_B_sel = w_B_gap_sel = w_sel = sig_A_sel = sig_B_sel = sig_sel = C_m_sel = G_leak_sel = U_leak_sel = np.array([])
    w_limit, sig_limit, C_m_limit, G_leak_limit, U_leak_limit = initialize_limits()

    # Some plotting Arrays
    w_limit_arr_low = w_limit_arr_high = sig_limit_arr_low = sig_limit_arr_high = C_m_limit_arr_low = C_m_limit_arr_high = G_leak_limit_arr_low = G_leak_limit_arr_high = U_leak_limit_arr_low = U_leak_limit_arr_high = reward_arr_plot = best_reward_arr_plot = np.array([])


    while True:
        initialize(Default_U_leak) # Initializing all Sensory- and Interneurons with the desired leakage voltage [-70mV]

        episodes += 1 # Episode Counter
        gen_episodes +=1

        if IsSymmetrical == True:
            w_A_rnd, w_B_rnd, w_B_gap_rnd, sig_A_rnd, sig_B_rnd, C_m_rnd, G_leak_rnd, U_leak_rnd = random_parameters_symm(w_limit, sig_limit, C_m_limit, G_leak_limit, U_leak_limit) # Make some new random parameter Matrices
        elif IsSymmetrical == False:
            w_A_rnd, w_B_rnd, w_B_gap_rnd, sig_A_rnd, sig_B_rnd, C_m_rnd, G_leak_rnd, U_leak_rnd = random_parameters(w_limit, sig_limit, C_m_limit, G_leak_limit, U_leak_limit) # Make some new random parameter Matrices

        reward = run_episode(env, w_A_rnd, w_B_rnd, w_B_gap_rnd, sig_A_rnd, sig_B_rnd, C_m_rnd, G_leak_rnd, U_leak_rnd)

        # Genetic stuff --------------------------

        # Generates a Matrix of Parameters in the current Generation (100x8)
        current_parameters_syn = np.array([w_A_rnd, w_B_rnd, np.append(w_B_gap_rnd,[0,0,0,0,0,0]), sig_A_rnd, sig_B_rnd])
        current_parameters_neuro = np.array([C_m_rnd, G_leak_rnd, U_leak_rnd])

        # Generate Tesors for Matrices through time
        dict_syn.update({id_syn:current_parameters_syn}) # 5x8x100
        dict_neuro.update({id_neuro:current_parameters_neuro}) # 3x4x100
        id_syn += 1
        id_neuro += 1

        # Generates a Vector of Rewards in the current Generation (1x100)
        gen_reward_arr = np.append(gen_reward_arr, reward)

        if gen_episodes >= 100:
            # Selection of the current Generation (10 out of 100 Episodes with the best Rewards)
            gen_selection_reward_arr = np.argpartition(gen_reward_arr, -10)[-10:]
            for i in range(10):
                # Get indices of the best 10 Episodes
                j = gen_selection_reward_arr[i]

                w_A_sel = np.append(w_A_sel, dict_syn.get(j)[0,:])
                w_B_sel = np.append(w_B_sel, dict_syn.get(j)[1,:])
                w_B_gap_sel = np.append(w_B_gap_sel, dict_syn.get(j)[2,(0,1)])
                sig_A_sel = np.append(sig_A_sel, dict_syn.get(j)[3,:])
                sig_B_sel = np.append(sig_B_sel, dict_syn.get(j)[4,:])

                C_m_sel = np.append(C_m_sel, dict_neuro.get(j)[0,:])
                G_leak_sel = np.append(G_leak_sel, dict_neuro.get(j)[1,:])
                U_leak_sel = np.append(U_leak_sel, dict_neuro.get(j)[2,:])

            # Get the Parameter-Vektors of the Selection (each 1x10)
            w_sel = np.append(w_A_sel, w_B_sel)
            w_sel = np.append(w_sel, w_B_gap_sel)
            sig_sel = np.append(sig_A_sel, sig_B_sel)

            # Get maximum/minimum values for new Limits
            w_limit = np.array([np.amin(w_sel), np.amax(w_sel)])
            sig_limit = np.array([np.amin(sig_sel), np.amax(sig_sel)])
            C_m_limit = np.array([np.amin(C_m_sel), np.amax(C_m_sel)])
            G_leak_limit = np.array([np.amin(G_leak_sel), np.amax(G_leak_sel)])
            U_leak_limit = np.array([np.amin(U_leak_sel), np.amax(U_leak_sel)])

            if plot_choice == 1:
                # For plotting purposes
                w_limit_arr_low = np.append(w_limit_arr_low, w_limit[0])
                w_limit_arr_high = np.append(w_limit_arr_high, w_limit[1])
                sig_limit_arr_low = np.append(sig_limit_arr_low, sig_limit[0])
                sig_limit_arr_high = np.append(sig_limit_arr_high, sig_limit[1])
                C_m_limit_arr_low = np.append(C_m_limit_arr_low, C_m_limit[0])
                C_m_limit_arr_high = np.append(C_m_limit_arr_high, C_m_limit[1])
                G_leak_limit_arr_low = np.append(G_leak_limit_arr_low, G_leak_limit[0])
                G_leak_limit_arr_high = np.append(G_leak_limit_arr_high, G_leak_limit[1])
                U_leak_limit_arr_low = np.append(U_leak_limit_arr_low, U_leak_limit[0])
                U_leak_limit_arr_high = np.append(U_leak_limit_arr_high, U_leak_limit[1])
                best_reward_arr_plot = np.append(best_reward_arr_plot, gen_reward_arr[np.argpartition(gen_reward_arr, -1)[-1:]])

            #Reset and Clear
            gen_episodes, gen_reward_arr, gen_selection_reward_arr, dict_syn, id_syn, dict_neuro, id_neuro, w_A_sel, w_B_sel, w_B_gap_sel, w_sel, sig_A_sel, sig_B_sel, sig_sel, C_m_sel, G_leak_sel, U_leak_sel = clear()


        if reward > best_reward:
            # Set current reward as new best_reward
            best_reward = reward
            # Save Results of the Run with the best reward
            Result = [w_A_rnd, w_B_rnd, w_B_gap_rnd, sig_A_rnd, sig_B_rnd, C_m_rnd, G_leak_rnd, U_leak_rnd]

        reward_arr_plot = np.append(reward_arr_plot, best_reward)

        if (time.time() - start_time) >= sim_time:
            # End Simulation-Run, if given Simulation time is elapsed.
            break

    # Prepare some Information to dump..
    date = datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S")
    best_reward_s = str(int(best_reward))
    episodes = str(int(episodes))
    elapsed_time = str((time.time() - start_time))

    hkl.dump(Result, (current_dir + "/parameter_dumps/" + date + "_ga_" + best_reward_s + ".hkl"), mode='w')
    # Information Text File
    file = open((current_dir + "/information/" + date + "_parameter_run_ga_" + best_reward_s + ".txt"), "w")
    file.write(("---GENETIC ALGORITHM---\nParameter run from " + date + " with Reward " + best_reward_s + " and " + episodes + " Episodes.\nSimulation Runtime was " + elapsed_time + "."))
    file.close()

    # Console Prints
    print ('The best Reward was:',best_reward)

    if plot_choice == 1:
        plot(w_limit_arr_low, w_limit_arr_high, sig_limit_arr_low, sig_limit_arr_high, C_m_limit_arr_low, C_m_limit_arr_high, G_leak_limit_arr_low, G_leak_limit_arr_high, U_leak_limit_arr_low, U_leak_limit_arr_high, reward_arr_plot, best_reward_arr_plot)

    print("--- %s seconds ---" % (time.time() - start_time))
    return date, best_reward_s

#-------------------------------------------------------------------------------------------


if __name__=="__main__":
    main()
