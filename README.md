# :mortar_board: Bachelor Thesis
at the [Chair of Automatic Control](https://www.control.tf.uni-kiel.de/en) ([Faculty of Engineering](http://www.tf.uni-kiel.de/en) - [CAU Kiel](https://www.uni-kiel.de/en/))

## Eine Anwendung des Reinforcement Learning zur Regelung dynamischer Systeme

This Git-Repository holds all the Code written for my Bachelor Thesis at the Chair of Automatic Control at TF Uni Kiel.
The Programming Language is Python.
NOW IN PYTHON 3!

---

### My Approach
This bachelor thesis implements a way to achieve a reliable and stable control of a dynamic system through approaches of reinforcement learning. For a neuronal network, the "Touch Withdrawal Circuit" of the worm C. Elegans is examined in great detail and the structures are transformed into a simulator. As a simulation environment, the inverted pendulum is being used with one degree of freedom (1 DOF).
To simulate the neural network and gurantee reliable control for the inverted pendulum, a simulator is being developed and implemented using the programming languarge Python. Using the well known Leaky Integrate and Fire model, simulation of internal neural dynamics and processing information within the network is made possible. Furthermore, Pparameters of the network are found using reinforcement learning algorithms and applied to the environment CartPole v0 from OpenAI Gym.
The result of this work shows, that it is possible to implement a functional simulator for biological neural networks and to link it with methods of reinforcement learning. After computing multiple simulations, suitable Parameters for the network, which ensure stable control of the inverse pendulum, are found. An application to other simulation environments or with similar neural networks is also possible due to the structure of the simulator.

<img src="https://github.com/J0nasW/BA/blob/master/docs/BA_Abgabe/LaTeX/figures/chap_neuron/Neural_Net_v3_plain.png" alt="TW-Neural Circuit of C. Elegans with some Modifications" vspace="10" align="middle" width="500px"/>

[![YT Video CartPole_v0](http://img.youtube.com/vi/NKdNCSEsll8/0.jpg)](http://www.youtube.com/watch?v=NKdNCSEsll8)

---

### ToDo's and Thoughts:
* Simulation for my neural Circuit was initally to use Roboschool's inverted Pendulum. Because of the high degree of non-linerarity, a new approach is being developed: A new Simulation Environment containing a Wagon which starts some distance out of its reference point. Control Theory would suggest a PT2, the neural Network should solve this on it's own using no control Theory at all. - Depricated due to the good integration of CartPole_v0 from OpenAI Gym
* Got the OpenAI Gym CartPole_v0 Environment up and running - now we have to develop a reliable Reward-System and some good reinforcement Learning algorithms
* Genetic Algorithms still under development.. Need to implement some tensor multiplications

Some Libraries that I used:
* [matplotlib](https://matplotlib.org/)
* [numpy](http://www.numpy.org/)
* [OpenAI Gym and Roboschool](https://gym.openai.com/)
* [Hickle](https://github.com/telegraphic/hickle)

Computing is getting more and more complex and takes a huge amount of resources. I'll be using the Google Cloud Plattform from now on to train parameters and weights of my neural Network. More Information in my Bachelor Thesis

---
#### Update Mi., 30.08.2018
* Bachelor Thesis is finally done and being reviewed by several people.
* Genetic Algorithm gets more and more complex
* Cleaning up this Repo - Everything under the master branch now

#### Update Di., 21.08.2018
* Implemented a GUI into main.py for Desktop use and created main_cmd.py for headless simulation use
* More writing for the BA Thesis (see Commit History)

#### Update So., 19.08.2018
* Wrote the last chapters in my Bachelor Thesis and made the first commit - more to come
* More writing for the BA Thesis (see Commit History)

#### Update Fr., 17.08.2018
* New Simulation Results from GCP came in - Parameter Run hit the 200 Reward - Mark and performed very well even without weight optimization
* Did a Screencast of a good run - see /footage
* More writing for the BA Thesis (see Commit History)

#### Update Mi., 15.08.2018
* New approach concerning Parameter sweeps - been simulating only half of the desired parameters and then duplicating them for symmetrical use - did improve performance very much
* More writing for the BA Thesis (see Commit History)

#### Update Mo., 13.08.2018
* Some little Performance tweaks as well as cleaning up the Repo..
* More writing for the BA Thesis (see Commit History)

#### Update, Sa., 11.08.2018
* Have been writing a lot this week and tewaking a little bit on my Observe-Method.
* Did some Commentarys on my code as well as some very important little changes
* Added LATEX Files containing my Bachelor Thesis. 4 Weeks until deadline.

#### Update Fr., 05.08.2018
* Made my Code ready for Google Cloud Plattform to get some serious simulation done
* Try merging to master-branch

#### Update Mi., 01.08.2018
* Merged NeuralCircuit_v2 into master-Branch and did some Debugging
* Finally switched to Python3 in order to use tensorflow for my next projects
* More Error-Debugging and logically thinking about the odd behaviour. Maybe I figure it out in the next days.
* New FileSaver "Hickle" for improved Performance and cross-version Compatibility.
* Got the Google Cloud Plattform Up and Running for training my Network - faster, efficient and free.
* Opened my first GitHub Project Kanban Board to keep track of my tasks

#### Update Di. 31.07.2018
* Created a new Feature-Branch "NeuralCircuit_v2" - rather a Version 2 of my original neural Circuit with many improvements - still trying to figure out this odd "LEFT, RIGHT, LEFT, ..." - Behaviour
* Added Runtimes to calculate how long the trainings will take - need to get some kind of cloud support :)

#### Update Mo., 30.07.2018
* Worked very hard on gathering information and writing my Thesis over the last week
* Made some serious changes in the Ecosystem - main.py as the caller-file for modules
* Created random_search_v2 with serious performance improvements and lightweight architecture
* Implemented a new weight-logic and increased my average score from 20/200 to 150/200 - still working on visiualizing the good simulations
* Still somme odd characteristics within the circuit. Maybe letting the RL-Algrithms take control over synapses...

#### Update Mi., 18.07.2018
* Halftime-Talk with Prof. Meurer and Alexander Schaum, showing my current Work and presenting the Slide (see *docs*)
* Got some more Simulation-Runs and I am pretty sure now, why the inverted Pendulum behaves incorrectly sometimes - implementation to be done

#### Update Di., 17.07.2018
* Made a new Inspection Module to print out the parameter Matrices
* First preperations for a genetic algorithm to learn new Parameters
* Did a Status Slide to present my current work on Wed., 18.07.2018 to Prof. Meurer - see *docs*

#### Update Mo., 16.07.2018
* Completely new structures within my Repo - Splitted the Code from main.py to some modules to ensure modular building structures and better work on features.
* Still need to increase my OpenAI Gym Score - it is still not higher than 30/200...

#### Update Fr., 13.07.2018
* Random Search for parameter matrices is finally working now. The OpenAI Gym Environment 'Cartpole_v0' performs well and leaves me with a reward of around 50/200 after 10.000 Simulations. It is important to know, that we are randomly simulating 6 [4x4] Matrices and 4 [4x1] vectors.
* New Ideas for further tuning: Genetic Algorithms and Function minimizing through Cost-Function

#### Update Do., 12.07.2018
* Yet another feature folder added: TW-Circuit/new_circuit_symm now with integration on OpenAI Gym CartPole_v0. Still working on tuning the Parameters and weights, still haven't got a good reward system.. but we are getting there!
* Did some Clean-Up to this Repository - only the important code is hosted, other things will remain on my private NextCloud Server. Also tried Branching - need some more info.

#### Update Di., 10.07.2018
* TW-Circuit/new_circuit_SIMCE released based on SIM-CE Simulink Simulation - New neual Network with potential!
* TW-Circuit/new_circuit_TWCIRCUIT released based on Worm Level Control Paper with advanced Matrices and more computational power - Working on this for the next days and maybe implement some kind of simple Simulation (see ToDo's)

#### Update Mo., 09.07.2018
* TW-Circuit now takes Gap-Junction-Currents - BUT still outputs weird Plots and Results - PLEASE HELP!

#### Update Fr. 29.06.2018
* Some breakthroughs at modelling the formulas from the Paper "TW Neuronal Circuit" - got a very promising LIF Simulation out of variating the Constants in Project "TW-Circuit" - will continue this Idea and maybe implement a Network from there.

#### Update Fr. 22.06.2018
* Added a new Project "WaterLevel_RL" which was a first try to implement a Water Level regulator with a nerual network - still under development and not quite sure, if I continue this Idea

#### Update Mo. 18.06.2018
* Implementation of a very substantial neural network in lif_model/single_neuron_net.py


#### Update Fr. 15.06.2018
* New Library "DyNetwork" for my own neural Network and some other purposes initialized - Still no serious content.
