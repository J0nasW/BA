# :mortar_board: Bachelor Thesis
## Eine Anwendung des Reinforcement Learning zur Regelung dynamischer Systeme
at the [Chair of Automatic Control](https://www.control.tf.uni-kiel.de/en) ([Faculty of Engineering](http://www.tf.uni-kiel.de/en) - [CAU Kiel](https://www.uni-kiel.de/en/))

This Git-Repository holds all the Code written for my Bachelor Thesis at the Chair of Automatic Control at TF Uni Kiel.
The Programming Language is Python.

---

### My Approach
To outperform established control-circuits and well known models in Control-Theroy, it takes more than a new Approach. But maybe this Idea is a new Way to solve problems like the Cart Pole "Inverted Pendulum" Problem. But instead of generating a neuronal Circuit from scratch like nearly every artificial Intelligence project, we use a already discovered neuronal Circuit: the touch-withdrawal neural circuit from C.Elegans []. With some modification and even more programming, we are able to simulate the well known leaky integrate and fire models and closing the loop to simulation environnments like OpenAI's GYM.

This Circuit is being used here and presented by many transition-matrices and state-vectors:
![TW-Neural Circuit of C. Elegans with some Modifications](https://github.com/J0nasW/BA/blob/master/img/Neural_Net.png "TW-Neural Circuit of C. Elegans with some Modifications")

---

### ToDo's and Thoughts:
* Simulation for my neural Circuit was initally to use Roboschool's inverted Pendulum. Because of the high degree of non-linerarity, a new approach is being developed: A new Simulation Environment containing a Wagon which starts some distance out of its reference point. Control Theory would suggest a PT2, the neural Network should solve this on it's own using no control Theory at all.
* Got the OpenAI Gym CartPole_v0 Environment up and running - now we have to develop a reliable Reward-System and some good reinforcement Learning algorithms - Maybe even getting to Epsilon-Greedy soon.. :blush:

Some Libraries that I used:
* [matplotlib](https://matplotlib.org/)
* [numpy](http://www.numpy.org/)
* [OpenAI Gym and Roboschool](https://gym.openai.com/)

---

#### Update Do., 12.07.2018
* Yet another feature folder added: TW-Circuit/new_circuit_symm now with integration on OpenAI Gym CartPole_v0. Still working on tuning the Parameters and weights, still haven't got a good reward system.. but we are getting there!

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
