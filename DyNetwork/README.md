# DyNetwork - A Python library for generating dynamic neural Networks based on the TW neural Circuit of C. Elegans

## main.py - Callable Functions and Documentation

tbd

---

### Ingriedients

#### synapse.py - Defining Synapses
This class is defining some useful functions to create synapses.

SynapseType:

|Key  |Type         |
|-----|-------------|
|1    |Excitatory   |
|2    |Inhitory     |
|3    |Gap-Junction |

#### neuron.py - Defining Neurons
Similar to synapse.py, this class defines Neurons and helps to create a network of them.

NeuronType:

|Key  |Type                                      |
|-----|------------------------------------------|
|1    |FWD (Locomotion)                          |
|2    |REV (Locomotion)                          |
|3    |PVD (Sensory - Cart Position (negative))  |
|4    |PLM (Sensory - Pendulum Angle (positive)) |
|5    |AVM (Sensory - Pendulum Angle (negative)) |
|6    |PVD (Sensory - Cart Position (positive))  |
|7    |AVA (Interneuron)                         |
|8    |AVB (Interneuron)                         |
|9    |AVD (Interneuron)                         |
|10   |PVC (Interneuron)                         |
|11   |DVA (Interneuron)                         |


---

### Dependencies

Packages used by this library:
* numpy
* matplotlib
