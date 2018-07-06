# TW Circuit implementation in Python
Status: Under Construction
Current Step: Implement Computation Segment

---

The TW Circuit of C.Elegans is implemented using transition Matrices between Neurons:

* **A**: Neuron -> Neuron [5x5]
  * 1 = Excitatory Synapses
  * -1 = Inhibitory Synapse
* **B**: Sensor -> Neuron [4x5]
  * -1 = Inhibitory Synapse
* **x**: Neuron State [1x5] [mV]
* **u**: Sensor State [1x5] [mV]

> Weights and other Parameters are stored in Matrices the same Dimensions as Neuron and Sensory Matrices
