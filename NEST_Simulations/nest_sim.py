"""
Simulation of the TW Neural Circuit with NEST
https://github.com/nest/nest-simulator
"""

import pylab
import nest

neuron_1 = nest.Create("iaf_psc_alpha")
nest.SetStatus(neuron_1, {"I_e": 376.0})

multimeter = nest.Create("multimeter")
nest.SetStatus(multimeter, {"withtime":True, "record_from":["V_m"]})

spikedetector = nest.Create("spike_detector",
                params={"withgid": True, "withtime": True})

# Detect some LIF Action
nest.Connect(multimeter, neuron_1)
nest.Connect(neuron_1, spikedetector)

# Start the Simulation
nest.Simulate(1000.0)

# Data has been gathered within the Simulation and now we want to display it
dmm = nest.GetStatus(multimeter)[0]
Vms = dmm["events"]["V_m"]
ts = dmm["events"]["times"]

pylab.figure(1)
pylab.plot(ts, Vms)
dSD = nest.GetStatus(spikedetector,keys="events")[0]
evs = dSD["senders"]
ts = dSD["times"]
pylab.figure(2)
pylab.plot(ts, evs, ".")
pylab.show()
