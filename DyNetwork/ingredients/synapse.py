"""
DyNetwork - Synapse

Defines a Synapse and can be imported easily.

Parameters:
 - Weight (Maximal conductanse of the Synapse)
 - SynapseType (Excitatory, Inhitory or Gap Junction)
 - Source (src)
 - sigma

 - id - for existing Synapses
"""

import os
import cPickle as pickle #To store and load binary files

SynapseType = {'Excitatory': 1, 'Inhitory': 2, 'GapJunction': 3}
i = 0

def AddSynapse(weight, SynapseType, src, sigma, i, network_id):
    synapse = {"id": i, "weight": weight, "SynapseType": SynapseType, "src": src, "sigma": sigma}

    try:
        os.makedirs('neural_networks/mynetwork',network_id)
    except OSError:
        if not os.path.isdir('neural_networks/mynetwork',network_id):
            raise

    pickle.dump(synapse, open(("s",i,".p"), "wb"))
    i += 1
    print SynapseType

def ChangeSynapse(id):
    print SynapseType

def DeleteSynapse(id):
    print "Synapse ",id," deleted."
