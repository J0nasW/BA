"""
DyNetwork - Synapse

Defines a Synapse and can be imported easily.

Parameters:
 - Weight (Maximal conductanse of the Synapse)
 - SynapseType (Excitatory, Inhitory or Gap Junction)
 - Source (src)
 - sigma
"""

SynapseType = {'Excitatory': 1, 'Inhitory': 2, 'GapJunction': 3}

def AddSynapse(weight, SynapseType, src, sigma):
    print SynapseType
