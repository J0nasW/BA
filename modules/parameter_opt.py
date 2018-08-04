"""
PARAMETER OPTIMIZATION with Hyperopt

IMPORT BY:  <parameter_opt.py>

RETURN:     Several Parameters

INFO:       All Parameters can be changed only here and spread over the whole Project
            Motor Neurons: FWD, REV
            Sensory Neurons: PVD, PLM, AVM, ALM
            Inter Neurons: AVA, AVD, PVC, AVB
"""

from hyperopt import fmin, tpe, hp


best = fmin(fn=lambda x: x ** 2,
    space=hp.uniform('x', -10, 10),
    algo=tpe.suggest,
    max_evals=100)
print (best)
