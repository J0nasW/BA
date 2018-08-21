"""
TEST Module for testing Purposes

CALL BY:    <TEST.py>

RETURN:     -

INFO:       -
"""

import os
import numpy as np

w_A_rnd = np.squeeze(np.random.uniform(low = 0.5, high = 3, size = (1,5)))
print (w_A_rnd)
w_A_rnd = np.append(w_A_rnd, w_A_rnd)
print (w_A_rnd)
