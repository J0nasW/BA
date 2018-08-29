"""
TEST Module for testing Purposes

CALL BY:    <TEST.py>

RETURN:     -

INFO:       -
"""

import os
import numpy as np


A = np.arange(9).reshape((3,3))

B = np.array([0, 1, 3, 4, 6])

C = np.empty((0,5), int)

D = B.reshape((1, -1))

print (D)
