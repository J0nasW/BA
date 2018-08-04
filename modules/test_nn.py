"""
TEST Module for testing Purposes

CALL BY:    <TEST.py>

RETURN:     -

INFO:       -
"""

from lif import *


u_plm = -70
u_pvc = -70
u_avb = -70
u_syn_pvc = -70
u_syn_avb = -70

I_gap_1 = I_gap_calc(u_plm, u_pvc, 1)
I_syn_2 = I_syn_calc(u_pvc, u_avb, 0, 1, 0.2, -40)

u_syn_pvc, fire = U_neuron_calc(u_syn_pvc, 0, I_gap_1, 0, 0, 0.1, 3, -90, -20, 1)
u_syn_avb, fire = U_neuron_calc(u_syn_avb, I_syn_2, 0, 0, 0, 0.1, 3, -90, -20, 1)

print I_gap_1
print I_syn_2
print u_syn_pvc
print u_syn_avb
