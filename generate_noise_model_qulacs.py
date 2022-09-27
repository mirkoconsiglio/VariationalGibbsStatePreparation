import numpy as np
import json
# Timings
gate_time_one_qubit = 50e-9
gate_time_two_qubit = 100e-9
t1 = 100e-6
t2 = 100e-6
# Errors
one_qubit_t1 = 1 - np.exp(-gate_time_one_qubit / t1)
one_qubit_t2 = 0.5 * (1 - np.exp(-gate_time_one_qubit / t2))
one_qubit_depo = 1e-4
two_qubit_t1 = 1 - np.exp(-gate_time_two_qubit / t1)
two_qubit_t2 = 0.5 * (1 - np.exp(-gate_time_two_qubit / t2))
two_qubit_depo = 1e-3
readout = 1e-3

noise_model = dict(one_qubit_t1=one_qubit_t1,
                   one_qubit_t2=one_qubit_t2,
                   one_qubit_depo=one_qubit_depo,
                   two_qubit_t1=two_qubit_t1,
                   two_qubit_t2=two_qubit_t2,
                   two_qubit_depo=two_qubit_depo,
                   readout=readout)

with open('noise_model.json', 'w') as f:
	json.dump(noise_model, f)
