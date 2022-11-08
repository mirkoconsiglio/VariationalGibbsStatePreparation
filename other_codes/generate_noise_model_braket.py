import json

import numpy as np
from braket.circuits.noise_model import NoiseModel, GateCriteria, ObservableCriteria
from braket.circuits.noises import AmplitudeDamping, PhaseDamping, Depolarizing, BitFlip, \
	TwoQubitDepolarizing, TwoQubitDephasing

# number of qubits
n = 4
# Timings
gate_time_one_qubit = 0.00001
gate_time_two_qubit = 0.0002
t1 = 10000
t2 = 0.2
# Errors
one_qubit_t1 = 1 - np.exp(-gate_time_one_qubit / t1)
one_qubit_t2 = 0.5 * (1 - np.exp(-gate_time_one_qubit / t2))
one_qubit_depo = 1 - 0.9986
two_qubit_t1 = 1 - np.exp(-gate_time_two_qubit / t1)
two_qubit_t2 = 0.5 * (1 - np.exp(-gate_time_two_qubit / t2))
two_qubit_depo = 1 - 0.9695
avg_readout = 1 - 0.99752
# Create noise model
noise_model = NoiseModel()
# One qubit errors
for i in range(n):
	noise_model.add_noise(AmplitudeDamping(one_qubit_t1), GateCriteria(qubits=i))
	noise_model.add_noise(PhaseDamping(one_qubit_t2), GateCriteria(qubits=i))
	noise_model.add_noise(Depolarizing(one_qubit_depo), GateCriteria(qubits=i))
# Two qubit errors
for i in range(n):
	for j in range(i + 1, n):
		noise_model.add_noise(TwoQubitDephasing(two_qubit_t2), GateCriteria(qubits=[(i, j), (j, i)]))
		noise_model.add_noise(TwoQubitDepolarizing(two_qubit_depo), GateCriteria(qubits=[(i, j), (j, i)]))
# Read out errors
noise_model.add_noise(BitFlip(avg_readout), ObservableCriteria())
# Save noise model
with open('../qulacs/noise_model.json', 'w') as f:
	json.dump(noise_model.to_dict(), f)
