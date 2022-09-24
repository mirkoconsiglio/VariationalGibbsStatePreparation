import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import XXPlusYYGate
from qiskit.quantum_info import Pauli, Statevector, partial_trace, state_fidelity, DensityMatrix, entropy
from qiskit.opflow.primitive_ops import PauliOp
from qiskit.opflow.list_ops import SummedOp
from qiskit.extensions import HamiltonianGate
from scipy.linalg import expm

def trace_distance(rho1, rho2):
	diff = rho1 - rho2
	eigs = np.linalg.eig(diff)[0]
	return np.sum([np.abs(i) for i in eigs]) / 2

def free_energy(dm, operator, beta):
	return dm.expectation_value(operator).real - entropy(dm) / beta

def gibbs_state(hamiltonian, beta):
	dm = expm(-beta * hamiltonian)
	return dm / np.trace(dm)

def hamiltonian(n, J, h):
	ham = []
	for i in range(n):
		# Interaction terms
		if i != n - 1:
			XX = 'I' * i + 'XX' + 'I' * (n - i - 2)
		else:
			XX = 'X' + 'I' * (n - 2) + 'X'
		ham.append(PauliOp(Pauli(XX), -J))
		# Magnetic terms
		Z = 'I' * i + 'Z' + 'I' * (n - i - 1)
		ham.append(PauliOp(Pauli(Z), -h))

	# Build up the Hamiltonian
	ham = SummedOp(ham).reduce()

	print(ham)

	return ham

def cm(UI, Um, UA, n, collisions):
	qc = QuantumCircuit(n + 2)
	system = list(range(n))
	ancilla = [n, n + 1]  # Since ancilla is a mixed state we actually need another ancilla

	for c in range(collisions):
		qc.append(UA, ancilla)  # unitary gate to thermalize ancilla
		for i in range(n):
			qc.append(Um, [i, n])  # collision gates
		qc.append(UI, system)  # Ising gate
		# for i in range(n):
		# 	qc.append(Um, [i, n])  # collision gates
		qc.reset(ancilla)  # Reset ancilla

	print(qc)

	dm = partial_trace(DensityMatrix(qc), ancilla)

	return dm

def ising_unitary(operator, t):
	return HamiltonianGate(operator, t, label='Ising Gate')

def collision_unitary(g, t):
	return XXPlusYYGate(g * t, 0, label='Collision')

def ancilla_unitary(beta, omega):
	theta = 2 * np.arccos(1 / np.sqrt(1 + np.exp(beta * omega)))
	qc = QuantumCircuit(2, name='Ancilla gate')
	qc.ry(theta, 0)
	qc.cnot(0, 1)
	return qc

def main():
	# Parameters
	n = 3  # number of qubits
	J = 1  # interaction
	h = 0.5  # magnetic field
	beta = 0.01  # inverse temperature
	collisions = 10  # number of collisions
	g = 4
	t = 1
	omega = 1
	# Define Hamiltonian
	operator = hamiltonian(n, J, h)
	# Define gates
	UI = ising_unitary(operator, t)
	Um = collision_unitary(g, t / 2)
	UA = ancilla_unitary(beta, omega)
	# Output density matrix
	dm = cm(UI, Um, UA, n, collisions)
	# Calculate exact thermal state
	thermal_state = gibbs_state(operator.to_matrix(), beta)
	# Print Results
	print(f'Density Matrix Eigenvalues: {np.linalg.eigh(dm.data)[0]}')
	print(f'Thermal State Eigenvalues: {np.linalg.eigh(thermal_state)[0]}')
	print(f'Density Matrix Free Energy: {free_energy(dm, operator, beta)}')
	print(f'Thermal State Free Energy: {free_energy(DensityMatrix(thermal_state), operator, beta)}')
	print(f'Fidelity: {state_fidelity(DensityMatrix(thermal_state), dm)}')
	print(f'Trace Distance: {trace_distance(thermal_state, dm.data)}')

if __name__ == '__main__':
	main()
