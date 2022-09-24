import pickle
import os

import numpy as np
from qiskit.opflow.list_ops import SummedOp
from qiskit.opflow.primitive_ops import PauliOp
from qiskit.quantum_info import Pauli
from scipy.linalg import expm, sqrtm, logm, norm
from gibbs_qiskit import Gibbs


def save_data(folder, data):  # function to save kwargs as pickle file
	os.makedirs(f'{folder}', exist_ok=True)
	with open(f'{folder}/data.pkl', 'wb') as file:
		pickle.dump(data, file)


def _trace_distance(rho, sigma):
	return norm(rho - sigma, 1) / 2


def _fidelity(rho, sigma):
	sqrt_rho = sqrtm(rho)
	return sqrtm(sqrt_rho @ sigma @ sqrt_rho).diagonal().sum().real ** 2


def _relative_entropy(rho, sigma):
	return (rho @ (logm(rho) - logm(sigma))).diagonal().sum().real


def von_neumann_entropy(dm):
	return -np.sum([0 if i <= 0 else i * np.log(i) for i in np.linalg.eigh(dm)[0]])


def exact_gibbs_state(hamiltonian, beta):
	dm = expm(-beta * hamiltonian).real
	return dm / dm.diagonal().sum()


def xx_hamiltonian(n, J=1., h=1.):
	ham = []
	for i in range(n):
		# Interaction terms
		if i != n - 1:
			XX = 'I' * i + 'XX' + 'I' * (n - i - 2)
			ham.append(PauliOp(Pauli(XX), -J))
		elif n > 2:
			XX = 'X' + 'I' * (n - 2) + 'X'
			ham.append(PauliOp(Pauli(XX), -J))
		# Magnetic terms
		Z = 'I' * i + 'Z' + 'I' * (n - i - 1)
		ham.append(PauliOp(Pauli(Z), -h))

	# Build up the Hamiltonian
	ham = SummedOp(ham)

	print(ham)

	return ham

class ExactResult:
	def __init__(self, hamiltonian, beta):
		self.hamiltonian_matrix = hamiltonian.to_matrix()
		self.gibbs_state = exact_gibbs_state(self.hamiltonian_matrix, beta)
		self.energy = (self.gibbs_state @ self.hamiltonian_matrix).diagonal().sum().real
		self.entropy = von_neumann_entropy(self.gibbs_state)
		self.cost = self.energy - self.entropy / beta
		self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.gibbs_state)
		self.hamiltonian_eigenvalues = np.linalg.eigh(self.hamiltonian_matrix)[0]


def main():
	# Parameters
	n = 2  # number of qubits
	J = 1
	h = 0.5
	beta = 10
	shots = None  # Number of shots to sample
	# Define Hamiltonian
	hamiltonian = xx_hamiltonian(n, J, h)
	# Define optimizer and kwargs
	min_kwargs = dict(maxfun=10000, initial_temp=0.1, restart_temp_ratio=1e-10,
	                  no_local_search=True, visit=2, accept=-100)
	# Run VQA
	gibbs = Gibbs(n, min_kwargs)
	calculated_result = gibbs.run(hamiltonian, beta, shots=shots)
	# Calculate exact results
	exact_result = ExactResult(hamiltonian, beta)
	# Calculate comparative results
	fidelity = _fidelity(exact_result.gibbs_state, calculated_result.gibbs_state)
	trace_distance = _trace_distance(exact_result.gibbs_state, calculated_result.gibbs_state)
	relative_entropy = _relative_entropy(exact_result.gibbs_state, calculated_result.gibbs_state)
	overlaps = [np.dot(calculated_result.eigenvectors[:, i], exact_result.eigenvectors[:, i]).real ** 2 for i in
	            range(2 ** n)]
	# Print results
	print(calculated_result.result)
	print()
	print(f'VQA cost: {calculated_result.cost}')
	print(f'Exact cost: {exact_result.cost}')
	print()
	print(f'VQA energy: {calculated_result.energy}')
	print(f'Exact energy: {exact_result.energy}')
	print()
	print(f'VQA entropy: {calculated_result.entropy}')
	print(f'Exact entropy: {exact_result.entropy}')
	print()
	print(f'VQA eigenvalues: {calculated_result.eigenvalues}')
	print(f'Exact eigenvalues: {exact_result.eigenvalues}')
	print()
	print(f'VQA Hamiltonian eigenvalues: {calculated_result.hamiltonian_eigenvalues}')
	print(f'Exact Hamiltonian eigenvalues: {exact_result.hamiltonian_eigenvalues}')
	print()
	print(f'Fidelity: {fidelity}')
	print(f'Trace Distance: {trace_distance}')
	print(f'Relative Entropy: {relative_entropy}')
	print()
	for i, overlap in enumerate(overlaps):
		print(f'Eigenvector {i} overlap: {overlap}')


if __name__ == '__main__':
	main()
