import pickle
import os

import numpy as np
from importlib import abc
from numpy.linalg import eigh
from openfermion import qubit_operator_sparse
from openfermion.ops import QubitOperator
from scipy.linalg import norm, expm, logm
from gibbs_tket import Gibbs


def funm_psd(A, func):
	A = np.asarray(A)
	if len(A.shape) != 2:
		raise ValueError("Non-matrix input to matrix function.")
	w, v = eigh(A)
	w = np.maximum(w, 0)
	return (v * func(w)).dot(v.conj().T)


def save_data(folder, data):  # function to save kwargs as pickle file
	os.makedirs(f'{folder}', exist_ok=True)
	with open(f'{folder}/data.pkl', 'wb') as file:
		pickle.dump(data, file)


def _trace_distance(rho, sigma):
	return norm(rho - sigma, 1) / 2


def _fidelity(rho, sigma):
	sqrt_rho = funm_psd(rho, np.sqrt)
	return funm_psd(sqrt_rho @ sigma @ sqrt_rho, np.sqrt).diagonal().sum().real ** 2


def _relative_entropy(rho, sigma):
	return (rho @ (logm(rho) - logm(sigma))).diagonal().sum().real


def von_neumann_entropy(dm):
	return -np.sum([0 if i <= 0 else i * np.log(i) for i in np.linalg.eigh(dm)[0]])


def exact_gibbs_state(hamiltonian, beta):
	dm = expm(-beta * hamiltonian).real
	return dm / dm.diagonal().sum()


def xx_hamiltonian(n, J=1., h=0.5):
	hamiltonian = QubitOperator()
	for i in range(n):
		# Interaction terms
		if i != n - 1:
			hamiltonian += QubitOperator(f'X{i} X{i + 1}', -J)
		elif n > 2:
			hamiltonian += QubitOperator(f'X0 X{n - 1}', -J)
		# Magnetic terms
		hamiltonian += QubitOperator(f'Z{i}', -h)

	return hamiltonian


def xx_hamiltonian_commuting_terms(n, J=1., h=0.5):
	terms = dict()
	if J != 0:
		terms.update(xx_even_odd=[-J, [[i, i + 1] for i in range(n, 2 * n - 1, 2)]])
		if n > 2:
			terms.update(xx_odd_even=[-J, [[i, i + 1] for i in range(n + 1, 2 * n - 1, 2)]])
		if n % 2 == 1:
			terms.update(xx_closed=[-J, [[n, 2 * n - 1]]])
	if h != 0:
		terms.update(z=[-h, list(range(n, 2 * n))])

	return terms


class ExactResult:
	def __init__(self, hamiltonian, beta):
		self.hamiltonian_matrix = qubit_operator_sparse(hamiltonian).todense()
		self.gibbs_state = exact_gibbs_state(self.hamiltonian_matrix, beta)
		self.energy = (self.gibbs_state @ self.hamiltonian_matrix).diagonal().sum().real
		self.entropy = von_neumann_entropy(self.gibbs_state)
		self.cost = self.energy - self.entropy / beta
		self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.gibbs_state)
		self.hamiltonian_eigenvalues = np.linalg.eigh(self.hamiltonian_matrix)[0]


def main_multiple_beta():
	# Parameters
	n = 5  # number of qubits
	J = 1
	h = 1
	betas = [10 ** i for i in range(-3, 3)]
	shots = None  # Number of shots to sample
	# Define Hamiltonian
	hamiltonian = xx_hamiltonian(n, J, h)
	commuting_terms = xx_hamiltonian_commuting_terms(n, J, h)
	# Define minimizer kwargs
	min_kwargs = dict()
	# Run VQA
	gibbs = Gibbs(n)
	# Run over betas
	data = []
	for i, beta in enumerate(betas):
		calculated_result = gibbs.run(hamiltonian, beta, min_kwargs, shots=shots, commuting_terms=commuting_terms)
		# Calculate exact results
		exact_result = ExactResult(hamiltonian, beta)
		# Calculate comparative results
		fidelity = _fidelity(exact_result.gibbs_state, calculated_result.gibbs_state)
		trace_distance = _trace_distance(exact_result.gibbs_state, calculated_result.gibbs_state)
		relative_entropy = _relative_entropy(exact_result.gibbs_state, calculated_result.gibbs_state)
		overlaps = [np.dot(calculated_result.eigenvectors[:, i], exact_result.eigenvectors[:, i]) ** 2 for i in
		            range(2 ** n)]
		# Print results
		print()
		print(beta)
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
		for j, overlap in enumerate(overlaps):
			print(f'Eigenvector {j} overlap: {overlap}')
		# Save data
		data.append(dict(beta=beta, calculated_result=calculated_result, exact_result=exact_result, fidelity=fidelity,
		                 trace_distance=trace_distance, relative_entropy=relative_entropy, overlaps=overlaps))
	folder = f'{f"sampling_{shots}" if shots else "statevector"}_h_{h:.1f}/Gibbs_{n}'
	save_data(folder, data)


def main():
	# Parameters
	n = 2  # number of qubits
	J = 1
	h = 0.5
	beta = 1
	shots = 8192  # Number of shots to sample
	# Define Hamiltonian
	hamiltonian = xx_hamiltonian(n, J, h)
	commuting_terms = xx_hamiltonian_commuting_terms(n, J, h)
	# Define minimizer kwargs
	min_kwargs = dict(maxfev=5000)
	# Run VQA
	gibbs = Gibbs(n)
	calculated_result = gibbs.run(hamiltonian, beta, min_kwargs, shots=shots, commuting_terms=commuting_terms)
	# Calculate exact results
	exact_result = ExactResult(hamiltonian, beta)
	# Calculate comparative results
	fidelity = _fidelity(exact_result.gibbs_state, calculated_result.gibbs_state)
	trace_distance = _trace_distance(exact_result.gibbs_state, calculated_result.gibbs_state)
	relative_entropy = _relative_entropy(exact_result.gibbs_state, calculated_result.gibbs_state)
	overlaps = [np.abs(np.vdot(calculated_result.eigenvectors[:, i], exact_result.eigenvectors[:, i])) ** 2 for i in
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
	print(calculated_result.eigenvectors)
	print(exact_result.eigenvectors)
	for i, overlap in enumerate(overlaps):
		print(f'Eigenvector {i} overlap: {overlap}')


if __name__ == '__main__':
	main()
