import os
import pickle

import numpy as np
from numpy.linalg import eigh
from scipy.linalg import norm, logm, expm
from openfermion import QubitOperator
from openfermion.linalg import qubit_operator_sparse


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


def trace_distance(rho, sigma):
	return norm(rho - sigma, 1) / 2


def fidelity(rho, sigma):
	sqrt_rho = funm_psd(rho, np.sqrt)
	return funm_psd(sqrt_rho @ sigma @ sqrt_rho, np.sqrt).diagonal().sum().real ** 2


def relative_entropy(rho, sigma):
	return (rho @ (logm(rho) - logm(sigma))).diagonal().sum().real


def von_neumann_entropy(dm):
	return -np.sum([0 if i <= 0 else i * np.log(i) for i in np.linalg.eigh(dm)[0]])


def exact_gibbs_state(hamiltonian, beta):
	dm = expm(-beta * hamiltonian).real
	return dm / dm.diagonal().sum()


def ising_hamiltonian(n, J=1., h=0.5):
	hamiltonian = QubitOperator()
	for i in range(n):
		# Interaction terms
		if i != n - 1:
			hamiltonian += QubitOperator(f'X{i} X{i + 1}', -J)
		elif n > 2:
			hamiltonian += QubitOperator(f'X0 X{n - 1}', -J)
		# Magnetic terms
		hamiltonian += QubitOperator(f'Z{i}', -h)

	print(hamiltonian)

	return hamiltonian


def ising_hamiltonian_commuting_terms(n, J=1., h=0.5, q_reg=None):
	if q_reg is None:
		q_reg = range(n)
	terms = dict()
	if J != 0:
		terms.update(ising_even_odd=[-J, [[q_reg[i], q_reg[i + 1]] for i in range(0, n - 1, 2)]])
		if n > 2:
			if n % 2 == 0:
				terms.update(ising_odd_even=[-J, [[q_reg[i], q_reg[i + 1]] for i in range(1, n - 2, 2)] + [[0, n - 1]]])
			else:
				terms.update(ising_odd_even=[-J, [[q_reg[i], q_reg[i + 1]] for i in range(1, n - 2, 2)]])
				terms.update(ising_closed=[-J, [[0, n - 1]]])
	if h != 0:
		terms.update(z=[-h, q_reg])

	return terms


class ExactResult:
	def __init__(self, hamiltonian, beta):
		self.hamiltonian_matrix = qubit_operator_sparse(hamiltonian).todense()
		self.gibbs_state = exact_gibbs_state(self.hamiltonian_matrix, beta)
		self.energy = (self.gibbs_state @ self.hamiltonian_matrix).diagonal().sum().real
		self.entropy = von_neumann_entropy(self.gibbs_state)
		self.cost = self.energy - self.entropy / beta
		self.eigenvalues = np.linalg.eigh(self.gibbs_state)[0]
		self.hamiltonian_eigenvalues = np.linalg.eigh(self.hamiltonian_matrix)[0]
