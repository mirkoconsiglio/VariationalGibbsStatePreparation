import os
import pickle
from typing import Callable

import numpy as np
from numpy.linalg import eigh
from openfermion import QubitOperator
from openfermion.linalg import qubit_operator_sparse
from scipy.linalg import norm, logm, expm
from scipy.optimize import OptimizeResult


def funm_psd(A: list[list[complex]], func: Callable[[list[list[complex]]], complex]) -> list[list[complex]]:
	A = np.asarray(A)
	if len(A.shape) != 2:
		raise ValueError("Non-matrix input to matrix function.")
	w, v = eigh(A)
	w = np.maximum(w, 0)
	return (v * func(w)).dot(v.conj().T)


def save_data(folder: str, data) -> None:  # function to save kwargs as pickle file
	os.makedirs(f'{folder}', exist_ok=True)
	with open(f'{folder}/data.pkl', 'wb') as file:
		pickle.dump(data, file)


def trace_distance(rho: list[list[complex]], sigma: list[list[complex]]) -> float:
	return norm(np.asarray(rho) - np.asarray(sigma), 1) / 2


def fidelity(rho: list[list[complex]], sigma: list[list[complex]]) -> float:
	sqrt_rho = funm_psd(np.asarray(rho), np.sqrt)
	return funm_psd(sqrt_rho @ np.asarray(sigma) @ sqrt_rho, np.sqrt).diagonal().sum().real ** 2


def relative_entropy(rho: list[list[complex]], sigma: list[list[complex]]) -> float:
	rho = np.asarray(rho)
	sigma = np.asarray(sigma)
	return (rho @ (logm(rho) - logm(sigma))).diagonal().sum().real


def purity(rho: list[list[complex]]) -> float:
	rho = np.asarray(rho)
	return (rho @ rho).diagonal().sum().real


def von_neumann_entropy(rho: list[list[complex]]) -> float:
	return -np.sum([0 if i <= 0 else i * np.log(i) for i in np.linalg.eigh(rho)[0]])


def exact_gibbs_state(hamiltonian: list[list[complex]], beta: float) -> list[list[float]]:
	rho = expm(-beta * hamiltonian).real
	return rho / rho.diagonal().sum()


def ising_hamiltonian(n: int, J: float = 1., h: float = 0.5) -> list[list[complex]]:
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


def ising_hamiltonian_commuting_terms(n: int, J: float = 1., h: float = 0.5) -> dict:
	terms = dict()
	if J != 0:
		terms.update(ising_even_odd=[-J, [[i, i + 1] for i in range(0, n - 1, 2)]])
		if n > 2:
			if n % 2 == 0:
				terms.update(ising_odd_even=[-J, [[i, i + 1] for i in range(1, n - 2, 2)] + [[0, n - 1]]])
			else:
				terms.update(ising_odd_even=[-J, [[i, i + 1] for i in range(1, n - 2, 2)]])
				terms.update(ising_closed=[-J, [[0, n - 1]]])
	if h != 0:
		terms.update(z=[-h, list(range(n))])

	return terms


def _exact_result(hamiltonian, beta):
	hamiltonian_matrix = qubit_operator_sparse(hamiltonian).todense()
	gibbs_state = exact_gibbs_state(hamiltonian_matrix, beta)
	energy = (gibbs_state @ hamiltonian_matrix).diagonal().sum().real
	entropy = von_neumann_entropy(gibbs_state)
	cost = energy - entropy / beta
	eigenvalues = np.linalg.eigh(gibbs_state)[0]
	hamiltonian_eigenvalues = np.linalg.eigh(hamiltonian_matrix)[0]
	return dict(
		hamiltonian_matrix=hamiltonian_matrix,
		gibbs_state=gibbs_state,
		energy=energy,
		entropy=entropy,
		cost=cost,
		eigenvalues=eigenvalues,
		hamiltonian_eigenvalues=hamiltonian_eigenvalues
	)


class ExactResult(OptimizeResult):
	def __init__(self, hamiltonian, beta) -> OptimizeResult:
		super().__init__(_exact_result(hamiltonian, beta))


def _gibbs_result(gibbs):
	if isinstance(gibbs, dict):
		return dict(
			# ancilla_unitary_params=gibbs.ancilla_params(),
			# system_unitary_params=gibbs.system_params(),
			params=gibbs['params'],
			# ancilla_unitary=gibbs.ancilla_unitary_matrix(),
			# system_unitary=gibbs.system_unitary_matrix(),
			cost=gibbs['cost'],
			energy=gibbs['energy'],
			entropy=gibbs['entropy'],
			rho=gibbs['rho'],
			noiseless_rho=gibbs['noiseless_rho'],
			sigma=gibbs['sigma'],
			noiseless_sigma=gibbs['noiseless_sigma'],
			eigenvalues=gibbs['eigenvalues'],
			noiseless_eigenvalues=gibbs['noiseless_eigenvalues'],
			# eigenvectors=gibbs.eigenvectors,
			hamiltonian_eigenvalues=gibbs['hamiltonian_eigenvalues'],
			noiseless_hamiltonian_eigenvalues=gibbs['noiseless_hamiltonian_eigenvalues']
		)
	return dict(
		result=gibbs.result,
		ancilla_unitary_params=gibbs.ancilla_params(),
		system_unitary_params=gibbs.system_params(),
		params=gibbs.params,
		ancilla_unitary=gibbs.ancilla_unitary_matrix(),
		system_unitary=gibbs.system_unitary_matrix(),
		cost=gibbs.cost,
		energy=gibbs.energy,
		entropy=gibbs.entropy,
		rho=gibbs.rho,
		noiseless_rho=gibbs.noiseless_rho,
		sigma=gibbs.sigma,
		noiseless_sigma=gibbs.noiseless_sigma,
		eigenvalues=gibbs.eigenvalues,
		noiseless_eigenvalues=gibbs.noiseless_eigenvalues,
		eigenvectors=gibbs.eigenvectors,
		hamiltonian_eigenvalues=gibbs.hamiltonian_eigenvalues,
		noiseless_hamiltonian_eigenvalues=gibbs.noiseless_hamiltonian_eigenvalues
	)


class GibbsResult(OptimizeResult):
	def __init__(self, gibbs) -> OptimizeResult:
		super().__init__(_gibbs_result(gibbs))
