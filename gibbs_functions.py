import json
import os
from typing import Callable

import numpy as np
from numpy.linalg import eigh
from openfermion import QubitOperator
from openfermion.linalg import qubit_operator_sparse
from qiskit_ibm_runtime import RuntimeEncoder
from scipy.linalg import norm, logm, expm
from scipy.optimize import OptimizeResult


def funm_psd(A: list[list[complex]], func: Callable[[list[list[complex]]], complex]) -> list[list[complex]]:
	A = np.asarray(A)
	if len(A.shape) != 2:
		raise ValueError("Non-matrix input to matrix function.")
	w, v = eigh(A)
	w = np.maximum(w, 0)
	return (v * func(w)).dot(v.conj().T)


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


def print_results(results, output_folder=None):
	if not isinstance(results, list):
		results = [results]
	if output_folder:
		os.makedirs(f'{output_folder}', exist_ok=True)
	for i, result in enumerate(results):
		n = result['n']
		J = result['J']
		h = result['h']
		beta = result['beta']
		hamiltonian = ising_hamiltonian(n, J, h)
		# get calculated results
		calculated_result = _gibbs_result(result)
		# Calculate exact results
		exact_result = _exact_result(hamiltonian, beta)
		# Calculate comparative results
		ep = purity(exact_result['gibbs_state'])
		cf = fidelity(exact_result['gibbs_state'], calculated_result['rho'])
		ctd = trace_distance(exact_result['gibbs_state'], calculated_result['rho'])
		cre = relative_entropy(exact_result['gibbs_state'], calculated_result['rho'])
		cp = purity(calculated_result['rho'])
		nf = fidelity(exact_result['gibbs_state'], calculated_result['noiseless_rho'])
		ntd = trace_distance(exact_result['gibbs_state'], calculated_result['noiseless_rho'])
		nre = relative_entropy(exact_result['gibbs_state'], calculated_result['noiseless_rho'])
		np = purity(calculated_result['noiseless_rho'])
		# Print results
		print()
		print(f"n: {n}")
		print(f"J: {J}")
		print(f"h: {h}")
		print(f"beta: {beta}")
		print()
		print("Exact Gibbs state: ")
		print(exact_result['gibbs_state'])
		print()
		print("Calculated Gibbs state: ")
		print(calculated_result['rho'])
		print()
		print("Noiseless Calculated Gibbs state: ")
		print(calculated_result['noiseless_rho'])
		print()
		print(f"VQA cost: {calculated_result['cost']}")
		print(f"Exact cost: {exact_result['cost']}")
		print()
		print(f"VQA energy: {calculated_result['energy']}")
		print(f"Exact energy: {exact_result['energy']}")
		print()
		print(f"VQA entropy: {calculated_result['entropy']}")
		print(f"Exact entropy: {exact_result['entropy']}")
		print()
		print(f"VQA eigenvalues: {calculated_result['eigenvalues']}")
		print(f"Exact eigenvalues: {exact_result['eigenvalues']}")
		print()
		print(f"VQA Hamiltonian eigenvalues: {calculated_result['hamiltonian_eigenvalues']}")
		print(f"Exact Hamiltonian eigenvalues: {exact_result['hamiltonian_eigenvalues']}")
		print()
		print(f"Exact Purity: {ep}")
		print()
		print(f"Calculated Fidelity: {cf}")
		print(f"Calculated Trace Distance: {ctd}")
		print(f"Calculated Relative Entropy: {cre}")
		print(f"Calculated Purity: {cp}")
		print()
		print(f"Noiseless Fidelity: {nf}")
		print(f"Noiseless Trace Distance: {ntd}")
		print(f"Noiseless Relative Entropy: {nre}")
		print(f"Noiseless Purity: {np}")
		print()

		if output_folder:
			data = dict(
				metadata=dict(
					n=n,
					J=J,
					h=h,
					beta=beta,
					ancilla_reps=result['ancilla_reps'],
					system_reps=result['system_reps'],
					skip_transpilation=result['skip_transpilation'],
					use_measurement_mitigation=result['use_measurement_mitigation'],
					ansatz=result['ansatz'],
					optimizer=result['optimizer'],
					min_kwargs=result['min_kwargs'],
					shots=result['shots'],
					backend=result['backend']
				),
				calculated_result=calculated_result,
				exact_result=exact_result,
				metrics=dict(
					exact_purity=ep,
					calculated_fidelity=cf,
					calculated_trace_distance=ctd,
					calculated_relative_entropy=cre,
					calculated_purity=cp,
					noiseless_fidelity=nf,
					noiseless_trace_distance=ntd,
					noiseless_relative_entropy=nre,
					noiseless_purity=np,
				)
			)

			with open(f'{output_folder}/{i}.json', 'w') as f:
				json.dump(data, f, indent=4, cls=ResultsEncoder)


class ResultsEncoder(RuntimeEncoder):
	def default(self, obj):
		if isinstance(obj, np.ndarray):
			return obj.tolist()
		return RuntimeEncoder.default(self, obj)
