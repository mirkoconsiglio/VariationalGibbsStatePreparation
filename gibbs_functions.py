import json
import os
from typing import List, Callable

import numpy as np
from numpy.linalg import eigh
from qiskit.opflow import SummedOp
from qiskit.opflow.primitive_ops import PauliOp
from qiskit.quantum_info import Pauli
from qiskit_ibm_runtime import RuntimeEncoder
from scipy.linalg import norm, logm, expm

from qulacs import ParametricQuantumCircuit


def funm_psd(A: List[List[complex]], func: Callable[[List[List[complex]]], complex]) -> List[List[complex]]:
	A = np.asarray(A)
	if len(A.shape) != 2:
		raise ValueError("Non-matrix input to matrix function.")
	w, v = eigh(A)
	w = np.maximum(w, 0)
	return (v * func(w)).dot(v.conj().T)


def trace_distance(rho: List[List[complex]], sigma: List[List[complex]]) -> float:
	return norm(np.asarray(rho) - np.asarray(sigma), 1) / 2


def fidelity(rho: List[List[complex]], sigma: List[List[complex]]) -> float:
	sqrt_rho = funm_psd(np.asarray(rho), np.sqrt)
	return funm_psd(sqrt_rho @ np.asarray(sigma) @ sqrt_rho, np.sqrt).diagonal().sum().real ** 2


def relative_entropy(rho: List[List[complex]], sigma: List[List[complex]]) -> float:
	rho = np.asarray(rho)
	sigma = np.asarray(sigma)
	return (rho @ (logm(rho) - logm(sigma))).diagonal().sum().real


def purity(rho: List[List[complex]]) -> float:
	rho = np.asarray(rho)
	return (rho @ rho).diagonal().sum().real


def von_neumann_entropy(rho: List[List[complex]]) -> float:
	return -np.sum([0 if i <= 0 else i * np.log(i) for i in np.linalg.eigh(rho)[0]])


def exact_gibbs_state(hamiltonian: List[List[complex]], beta: float) -> List[List[float]]:
	rho = expm(-beta * hamiltonian).real
	return rho / rho.diagonal().sum()


def ising_hamiltonian(n: int, J: float = 1., h: float = 0.5) -> List[List[complex]]:
	hamiltonian = []
	for i in range(n):
		# Interaction terms
		if i != n - 1:
			XX = 'I' * i + 'XX' + 'I' * (n - i - 2)
			hamiltonian.append(PauliOp(Pauli(XX), -J))
		elif n > 2:
			XX = 'X' + 'I' * (n - 2) + 'X'
			hamiltonian.append(PauliOp(Pauli(XX), -J))
		# Magnetic terms
		Z = 'I' * i + 'Z' + 'I' * (n - i - 1)
		hamiltonian.append(PauliOp(Pauli(Z), -h))

	# Build up the Hamiltonian
	return SummedOp(hamiltonian)


def ising_hamiltonian_commuting_terms(n: int, J: float = 1., h: float = 0.5) -> dict:
	terms = dict()
	if J != 0:
		if n == 2:
			terms.update(x=[-J, [[0, 1]]])
		elif n > 2:
			terms.update(x=[-J, [[i, (i + 1) % n] for i in range(n)]])
	if h != 0:
		terms.update(z=[-h, list(range(n))])

	return terms


def _exact_result(hamiltonian, beta):
	hamiltonian_matrix = hamiltonian.to_matrix()
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
					ancilla_reps=result.get('ancilla_reps'),
					system_reps=result.get('system_reps'),
					skip_transpilation=result.get('skip_transpilation'),
					use_measurement_mitigation=result.get('use_measurement_mitigation'),
					ansatz=result.get('ansatz'),
					optimizer=result.get('optimizer'),
					min_kwargs=result.get('min_kwargs'),
					shots=result.get('shots'),
					backend=result.get('backend')
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

			with open(f'{output_folder}/{beta:.2f}.json', 'w') as f:
				json.dump(data, f, indent=4, cls=ResultsEncoder)


def print_multiple_results(multiple_results, output_folder=None, job_id=None, backend=None):
	if not isinstance(multiple_results, list):
		multiple_results = [multiple_results]
	if output_folder:
		os.makedirs(f'{output_folder}', exist_ok=True)
	for results in multiple_results:
		n = None
		J = None
		h = None
		beta = None
		hamiltonian = None
		ancilla_reps = None
		system_reps = None
		optimizer = None
		min_kwargs = None
		shots = None
		ep = None
		cf_list = []
		ctd_list = []
		cre_list = []
		cp_list = []
		nf_list = []
		ntd_list = []
		nre_list = []
		np_list = []
		for i, result in enumerate(results):
			if i == 0:
				n = result['n']
				J = result['J']
				h = result['h']
				beta = result['beta']
				hamiltonian = ising_hamiltonian(n, J, h)
				ancilla_reps = result.get('ancilla_reps')
				optimizer = result.get('optimizer')
				min_kwargs = result.get('min_kwargs')
				shots = result.get('shots')
			# get calculated results
			calculated_result = _gibbs_result(result)
			# Calculate exact results
			exact_result = _exact_result(hamiltonian, beta)
			# Calculate comparative results
			if i == 0:
				ep = purity(exact_result['gibbs_state'])
			cf_list.append(fidelity(exact_result['gibbs_state'], calculated_result['rho']))
			ctd_list.append(trace_distance(exact_result['gibbs_state'], calculated_result['rho']))
			cre_list.append(relative_entropy(exact_result['gibbs_state'], calculated_result['rho']))
			cp_list.append(purity(calculated_result['rho']))
			nf_list.append(fidelity(exact_result['gibbs_state'], calculated_result['noiseless_rho']))
			ntd_list.append(trace_distance(exact_result['gibbs_state'], calculated_result['noiseless_rho']))
			nre_list.append(relative_entropy(exact_result['gibbs_state'], calculated_result['noiseless_rho']))
			np_list.append(purity(calculated_result['noiseless_rho']))
		# Print results
		print()
		print(f"n: {n}")
		print(f"J: {J}")
		print(f"h: {h}")
		print(f"beta: {beta}")
		print()
		print(f"Exact Purity: {np.min(ep)}")
		print()
		print(f"Calculated Fidelity min: {np.min(cf_list)}")
		print(f"Calculated Fidelity avg: {np.average(cf_list)}")
		print(f"Calculated Fidelity max: {np.max(cf_list)}")
		print(f"Calculated Trace Distance min: {np.min(ctd_list)}")
		print(f"Calculated Trace Distance avg: {np.average(ctd_list)}")
		print(f"Calculated Trace Distance max: {np.max(ctd_list)}")
		print(f"Calculated Relative Entropy min: {np.min(cre_list)}")
		print(f"Calculated Relative Entropy avg: {np.average(cre_list)}")
		print(f"Calculated Relative Entropy max: {np.max(cre_list)}")
		print(f"Calculated Purity min: {np.min(cp_list)}")
		print(f"Calculated Purity min: {np.average(cp_list)}")
		print(f"Calculated Purity min: {np.max(cp_list)}")
		print()
		print(f"Noiseless Fidelity min: {np.min(nf_list)}")
		print(f"Noiseless Fidelity avg: {np.average(nf_list)}")
		print(f"Noiseless Fidelity max: {np.max(nf_list)}")
		print(f"Noiseless Trace Distance min: {np.min(ntd_list)}")
		print(f"Noiseless Trace Distance avg: {np.average(ntd_list)}")
		print(f"Noiseless Trace Distance max: {np.max(ntd_list)}")
		print(f"Noiseless Relative Entropy min: {np.min(nre_list)}")
		print(f"Noiseless Relative Entropy avg: {np.average(nre_list)}")
		print(f"Noiseless Relative Entropy max: {np.max(nre_list)}")
		print(f"Noiseless Purity min: {np.min(np_list)}")
		print(f"Noiseless Purity avg: {np.average(np_list)}")
		print(f"Noiseless Purity max: {np.max(np_list)}")
		print()

		if output_folder:
			data = dict(
				metadata=dict(
					job_id=job_id,
					backend=backend,
					n=n,
					J=J,
					h=h,
					beta=beta,
					ancilla_reps=ancilla_reps,
					system_reps=system_reps,
					optimizer=optimizer,
					min_kwargs=min_kwargs,
					shots=shots,
				),
				metrics=dict(
					exact_purity=ep,
					calculated_fidelity=cf_list,
					calculated_trace_distance=ctd_list,
					calculated_relative_entropy=cre_list,
					calculated_purity=cp_list,
					noiseless_fidelity=nf_list,
					noiseless_trace_distance=ntd_list,
					noiseless_relative_entropy=nre_list,
					noiseless_purity=np_list,
				)
			)

			with open(f'{output_folder}/{beta:.2f}.json', 'w') as f:
				json.dump(data, f, indent=4, cls=ResultsEncoder)


class ResultsEncoder(RuntimeEncoder):
	def default(self, obj):
		if isinstance(obj, np.ndarray):
			return obj.tolist()
		if isinstance(obj, ParametricQuantumCircuit):
			return None
		return RuntimeEncoder.default(self, obj)
