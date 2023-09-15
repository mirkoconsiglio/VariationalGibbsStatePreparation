import gzip
import json
import os
import warnings

import numpy as np
from numpy.linalg import eigh
from qiskit.quantum_info.operators import SparsePauliOp
from qiskit_ibm_runtime import RuntimeEncoder
from scipy.linalg import norm, logm, expm
from scipy.stats import entropy

from qulacs import ParametricQuantumCircuit


def funm_psd(A, func):
	A = np.asarray(A)
	if len(A.shape) != 2:
		raise ValueError("Non-matrix input to matrix function.")
	w, v = eigh(A)
	w = np.maximum(w, 0)
	return (v * func(w)).dot(v.conj().T)


def trace_distance(rho, sigma):
	return norm(np.asarray(rho) - np.asarray(sigma), 1) / 2


def fidelity(rho, sigma):
	sqrt_rho = funm_psd(np.asarray(rho), np.sqrt)
	return funm_psd(sqrt_rho @ np.asarray(sigma) @ sqrt_rho, np.sqrt).diagonal().sum().real ** 2


def relative_entropy(rho, sigma):
	rho = np.asarray(rho)
	sigma = np.asarray(sigma)
	return (rho @ (logm(rho) - logm(sigma))).diagonal().sum().real


def purity(rho):
	rho = np.asarray(rho)
	return (rho @ rho).diagonal().sum().real


def von_neumann_entropy(rho):
	return -np.sum([0 if i <= 0 else i * np.log(i) for i in np.linalg.eigh(rho)[0]])


def exact_gibbs_state(hamiltonian, beta):
	rho = expm(-beta * hamiltonian).real
	return rho / rho.diagonal().sum()


def XXZ_hamiltonian(n, gamma=1., delta=1., h=0.):
	hamiltonian = []
	J_x = 1 + gamma
	J_y = 1 - gamma
	for i in range(n):
		# Interaction terms
		if i != n - 1:
			if J_x != 0:
				hamiltonian.append(('I' * i + 'XX' + 'I' * (n - i - 2), -0.25 * J_x))
			if J_y != 0:
				hamiltonian.append(('I' * i + 'YY' + 'I' * (n - i - 2), -0.25 * J_y))
			if delta != 0:
				hamiltonian.append(('I' * i + 'ZZ' + 'I' * (n - i - 2), -0.25 * delta))
		elif n > 2:
			if J_x != 0:
				hamiltonian.append(('X' + 'I' * (n - 2) + 'X', -0.25 * J_x))
			if J_y != 0:
				hamiltonian.append(('Y' + 'I' * (n - 2) + 'Y', -0.25 * J_y))
			if delta != 0:
				hamiltonian.append(('Z' + 'I' * (n - 2) + 'Z', -0.25 * delta))
		# Magnetic terms
		if h != 0:
			hamiltonian.append(('I' * i + 'Z' + 'I' * (n - i - 1), -0.25 * h))

	# Build up the Hamiltonian
	return SparsePauliOp.from_list(hamiltonian)


def analytical_result(hamiltonian, beta):
	hamiltonian_matrix = hamiltonian.to_matrix()
	gibbs_state = exact_gibbs_state(hamiltonian_matrix, beta)
	energy = (gibbs_state @ hamiltonian_matrix).diagonal().sum().real
	vn_entropy = von_neumann_entropy(gibbs_state)
	cost = beta * energy - vn_entropy
	eigenvalues = np.linalg.eigh(gibbs_state)[0]
	hamiltonian_eigenvalues = np.linalg.eigh(hamiltonian_matrix)[0]
	return dict(
		hamiltonian_matrix=hamiltonian_matrix,
		gibbs_state=gibbs_state,
		energy=energy,
		entropy=vn_entropy,
		cost=cost,
		eigenvalues=eigenvalues,
		hamiltonian_eigenvalues=hamiltonian_eigenvalues
	)


def gibbs_result(gibbs):
	if isinstance(gibbs, dict):
		return dict(
			params=gibbs['params'],
			cost=gibbs['cost'],
			energy=gibbs['energy'],
			entropy=gibbs['entropy'],
			rho=gibbs['rho'],
			noiseless_rho=gibbs['noiseless_rho'],
			sigma=gibbs['sigma'],
			noiseless_sigma=gibbs['noiseless_sigma'],
			eigenvalues=gibbs['eigenvalues'],
			noiseless_eigenvalues=gibbs['noiseless_eigenvalues'],
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


def save_result(result, output_folder, job_id=None, backend=None, append=True):
	if output_folder:
		os.makedirs(output_folder, exist_ok=True)
	# Get results
	n = result.get('n')
	gamma = result.get('gamma')
	delta = result.get('delta')
	h = result.get('h')
	beta = result.get('beta')
	hamiltonian = XXZ_hamiltonian(n, gamma, delta, h)
	ancilla_reps = result.get('ancilla_reps')
	system_reps = result.get('system_reps')
	optimizer = result.get('optimizer')
	min_kwargs = result.get('min_kwargs')
	shots = result.get('shots')
	noise_model = result.get('noise_model_backend')
	exact_result = analytical_result(hamiltonian, beta)
	ep = purity(exact_result['gibbs_state'])
	# load data if append is True
	data = dict()
	if append:
		try:
			with open(f'{output_folder}/{beta:.2f}.json', 'r') as f:
				data = json.load(f)
		except FileNotFoundError:
			pass
	# Keep job ids and backends
	metadata = data.get('metadata', dict())
	job_ids = metadata.get('job_ids', [])
	backends = metadata.get('backends', [])
	job_ids.append(job_id)
	backends.append(backend)
	# Do not add backends or job ids if they are already added
	job_ids = list(set(job_ids))
	backends = list(set(backends))
	# Set up metrics
	metrics = data.get('metrics', dict())
	cf_list = metrics.get('calculated_fidelity', [])
	ctd_list = metrics.get('calculated_trace_distance', [])
	cre_list = metrics.get('calculated_relative_entropy', [])
	cp_list = metrics.get('calculated_purity', [])
	ckld_list = metrics.get('calculated_kullback_leibler_divergence', [])
	nf_list = metrics.get('noiseless_fidelity', [])
	ntd_list = metrics.get('noiseless_trace_distance', [])
	nre_list = metrics.get('noiseless_relative_entropy', [])
	np_list = metrics.get('noiseless_purity', [])
	nkld_list = metrics.get('noiseless_kullback_leibler_divergence', [])
	# Calculate and save data
	calculated_results = data.get('calculated_result', [])
	# get calculated results
	calculated_result = gibbs_result(result)
	calculated_results.append(calculated_result)
	# Calculate comparative results
	cf_list.append(fidelity(exact_result['gibbs_state'], calculated_result['rho']))
	ctd_list.append(trace_distance(exact_result['gibbs_state'], calculated_result['rho']))
	cre_list.append(relative_entropy(exact_result['gibbs_state'], calculated_result['rho']))
	cp_list.append(purity(calculated_result['rho']))
	ckld_list.append(entropy(exact_result['eigenvalues'], calculated_result['eigenvalues']))
	nf_list.append(fidelity(exact_result['gibbs_state'], calculated_result['noiseless_rho']))
	ntd_list.append(trace_distance(exact_result['gibbs_state'], calculated_result['noiseless_rho']))
	nre_list.append(relative_entropy(exact_result['gibbs_state'], calculated_result['noiseless_rho']))
	np_list.append(purity(calculated_result['noiseless_rho']))
	nkld_list.append(entropy(exact_result['eigenvalues'], calculated_result['noiseless_eigenvalues']))
	# Set up data dictionary
	data = dict(
		metrics=dict(
			exact_purity=ep,
			calculated_fidelity=cf_list,
			calculated_trace_distance=ctd_list,
			calculated_relative_entropy=cre_list,
			calculated_purity=cp_list,
			calculated_kullback_leibler_divergence=ckld_list,
			noiseless_fidelity=nf_list,
			noiseless_trace_distance=ntd_list,
			noiseless_relative_entropy=nre_list,
			noiseless_purity=np_list,
			noiseless_kullback_leibler_divergence=nkld_list,
		),
		metadata=dict(
			job_ids=job_ids,
			backends=backends,
			n=n,
			gamma=gamma,
			delta=delta,
			h=h,
			beta=beta,
			ancilla_reps=ancilla_reps,
			system_reps=system_reps,
			optimizer=optimizer,
			min_kwargs=min_kwargs,
			shots=shots,
			noise_model=noise_model,
		),
		calculated_result=calculated_results,
		exact_result=exact_result
	)

	with open(f'{output_folder}/{beta:.2f}.json', 'w') as f:
		json.dump(data, f, indent=4, cls=ResultsEncoder)


def print_multiple_results(multiple_results, output_folder=None, job_id=None, backend=None, append=True, zip_file=True,
                           hamiltonian='XXZ'):
	if output_folder:
		os.makedirs(output_folder, exist_ok=True)
	# Check if there is already data saved to add to it rather than overwrite
	if append and zip_file and os.path.exists(f'{output_folder}/data.gz'):
		with gzip.open(f'{output_folder}/data.gz', 'r') as f:
			saved_data = json.loads(f.read().decode('utf-8'))
	else:
		saved_data = []
	if zip_file:
		saved_data.append(multiple_results)
		# Save results in a compressed format
		with gzip.open(f'{output_folder}/data.gz', 'w') as f:
			f.write(json.dumps(saved_data, cls=ResultsEncoder).encode('utf-8'))
	for results in multiple_results:  # Different beta
		# Calculate exact results
		# Assumes these will be the same for each job
		result = results[0]
		n = result.get('n')
		if hamiltonian == 'Ising':
			gamma = 1
		else:
			gamma = result.get('gamma')
		if hamiltonian != 'XXZ':
			delta = 0
		else:
			delta = result.get('delta')
		h = result.get('h')
		beta = result.get('beta')
		hamiltonian = XXZ_hamiltonian(n, gamma, delta, h)
		ancilla_reps = result.get('ancilla_reps')
		system_reps = result.get('system_reps')
		optimizer = result.get('optimizer')
		min_kwargs = result.get('min_kwargs')
		shots = result.get('shots')
		noise_model = result.get('noise_model_backend')
		exact_result = analytical_result(hamiltonian, beta)
		ep = purity(exact_result['gibbs_state'])
		# load data if append is True
		data = dict()
		if append:
			try:
				with open(f'{output_folder}/{beta:.2f}.json', 'r') as f:
					data = json.load(f)
			except FileNotFoundError:
				pass
		# Keep job ids and backends
		metadata = data.get('metadata', dict())
		job_ids = metadata.get('job_ids', [])
		backends = metadata.get('backends', [])
		job_ids.append(job_id)
		backends.append(backend)
		# Do not add backends or job ids if they are already added
		job_ids = list(set(job_ids))
		backends = list(set(backends))
		# Set up metrics
		metrics = data.get('metrics', dict())
		cf_list = metrics.get('calculated_fidelity', [])
		ctd_list = metrics.get('calculated_trace_distance', [])
		cre_list = metrics.get('calculated_relative_entropy', [])
		cp_list = metrics.get('calculated_purity', [])
		ckld_list = metrics.get('calculated_kullback_leibler_divergence', [])
		nf_list = metrics.get('noiseless_fidelity', [])
		ntd_list = metrics.get('noiseless_trace_distance', [])
		nre_list = metrics.get('noiseless_relative_entropy', [])
		np_list = metrics.get('noiseless_purity', [])
		nkld_list = metrics.get('noiseless_kullback_leibler_divergence', [])
		# Calculate and save data
		calculated_results = data.get('calculated_result', [])
		for result in results:  # Different runs
			# get calculated results
			calculated_result = gibbs_result(result)
			calculated_results.append(calculated_result)
			print()
			print(calculated_result)
			print(exact_result)
			print()
			# Calculate comparative results
			cf_list.append(fidelity(exact_result['gibbs_state'], calculated_result['rho']))
			ctd_list.append(trace_distance(exact_result['gibbs_state'], calculated_result['rho']))
			cre_list.append(relative_entropy(exact_result['gibbs_state'], calculated_result['rho']))
			cp_list.append(purity(calculated_result['rho']))
			ckld_list.append(entropy(exact_result['eigenvalues'], calculated_result['eigenvalues']))
			nf_list.append(fidelity(exact_result['gibbs_state'], calculated_result['noiseless_rho']))
			ntd_list.append(trace_distance(exact_result['gibbs_state'], calculated_result['noiseless_rho']))
			nre_list.append(relative_entropy(exact_result['gibbs_state'], calculated_result['noiseless_rho']))
			np_list.append(purity(calculated_result['noiseless_rho']))
			nkld_list.append(entropy(exact_result['eigenvalues'], calculated_result['noiseless_eigenvalues']))
		# Print results
		print()
		print(f"n: {n}")
		print(f"gamma: {gamma}")
		print(f"delta: {delta}")
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
		print(f"Calculated Purity avg: {np.average(cp_list)}")
		print(f"Calculated Purity max: {np.max(cp_list)}")
		print(f"Calculated Kullback-Leibler Divergence min: {np.min(ckld_list)}")
		print(f"Calculated Kullback-Leibler Divergence avg: {np.average(ckld_list)}")
		print(f"Calculated Kullback-Leibler Divergence max: {np.max(ckld_list)}")
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
		print(f"Noiseless Kullback-Leibler Divergence min: {np.min(nkld_list)}")
		print(f"Noiseless Kullback-Leibler Divergence avg: {np.average(nkld_list)}")
		print(f"Noiseless Kullback-Leibler Divergence max: {np.max(nkld_list)}")
		print()

		if output_folder:
			data = dict(
				metrics=dict(
					exact_purity=ep,
					calculated_fidelity=cf_list,
					calculated_trace_distance=ctd_list,
					calculated_relative_entropy=cre_list,
					calculated_purity=cp_list,
					calculated_kullback_leibler_divergence=ckld_list,
					noiseless_fidelity=nf_list,
					noiseless_trace_distance=ntd_list,
					noiseless_relative_entropy=nre_list,
					noiseless_purity=np_list,
					noiseless_kullback_leibler_divergence=nkld_list,
				),
				metadata=dict(
					job_ids=job_ids,
					backends=backends,
					n=n,
					gamma=gamma,
					delta=delta,
					h=h,
					beta=beta,
					ancilla_reps=ancilla_reps,
					system_reps=system_reps,
					optimizer=optimizer,
					min_kwargs=min_kwargs,
					shots=shots,
					noise_model=noise_model,
				),
				calculated_result=calculated_results,
				exact_result=exact_result
			)

			with open(f'{output_folder}/{beta:.2f}.json', 'w') as f:
				json.dump(data, f, indent=4, cls=ResultsEncoder)


class ResultsEncoder(RuntimeEncoder):
	def default(self, obj):
		if isinstance(obj, ParametricQuantumCircuit):
			warnings.warn(f"ParametricQuantumCircuit {obj} is not JSON serializable and will be set to None.")
			return None
		return RuntimeEncoder.default(self, obj)
