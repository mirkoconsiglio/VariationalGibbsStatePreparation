import json
import os
import warnings

import numpy as np
from qiskit_ibm_runtime import QiskitRuntimeService, RuntimeDecoder

from gibbs_functions import (_exact_result, _gibbs_result, ising_hamiltonian, fidelity, trace_distance,
                             relative_entropy, purity)


class ResultsEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, complex):
			return [obj.real, obj.imag]
		if isinstance(obj, np.ndarray):
			return obj.tolist()
		return json.JSONEncoder.default(self, obj)


def print_results(results, save=False, backend_name=None, job_id=None, directory='jobs'):
	if not isinstance(results, list):
		results = [results]
	if save:
		os.makedirs(f'{directory}/{backend_name}', exist_ok=True)
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

		if save:
			data = dict(
				n=n,
				J=J,
				h=h,
				beta=beta,
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

			with open(f'{directory}/{backend_name}/{job_id}_{i}.json', 'w') as f:
				json.dump(data, f, indent=4, cls=ResultsEncoder)


def decode_interim_results(data):
	return [json.loads(i, cls=RuntimeDecoder) for i in data]


if __name__ == '__main__':
	service = QiskitRuntimeService()
	# Put job ID here after it is finished to retrieve it
	job_id = 'ccu1k4sh1rm1j15ggbvg'
	job = service.job(job_id)
	backend_name = job.backend.name
	# Get job results
	try:
		results = job.result()
	except:
		results = dict()
		warnings.warn("Results could not be retrieved, might be an error.")
	interim_results = decode_interim_results(job.interim_results())
	# Print results
	print_results(results, save=True, backend_name=backend_name, job_id=job_id, directory='jobs')
