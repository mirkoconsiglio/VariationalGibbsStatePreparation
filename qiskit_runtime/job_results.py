import json
from json import JSONDecodeError

from qiskit_ibm_runtime import QiskitRuntimeService, RuntimeDecoder, RuntimeJobFailureError, RuntimeJobMaxTimeoutError

from gibbs_functions import print_multiple_results


# Deprecated
def decode_interim_results(data, N=1):
	results = []
	for i in reversed(data):
		try:
			line = json.loads(i, cls=RuntimeDecoder)
		except JSONDecodeError:
			pass
		else:
			# Only append dictionary interim results
			if isinstance(line, dict) and line.get('final'):
				results.append(line)

	multiple_results = [results[i:i + N] for i in range(0, len(results), N)]

	return multiple_results


def main():
	service = QiskitRuntimeService(name='personal')
	jobs = service.jobs()
	append = True  # Append results or overwrite
	for job in jobs:
		job_id = job.job_id()
		# Get job results
		try:
			results = job.result()
		except RuntimeJobMaxTimeoutError:
			print(f"Runtime job {job_id} timed out.")
			continue
		except RuntimeJobFailureError:
			print(f"Runtime job {job_id} failed.")
			continue
		else:
			print(f"Runtime job {job_id} succeeded.")

		# These will be the same for every batched run in the job
		backend_name = results[0][0].get('backend')
		n = results[0][0].get('n')
		J = results[0][0].get('J')
		h = results[0][0].get('h')
		shots = results[0][0].get('shots')
		noise_model = results[0][0].get('noise_model')

		folder = f'jobs/{backend_name}'
		if isinstance(noise_model, str):
			folder += f'_{noise_model}'
		folder += f'/n_{n}_J_{J:.2f}_h_{h:.2f}_shots_{shots}'

		print_multiple_results(results, output_folder=folder, job_id=job_id, backend=backend_name, append=append,
		                       zip_file=False, hamiltonian='Ising')


if __name__ == '__main__':
	main()
