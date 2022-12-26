import json
from json import JSONDecodeError

from qiskit_ibm_runtime import QiskitRuntimeService, RuntimeDecoder, RuntimeJobFailureError, RuntimeJobMaxTimeoutError

from gibbs_functions import print_multiple_results
from plotter import plot_result_min_avg_max


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
	service = QiskitRuntimeService()
	# jobs = service.jobs(limit=10, skip=20)
	jobs = [service.job('cei3i95iuboftdmpnc70')]
	append = True  # Append results or overwrite
	folders = set()
	for job in jobs:
		job_id = job.job_id()
		backend_name = job.backend.name
		n = job.inputs.get('n')
		J = job.inputs.get('J')
		h = job.inputs.get('h')
		shots = job.inputs.get('shots')
		noise_model = job.inputs.get('noise_model')

		# Get job results
		try:
			results = job.result()
		except (RuntimeJobFailureError, RuntimeJobMaxTimeoutError) as err:
			print(err.message)
			print("Since final results could not be retrieved, getting interim results instead.")
			results = decode_interim_results(job.interim_results(), job.inputs.get('N'))

		folder = f'jobs/{backend_name}'
		if isinstance(noise_model, str):
			folder += f'_{noise_model}'
		folder += f'/n_{n}_J_{J:.2f}_h_{h:.2f}_shots_{shots}'
		folders.add(folder)

		print_multiple_results(results, output_folder=folder, job_id=job_id, backend=backend_name, append=append)

	for folder in folders:
		plot_result_min_avg_max(folder)


if __name__ == '__main__':
	main()
