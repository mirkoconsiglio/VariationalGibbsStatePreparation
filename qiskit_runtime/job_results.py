import json
from json import JSONDecodeError

from qiskit_ibm_runtime import QiskitRuntimeService, RuntimeDecoder, RuntimeJobFailureError, RuntimeJobMaxTimeoutError

from gibbs_functions import print_multiple_results
from plotter import plot_result_min_avg_max


def decode_interim_results(data, N):
	multiple_results = []
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
			# Structure interim results
			if len(results) == N:
				multiple_results.append(results)
				results = []

	return multiple_results


def main():
	service = QiskitRuntimeService()
	jobs = service.jobs(limit=10)
	append = True  # Append results or overwrite
	show = False  # Show plot after every job
	# Put job ID here after it is finished to retrieve it
	for job in jobs:
		job_id = job.job_id()
		backend_name = job.backend.name
		n = job.inputs.get('n')
		J = job.inputs.get('J')
		h = job.inputs.get('h')
		shots = job.inputs.get('shots')
		N = job.inputs.get('N')
		
		# Get job results
		try:
			results = job.result()
		except (RuntimeJobFailureError, RuntimeJobMaxTimeoutError) as err:
			print(err.message)
			print("Since final results could not be retrieved, getting interim results instead.")
			results = decode_interim_results(job.interim_results(), N)
		
		folder = f'jobs/{backend_name}/n_{n}_J_{J:.2f}_h_{h:.2f}_shots_{shots}'
		
		print_multiple_results(results, output_folder=folder, job_id=job_id, backend=backend_name, append=append)
		
		plot_result_min_avg_max(folder, show=show)


if __name__ == '__main__':
	main()
