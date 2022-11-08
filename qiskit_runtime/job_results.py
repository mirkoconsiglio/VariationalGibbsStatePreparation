import json
import warnings

from qiskit_ibm_runtime import QiskitRuntimeService, RuntimeDecoder, RuntimeJobFailureError

from gibbs_functions import print_multiple_results
from plotter import plot_result


def decode_interim_results(data):
	return list(reversed([json.loads(i, cls=RuntimeDecoder) for i in data]))


def main():
	service = QiskitRuntimeService()
	# Put job ID here after it is finished to retrieve it
	job_id = 'cdl6b0pg1234sbmvqki0'
	job = service.job(job_id)
	backend_name = job.backend.name
	folder = f'jobs/{backend_name}/{job_id}'
	# Get job results
	try:
		results = job.result()
	except RuntimeJobFailureError:
		warnings.warn("Results could not be retrieved, might be an error, getting interim results instead.")
		interim_results = decode_interim_results(job.interim_results())
		results = [interim_result for interim_result in interim_results if interim_result.get('final')]

	print_multiple_results(results, output_folder=folder, job_id=job_id, backend=backend_name)

	plot_result(folder)


if __name__ == '__main__':
	main()
