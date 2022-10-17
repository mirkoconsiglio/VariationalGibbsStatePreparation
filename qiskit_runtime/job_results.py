import json
import warnings

from qiskit_ibm_runtime import QiskitRuntimeService, RuntimeDecoder

from gibbs_functions import print_results


def decode_interim_results(data):
	return list(reversed([json.loads(i, cls=RuntimeDecoder) for i in data]))


if __name__ == '__main__':
	service = QiskitRuntimeService()
	# Put job ID here after it is finished to retrieve it
	job_id = 'ccvfuro9ujl45d6clog0'
	job = service.job(job_id)
	backend_name = job.backend.name
	folder = f'jobs/{backend_name}/{job_id}'
	# Get job results
	try:
		results = job.result()
	except:
		warnings.warn("Results could not be retrieved, might be an error, getting interim results instead.")
		interim_results = decode_interim_results(job.interim_results())
		results = [interim_result for interim_result in interim_results if interim_result.get('n')]

	print_results(results, output_folder=folder)
