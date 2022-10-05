import json
import warnings

from qiskit_ibm_runtime import QiskitRuntimeService, RuntimeDecoder

from gibbs_functions import print_results


def decode_interim_results(data):
	return [json.loads(i, cls=RuntimeDecoder) for i in data]


if __name__ == '__main__':
	service = QiskitRuntimeService()
	# Put job ID here after it is finished to retrieve it
	job_id = 'ccuo7pmhs7ee9ds6omd0'
	job = service.job(job_id)
	backend_name = job.backend.name
	# Get job results
	try:
		results = job.result()
	except:
		results = dict()
		warnings.warn("Results could not be retrieved, might be an error.")
	interim_results = decode_interim_results(job.interim_results())
	print(interim_results)
	folder = f'jobs/{backend_name}/{job_id}'
	print_results(results, output_folder=folder)
