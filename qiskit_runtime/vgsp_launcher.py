from qiskit import IBMQ
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import QiskitRuntimeService

from job_results import print_results


# noinspection PyUnusedLocal
def callback(job_id, interim_result):
	print(interim_result)


if __name__ == '__main__':
	service = QiskitRuntimeService()
	program_id = 'vgsp-ising-xVBvWpWqyV'  # program id once it is uploaded
	inputs = dict(n=3, noise_model=NoiseModel.from_backend(IBMQ.load_account().get_backend('ibmq_jakarta')).to_dict())

	backend_name = "ibmq_qasm_simulator"
	options = dict(backend_name=backend_name)  # Choose backend (required)
	# Run job
	job = service.run(program_id, options=options, inputs=inputs)
	job_id = job.job_id
	print(f"Job sent to {backend_name} with job ID: {job_id}")
	# Start streaming results
	job.stream_results(callback)
	# Get results
	results = job.result()
	# Print and save results
	folder = f'jobs/{backend_name}/{job_id}'
	print_results(results, output_folder=folder)

# for backend in service.backends(simulator=False, min_num_qubits=4):
# 	try:
# 		options = dict(backend_name=backend.name)  # Choose backend (required)
# 		# Run job
# 		job = service.run(program_id, options=options, inputs=inputs)
# 		print(f"Job sent to {backend.name} with job ID: {job.job_id}")
# 	except:
# 		print(f"Job could not be sent to {backend.name}")
