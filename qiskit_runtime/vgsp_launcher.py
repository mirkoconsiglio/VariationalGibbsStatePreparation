from qiskit_ibm_runtime import QiskitRuntimeService


def callback(job_id, interim_result):
	print(interim_result)


if __name__ == '__main__':
	service = QiskitRuntimeService()
	program_id = 'vgsp-ising-xVBvWpWqyV'  # program id once it is uploaded
	backends = service.backends(simulator=False, min_num_qubits=4)
	inputs = dict()

	backend_name = "ibmq_qasm_simulator"
	options = dict(backend_name=backend_name)  # Choose backend (required)
	# Run job
	job = service.run(program_id, options=options, inputs=inputs)
	print(f"Job sent to {backend_name} with job ID: {job.job_id}")

	job.stream_results(callback)

# for backend in backends:
# 	try:
# 		options = dict(backend_name=backend.name)  # Choose backend (required)
# 		# Run job
# 		job = service.run(program_id, options=options, inputs=inputs)
# 		print(f"Job sent to {backend.name} with job ID: {job.job_id}")
# 	except:
# 		print(f"Job could not be sent to {backend.name}")
