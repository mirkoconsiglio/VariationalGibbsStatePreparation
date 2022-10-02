from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService()
program_id = 'vgsp-ising-XNvjgxPY4N'  # program id once it is uploaded
backends = service.backends(simulator=False, min_num_qubits=4)
backend_names = [backend.name for backend in backends]
inputs = {}

for backend_name in backend_names:
	try:
		options = {"backend_name": backend_name}  # Choose backend (required)
		# Run job
		job = service.run(program_id, options=options, inputs=inputs)
		print(f"Job sent to {backend_name} with job ID: {job.job_id}")
	except:
		print(f"Job could not be sent to {backend_name}")
