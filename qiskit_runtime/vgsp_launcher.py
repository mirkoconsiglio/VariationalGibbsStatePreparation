from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService()
program_id = 'vgsp-ising-xVBvWpWqyV'  # program id once it is uploaded
backends = service.backends(simulator=False, min_num_qubits=4)
inputs = dict(adiabatic_assistance=True)

backend_name = "ibmq_qasm_simulator"
options = dict(backend_name=backend_name)  # Choose backend (required)
# Run job
job = service.run(program_id, options=options, inputs=inputs)
print(f"Job sent to {backend_name} with job ID: {job.job_id}")

# TODO: add streaming of results with callback function

# for backend in backends:
# 	try:
# 		options = dict(backend_name=backend.name)  # Choose backend (required)
# 		# Run job
# 		job = service.run(program_id, options=options, inputs=inputs)
# 		print(f"Job sent to {backend.name} with job ID: {job.job_id}")
# 	except:
# 		print(f"Job could not be sent to {backend.name}")
