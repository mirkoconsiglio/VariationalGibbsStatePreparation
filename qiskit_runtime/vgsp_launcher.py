from qiskit_ibm_runtime import QiskitRuntimeService

from gibbs_functions import print_multiple_results
from plotter import plot_result_min_avg_max


# noinspection PyUnusedLocal
def callback(job_id, interim_result):
	print(interim_result)


def main():
	n = 2
	J = 1.
	h = 0.5
	beta = [1e-10, 0.2, 0.5, 0.8, 1., 1.2, 2., 3., 4., 5.]
	shots = 1024
	N = 10

	service = QiskitRuntimeService()
	program_id = 'vgsp-ising-n6W7RRa7W6'  # program id once it is uploaded
	inputs = dict(n=n, J=J, h=h, beta=beta, shots=shots, N=N)

	backend_name = 'ibmq_qasm_simulator'
	options = dict(backend_name=backend_name)  # Choose backend (required)
	# Run job
	job = service.run(program_id, options=options, inputs=inputs)
	job_id = job.job_id()
	print(f"Job sent to {backend_name} with job ID: {job_id}")
	# Start streaming results
	job.stream_results(callback)
	# Get results
	results = job.result()
	# Print and save results
	folder = f'jobs/{backend_name}/n_{n}_J_{J:.2f}_h_{h:.2f}_shots_{shots}'
	print_multiple_results(results, output_folder=folder, job_id=job_id, backend=backend_name)

	plot_result_min_avg_max(folder)


# for backend in service.backends(simulator=False, min_num_qubits=4):
# 	try:
# 		options = dict(backend_name=backend.name)  # Choose backend (required)
# 		# Run job
# 		job = service.run(program_id, options=options, inputs=inputs)
# 		print(f"Job sent to {backend.name} with job ID: {job.job_id}")
# 	except:
# 		print(f"Job could not be sent to {backend.name}")


if __name__ == '__main__':
	main()
