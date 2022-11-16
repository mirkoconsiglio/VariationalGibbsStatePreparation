from qiskit import IBMQ
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import QiskitRuntimeService

from gibbs_functions import print_multiple_results
from plotter import plot_result_min_avg_max


def main():
	n = 2
	J = 1.
	h = 0.5
	beta = [1e-10, 0.2, 0.5, 0.8, 1., 1.2, 2., 3., 4., 5.]
	shots = 1024
	N = 10  # Can be split manually into a list
	split_betas = False  # Split each beta into a separate job with N runs each
	
	service = QiskitRuntimeService()
	program_id = 'vgsp-ising-n6W7RRa7W6'  # program id once it is uploaded
	backend_name = 'ibmq_qasm_simulator'
	options = dict(backend_name=backend_name)  # Choose backend (required)
	
	noise_model = NoiseModel.from_backend(IBMQ.load_account().get_backend('ibmq_jakarta')).to_dict()
	
	job = None
	if not split_betas:
		beta = [beta]
	if not isinstance(N, list):
		N = [N]
	for b in beta:
		for i in N:
			# inputs
			inputs = dict(n=n, J=J, h=h, beta=b, shots=shots, N=i, noise_model=noise_model)
			# Run job
			job = service.run(program_id, options=options, inputs=inputs)
			job_id = job.job_id()
			print(f"Job sent to {backend_name} with job ID: {job_id} for beta: {b}, run: {i}")
	# If only one job is sent, stream it
	if len(beta) == 1 and len(N) == 1:
		stream_results(job)


def stream_results(job):
	job_id = job.job_id()
	backend_name = job.backend.name
	n = job.inputs.get('n')
	J = job.inputs.get('J')
	h = job.inputs.get('h')
	shots = job.inputs.get('shots')
	# Start streaming results
	print("Streaming results")
	job.stream_results(callback)
	# Get results
	results = job.result()
	# Print and save results
	folder = f'jobs/{backend_name}/n_{n}_J_{J:.2f}_h_{h:.2f}_shots_{shots}'
	print_multiple_results(results, output_folder=folder, job_id=job_id, backend=backend_name)

	plot_result_min_avg_max(folder)


# noinspection PyUnusedLocal
def callback(job_id, interim_result):
	print(interim_result)


if __name__ == '__main__':
	main()
