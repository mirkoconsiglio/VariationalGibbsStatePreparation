from qiskit_ibm_runtime import QiskitRuntimeService

from gibbs_functions import print_multiple_results
from plotter import plot_result_min_avg_max


def main():
	n = 2
	J = 1.
	h = 0.5
	beta = [1e-10, 0.2, 0.5, 0.8, 1., 1.2, 2., 3., 4., 5.]
	shots = 10000
	tomography_shots = 100000
	N = 1  # Can be split manually into a list
	split_betas = True  # Split each beta into a separate job with N runs each
	program_id = 'vgsp-ising-2N4DRAD24N'  # program id
	backend_name = 'ibmq_qasm_simulator'
	noise_model = 'ibm_hanoi'
	options = dict(backend=backend_name)  # Choose backend (required)
	# Initiate service
	service = QiskitRuntimeService(name='personal')
	if isinstance(noise_model, str):  # needed to simulate noise model based on backend you have access to
		account = service.active_account()  # need to have credentials stored locally
		token = account['token']
	else:
		token = None
	# Submit job/s
	job = None
	if not split_betas or not isinstance(beta, list):
		beta = [beta]
	if not isinstance(N, list):
		N = [N]
	for b in beta:
		for i in N:
			# inputs
			inputs = dict(n=n, J=J, h=h, beta=b, shots=shots, tomography_shots=tomography_shots,
			              N=i, noise_model=noise_model, token=token)
			# Run job
			job = service.run(program_id, inputs, options=options)
			job_id = job.job_id()
			print(f"Job sent to {backend_name} with job ID: {job_id} for beta: {b}, runs: {i}")
	# If only one job is sent, stream it
	if len(beta) == 1 and len(N) == 1:
		stream_results(job)


def stream_results(job):
	job_id = job.job_id()
	backend_name = job.backend().name
	n = job.inputs.get('n')
	J = job.inputs.get('J')
	h = job.inputs.get('h')
	shots = job.inputs.get('shots')
	noise_model = job.inputs.get('noise_model')
	# Start streaming results
	print("Streaming results")
	job.stream_results(callback)
	# Get results
	results = job.result()
	# Print and save results
	folder = f'jobs/{backend_name}'
	if isinstance(noise_model, str):
		folder += f'_{noise_model}'
	folder += f'/n_{n}_J_{J:.2f}_h_{h:.2f}_shots_{shots}'

	print_multiple_results(results, output_folder=folder, job_id=job_id, backend=backend_name, zip_file=False,
	                       hamiltonian='Ising')

	plot_result_min_avg_max(folder)


# noinspection PyUnusedLocal
def callback(job_id, interim_result):
	print(interim_result)


if __name__ == '__main__':
	main()
