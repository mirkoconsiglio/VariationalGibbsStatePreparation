from qiskit import IBMQ
from qiskit_ibm_runtime import QiskitRuntimeService

from gibbs_functions import print_multiple_results
from plotter import plot_result_min_avg_max


def main():
	n = 2
	J = 1.
	h = 0.5
	beta = [1e-10, 0.2, 0.5, 0.8, 1.2, 1., 2., 3., 4., 5.]
	shots = 1024
	N = 10  # Can be split manually into a list
	split_betas = True  # Split each beta into a separate job with N runs each
	program_id = 'vgsp-ising-qGq4q73MaV'  # program id once it is uploaded
	backend_name = 'ibmq_qasm_simulator'
	noise_model = 'ibmq_guadalupe'
	options = dict(backend_name=backend_name)  # Choose backend (required)
	if isinstance(noise_model, str):  # needed to simulate noise model based on backend you have access to
		provider = IBMQ.load_account()  # need to have credentials stored locally
		credentials = dict(
			token=provider.credentials.token,
			hub=provider.credentials.hub,
			group=provider.credentials.group,
			project=provider.credentials.project
		)
	else:
		credentials = None
	# Initiate service
	service = QiskitRuntimeService()
	# Submit job/s
	job = None
	if not split_betas or not isinstance(beta, list):
		beta = [beta]
	if not isinstance(N, list):
		N = [N]
	for b in beta:
		for i in N:
			# inputs
			inputs = dict(n=n, J=J, h=h, beta=b, shots=shots, N=i, noise_model=noise_model, credentials=credentials)
			# Run job
			job = service.run(program_id, options=options, inputs=inputs)
			job_id = job.job_id()
			print(f"Job sent to {backend_name} with job ID: {job_id} for beta: {b}, runs: {i}")
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

	print_multiple_results(results, output_folder=folder, job_id=job_id, backend=backend_name, append=True)

	plot_result_min_avg_max(folder)


# noinspection PyUnusedLocal
def callback(job_id, interim_result):
	print(interim_result)


if __name__ == '__main__':
	main()
