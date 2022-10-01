import json

from braket.circuits.noise_model import NoiseModel
from braket.devices import LocalSimulator

from gibbs_functions import ExactResult, fidelity, trace_distance, relative_entropy
from gibbs_ising_braket import GibbsIsing


def main():
	# Parameters
	n = 2  # number of qubits
	J = 1.
	h = 0.5
	beta = 1.
	shots = 512  # Number of shots to sample
	seed = 0
	use_noise_model = True
	# Load noise_model
	if use_noise_model:
		with open('../qulacs/noise_model.json', 'r') as f:
			data = json.load(f)
			noise_model = NoiseModel().from_dict(data)
	else:
		noise_model = None
	backend = LocalSimulator('braket_dm')
	# Define minimizer kwargs
	min_kwargs = dict(maxiter=100)
	# Run VQA
	gibbs = GibbsIsing(n, J, h, beta, backend=backend, noise_model=noise_model)
	calculated_result = gibbs.run(min_kwargs, shots=shots, seed=seed)
	# Calculate exact results
	exact_result = ExactResult(gibbs.hamiltonian, beta)
	# Calculate comparative results
	f = fidelity(exact_result.gibbs_state, calculated_result.gibbs_state)
	td = trace_distance(exact_result.gibbs_state, calculated_result.gibbs_state)
	re = relative_entropy(exact_result.gibbs_state, calculated_result.gibbs_state)
	# Print results
	print(calculated_result.result)
	print()
	print('Exact Gibbs state: ')
	print(exact_result.gibbs_state)
	print()
	print('Calculated Gibbs state: ')
	print(calculated_result.gibbs_state)
	print()
	print(f'VQA cost: {calculated_result.cost}')
	print(f'Exact cost: {exact_result.cost}')
	print()
	print(f'VQA energy: {calculated_result.energy}')
	print(f'Exact energy: {exact_result.energy}')
	print()
	print(f'VQA entropy: {calculated_result.entropy}')
	print(f'Exact entropy: {exact_result.entropy}')
	print()
	print(f'VQA eigenvalues: {calculated_result.eigenvalues}')
	print(f'Exact eigenvalues: {exact_result.eigenvalues}')
	print()
	print(f'VQA Hamiltonian eigenvalues: {calculated_result.hamiltonian_eigenvalues}')
	print(f'Exact Hamiltonian eigenvalues: {exact_result.hamiltonian_eigenvalues}')
	print()
	print(f'Fidelity: {f}')
	print(f'Trace Distance: {td}')
	print(f'Relative Entropy: {re}')


if __name__ == '__main__':
	main()
