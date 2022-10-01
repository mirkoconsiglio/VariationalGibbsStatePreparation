from gibbs_functions import ExactResult, fidelity, trace_distance, relative_entropy
from gibbs_ising_qulacs import GibbsIsing


def main():
	# Parameters
	n = 2  # number of qubits
	J = 1
	h = 0.5
	beta = 1
	shots = 1024  # Number of shots to sample
	seed = None
	noise_model = True
	# Define minimizer kwargs
	min_kwargs = dict()
	# Run VQA
	gibbs = GibbsIsing(n, J, h, beta)
	calculated_result = gibbs.run(min_kwargs, shots=shots, seed=seed, noise_model=noise_model)
	# Calculate exact results
	exact_result = ExactResult(gibbs.hamiltonian, beta)
	# Calculate comparative results
	f = fidelity(exact_result.gibbs_state, calculated_result.gibbs_state)
	td = trace_distance(exact_result.gibbs_state, calculated_result.gibbs_state)
	re = relative_entropy(exact_result.gibbs_state, calculated_result.gibbs_state)
	nf = fidelity(exact_result.gibbs_state, calculated_result.noiseless_gibbs_state)
	ntd = trace_distance(exact_result.gibbs_state, calculated_result.noiseless_gibbs_state)
	nre = relative_entropy(exact_result.gibbs_state, calculated_result.noiseless_gibbs_state)
	# Print results
	print()
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
	print()
	print(f'Noiseless Fidelity: {nf}')
	print(f'Noiseless Trace Distance: {ntd}')
	print(f'Noiseless Relative Entropy: {nre}')


if __name__ == '__main__':
	main()
