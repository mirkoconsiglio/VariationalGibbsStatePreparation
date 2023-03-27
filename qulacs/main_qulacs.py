# noinspection PyUnresolvedReferences
import numpy as np

from gibbs_functions import print_multiple_results
from gibbs_ising_qulacs import GibbsIsing
from plotter import plot_result_min_avg_max


def main():
	# Parameters
	n = 4  # number of qubits
	J = 1.
	h = 0.5
	beta = [1e-10, 0.2, 0.5, 0.8, 1., 1.2, 2., 3., 4., 5.]
	shots = 8192  # Number of shots to sample
	seed = None
	ancilla_reps = None
	system_reps = None
	commuting_terms = True
	# Repeated runs
	N = 10
	# Define optimizer
	optimizer = 'SPSA'
	# min_kwargs = dict()
	min_kwargs = dict(maxiter=100 * n)
	# min_kwargs = dict(initial_temp=10., visit=2, accept=-10., maxfun=1000,
	# no_local_search=True, restart_temp_ratio=1e-10) Folder name
	folder = f'{optimizer}/'
	if shots:
		folder += f'{shots}_shots_'
	else:
		folder += 'exact_'
	folder += f'n_{n}_J_{J:.2f}_h_{h:.2f}'
	if not isinstance(beta, list):
		beta = [beta]
	multiple_results = []
	# Run VQA
	for b in beta:
		results = []
		for i in range(N):
			print(f'beta: {b}, run: {i}')
			gibbs = GibbsIsing(n, J, h, b)
			calculated_result = gibbs.run(
				optimizer=optimizer,
				min_kwargs=min_kwargs,
				shots=shots,
				ancilla_reps=ancilla_reps,
				system_reps=system_reps,
				seed=seed,
				commuting_terms=commuting_terms
			)

			results.append(calculated_result)

		multiple_results.append(results)

	print_multiple_results(multiple_results, folder)

	plot_result_min_avg_max(folder)


if __name__ == '__main__':
	main()
