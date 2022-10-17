# noinspection PyUnresolvedReferences
import numpy as np

from gibbs_functions import print_results
from gibbs_ising_qulacs import GibbsIsing


def main():
	# Parameters
	n = 4  # number of qubits
	J = 1.
	h = 0.5
	beta = [1e-8, 0.2, 0.5, 0.8, 1., 1.2, 2., 5.]
	shots = None  # Number of shots to sample
	seed = None
	noise_model = False
	adiabatic_assistance = False
	# Define optimizer
	optimizer = 'SciPyOptimizer'
	min_kwargs = dict(method='BFGS')
	x0 = None
	# Folder name
	folder = 'data/'
	if shots:
		folder += f'{shots}_shots_'
	else:
		folder += 'exact_'
	if noise_model:
		folder += 'noise_'
	folder += f'n_{n}_J_{J:.2f}_h_{h:.2f}'
	# Run VQA
	results = []
	if not isinstance(beta, list):
		beta = [beta]
	for b in beta:
		gibbs = GibbsIsing(n, J, h, b)
		calculated_result = gibbs.run(
			optimizer=optimizer,
			min_kwargs=min_kwargs,
			x0=x0,
			shots=shots,
			seed=seed,
			noise_model=noise_model
		)

		results.append(calculated_result)

		if adiabatic_assistance:
			x0 = calculated_result['params']

	print_results(results, folder)

if __name__ == '__main__':
	main()
