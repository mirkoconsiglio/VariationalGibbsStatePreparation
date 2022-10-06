from gibbs_functions import print_results
from gibbs_ising_qulacs import GibbsIsing


def main():
	# Parameters
	n = 3  # number of qubits
	J = 1.
	h = 0.5
	betas = [1e-8, 0.2, 0.5, 0.8, 1., 1.2, 2., 5.]
	shots = 1024  # Number of shots to sample
	seed = None
	noise_model = False
	adiabatic_assistance = True
	# Define minimizer kwargs
	min_kwargs = dict(maxfev=n * 500)
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
	for beta in betas:
		gibbs = GibbsIsing(n, J, h, beta)
		calculated_result = gibbs.run(
			min_kwargs=min_kwargs,
			x0=x0,
			shots=shots,
			seed=seed,
			noise_model=noise_model
		)

		results.append(calculated_result)

		if adiabatic_assistance:
			x0 = calculated_result['params']

	print_results(results, folder),

if __name__ == '__main__':
	main()
