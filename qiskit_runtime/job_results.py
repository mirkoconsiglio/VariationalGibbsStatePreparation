from qiskit import IBMQ

from gibbs_functions import *


def print_results(calculated_result):
	n = calculated_result['n']
	J = calculated_result['J']
	h = calculated_result['h']
	beta = calculated_result['beta']
	hamiltonian = ising_hamiltonian(n, J, h)
	# Calculate exact results
	exact_result = ExactResult(hamiltonian, beta)
	# Calculate comparative results
	ef = fidelity(exact_result.gibbs_state, exact_result.gibbs_state)
	etd = trace_distance(exact_result.gibbs_state, exact_result.gibbs_state)
	ere = relative_entropy(exact_result.gibbs_state, exact_result.gibbs_state)
	ep = purity(exact_result.gibbs_state)
	cf = fidelity(exact_result.gibbs_state, calculated_result['rho'])
	ctd = trace_distance(exact_result.gibbs_state, calculated_result['rho'])
	cre = relative_entropy(exact_result.gibbs_state, calculated_result['rho'])
	cp = purity(calculated_result['rho'])
	nf = fidelity(exact_result.gibbs_state, calculated_result['noiseless_rho'])
	ntd = trace_distance(exact_result.gibbs_state, calculated_result['noiseless_rho'])
	nre = relative_entropy(exact_result.gibbs_state, calculated_result['noiseless_rho'])
	np = purity(calculated_result['noiseless_rho'])
	# Print results
	print()
	print(f"n: {n}")
	print(f"J: {J}")
	print(f"h: {h}")
	print(f"beta: {beta}")
	print()
	print("Exact Gibbs state: ")
	print(exact_result.gibbs_state)
	print()
	print("Calculated Gibbs state: ")
	print(calculated_result['rho'])
	print()
	print("Noiseless Calculated Gibbs state: ")
	print(calculated_result['noiseless_rho'])
	print()
	print(f"VQA cost: {calculated_result['cost']}")
	print(f"Exact cost: {exact_result.cost}")
	print()
	print(f"VQA energy: {calculated_result['energy']}")
	print(f"Exact energy: {exact_result.energy}")
	print()
	print(f"VQA entropy: {calculated_result['entropy']}")
	print(f"Exact entropy: {exact_result.entropy}")
	print()
	print(f"VQA eigenvalues: {calculated_result['eigenvalues']}")
	print(f"Exact eigenvalues: {exact_result.eigenvalues}")
	print()
	print(f"VQA Hamiltonian eigenvalues: {calculated_result['hamiltonian_eigenvalues']}")
	print(f"Exact Hamiltonian eigenvalues: {exact_result.hamiltonian_eigenvalues}")
	print()
	print(f"Exact Fidelity: {ef}")
	print(f"Exact Trace Distance: {etd}")
	print(f"Exact Relative Entropy: {ere}")
	print(f"Exact Purity: {ep}")
	print()
	print(f"Calculated Fidelity: {cf}")
	print(f"Calculated Trace Distance: {ctd}")
	print(f"Calculated Relative Entropy: {cre}")
	print(f"Calculated Purity: {cp}")
	print()
	print(f"Noiseless Fidelity: {nf}")
	print(f"Noiseless Trace Distance: {ntd}")
	print(f"Noiseless Relative Entropy: {nre}")
	print(f"Noiseless Purity: {np}")
	print()


if __name__ == '__main__':
	provider = IBMQ.load_account()
	# Put job name here after it is finished to retrieve it
	job = provider.runtime.job('ccspn5emg4jqer5qtav0')
	# Get job results
	results = job.result()
	# Obtain results
	for result in results:
		print_results(result)
