from qiskit import IBMQ

from gibbs_functions import *


def old_to_new(calculated_result):
	from qiskit_runtime.vgsp_ising_program import GibbsIsing
	from qiskit.quantum_info import Statevector, partial_trace

	gibbs = GibbsIsing()
	circuit = gibbs.ansatz.bind_parameters(calculated_result['params'])
	statevector = Statevector(circuit)
	noiseless_rho = partial_trace(statevector, gibbs.ancilla_qubits).data.real
	hamiltonian_eigenvalues = np.sort(calculated_result['cost'] -
	                                  1.0 * np.log(calculated_result['eigenvalues']))
	calculated_result.update(
		n=2,
		J=1.0,
		h=0.5,
		beta=1.0,
		ancilla_reps=1,
		system_reps=1,
		noiseless_rho=noiseless_rho.tolist(),
		hamiltonian_eigenvalues=hamiltonian_eigenvalues.tolist(),
	)


if __name__ == '__main__':
	provider = IBMQ.load_account()
	# Put job name here after it is finished to retrieve it
	job = provider.runtime.job('ccs5mb0spkt6bprua7o0')
	# Get job result
	calculated_result = job.result()
	# Soon-to-be deprecated function since I will update the program soon
	# old_to_new(calculated_result)
	n = calculated_result['n']
	J = calculated_result['J']
	h = calculated_result['h']
	beta = calculated_result['beta']
	hamiltonian = ising_hamiltonian(n, J, h)
	# Calculate exact results
	exact_result = ExactResult(hamiltonian, beta)
	# Calculate comparative results
	f = fidelity(exact_result.gibbs_state, calculated_result['rho'])
	td = trace_distance(exact_result.gibbs_state, calculated_result['rho'])
	re = relative_entropy(exact_result.gibbs_state, calculated_result['rho'])
	nf = fidelity(exact_result.gibbs_state, calculated_result['noiseless_rho'])
	ntd = trace_distance(exact_result.gibbs_state, calculated_result['noiseless_rho'])
	nre = relative_entropy(exact_result.gibbs_state, calculated_result['noiseless_rho'])
	# Print results
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
	print(f'Fidelity: {f}')
	print(f'Trace Distance: {td}')
	print(f'Relative Entropy: {re}')
	print()
	print(f"Noiseless Fidelity: {nf}")
	print(f"Noiseless Trace Distance: {ntd}")
	print(f"Noiseless Relative Entropy: {nre}")
