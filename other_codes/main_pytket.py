import os

import numpy as np
import scipy
from pytket import Qubit
from pytket.pauli import Pauli, QubitPauliString
from pytket.utils.operators import QubitPauliOperator
from qiskit.algorithms.optimizers import SLSQP, SPSA
from scipy.linalg import expm, sqrtm

from gibbs_pytket import Gibbs


def state_trace_distance(rho, sigma):
	return scipy.linalg.norm(rho - sigma, 1) / 2


def state_fidelity(rho, sigma):
	sqrt_rho = sqrtm(rho)
	return sqrtm(sqrt_rho @ sigma @ sqrt_rho).diagonal().sum().real ** 2


def von_neumann_entropy(dm):
	return -np.sum([0 if i <= 0 else i * np.log(i) for i in np.linalg.eigh(dm)[0]])


def exact_gibbs_state(hamiltonian, beta):
	dm = expm(-beta * hamiltonian).real
	return dm / dm.diagonal().sum()


def xx_hamiltonian(n, J=1., h=1.):
	hamiltonian = {}
	for i in range(n):
		# Interaction terms
		if i != n - 1:
			XX = {Qubit(i): Pauli.X, Qubit(i + 1): Pauli.X}
			hamiltonian[QubitPauliString(XX)] = -J
		elif n > 2:
			XX = {Qubit(i): Pauli.X, Qubit(0): Pauli.X}
			hamiltonian[QubitPauliString(XX)] = -J
		# Magnetic terms
		Z = {Qubit(i): Pauli.Z}
		hamiltonian[QubitPauliString(Z)] = -h

	return QubitPauliOperator(hamiltonian)


def main():
	# Parameters
	n = 2 # number of qubits
	J = 1
	h = 0.5
	betas = [1]
	shots = None  # Number of shots to sample
	# Define Hamiltonian
	hamiltonian = xx_hamiltonian(n, J, h)
	# Define optimizer
	optimizer = SPSA(maxiter=1000) if shots else SLSQP(ftol=1e-10)
	# Loop
	vqa_cost = []
	exact_cost = []
	fidelity = []
	trace_distance = []
	for beta in betas:
		# Run VQA
		gibbs = Gibbs(n, optimizer)
		result = gibbs.run(hamiltonian, beta, shots=shots)
		print(result.result)
		# Generate xx_Gibbs State Preparation state
		calculated_state = result.gibbs_state
		calculated_state_energy = result.energy
		calculated_state_entropy = result.entropy
		calculated_state_cost = result.cost
		calculated_state_eigenvalues = result.eigenvalues
		calculated_hamiltonian_eigevalues = result.hamiltonian_eigenvalues
		# Calculate exact results
		exact_state = exact_gibbs_state(hamiltonian.to_sparse_matrix(), beta)
		exact_state_energy = (exact_state @ hamiltonian.to_sparse_matrix()).diagonal().sum().real
		exact_state_entropy = von_neumann_entropy(exact_state.todense())
		exact_state_cost = exact_state_energy - exact_state_entropy / beta
		exact_state_eigenvalues = np.linalg.eigh(exact_state.todense())[0]
		exact_hamiltonian_eigevalues = np.linalg.eigh(hamiltonian.to_sparse_matrix().todense())[0]
		# Print results
		f = state_fidelity(calculated_state, exact_state)
		td = state_trace_distance(calculated_state, exact_state)
		print()
		print(f'VQA cost: {calculated_state_cost}')
		print(f'Exact cost: {exact_state_cost}')
		print(f'VQA energy: {calculated_state_energy}')
		print(f'Exact energy: {exact_state_energy}')
		print(f'VQA entropy: {calculated_state_entropy}')
		print(f'Exact entropy: {exact_state_entropy}')
		print(f'VQA eigenvalues: {calculated_state_eigenvalues}')
		print(f'Exact eigenvalues: {exact_state_eigenvalues}')
		print(f'VQA Hamiltonian eigenvalues: {calculated_hamiltonian_eigevalues}')
		print(f'Exact Hamiltonian eigenvalues: {exact_hamiltonian_eigevalues}')
		print(f'Fidelity: {f}')
		print(f'Trace Distance: {td}')
		# Append statevector_h_0.5
		vqa_cost.append(calculated_state_cost)
		exact_cost.append(exact_state_cost)
		fidelity.append(f)
		trace_distance.append(td)

	directory = f'{f"sampling_{shots}" if shots else "statevector"}_data'
	filename = f'Gibbs_{n}'
	os.makedirs(f'{directory}/{filename}', exist_ok=True)
	np.savetxt(f'{directory}/{filename}/beta', betas)
	np.savetxt(f'{directory}/{filename}/vqa_cost', vqa_cost)
	np.savetxt(f'{directory}/{filename}/exact_cost', exact_cost)
	np.savetxt(f'{directory}/{filename}/fidelity', fidelity)
	np.savetxt(f'{directory}/{filename}/trace_distance', trace_distance)


if __name__ == '__main__':
	main()
