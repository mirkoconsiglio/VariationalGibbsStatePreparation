import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import TwoLocal
from qiskit.opflow.list_ops import SummedOp
from qiskit.opflow.primitive_ops import PauliOp
from qiskit.quantum_info import Statevector, Pauli, state_fidelity, DensityMatrix, partial_trace
from scipy.linalg import expm
from scipy.optimize import minimize, LinearConstraint


def callback(args):
	pass


def generate_gibbs_state(n, params, unitary):
	lambdas = params[:2 ** n]
	thetas = params[2 ** n:]

	unit = unitary.assign_parameters(thetas)

	dm = np.zeros((2 ** n, 2 ** n), dtype=np.complex128)
	for i, l in enumerate(lambdas):
		dm += l * DensityMatrix(basis_qc(i, n, unit)).data

	return DensityMatrix(dm)


def gibbs_state(hamiltonian, beta):
	dm = expm(-beta * hamiltonian)
	return dm / np.trace(dm)


def hamiltonian(n, J, h):
	ham = []
	for i in range(n):
		# Interaction terms
		if i != n - 1:
			XX = 'I' * i + 'XX' + 'I' * (n - i - 2)
		else:
			XX = 'X' + 'I' * (n - 2) + 'X'
		ham.append(PauliOp(Pauli(XX), -J))
		# Magnetic terms
		Z = 'I' * i + 'Z' + 'I' * (n - i - 1)
		ham.append(PauliOp(Pauli(Z), -h))

	# Build up the Hamiltonian
	ham = SummedOp(ham)

	print(ham)

	return ham


def basis_qc(i, n, unitary):
	b = f'{i:0{n}b}'
	qc = QuantumCircuit(n)
	for j, k in enumerate(b):
		if k == '1':
			qc.x(j)
	qc.append(unitary, list(range(n)))
	print(qc)
	return qc


def measure_overlaps(unitary, n, hamiltonian):
	overlaps = []
	for i in range(2 ** n):
		qc = basis_qc(i, n, unitary)
		overlaps.append(Statevector(qc).expectation_value(hamiltonian).real)
	return overlaps


def cost(x, n, operator, unitary, inverse_beta):
	lambdas = x[:2 ** n]
	circuit_params = x[2 ** n:]
	overlaps = measure_overlaps(unitary.assign_parameters(circuit_params), n, operator)
	return np.sum([i * j + inverse_beta * ent(i) for i, j in zip(lambdas, overlaps)])

def ent(i):
	return 0 if i <= 0 else i * np.log(i)

def prepare(n, unitary, eigvals): # So far for n = 2 only
	alpha = get_alpha(eigvals)
	qc = QuantumCircuit(2 * n)
	qc.ry(alpha[0], 0)
	qc.cry(alpha[1], 0, 1, ctrl_state='0')
	qc.cry(alpha[2], 0, 1, ctrl_state='1')
	for i in range(n):
		qc.cx(i, n + i)
	qc.append(unitary, [2, 3])

	print(qc)

	return partial_trace(Statevector(qc), [0, 1]).data

def get_alpha(eigvals):
	num_params = len(eigvals) - 1
	x0 = np.random.uniform(-2 * np.pi, 2 * np.pi, num_params)
	args = (eigvals,)
	bounds = [[-2 * np.pi, 2 * np.pi] for _ in range(num_params)]
	result = minimize(fun, x0=x0, args=args, bounds=bounds, tol=1e-10)
	print(result)

	return result.x

def fun(x, eigvals):
	return np.sum([
		np.abs(np.cos(x[0]) ** 2 * np.cos(x[1]) ** 2 - eigvals[0]),
		np.abs(np.cos(x[0]) ** 2 * np.sin(x[1]) ** 2 - eigvals[1]),
		np.abs(np.sin(x[0]) ** 2 * np.cos(x[2]) ** 2 - eigvals[2]),
		np.abs(np.sin(x[0]) ** 2 * np.sin(x[2]) ** 2 - eigvals[3])
	])

def main():
	# Parameters
	n = 2  # number of qubits
	J = 1  # interaction
	h = 0.5  # magnetic field
	beta = 1  # inverse temperature
	reps = 2  # Number of ansatz layers
	# Define Hamiltonian
	operator = hamiltonian(n, J, h)
	# Define unitary
	unitary = TwoLocal(num_qubits=n, rotation_blocks='ry', entanglement_blocks='cx', entanglement='sca', reps=reps)
	print(unitary.decompose())
	# Define initial point
	x0 = np.random.uniform(0, 1, 2 ** n)
	x0 /= np.sum(x0)
	x0 = np.append(x0, np.random.uniform(-2 * np.pi, 2 * np.pi, unitary.num_parameters))
	# Define args
	args = (n, operator, unitary, 1 / beta)
	# Define method
	method = 'SLSQP'
	# Define bounds
	bounds = [[0, 1] for _ in range(2 ** n)]
	bounds.extend([[-2 * np.pi, 2 * np.pi] for _ in range(unitary.num_parameters)])
	# Define constraints
	constraints = [LinearConstraint(np.append(np.ones(2 ** n), np.zeros(unitary.num_parameters)), 1, 1)]
	# Optimize
	result = minimize(cost, x0, args=args, method=method, constraints=constraints,
	                  bounds=bounds, callback=callback, options=dict(ftol=1e-10, maxiter=1000))
	# Calculate xx_Gibbs State Preparation states
	calculated_gibbs_state = generate_gibbs_state(n, result.x, unitary)
	exact_gibbs_state = gibbs_state(operator.to_matrix(), beta)
	# Print results
	print(result)

	print(np.sort(result.x[:2 ** n]))
	print(np.linalg.eigh(gibbs_state(operator.to_matrix(), beta))[0])
	print(state_fidelity(calculated_gibbs_state, exact_gibbs_state))

	# Prepare xx_Gibbs State Preparation state
	eigvals = result.x[:2 ** n]
	theta = result.x[2 ** n:]
	dm = prepare(n, unitary.assign_parameters(theta), eigvals)

	# Print results
	print(np.linalg.eigh(dm)[0])
	print(np.linalg.eigh(gibbs_state(operator.to_matrix(), beta))[0])
	print(state_fidelity(dm, exact_gibbs_state))


if __name__ == '__main__':
	main()
