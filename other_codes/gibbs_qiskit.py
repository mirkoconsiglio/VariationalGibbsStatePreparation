import numpy as np
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.circuit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.quantum_info import Statevector, partial_trace, Operator
from qiskit.circuit.library import TwoLocal
from qiskit.opflow import StateFn, PauliExpectation, CircuitSampler, CircuitStateFn
from scipy.optimize import dual_annealing

class Gibbs:
	def __init__(self, n, min_kwargs=None):
		# Initialise parameters
		self.nqubits = n
		self.min_kwargs = min_kwargs if min_kwargs else dict()
		self.min_kwargs.update(callback=self.callback)
		self.dimension = 2 ** self.nqubits
		self.ancilla_qubits = list(range(self.nqubits))
		self.system_qubits = list(range(self.nqubits, 2 * self.nqubits))
		self.circuit = None
		self.statevector = None
		self.state_fn = None
		self.circuit_state_fn = None
		self.circuit_sampler = None
		self.pauli_exp = None
		self.backend = None
		self.quantum_instance = None
		self.ansatz = None
		self.num_params = None
		self.num_ancilla_params = None
		self.num_system_params = None
		self.hamiltonian = None
		self.inverse_beta = None
		self.x0 = None
		self.bounds = None
		self.ancilla_reps = None
		self.system_reps = None
		self.params = None
		self.energy = None
		self.entropy = None
		self.cost = None
		self.iter = None
		self.nfev = None
		self.result = None
		self.shots = None
		self.eigenvalues = None
		self.eigenvectors = None

	def run(self, hamiltonian, beta, x0=None, bounds=None, shots=None, ancilla_reps=None, system_reps=None):
		self.hamiltonian = hamiltonian
		self.inverse_beta = 1 / beta
		self.ancilla_reps = ancilla_reps if ancilla_reps else int(
			np.ceil((self.dimension - self.nqubits - 1) / self.nqubits))
		self.system_reps = system_reps if system_reps else int(
			np.ceil((self.dimension * (self.dimension - 1) / (2 * self.nqubits) - 1)))
		self.ansatz = self.var_ansatz()
		self.x0 = x0 if x0 else np.random.uniform(-2 * np.pi, 2 * np.pi, self.num_params)
		self.bounds = bounds if bounds else [(-2 * np.pi, 2 * np.pi) for _ in range(self.num_params)]
		self.shots = shots
		# If we have shots simulate using qasm
		if self.shots:
			self.backend = Aer.get_backend('qasm_simulator')
			self.quantum_instance = QuantumInstance(self.backend, self.shots)
			self.state_fn = StateFn(self.hamiltonian, is_measurement=True)
			self.circuit_sampler = CircuitSampler(self.quantum_instance, param_qobj=True, caching='all')
			self.pauli_exp = PauliExpectation()
		# set iter and fev to zero
		self.iter = 0
		self.nfev = 0
		print('iter     nfev     cost      energy     entropy')
		# Run the optimiser
		self.result = dual_annealing(func=self.cost_fun, x0=self.x0, bounds=self.bounds,
		                             **self.min_kwargs)

		return GibbsResult(self)

	def callback(self, *args):
		self.iter += 1
		print(f'{self.iter}', f'{self.nfev}', f'{self.cost:.8f}', f'{self.energy:.8f}', f'{self.entropy:.8f}')

	def cost_fun(self, x): # Cost function
		self.nfev += 1
		self.params = x
		self.circuit = self.ansatz.assign_parameters(self.params)
		self.statevector = Statevector(self.circuit)
		self.circuit_state_fn = CircuitStateFn(self.circuit) # change the circuit to a circuit state fn
		self.energy = self.energy_fun()
		self.entropy = self.entropy_fun()
		self.cost = self.energy - self.inverse_beta * self.entropy
		return self.cost

	def energy_fun(self):
		# Calculate the energy of the system qubits
		if self.shots:
			# Compose the Hamiltonian (state_fn) as a measurement operator with the circuit state fn
			# Note: when composing a measurement operator of n qubits with a state fn that has a larger number of qubits,
			# identities will be appended to the end of the Pauli strings, i.e. the Hamiltonian will be
			# evaluated on the last n qubits, which is why the system qubits are in the last half of the circuit.
			measurable_expression = self.state_fn.compose(self.circuit_state_fn)
			# Use a Pauli expectation convertor to measure circuits in the Z basis
			expectation = self.pauli_exp.convert(measurable_expression)
			# Use a circuit sampler to obtain the binary strings of the different circuits
			sampler = self.circuit_sampler.convert(expectation)
			# Evaluate the expectation values and sum them to return the energy
			return np.sum(sampler.eval()).real
		return self.statevector.expectation_value(self.hamiltonian, self.system_qubits).real

	def entropy_fun(self):
		# Calculate the entropy of the ancilla qubits
		if self.shots:
			counts = self.statevector.sample_counts(self.shots, self.ancilla_qubits)
			eigvals = [i / self.shots for i in counts.values()]
		else:
			eigvals = np.diagonal(partial_trace(self.statevector, self.system_qubits).data).real
		return -np.sum([i * np.log(i) for i in eigvals])

	def var_ansatz(self): # Function to generate the variational ansatz
		qc = QuantumCircuit(2 * self.nqubits)
		UA = self.ancilla_unitary()
		US = self.system_unitary()
		self.num_ancilla_params = UA.num_parameters
		self.num_system_params = US.num_parameters
		self.num_params = self.num_ancilla_params + self.num_system_params
		qc.append(UA.to_instruction(), self.ancilla_qubits)
		for i in range(self.nqubits):
			qc.cx(i, i + self.nqubits)
		qc.append(US.to_instruction(), self.system_qubits)
		qc.draw(output='mpl', filename=f'figures/circuit_{self.nqubits}')
		transpile(qc, basis_gates=['cx', 'id', 'rz', 'sx', 'x'],
		          optimization_level=3).draw(output='mpl', filename=f'figures/decomposed_circuit_{self.nqubits}')
		return qc

	def ancilla_unitary(self): # Ancilla ansatz
		return TwoLocal(num_qubits=self.nqubits, rotation_blocks='ry', entanglement_blocks='cx', entanglement='sca',
		                parameter_prefix='θ', reps=self.ancilla_reps)

	def system_unitary(self): # System ansatz
		return TwoLocal(num_qubits=self.nqubits, rotation_blocks='ry', entanglement_blocks='cx', entanglement='sca',
		                parameter_prefix='ϕ', reps=self.system_reps)

	def ancilla_unitary_params(self):
		return self.params[:self.num_ancilla_params]

	def system_unitary_params(self):
		return self.params[self.num_ancilla_params:]

	def ancilla_unitary_matrix(self):
		unitary = self.ancilla_unitary()
		params = self.ancilla_unitary_params()
		return Operator(unitary.assign_parameters(params)).data

	def system_unitary_matrix(self):
		unitary = self.system_unitary()
		params = self.system_unitary_params()
		return Operator(unitary.assign_parameters(params)).data

	def eigensystem(self): # Get eigensystem of probabilities and eigenvectors
		eigenvalues = partial_trace(self.statevector, self.system_qubits).data.diagonal().real
		eigenvectors = self.system_unitary_matrix().T
		indices = np.argsort(eigenvalues)

		return eigenvalues[indices], eigenvectors[indices].T


class GibbsResult:
	def __init__(self, gibbs):
		self.result = gibbs.result
		self.optimal_parameters = gibbs.params
		self.ancilla_unitary = gibbs.ancilla_unitary_matrix()
		self.system_unitary = gibbs.system_unitary_matrix()
		self.cost = gibbs.cost
		self.energy = gibbs.energy
		self.entropy = gibbs.entropy
		self.gibbs_state = partial_trace(gibbs.statevector, gibbs.ancilla_qubits).data
		self.eigenvalues, self.eigenvectors = gibbs.eigensystem()
		self.hamiltonian_eigenvalues = np.sort(gibbs.cost - gibbs.inverse_beta * np.log(self.eigenvalues))
