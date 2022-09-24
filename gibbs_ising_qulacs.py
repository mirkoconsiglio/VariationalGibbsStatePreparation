import json
import numpy as np
from collections import Counter
from itertools import product
from functools import reduce
from qiskit.algorithms.optimizers import *
from qiskit.utils import algorithm_globals
from qulacs.circuit import QuantumCircuitOptimizer
from qulacs import ParametricQuantumCircuit, QuantumState
from qulacs.gate import (ParametricRY, CNOT, H, Sdag, AmplitudeDampingNoise, DephasingNoise, DepolarizingNoise,
                         BitFlipNoise, TwoQubitDepolarizingNoise, ParametricPauliRotation)
from qulacs.observable import create_observable_from_openfermion_text
from qulacs.state import partial_trace
from scipy.optimize import dual_annealing
from scipy.special import xlogy
from gibbs_functions import ising_hamiltonian, ising_hamiltonian_commuting_terms
from qulacsvis import circuit_drawer


class GibbsIsing:
	def __init__(self, n, J, h, beta):
		self.n = n
		self.J = J
		self.h = h
		self.beta = beta
		self.seed = None
		self.hamiltonian = ising_hamiltonian(self.n, self.J, self.h)
		self.observable = create_observable_from_openfermion_text(str(self.hamiltonian))
		self.inverse_beta = 1 / self.beta
		self.total_n = 2 * self.n
		self.dimension = 2 ** self.n
		self.ancilla_qubits = list(range(self.n))
		self.system_qubits = list(range(self.n, self.total_n))
		self.state = QuantumState(self.total_n)
		self.grad_state = QuantumState(self.n)
		self.ancilla_ansatz = None
		self.system_ansatz = None
		self.ansatz = None
		self.num_params = None
		self.num_ancilla_params = None
		self.num_system_params = None
		self.min_kwargs = None
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
		self.rho = None
		self.sigma = None
		self.result = None
		self.shots = None
		self.eigenvalues = None
		self.eigenvectors = None
		self.commuting_terms = None
		self.pauli_circuits = None
		self.total_shots = None
		self.cost_fun = None
		self.gradient_fun = None
		self.hamiltonian_eigenvalues = None
		self.noise_model = None

	def run(self, min_kwargs=None, x0=None, shots=None, ancilla_reps=None, system_reps=None, seed=None,
	        noise_model=None):
		self.seed = seed if seed is not None else np.random.randint(2 ** 32)
		np.random.seed(self.seed)
		algorithm_globals.random_seed = self.seed
		if noise_model:
			with open('noise_model.json', 'r') as f:
				self.noise_model = json.load(f)

		self.ancilla_reps = ancilla_reps or self.n - 1
		self.system_reps = system_reps or self.n - 1
		self.init_var_ansatz()
		self.num_ancilla_params = self.n * (self.ancilla_reps + 1)
		self.num_params = self.ansatz.get_parameter_count()
		self.num_system_params = self.num_params - self.num_ancilla_params
		self.x0 = x0 if x0 else np.random.uniform(0, 2 * np.pi, self.num_params)
		self.bounds = [(0, 2 * np.pi)] * self.num_params
		# Set up minimizer kwargs
		self.min_kwargs = min_kwargs if min_kwargs else dict()
		self.min_kwargs.update(callback=self.callback)
		self.shots = shots
		if self.shots:
			self.cost_fun = self.sampled_cost_fun
			self.gradient_fun = None  # self.sampled_gradient_fun
			self.commuting_terms = ising_hamiltonian_commuting_terms(self.n, self.J, self.h, self.system_qubits)
			self.pauli_circuits = self.generate_measurement_circuits()
			self.total_shots = self.shots * len(self.commuting_terms)
		else:
			self.cost_fun = self.statevector_cost_fun
			self.gradient_fun = self.statevector_gradient_fun

		self.iter = 0
		self.nfev = 0
		print('| iter | nfev | Cost | Energy | Entropy |')
		# self.result = SPSA(**self.min_kwargs).minimize(fun=self.cost_fun, x0=self.x0, bounds=self.bounds)
		self.result = dual_annealing(func=self.cost_fun, x0=self.x0, bounds=self.bounds, **self.min_kwargs)
		# Update
		self.params = self.result.x
		self.cost = self.result.fun
		self.rho, self.sigma = self.tomography()
		self.eigenvalues = np.sort(self.sigma)
		self.hamiltonian_eigenvalues = np.sort(self.cost - self.inverse_beta * np.log(self.eigenvalues))

		return GibbsResult(self)

	@staticmethod
	def set_parameters(circuit, params):
		for i, param in enumerate(params):
			circuit.set_parameter(i, param)

	@staticmethod
	def entropy_fun(p):
		return -np.sum([xlogy(i, i) for i in p])

	@staticmethod
	def get_bit_string(i, n):
		return f'{i:0{n}b}'[::-1]

	def get_bit_list(self, i, n):
		return [int(k) for k in self.get_bit_string(i, n)]

	def callback(self, *args, **kwargs):
		self.iter += 1
		print(f'| {self.iter} | {self.nfev} | {self.cost:.8f} | {self.energy:.8f} | {self.entropy:.8f} |')

	def add_gate_to_circuit(self, circuit, gate):
		if gate.is_parametric():
			circuit.add_parametric_gate(gate)
		else:
			circuit.add_gate(gate)
		if self.noise_model:
			match gate.get_name():
				case 'ParametricPauliRotation':
					[q1, q2] = gate.get_target_index_list()
					circuit.add_gate(AmplitudeDampingNoise(q1, self.noise_model['two_qubit_t1']))
					circuit.add_gate(AmplitudeDampingNoise(q2, self.noise_model['two_qubit_t1']))
					circuit.add_gate(DephasingNoise(q1, self.noise_model['two_qubit_t2']))
					circuit.add_gate(DephasingNoise(q2, self.noise_model['two_qubit_t2']))
					circuit.add_gate(TwoQubitDepolarizingNoise(q1, q2, self.noise_model['two_qubit_depo']))
				case _:
					q = gate.get_target_index_list()[0]
					circuit.add_gate(AmplitudeDampingNoise(q, self.noise_model['one_qubit_t1']))
					circuit.add_gate(DephasingNoise(q, self.noise_model['one_qubit_t2']))
					circuit.add_gate(DepolarizingNoise(q, self.noise_model['one_qubit_depo']))

	def add_readout_noise(self, circuit):
		for i in range(circuit.get_qubit_count()):
			circuit.add_gate(BitFlipNoise(i, self.noise_model['readout']))

	def init_var_ansatz(self):
		self.ansatz = ParametricQuantumCircuit(self.total_n)

		# Ancilla ansatz
		self.ancilla_ansatz = self.ancilla_circuit()
		self.ancilla_circuit(self.ansatz, self.ancilla_qubits)

		# Connecting CNOTs
		for i in range(self.n):
			self.add_gate_to_circuit(self.ansatz, CNOT(i, i + self.n))

		# System ansatz
		self.system_ansatz = self.system_circuit()
		self.system_circuit(self.ansatz, self.system_qubits)

		# circuit_drawer(self.ansatz, 'mpl')

	def ancilla_circuit(self, qc=None, qubits=None):
		if qc is None:
			qc = ParametricQuantumCircuit(self.n)
		if qubits is None:
			qubits = range(qc.get_qubit_count())

		# Layers
		for _ in range(self.ancilla_reps):
			for i in qubits:
				self.add_gate_to_circuit(qc, ParametricRY(i, 0))
				if i > qubits[0]:
					self.add_gate_to_circuit(qc, CNOT(i - 1, i))

		# Last one-qubit layer
		for i in qubits:
			self.add_gate_to_circuit(qc, ParametricRY(i, 0))

		return qc

	def system_circuit(self, qc=None, qubits=None):
		if qc is None:
			qc = ParametricQuantumCircuit(self.n)
		if qubits is None:
			qubits = range(qc.get_qubit_count())

		for _ in range(self.system_reps):
			for i in range(0, len(qubits) - 1, 2):
				self.add_gate_to_circuit(qc, ParametricPauliRotation([qubits[i], qubits[i + 1]], [1, 2], 0))
				self.add_gate_to_circuit(qc, ParametricPauliRotation([qubits[i], qubits[i + 1]], [2, 1], 0))
			for i in range(1, len(qubits) - 1, 2):
				self.add_gate_to_circuit(qc, ParametricPauliRotation([qubits[i], qubits[i + 1]], [1, 2], 0))
				self.add_gate_to_circuit(qc, ParametricPauliRotation([qubits[i], qubits[i + 1]], [2, 1], 0))

		# # Layers
		# for _ in range(self.system_reps):
		# 	for i in qubits:
		# 		self.add_gate_to_circuit(qc, ParametricRY(i, 0))
		# 		if i > qubits[0]:
		# 			self.add_gate_to_circuit(qc, CNOT(i - 1, i))
		#
		# # Last one-qubit layer
		# for i in qubits:
		# 	self.add_gate_to_circuit(qc, ParametricRY(i, 0))

		return qc

	def ancilla_params(self):
		return self.params[:self.num_ancilla_params]

	def system_params(self):
		return self.params[self.num_ancilla_params:]

	def ancilla_unitary_matrix(self):
		unitary = np.zeros((self.dimension, self.dimension), dtype=np.float64)
		state = QuantumState(self.n)
		qc = self.ancilla_circuit()
		self.set_parameters(qc, self.ancilla_params())

		for i in range(self.dimension):
			state.set_computational_basis(i)
			qc.update_quantum_state(state)
			unitary[i] = state.get_vector().real

		return unitary

	def system_unitary_matrix(self):
		unitary = np.zeros((self.dimension, self.dimension), dtype=np.float64)
		state = QuantumState(self.n)
		qc = self.system_circuit()
		self.set_parameters(qc, self.system_params())

		eigvals = partial_trace(self.state, self.system_qubits).get_matrix().diagonal()
		for i in np.argsort(eigvals):
			state.set_computational_basis(i)
			qc.update_quantum_state(state)
			unitary[i] = state.get_vector().real

		return unitary

	def generate_measurement_circuits(self):
		pauli_circuits = []
		for label, (_, terms) in self.commuting_terms.items():
			pauli_circ = self.ansatz.copy()
			if label != 'z':
				for qubits in terms:
					for q in qubits:
						self.add_gate_to_circuit(pauli_circ, H(q))
			if self.noise_model:
				self.add_readout_noise(pauli_circ)
			pauli_circuits.append(pauli_circ)

		return pauli_circuits

	def tomography(self):
		if self.shots:
			post_process_strings = {
				'I': np.array([1, 1]),
				'X': np.array([1, -1]),
				'Y': np.array([1, -1]),
				'Z': np.array([1, -1])
			}
			pauli = {
				'I': np.array([[1, 0], [0, 1]]),
				'X': np.array([[0, 1], [1, 0]]),
				'Y': np.array([[0, -1j], [1j, 0]]),
				'Z': np.array([[1, 0], [0, -1]])
			}

			# Inefficient recursive function to also get the Pauli strings including 'I' whenever we measure in 'Z'
			def pauli_helper(m, s=None):
				if s is None:
					s = set()
				for i, t in enumerate(m):
					_m = list(m).copy()
					if t == 'Z':
						_m[i] = 'I'
						pauli_helper(_m, s)
					s.add(tuple(m))
				return s

			sigma = np.zeros(self.dimension, dtype=np.float64)
			rho = np.zeros((self.dimension, self.dimension), dtype=np.complex128)
			# Construct measurement circuits
			for m in product(['X', 'Y', 'Z'], repeat=self.n):
				circuit = self.ansatz.copy()
				for i, t in enumerate(m):
					if t == 'X':
						self.add_gate_to_circuit(circuit, H(self.system_qubits[i]))
					elif t == 'Y':
						self.add_gate_to_circuit(circuit, Sdag(self.system_qubits[i]))
						self.add_gate_to_circuit(circuit, H(self.system_qubits[i]))
					self.set_parameters(circuit, self.params)
				if self.noise_model:
					self.add_readout_noise(circuit)
				# Update quantum state
				self.state.set_zero_state()
				self.set_parameters(circuit, self.params)
				circuit.update_quantum_state(self.state)
				# Sample
				samples = Counter(self.state.sampling(self.shots, self.seed)).items()
				counts = [[self.get_bit_string(i, self.total_n), j] for (i, j) in samples]
				# Compute rho and sigma
				for shot, n in counts:
					# rho
					for pauli_string in pauli_helper(m):
						post_process_vector = reduce(np.kron, [post_process_strings[i] for i in pauli_string])
						pauli_matrix = reduce(np.kron, [pauli[i] for i in pauli_string])
						coef = post_process_vector[int(shot[self.n:], 2)]
						rho += coef * pauli_matrix * n
					# sigma (assuming the state is diagonal)
					sigma[int(shot[:self.n], 2)] += n
			rho /= self.shots * self.dimension
			sigma /= self.shots * 3 ** self.n  # might as well use the 3^n measurements to compute with better accurary
		else:
			self.state.set_zero_state()
			self.set_parameters(self.ansatz, self.params)
			self.ansatz.update_quantum_state(self.state)
			rho = partial_trace(self.state, self.ancilla_qubits).get_matrix()
			sigma = partial_trace(self.state, self.system_qubits).get_matrix().diagonal().real

		return rho, sigma

	def statevector_cost_fun(self, x):
		self.nfev += 1
		self.params = x
		self.state.set_zero_state()
		self.set_parameters(self.ansatz, self.params)
		self.ansatz.update_quantum_state(self.state)
		self.energy = self.observable.get_expectation_value(partial_trace(self.state, self.ancilla_qubits))
		self.eigenvalues = [partial_trace(self.state, self.system_qubits).get_marginal_probability(
			self.get_bit_list(i, self.n)) for i in range(self.dimension)]
		self.entropy = self.entropy_fun(self.eigenvalues)
		self.cost = self.energy - self.inverse_beta * self.entropy
		# if self.nfev % 100 == 0:
		# 	print(f'| {self.iter} | {self.nfev} | {self.cost:.8f} | {self.energy:.8f} | {self.entropy:.8f} |')
		return self.cost

	# TODO: statevector_gradient_fun
	def statevector_gradient_fun(self, x):  # TODO: Check qulacs gates for parameter shift rule
		self.params = x
		self.nfev += self.num_params + 2 * self.num_params ** 2
		r = 0.5
		shift = np.pi / (4 * r)
		p = np.zeros(self.dimension)
		p_gradient = np.zeros((self.dimension, self.num_ancilla_params))
		U = np.zeros(self.dimension)
		U_gradient = np.zeros((self.dimension, self.num_system_params))
		# Evaluate ancilla unitary stuff
		for i in range(self.dimension):
			bit_list = self.get_bit_list(i, self.n)
			# default ancilla
			params = self.ancilla_params().copy()
			self.grad_state.set_zero_state()
			self.set_parameters(self.ancilla_ansatz, params)
			self.ancilla_ansatz.update_quantum_state(self.grad_state)
			p[i] = self.grad_state.get_marginal_probability(bit_list)
			for j in range(self.num_ancilla_params):
				# p_plus
				params = self.ancilla_params().copy()
				params[j] += shift
				self.grad_state.set_zero_state()
				self.set_parameters(self.ancilla_ansatz, params)
				self.ancilla_ansatz.update_quantum_state(self.grad_state)
				p_plus = self.grad_state.get_marginal_probability(bit_list)
				# p_minus
				params = self.ancilla_params().copy()
				params[j] -= shift
				self.grad_state.set_zero_state()
				self.set_parameters(self.ancilla_ansatz, params)
				self.ancilla_ansatz.update_quantum_state(self.grad_state)
				p_minus = self.grad_state.get_marginal_probability(bit_list)
				# p_gradient
				p_gradient[i, j] = r * (p_plus - p_minus)
		# Evaluate system unitary stuff
		for i in range(self.dimension):
			# default system
			params = self.system_params().copy()
			self.grad_state.set_computational_basis(i)
			self.set_parameters(self.system_ansatz, params)
			self.system_ansatz.update_quantum_state(self.grad_state)
			U[i] = self.observable.get_expectation_value(self.grad_state)
			for j in range(self.num_system_params):
				# U_plus
				params = self.system_params().copy()
				params[j] += shift
				self.grad_state.set_computational_basis(i)
				self.set_parameters(self.system_ansatz, params)
				self.system_ansatz.update_quantum_state(self.grad_state)
				U_plus = self.observable.get_expectation_value(self.grad_state)
				# U_minus
				params = self.system_params().copy()
				params[j] -= shift
				self.grad_state.set_computational_basis(i)
				self.set_parameters(self.system_ansatz, params)
				self.system_ansatz.update_quantum_state(self.grad_state)
				U_minus = self.observable.get_expectation_value(self.grad_state)
				# U_gradient
				U_gradient[i, j] = r * (U_plus - U_minus)
		# Evaluate ancilla_params_gradient
		ancilla_params_gradient = [np.sum([p_gradient[k, i] * (U[k] - self.inverse_beta * (np.log(p[k]) + 1))
		                                   for k in range(self.dimension)]) for i in range(self.num_ancilla_params)]
		# Evaluate system_params_gradient
		system_params_gradient = [np.sum([p[k] * U_gradient[k, i] for k in range(self.dimension)]) for i in
		                          range(self.num_system_params)]

		return np.concatenate((ancilla_params_gradient, system_params_gradient))

	def sampled_cost_fun(self, x):
		self.nfev += 1
		self.params = x

		def all_z_expectation(shot):
			return 2 * shot[self.n:].count('0') - self.n

		def xx_expectation(shot, q1, q2):
			return 1 if shot[q1] == shot[q2] else -1

		def entropy(p, shot):
			j = 0
			for b in shot[:self.n]:
				j = (j << 1) | int(b)
			p[j] += n

		energy = 0
		p = np.zeros(self.dimension)
		for pauli_circ, (label, (coef, qubits)) in zip(self.pauli_circuits, self.commuting_terms.items()):
			# Update quantum state
			self.state.set_zero_state()
			self.set_parameters(pauli_circ, self.params)
			pauli_circ.update_quantum_state(self.state)
			# Sample
			samples = Counter(self.state.sampling(self.shots, self.seed)).items()
			counts = [[self.get_bit_string(i, self.total_n), j] for (i, j) in samples]
			# Evaluate energy and entropy
			if label == 'z':
				for shot, n in counts:
					# Energy
					energy += all_z_expectation(shot) * coef * n
					# Entropy
					entropy(p, shot)
			else:
				for shot, n in counts:
					# Energy
					for q1, q2 in qubits:
						energy += xx_expectation(shot, q1, q2) * coef * n
					# Entropy
					entropy(p, shot)

		self.energy = energy / self.shots
		self.eigenvalues = p / self.total_shots
		self.entropy = self.entropy_fun(self.eigenvalues)
		self.cost = self.energy - self.inverse_beta * self.entropy

		return self.cost

	# TODO: sampled_gradient_fun
	def sampled_gradient_fun(self, x):
		pass


class GibbsResult:
	def __init__(self, gibbs):
		self.result = gibbs.result
		self.ancilla_unitary_params = gibbs.ancilla_params()
		self.system_unitary_params = gibbs.system_params()
		self.optimal_parameters = gibbs.params
		self.ancilla_unitary = gibbs.ancilla_unitary_matrix()
		self.system_unitary = gibbs.system_unitary_matrix()
		self.cost = gibbs.cost
		self.energy = gibbs.energy
		self.entropy = gibbs.entropy
		self.gibbs_state = gibbs.rho
		self.eigenvalues = gibbs.eigenvalues
		self.eigenvectors = gibbs.eigenvectors
		self.hamiltonian_eigenvalues = gibbs.hamiltonian_eigenvalues
