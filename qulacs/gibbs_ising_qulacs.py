import json
from collections import Counter
from functools import reduce
from itertools import product

from qiskit.algorithms.optimizers import POWELL
from qiskit.utils import algorithm_globals
from qulacs import ParametricQuantumCircuit, QuantumState
from qulacs.gate import (sqrtX, RZ, ParametricRZ, CNOT, AmplitudeDampingNoise, DephasingNoise, DepolarizingNoise,
                         BitFlipNoise, TwoQubitDepolarizingNoise)
from qulacs.observable import create_observable_from_openfermion_text
from qulacs.state import partial_trace
from scipy.special import xlogy
from sklearn.gaussian_process.kernels import *

from gibbs_functions import ising_hamiltonian, ising_hamiltonian_commuting_terms, GibbsResult


def powerseries(eta=0.01, power=2, offset=0):
	n = 1
	while True:
		yield eta / ((n + offset) ** power)
		n += 1


class GibbsIsing:
	def __init__(self, n: int, J: float, h: float, beta: float) -> object:
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
		self.noiseless_rho = None
		self.noiseless_sigma = None
		self.result = None
		self.shots = None
		self.eigenvalues = None
		self.noiseless_eigenvalues = None
		self.eigenvectors = None
		self.commuting_terms = None
		self.pauli_circuits = None
		self.grad_pauli_circuits = None
		self.total_shots = None
		self.cost_fun = None
		self.gradient_fun = None
		self.hamiltonian_eigenvalues = None
		self.noiseless_hamiltonian_eigenvalues = None
		self.noise = None
		self.noise_model = None

	def run(self,
	        min_kwargs: dict | None = None,
	        x0: list[float] | None = None,
	        shots: int | None = None,
	        ancilla_reps: int | None = None,
	        system_reps: int | None = None,
	        seed: int | None = None,
	        noise_model: bool = False
	        ) -> GibbsResult:
		"""
		Executes the variational quantum algorithm for determining the Gibbs states of the Ising model.

		:param min_kwargs: optional kwargs for the minimizer.
		:param x0: initial set of points for the minimizer, defaults to None. If None, a random list of
			params between 0 and 2π is chosen.
		:param shots: number of shots to simulate circuits, defaults to None. If None, then statevector
			simulations are carried out.
		:param ancilla_reps: number of layer repetitions for the ancillary circuit, defaults to number of qubits n - 1
		:param system_reps: number of layer repetitions for the ancillary circuit, defaults to None. If None, then it is
		    set to the number of qubits: n - 1
		:param seed: seed for the simulations, defaults to None. If None, it is set to a random number
			between 0 and 2 ** 32 - 1
		:param noise_model: whether to use noise_model.json as a noise model. Can be customized by
			generate_noise_model_qulacs.py.
		:return: GibbsResult object containing results of the simulation
		"""
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
			self.gradient_fun = self.sampled_gradient_fun
			self.commuting_terms = ising_hamiltonian_commuting_terms(self.n, self.J, self.h)
			self.pauli_circuits, self.grad_pauli_circuits = self.generate_measurement_circuits()
			self.total_shots = self.shots * len(self.commuting_terms)
		else:
			self.cost_fun = self.statevector_cost_fun
			self.gradient_fun = self.statevector_gradient_fun

		self.iter = 0
		self.nfev = 0
		print('| iter | nfev | Cost | Energy | Entropy |')
		self.result = POWELL(**self.min_kwargs).minimize(fun=self.cost_fun, x0=self.x0, bounds=self.bounds,
		                                                 jac=self.gradient_fun)
		# self.result = gp_minimize(func=self.cost_fun, dimensions=self.bounds, **self.min_kwargs)
		# Update
		self.params = self.result.x
		# noinspection PyArgumentList
		self.cost_fun(self.params)
		self.noiseless_rho, self.noiseless_sigma = self.statevector_tomography()
		if self.shots:
			self.rho, self.sigma = self.sampled_tomography()
		else:
			self.rho, self.sigma = self.noiseless_rho, self.noiseless_sigma
		self.eigenvalues = np.sort(self.sigma)
		self.noiseless_eigenvalues = np.sort(self.noiseless_sigma)
		self.hamiltonian_eigenvalues = np.sort(self.cost - self.inverse_beta * np.log(self.eigenvalues))
		self.noiseless_hamiltonian_eigenvalues = np.sort(self.cost - self.inverse_beta *
		                                                 np.log(self.noiseless_eigenvalues))

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
				case 'CNOT':
					q1 = gate.get_control_index_list()[0]
					q2 = gate.get_target_index_list()[0]
					circuit.add_gate(AmplitudeDampingNoise(q1, self.noise_model['two_qubit_t1']))
					circuit.add_gate(AmplitudeDampingNoise(q2, self.noise_model['two_qubit_t1']))
					circuit.add_gate(DephasingNoise(q1, self.noise_model['two_qubit_t2']))
					circuit.add_gate(DephasingNoise(q2, self.noise_model['two_qubit_t2']))
					circuit.add_gate(TwoQubitDepolarizingNoise(q1, q2, self.noise_model['two_qubit_depo']))
				case 'sqrtX':
					q = gate.get_target_index_list()[0]
					circuit.add_gate(AmplitudeDampingNoise(q, self.noise_model['one_qubit_t1']))
					circuit.add_gate(DephasingNoise(q, self.noise_model['one_qubit_t2']))
					circuit.add_gate(DepolarizingNoise(q, self.noise_model['one_qubit_depo']))

	def add_readout_noise(self, circuit):
		for i in range(circuit.get_qubit_count()):
			circuit.add_gate(BitFlipNoise(i, self.noise_model['readout']))

	def add_qiskit_ising_gate(self, qc, q1, q2):  # U = R_yx.R_xy
		self.add_gate_to_circuit(qc, sqrtX(q1))
		self.add_gate_to_circuit(qc, sqrtX(q2))
		self.add_gate_to_circuit(qc, RZ(q1, 3 * np.pi / 2))
		self.add_gate_to_circuit(qc, CNOT(q1, q2))
		self.add_gate_to_circuit(qc, ParametricRZ(q2, 0))
		self.add_gate_to_circuit(qc, sqrtX(q1))
		self.add_gate_to_circuit(qc, sqrtX(q2))
		self.add_gate_to_circuit(qc, ParametricRZ(q1, 0))
		self.add_gate_to_circuit(qc, RZ(q2, np.pi))
		self.add_gate_to_circuit(qc, sqrtX(q1))
		self.add_gate_to_circuit(qc, RZ(q1, np.pi / 2))
		self.add_gate_to_circuit(qc, CNOT(q1, q2))
		self.add_gate_to_circuit(qc, sqrtX(q1))
		self.add_gate_to_circuit(qc, RZ(q2, np.pi / 2))

	def add_qiskit_ry_gate(self, qc, q):
		self.add_gate_to_circuit(qc, sqrtX(q))
		self.add_gate_to_circuit(qc, ParametricRZ(q, 0))
		self.add_gate_to_circuit(qc, sqrtX(q))
		self.add_gate_to_circuit(qc, RZ(q, np.pi))

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

	# circuit_drawer(self.ansatz, output_method='mpl')

	def ancilla_circuit(self, qc=None, qubits=None):
		if qc is None:
			qc = ParametricQuantumCircuit(self.n)
		if qubits is None:
			qubits = range(qc.get_qubit_count())

		# Layers
		for _ in range(self.ancilla_reps):
			for i in qubits:
				self.add_qiskit_ry_gate(qc, i)
				if i > qubits[0]:
					self.add_gate_to_circuit(qc, CNOT(i - 1, i))

		# Last one-qubit layer
		for i in qubits:
			self.add_qiskit_ry_gate(qc, i)

		return qc

	def system_circuit(self, qc=None, qubits=None):
		if qc is None:
			qc = ParametricQuantumCircuit(self.n)
		if qubits is None:
			qubits = range(qc.get_qubit_count())

		for _ in range(self.system_reps):
			for i in range(0, len(qubits) - 1, 2):
				self.add_qiskit_ising_gate(qc, qubits[i], qubits[i + 1])
			for i in range(1, len(qubits) - 1, 2):
				self.add_qiskit_ising_gate(qc, qubits[i], qubits[i + 1])

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
		grad_pauli_circuits = []
		for label, (_, terms) in self.commuting_terms.items():
			pauli_circ = self.ansatz.copy()
			grad_pauli_circ = self.system_ansatz.copy()
			if label != 'z':
				for qubits in terms:
					for q in qubits:  # Hadamard gate
						self.add_gate_to_circuit(pauli_circ, RZ(q + self.n, np.pi / 2))
						self.add_gate_to_circuit(pauli_circ, sqrtX(q + self.n))
						self.add_gate_to_circuit(grad_pauli_circ, RZ(q, np.pi / 2))
						self.add_gate_to_circuit(grad_pauli_circ, sqrtX(q))
			if self.noise_model:
				self.add_readout_noise(pauli_circ)
				self.add_readout_noise(grad_pauli_circ)
			pauli_circuits.append(pauli_circ)
			grad_pauli_circuits.append(grad_pauli_circ)

		return pauli_circuits, grad_pauli_circuits

	def statevector_tomography(self):
		self.state.set_zero_state()
		self.set_parameters(self.ansatz, self.params)
		self.ansatz.update_quantum_state(self.state)
		rho = partial_trace(self.state, self.ancilla_qubits).get_matrix().real
		sigma = partial_trace(self.state, self.system_qubits).get_matrix().diagonal().real

		return rho, sigma

	def sampled_tomography(self):
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
		rho = np.zeros((self.dimension, self.dimension), dtype=np.float64)
		# Construct measurement circuits
		for measurement in list(product(['X', 'Y', 'Z'], repeat=self.n)):
			circuit = self.ansatz.copy()
			for i, t in enumerate(measurement):
				if t == 'X':  # Hadamard gate
					self.add_gate_to_circuit(circuit, RZ(self.system_qubits[i], np.pi / 2))
					self.add_gate_to_circuit(circuit, sqrtX(self.system_qubits[i]))
				elif t == 'Y':
					self.add_gate_to_circuit(circuit, sqrtX(self.system_qubits[i]))
				self.set_parameters(circuit, self.params)
			if self.noise_model:
				self.add_readout_noise(circuit)
			# Update quantum state
			self.state.set_zero_state()
			self.set_parameters(circuit, self.params)
			circuit.update_quantum_state(self.state)
			# Sample
			samples = Counter(self.state.sampling(self.shots, self.seed)).items()
			counts = [[self.get_bit_string(i, self.total_n), j] for i, j in samples]
			# Compute rho and sigma
			for shot, n in counts:
				# rho (assuming the state is real)
				for pauli_string in pauli_helper(measurement):
					post_process_vector = reduce(np.kron, [post_process_strings[i] for i in pauli_string])
					pauli_matrix = reduce(np.kron, [pauli[i] for i in pauli_string]).real
					coef = post_process_vector[int(shot[self.n:], 2)]
					rho += coef * pauli_matrix * n
				# sigma (assuming the state is diagonal)
				sigma[int(shot[:self.n], 2)] += n
		rho /= self.shots * self.dimension
		sigma /= self.shots * 3 ** self.n  # might as well use the 3^n measurements to compute with better accuracy

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

	def statevector_gradient_fun(self, x):
		self.params = x
		self.nfev += self.dimension * (1 + 2 * self.num_params)
		r = 0.5
		shift = np.pi / (4 * r)
		p = np.zeros(self.dimension)  # p evaluated at x
		p_gradient = np.zeros((self.dimension, self.num_ancilla_params))  # gradient of p at x
		U = np.zeros(self.dimension)  # expectation of U at x
		U_gradient = np.zeros((self.dimension, self.num_system_params))  # gradient of the expectation of U at x
		# Evaluate ancilla unitary stuff
		for i in range(self.dimension):
			bit_list = self.get_bit_list(i, self.n)
			# p_i evaluated at x
			params = self.ancilla_params().copy()
			self.grad_state.set_zero_state()
			self.set_parameters(self.ancilla_ansatz, params)
			self.ancilla_ansatz.update_quantum_state(self.grad_state)
			p[i] = self.grad_state.get_marginal_probability(bit_list)
			for j in range(self.num_ancilla_params):
				# p_ij_plus
				params = self.ancilla_params().copy()
				params[j] += shift
				self.grad_state.set_zero_state()
				self.set_parameters(self.ancilla_ansatz, params)
				self.ancilla_ansatz.update_quantum_state(self.grad_state)
				p_plus = self.grad_state.get_marginal_probability(bit_list)
				# p_ij_minus
				params = self.ancilla_params().copy()
				params[j] -= shift
				self.grad_state.set_zero_state()
				self.set_parameters(self.ancilla_ansatz, params)
				self.ancilla_ansatz.update_quantum_state(self.grad_state)
				p_minus = self.grad_state.get_marginal_probability(bit_list)
				# p_ij_gradient
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
		ancilla_params_gradient = [np.sum([p_gradient[k, i] * (U[k] + self.inverse_beta * (np.log(p[k]) + 1))
		                                   for k in range(self.dimension)]) for i in range(self.num_ancilla_params)]
		# Evaluate system_params_gradient
		system_params_gradient = [np.sum([p[k] * U_gradient[k, i] for k in range(self.dimension)]) for i in
		                          range(self.num_system_params)]

		return np.concatenate((ancilla_params_gradient, system_params_gradient))

	@staticmethod
	def all_z_expectation(shot, n):
		return 2 * shot.count('0') - n

	@staticmethod
	def xx_expectation(shot, q1, q2):
		return 1 if shot[q1] == shot[q2] else -1

	@staticmethod
	def probabilities(p, shot, n):
		j = 0
		for b in shot:
			j = (j << 1) | int(b)
		p[j] += n

	def sampled_cost_fun(self, x):
		self.nfev += 1
		self.params = x
		energy = 0
		p = np.zeros(self.dimension)
		for pauli_circ, (label, (coef, qubits)) in zip(self.pauli_circuits, self.commuting_terms.items()):
			# Update quantum state
			self.state.set_zero_state()
			self.set_parameters(pauli_circ, self.params)
			pauli_circ.update_quantum_state(self.state)
			# Sample
			samples = Counter(self.state.sampling(self.shots, self.seed)).items()
			counts = [[self.get_bit_string(i, self.total_n), j] for i, j in samples]
			# Evaluate energy and entropy
			if label == 'z':
				for shot, n in counts:
					# Energy
					energy += self.all_z_expectation(shot[self.n:], self.n) * coef * n
					# Entropy
					self.probabilities(p, shot[:self.n], n)
			else:
				for shot, n in counts:
					# Energy
					for q1, q2 in qubits:
						energy += self.xx_expectation(shot[self.n:], q1, q2) * coef * n
					# Entropy
					self.probabilities(p, shot[:self.n], n)

		self.energy = energy / self.shots
		self.eigenvalues = p / self.total_shots
		self.entropy = self.entropy_fun(self.eigenvalues)
		self.cost = self.energy - self.inverse_beta * self.entropy

		return self.cost

	def get_grad_sampled_marginal_probability(self, bit_string):
		samples = Counter(self.grad_state.sampling(self.shots, self.seed)).items()
		return next((i[1] for i in samples if self.get_bit_string(i[0], self.n) == bit_string), 0) / self.shots

	def get_grad_sampled_expectation_value(self, k, params):
		energy = 0
		for grad_pauli_circ, (label, (coef, qubits)) in zip(self.grad_pauli_circuits, self.commuting_terms.items()):
			# Update quantum state
			self.grad_state.set_computational_basis(k)
			self.set_parameters(grad_pauli_circ, params)
			grad_pauli_circ.update_quantum_state(self.grad_state)
			# Sample
			samples = Counter(self.grad_state.sampling(self.shots, self.seed)).items()
			counts = [[self.get_bit_string(i, self.n), j] for i, j in samples]
			# Evaluate energy
			if label == 'z':
				for shot, n in counts:
					energy += self.all_z_expectation(shot, self.n) * coef * n
			else:
				for shot, n in counts:
					for q1, q2 in qubits:
						energy += self.xx_expectation(shot, q1, q2) * coef * n

		return energy / self.shots

	def sampled_gradient_fun(self, x):
		self.params = x
		self.nfev += self.dimension * (1 + 2 * self.num_params)
		r = 0.5
		shift = np.pi / (4 * r)
		p = np.zeros(self.dimension)  # p evaluated at x
		p_gradient = np.zeros((self.dimension, self.num_ancilla_params))  # gradient of p at x
		U = np.zeros(self.dimension)  # expectation of U at x
		U_gradient = np.zeros((self.dimension, self.num_system_params))  # gradient of the expectation of U at x
		# Evaluate ancilla unitary stuff
		for i in range(self.dimension):
			bit_string = self.get_bit_string(i, self.n)
			# p_i evaluated at x
			params = self.ancilla_params().copy()
			self.grad_state.set_zero_state()
			self.set_parameters(self.ancilla_ansatz, params)
			self.ancilla_ansatz.update_quantum_state(self.grad_state)
			p[i] = self.get_grad_sampled_marginal_probability(bit_string)
			for j in range(self.num_ancilla_params):
				# p_ij_plus
				params = self.ancilla_params().copy()
				params[j] += shift
				self.grad_state.set_zero_state()
				self.set_parameters(self.ancilla_ansatz, params)
				self.ancilla_ansatz.update_quantum_state(self.grad_state)
				p_plus = self.get_grad_sampled_marginal_probability(bit_string)
				# p_ij_minus
				params = self.ancilla_params().copy()
				params[j] -= shift
				self.grad_state.set_zero_state()
				self.set_parameters(self.ancilla_ansatz, params)
				self.ancilla_ansatz.update_quantum_state(self.grad_state)
				p_minus = self.get_grad_sampled_marginal_probability(bit_string)
				# p_ij_gradient
				p_gradient[i, j] = r * (p_plus - p_minus)
		# Evaluate system unitary stuff
		for i in range(self.dimension):
			# default system
			params = self.system_params().copy()
			U[i] = self.get_grad_sampled_expectation_value(i, params)
			for j in range(self.num_system_params):
				# U_plus
				params = self.system_params().copy()
				params[j] += shift
				self.grad_state.set_computational_basis(i)
				U_plus = self.get_grad_sampled_expectation_value(i, params)
				# U_minus
				params = self.system_params().copy()
				params[j] -= shift
				self.grad_state.set_computational_basis(i)
				U_minus = self.get_grad_sampled_expectation_value(i, params)
				# U_gradient
				U_gradient[i, j] = r * (U_plus - U_minus)

		# Evaluate ancilla_params_gradient
		ancilla_params_gradient = [np.sum([p_gradient[k, i] * (U[k] + self.inverse_beta * (np.log(p[k]) + 1))
		                                   for k in range(self.dimension)]) for i in range(self.num_ancilla_params)]
		# Evaluate system_params_gradient
		system_params_gradient = [np.sum([p[k] * U_gradient[k, i] for k in range(self.dimension)]) for i in
		                          range(self.num_system_params)]

		return np.concatenate((ancilla_params_gradient, system_params_gradient))
