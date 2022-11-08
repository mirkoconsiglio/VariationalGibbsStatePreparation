import inspect
import json
from collections import Counter
from functools import reduce
from itertools import product
from typing import Callable, List

import numpy as np
from openfermion import QubitOperator
from qiskit.algorithms import optimizers
from qiskit.utils import algorithm_globals
from qulacs import ParametricQuantumCircuit, QuantumState, QuantumGateBase
from qulacs.gate import (H, sqrtX, RZ, ParametricRZ, CNOT, AmplitudeDampingNoise, DephasingNoise, DepolarizingNoise,
                         BitFlipNoise, TwoQubitDepolarizingNoise, ParametricRY, ParametricPauliRotation, Sdag)
from qulacs.observable import create_observable_from_openfermion_text
from qulacs.state import partial_trace, permutate_qubit
# noinspection PyUnresolvedReferences
from qulacsvis import circuit_drawer
from scipy.optimize import dual_annealing
from scipy.special import xlogy
from skopt import gp_minimize

_optimizers = dict(inspect.getmembers(optimizers, inspect.isclass))


class GibbsIsing:
	def __init__(self, n: int, J: float, h: float, beta: float) -> None:
		self.n = n
		self.J = J
		self.h = h
		self.beta = beta
		self.seed = None
		self.hamiltonian = self.ising_hamiltonian(self.n, self.J, self.h)
		self.observable = create_observable_from_openfermion_text(str(self.hamiltonian))
		self.inverse_beta = 1 / self.beta
		self.total_n = 2 * self.n
		self.dimension = 2 ** self.n
		self.ancilla_qubits = list(range(self.n))
		self.system_qubits = list(range(self.n, self.total_n))
		self.state = QuantumState(self.total_n)
		self.ancilla_ansatz = None
		self.system_ansatz = None
		self.ansatz = None
		self.num_params = None
		self.num_ancilla_params = None
		self.num_system_params = None
		self.optimizer = None
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
		self.terms = None
		self.commuting_terms = None
		self.pauli_circuits = None
		self.total_shots = None
		self.cost_fun = None
		self.hamiltonian_eigenvalues = None
		self.noiseless_hamiltonian_eigenvalues = None
		self.noise = None
		self.noise_model = None

	def run(self,
	        optimizer: str = 'SPSA',
	        min_kwargs: dict | None = None,
	        x0: List[float] | None = None,
	        shots: int | None = None,
	        ancilla_reps: int | None = None,
	        system_reps: int | None = None,
	        seed: int | None = None,
	        noise_model: bool = False,
	        commuting_terms: bool = True,
	        ) -> dict:
		"""
		Executes the variational quantum algorithm for determining the Gibbs states of the Ising model.

		:param optimizer: Qiskit optimizer given as a string.
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
		:param commuting_terms: whether to group commuting terms or not.
		:return: dict containing results of the simulation
		"""
		self.seed = seed if seed is not None else np.random.randint(2 ** 32)
		np.random.seed(self.seed)
		algorithm_globals.random_seed = self.seed
		if noise_model:
			with open('noise_model.json', 'r') as f:
				self.noise_model = json.load(f)
		if ancilla_reps is not None:
			self.ancilla_reps = ancilla_reps
		else:
			self.ancilla_reps = 1
		if system_reps is not None:
			self.system_reps = system_reps
		else:
			self.system_reps = self.n - 1
		self.ansatz = self.init_var_ansatz()
		self.num_ancilla_params = self.n * (self.ancilla_reps + 1)
		self.num_params = self.ansatz.get_parameter_count()
		self.num_system_params = self.num_params - self.num_ancilla_params
		self.x0 = x0 if x0 is not None else np.random.uniform(-np.pi, np.pi, self.num_params)
		self.bounds = [(-np.pi, np.pi)] * self.num_params
		# Set up optimizer
		self.min_kwargs = min_kwargs if min_kwargs else dict()
		opt = _optimizers.get(optimizer)
		if opt:
			parameters = inspect.signature(opt.__init__).parameters
			if 'kwargs' in parameters or 'callback' in parameters:
				if optimizer == 'QNSPSA':
					self.optimizer = opt(**self.min_kwargs, callback=self.callback,
					                     fidelity=self.get_fidelity(self.system_circuit))
				else:
					self.optimizer = opt(**self.min_kwargs, callback=self.callback)
			else:
				self.optimizer = opt(**self.min_kwargs)
		# Set up cost function
		self.shots = shots
		if self.shots:
			self.commuting_terms = self.ising_hamiltonian_commuting_terms(self.n, self.J, self.h)
			self.pauli_circuits = self.generate_measurement_circuits()
			if commuting_terms:
				self.total_shots = self.shots * len(self.commuting_terms)
				self.cost_fun = self.commuting_sampled_cost_fun
			else:
				self.total_shots = self.shots * np.sum(len(i[1][1]) for i in self.commuting_terms.items())
				self.cost_fun = self.sampled_cost_fun
		else:
			self.cost_fun = self.statevector_cost_fun

		self.iter = 0
		self.nfev = 0
		if optimizer == 'gp_minimize':
			self.optimizer = gp_minimize
			self.result = self.optimizer(func=self.cost_fun, dimensions=self.bounds, callback=self.callback,
			                             **self.min_kwargs)
		elif optimizer == 'dual_annealing':
			self.optimizer = dual_annealing
			self.result = self.optimizer(func=self.cost_fun, bounds=self.bounds, callback=self.callback,
			                             **self.min_kwargs)
		else:
			self.result = self.optimizer.minimize(fun=self.cost_fun, x0=self.x0, bounds=self.bounds)
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
		self.noiseless_hamiltonian_eigenvalues = np.sort(self.cost -
		                                                 self.inverse_beta * np.log(self.noiseless_eigenvalues))
		# Return results
		return dict(
			n=self.n,
			J=self.J,
			h=self.h,
			beta=self.beta,
			ancilla_reps=self.ancilla_reps,
			system_reps=self.system_reps,
			iter=self.iter,
			nfev=self.nfev,
			cost=self.cost,
			energy=self.energy,
			entropy=self.entropy,
			params=self.params,
			eigenvalues=self.eigenvalues,
			rho=self.rho,
			sigma=self.sigma,
			noiseless_rho=self.noiseless_rho,
			noiseless_sigma=self.noiseless_sigma,
			noiseless_eigenvalues=self.noiseless_eigenvalues,
			hamiltonian_eigenvalues=self.hamiltonian_eigenvalues,
			noiseless_hamiltonian_eigenvalues=self.noiseless_hamiltonian_eigenvalues,
			ansatz=self.ansatz,
			optimizer=self.optimizer.__class__.__name__,
			min_kwargs=self.min_kwargs,
			shots=self.shots,
			result=self.result
		)

	@staticmethod
	def ising_hamiltonian(n: int, J: float = 1., h: float = 0.5) -> QubitOperator:
		hamiltonian = QubitOperator()
		for i in range(n):
			# Interaction terms
			if i != n - 1:
				hamiltonian += QubitOperator(f'X{i} X{i + 1}', -J)
			elif n > 2:
				hamiltonian += QubitOperator(f'X0 X{n - 1}', -J)
			# Magnetic terms
			hamiltonian += QubitOperator(f'Z{i}', -h)

		return hamiltonian

	@staticmethod
	def ising_hamiltonian_commuting_terms(n: int, J: float = 1., h: float = 0.5) -> dict:
		terms = dict()
		if J != 0:
			if n == 2:
				terms.update(XX=[-J, [[0, 1]]])
			elif n > 2:
				terms.update(XX=[-J, [[i, (i + 1) % n] for i in range(n)]])
		if h != 0:
			terms.update(Z=[-h, [[i] for i in range(n)]])

		return terms

	@staticmethod
	def set_parameters(circuit: ParametricQuantumCircuit, params: List[float], idx_list: List[int] = None) -> None:
		if idx_list is not None:
			iterable = zip(idx_list, params)
		else:
			iterable = enumerate(params)
		for i, param in iterable:
			circuit.set_parameter(i, param)

	@staticmethod
	def entropy_fun(p: List[float]) -> float:
		# noinspection PyCallingNonCallable
		return -np.sum([xlogy(i, i) for i in p])

	@staticmethod
	def get_bit_string(i: int, n: int) -> str:
		return f'{i:0{n}b}'[::-1]

	def get_bit_list(self, i: int, n: int) -> List[int]:
		return [int(k) for k in self.get_bit_string(i, n)]

	# noinspection PyUnusedLocal
	def callback(self, *args: any, **kwargs: any) -> None:
		if self.iter == 0:
			print('| iter | nfev | Cost | Energy | Entropy |')
		self.iter += 1
		print(f'| {self.iter} | {self.nfev} | {self.cost:.8f} | {self.energy:.8f} | {self.entropy:.8f} |')

	def add_gate(self, circuit: ParametricQuantumCircuit, gate: QuantumGateBase) -> None:
		if gate.is_parametric():
			circuit.add_parametric_gate(gate)
		else:
			circuit.add_gate(gate)
		if self.noise_model and gate.get_name() not in ['Z-rotation', 'ParametricRZ']:  # Assume Z gates are virtual
			if len(gate.get_control_index_list()) != 0:
				q1 = gate.get_control_index_list()[0]
				q2 = gate.get_target_index_list()[0]
				circuit.add_gate(AmplitudeDampingNoise(q1, self.noise_model['two_qubit_t1']))
				circuit.add_gate(AmplitudeDampingNoise(q2, self.noise_model['two_qubit_t1']))
				circuit.add_gate(DephasingNoise(q1, self.noise_model['two_qubit_t2']))
				circuit.add_gate(DephasingNoise(q2, self.noise_model['two_qubit_t2']))
				circuit.add_gate(TwoQubitDepolarizingNoise(q1, q2, self.noise_model['two_qubit_depo']))
			else:
				q = gate.get_target_index_list()[0]
				circuit.add_gate(AmplitudeDampingNoise(q, self.noise_model['one_qubit_t1']))
				circuit.add_gate(DephasingNoise(q, self.noise_model['one_qubit_t2']))
				circuit.add_gate(DepolarizingNoise(q, self.noise_model['one_qubit_depo']))

	def add_readout_noise(self, circuit: ParametricQuantumCircuit) -> None:
		for i in range(circuit.get_qubit_count()):
			circuit.add_gate(BitFlipNoise(i, self.noise_model['readout']))

	def add_qiskit_ising_gate(self, qc: ParametricQuantumCircuit, q1: int, q2: int) -> None:  # U = R_yx.R_xy
		self.add_gate(qc, sqrtX(q1))
		self.add_gate(qc, sqrtX(q2))
		self.add_gate(qc, RZ(q1, 3 * np.pi / 2))
		self.add_gate(qc, CNOT(q1, q2))
		self.add_gate(qc, ParametricRZ(q2, 0))
		self.add_gate(qc, sqrtX(q1))
		self.add_gate(qc, sqrtX(q2))
		self.add_gate(qc, ParametricRZ(q1, 0))
		self.add_gate(qc, RZ(q2, np.pi))
		self.add_gate(qc, sqrtX(q1))
		self.add_gate(qc, RZ(q1, np.pi / 2))
		self.add_gate(qc, CNOT(q1, q2))
		self.add_gate(qc, sqrtX(q1))
		self.add_gate(qc, RZ(q2, np.pi / 2))

	def add_qiskit_ry_gate(self, qc: ParametricQuantumCircuit, q: int) -> None:
		self.add_gate(qc, sqrtX(q))
		self.add_gate(qc, ParametricRZ(q, 0))
		self.add_gate(qc, sqrtX(q))
		self.add_gate(qc, RZ(q, np.pi))

	def add_qiskit_h_gate(self, qc: ParametricQuantumCircuit, q: int) -> None:
		self.add_gate(qc, RZ(q, np.pi / 2))
		self.add_gate(qc, sqrtX(q))

	def add_qiskit_x_basis_transform(self, qc: ParametricQuantumCircuit, q: int) -> None:
		self.add_qiskit_h_gate(qc, q)

	def add_qiskit_y_basis_transform(self, qc: ParametricQuantumCircuit, q: int) -> None:
		self.add_gate(qc, sqrtX(q))

	def init_var_ansatz(self) -> ParametricQuantumCircuit:
		circuit = ParametricQuantumCircuit(self.total_n)

		# Ancilla ansatz
		self.ancilla_ansatz = self.ancilla_circuit()
		self.ancilla_circuit(circuit, self.ancilla_qubits)

		# Connecting CNOTs
		for i in range(self.n):
			self.add_gate(circuit, CNOT(i, i + self.n))

		# System ansatz
		self.system_ansatz = self.system_circuit()
		self.system_circuit(circuit, self.system_qubits)

		return circuit

	def ancilla_circuit(self, qc: ParametricQuantumCircuit = None,
	                    qubits: List[int] = None) -> ParametricQuantumCircuit:
		if qc is None:
			qc = ParametricQuantumCircuit(self.n)
		if qubits is None:
			qubits = range(qc.get_qubit_count())

		# Layers
		for _ in range(self.ancilla_reps):
			for i in qubits:
				if self.noise_model:
					self.add_qiskit_ry_gate(qc, i)
				else:
					self.add_gate(qc, ParametricRY(i, 0))
				if i > qubits[0]:
					self.add_gate(qc, CNOT(i - 1, i))

		# Last one-qubit layer
		for i in qubits:
			if self.noise_model:
				self.add_qiskit_ry_gate(qc, i)
			else:
				self.add_gate(qc, ParametricRY(i, 0))

		return qc

	def system_circuit(self, qc: ParametricQuantumCircuit = None, qubits: List[int] = None) -> ParametricQuantumCircuit:
		if qc is None:
			qc = ParametricQuantumCircuit(self.n)
		if qubits is None:
			qubits = range(qc.get_qubit_count())

		for _ in range(self.system_reps):
			for i in range(0, len(qubits) - 1, 2):
				j = (i + 1) % self.n
				if self.noise_model:
					self.add_qiskit_ising_gate(qc, qubits[i], qubits[j])
				else:
					self.add_gate(qc, ParametricPauliRotation([qubits[i], qubits[j]], [1, 2], 0))
					self.add_gate(qc, ParametricPauliRotation([qubits[i], qubits[j]], [2, 1], 0))
			if len(qubits) == 2:
				break
			for i in range(1, len(qubits), 2):
				j = (i + 1) % self.n
				if self.noise_model:
					self.add_qiskit_ising_gate(qc, qubits[i], qubits[j])
				else:
					self.add_gate(qc, ParametricPauliRotation([qubits[i], qubits[j]], [1, 2], 0))
					self.add_gate(qc, ParametricPauliRotation([qubits[i], qubits[j]], [2, 1], 0))

		return qc

	def get_fidelity(self, circuitFunction: Callable) -> Callable:  # TODO: Check correctness
		n = circuitFunction().get_qubit_count()
		reg1 = range(n)
		reg2 = range(n, 2 * n)
		circuit = ParametricQuantumCircuit(2 * n)
		circuitFunction(circuit, reg1)
		circuitFunction(circuit, reg2)
		# Destructive SWAP test
		for i in range(self.n):
			if self.noise_model:
				self.add_gate(circuit, CNOT(i, i + n))
				self.add_qiskit_h_gate(circuit, i)
			else:
				self.add_gate(circuit, CNOT(i, i + n))
				self.add_gate(circuit, H(i))
		# Define helper variables
		permutation = [i for j in zip(reg1, reg2) for i in j]
		operator = reduce(np.kron, [[1, 1, 1, -1]] * n)
		state = QuantumState(2 * n)

		# Fidelity function
		def fidelity(x: List[float], y: List[float]) -> float:
			self.nfev += 1
			# Set parameters
			self.set_parameters(circuit, x, reg1)
			self.set_parameters(circuit, y, reg2)
			# Update quantum state
			state.set_zero_state()
			circuit.update_quantum_state(state)
			# Permute qubits
			temp_state = permutate_qubit(state, permutation)
			# Calculate overlap
			if self.shots:
				overlap = np.sum(operator[i] for i in temp_state.sampling(self.shots)) / self.shots
			else:
				overlap = np.sum(i * np.abs(j) ** 2 for i, j in zip(operator, temp_state.get_vector()))

			return overlap

		return fidelity

	def ancilla_params(self) -> List[float]:
		return self.params[:self.num_ancilla_params]

	def system_params(self) -> List[float]:
		return self.params[self.num_ancilla_params:]

	def ancilla_unitary_matrix(self) -> List[List[float]]:
		unitary = np.zeros((self.dimension, self.dimension), dtype=np.float64)
		state = QuantumState(self.n)
		qc = self.ancilla_circuit()
		self.set_parameters(qc, self.ancilla_params())

		for i in range(self.dimension):
			state.set_computational_basis(i)
			qc.update_quantum_state(state)
			unitary[:, i] = state.get_vector().real

		return unitary

	def system_unitary_matrix(self) -> List[List[float]]:
		unitary = np.zeros((self.dimension, self.dimension), dtype=np.float64)
		state = QuantumState(self.n)
		qc = self.system_circuit()
		self.set_parameters(qc, self.system_params())

		eigvals = partial_trace(self.state, self.system_qubits).get_matrix().diagonal()
		for i in np.argsort(eigvals):
			state.set_computational_basis(i)
			qc.update_quantum_state(state)
			unitary[:, i] = state.get_vector()

		return unitary

	def generate_measurement_circuits(self) -> List[ParametricQuantumCircuit]:
		pauli_circuits = []
		for label, (_, terms) in self.commuting_terms.items():
			pauli_circ = self.ansatz.copy()
			if label != 'Z':
				for q in self.system_qubits:
					if self.noise_model:
						self.add_qiskit_x_basis_transform(pauli_circ, q)
					else:
						self.add_gate(pauli_circ, H(q))
			if self.noise_model:
				self.add_readout_noise(pauli_circ)
			pauli_circuits.append(pauli_circ)

		return pauli_circuits

	def statevector_tomography(self) -> (List[List[float]], List[List[float]]):
		self.state.set_zero_state()
		self.set_parameters(self.ansatz, self.params)
		self.ansatz.update_quantum_state(self.state)
		rho = partial_trace(self.state, self.ancilla_qubits).get_matrix()
		sigma = partial_trace(self.state, self.system_qubits).get_matrix().diagonal().real

		return rho, sigma

	def sampled_tomography(self) -> (List[List[float]], List[List[float]]):
		map_pauli = {
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
		for measurement in list(product(['X', 'Y', 'Z'], repeat=self.n)):
			circuit = self.ansatz.copy()
			for i, t in enumerate(measurement):
				if t == 'X':  # Hadamard gate
					if self.noise_model:
						self.add_qiskit_x_basis_transform(circuit, self.system_qubits[i])
					else:
						self.add_gate(circuit, H(self.system_qubits[i]))
				elif t == 'Y':
					if self.noise_model:
						self.add_qiskit_y_basis_transform(circuit, self.system_qubits[i])
					else:
						self.add_gate(circuit, Sdag(self.system_qubits[i]))
						self.add_gate(circuit, H(self.system_qubits[i]))
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
				# rho
				for pauli_string in pauli_helper(measurement):
					post_process_vector = reduce(np.kron, [map_pauli[i] for i in pauli_string])
					coef = post_process_vector[int(shot[self.n:], 2)]
					pauli_matrix = reduce(np.kron, [pauli[i] for i in pauli_string])
					rho += coef * pauli_matrix * n
				# sigma (assuming the state is diagonal)
				sigma[int(shot[:self.n], 2)] += n
		rho /= self.shots * self.dimension
		sigma /= self.shots * 3 ** self.n  # might as well use the 3^n measurements to compute with better accuracy

		return rho, sigma

	def statevector_cost_fun(self, x: List[float]) -> float:
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

		return self.cost

	@staticmethod
	def probabilities(p: Counter, shot: str, n: int) -> None:
		j = 0
		for b in shot:
			j = (j << 1) | int(b)
		p.update({j: n})

	@staticmethod
	def z_tensor_expectation(shot: str) -> int:
		return 1 if shot.count('1') % 2 == 0 else -1

	def sampled_cost_fun(self, x: List[float]) -> float:
		self.nfev += 1
		self.params = x
		energy = 0
		p = Counter()
		for pauli_circ, (label, (coef, terms)) in zip(self.pauli_circuits, self.commuting_terms.items()):
			# Update quantum state
			self.state.set_zero_state()
			self.set_parameters(pauli_circ, self.params)
			pauli_circ.update_quantum_state(self.state)
			# Iterate over pauli strings
			for qubits in terms:
				# Sample
				samples = Counter(self.state.sampling(self.shots, self.seed)).items()
				counts = [[self.get_bit_string(i, self.total_n), j] for i, j in samples]
				# Evaluate energy and entropy
				for shot, n in counts:
					# Energy
					energy += self.z_tensor_expectation([shot[q + self.n] for q in qubits]) * coef * n
					# Entropy
					self.probabilities(p, shot[:self.n], n)

		self.energy = energy.real / self.shots
		self.eigenvalues = np.array([i[1] for i in sorted(p.items())]) / self.total_shots
		self.entropy = self.entropy_fun(self.eigenvalues)
		self.cost = self.energy - self.inverse_beta * self.entropy

		return self.cost

	@staticmethod
	def all_z_expectation(shot: str, n: int) -> int:
		return 2 * shot.count('0') - n

	@staticmethod
	def xx_expectation(shot: str, q1: int, q2: int) -> int:
		return 1 if shot[q1] == shot[q2] else -1

	def commuting_sampled_cost_fun(self, x: List[float]) -> float:
		self.nfev += 1
		self.params = x
		energy = 0
		p = Counter()
		for pauli_circ, (label, (coef, qubits)) in zip(self.pauli_circuits, self.commuting_terms.items()):
			# Update quantum state
			self.state.set_zero_state()
			self.set_parameters(pauli_circ, self.params)
			pauli_circ.update_quantum_state(self.state)
			# Sample
			samples = Counter(self.state.sampling(self.shots, self.seed)).items()
			counts = [[self.get_bit_string(i, self.total_n), j] for i, j in samples]
			# Evaluate energy and entropy
			if label == 'Z':
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
		self.eigenvalues = np.array([i[1] for i in sorted(p.items())]) / self.total_shots
		self.entropy = self.entropy_fun(self.eigenvalues)
		self.cost = self.energy - self.inverse_beta * self.entropy

		return self.cost
