from functools import reduce
from itertools import product

import numpy as np
from braket.circuits import Circuit, FreeParameter
from braket.devices import LocalSimulator
from braket.jobs.metrics import log_metric
from qiskit.algorithms.optimizers import SPSA
from qiskit.utils import algorithm_globals
from scipy.special import xlogy

from gibbs_functions import ising_hamiltonian, ising_hamiltonian_commuting_terms


class GibbsIsing:
	def __init__(self, n, J, h, beta, backend=LocalSimulator(), noise_model=None):
		self.n = n
		self.J = J
		self.h = h
		self.beta = beta
		self.seed = None
		self.hamiltonian = ising_hamiltonian(self.n, self.J, self.h)
		self.inverse_beta = 1 / self.beta
		self.total_n = 2 * self.n
		self.dimension = 2 ** self.n
		self.ancilla_qubits = list(range(self.n))
		self.system_qubits = list(range(self.n, self.total_n))
		self.ansatz = None
		self.theta_n = None
		self.num_params = None
		self.num_ancilla_params = None
		self.num_system_params = None
		self.backend = backend
		self.min_kwargs: dict = None
		self.x0 = None
		self.bounds = None
		self.ancilla_reps = None
		self.system_reps = None
		self.params = None
		self.params_keys = None
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
		self.commuting_terms = None
		self.pauli_circuits = None
		self.total_shots = None
		self.cost_fun = None
		self.hamiltonian_eigenvalues = None
		self.noise_model = noise_model

	def run(self, min_kwargs=None, x0=None, shots=1024, ancilla_reps=None, system_reps=None, seed=None):
		self.seed = seed
		np.random.seed(self.seed)
		algorithm_globals.random_seed = self.seed
		self.ancilla_reps = ancilla_reps or self.n - 1
		self.system_reps = system_reps or 2 * (self.n - 1)
		self.num_ancilla_params = self.n * (self.ancilla_reps + 1)
		self.num_system_params = self.n * (self.system_reps + 1)
		self.num_params = self.num_ancilla_params + self.num_system_params
		self.params_keys = []
		self.init_var_ansatz()
		self.x0 = x0 if x0 else np.random.uniform(0, 2 * np.pi, self.num_params)
		self.bounds = [(0, 2 * np.pi)] * self.num_params
		# Set up minimizer kwargs
		self.min_kwargs = min_kwargs if min_kwargs else dict()
		self.min_kwargs.update(callback=self.callback)
		self.shots = shots
		self.cost_fun = self.sampled_cost_fun
		self.commuting_terms = ising_hamiltonian_commuting_terms(self.n, self.J, self.h)
		self.pauli_circuits = self.generate_measurement_circuits()
		self.total_shots = self.shots * len(self.commuting_terms)

		self.iter = 0
		self.nfev = 0
		print("| iter | nfev | Cost | Energy | Entropy |")
		self.result = SPSA(**self.min_kwargs).minimize(
			fun=self.cost_fun, x0=self.x0, bounds=self.bounds
		)
		# Update
		self.params = self.result.x
		self.cost = self.result.fun
		self.rho, self.sigma = self.tomography()
		self.eigenvalues = np.sort(self.sigma.diagonal().real)
		self.hamiltonian_eigenvalues = np.sort(self.cost - self.inverse_beta * np.log(self.eigenvalues))

		return GibbsResult(self)

	def params_dict(self):
		return dict(zip(self.params_keys, self.params))

	def ancilla_params_dict(self):
		return dict(
			zip(self.params_keys[: self.num_ancilla_params], self.ancilla_params())
		)

	def system_params_dict(self):
		return dict(
			zip(self.params_keys[self.num_ancilla_params:], self.system_params())
		)

	def theta(self):
		string = f"θ{self.theta_n}"
		self.params_keys.append(string)
		self.theta_n += 1
		return string

	@staticmethod
	def entropy_fun(p):
		return -np.sum([xlogy(i, i) for i in p])

	def callback(self, *args, **kwargs):
		self.iter += 1
		print(f"| {self.iter} | {self.nfev} | {self.cost:.8f} | {self.energy:.8f} | {self.entropy:.8f} |")
		log_metric(metric_name='cost', value=self.cost, iteration_number=self.iter)
		log_metric(metric_name='nfev', value=self.nfev, iteration_number=self.iter)
		log_metric(metric_name='energy', value=self.energy, iteration_number=self.iter)
		log_metric(metric_name='entropy', value=self.entropy, iteration_number=self.iter)

	def init_var_ansatz(self):
		self.ansatz = Circuit()

		# Ancilla ansatz
		self.ancilla_circuit(self.ansatz, self.ancilla_qubits)

		# Connecting CNOTs
		for i in range(self.n):
			self.ansatz.cnot(i, i + self.n)

		# System ansatz
		self.system_circuit(self.ansatz, self.system_qubits)

		return self.ansatz

	def ancilla_circuit(self, qc=None, qubits=None):
		if qc is None:
			qc = Circuit()
		if qubits is None:
			qubits = range(self.n)

		self.theta_n = 0

		# Layers
		for _ in range(self.ancilla_reps):
			for i in qubits:
				qc.ry(i, FreeParameter(self.theta()))
				if i > qubits[0]:
					qc.cnot(i - 1, i)

		# Last one-qubit layer
		for i in qubits:
			qc.ry(i, FreeParameter(self.theta()))

		return qc

	def system_circuit(self, qc=None, qubits=None):
		if qc is None:
			qc = Circuit()
		if qubits is None:
			qubits = range(self.n)

		self.theta_n = self.num_ancilla_params

		# Layers
		for _ in range(self.system_reps):
			for i in qubits:
				qc.ry(i, FreeParameter(self.theta()))
				if i > qubits[0]:
					qc.cnot(i - 1, i)

		# Last one-qubit layer
		for i in qubits:
			qc.ry(i, FreeParameter(self.theta()))

		return qc

	def ancilla_params(self):
		return self.params[:self.num_ancilla_params]

	def system_params(self):
		return self.params[self.num_ancilla_params:]

	def ancilla_unitary_matrix(self):
		self.ancilla_circuit().make_bound_circuit(self.ancilla_params_dict()).to_unitary()

	def system_unitary_matrix(self):
		return self.system_circuit().make_bound_circuit(self.system_params_dict()).to_unitary()

	def generate_measurement_circuits(self):
		pauli_circuits = []
		for label, (_, terms) in self.commuting_terms.items():
			pauli_circ = self.ansatz.copy()
			if label != 'z':
				for qubits in terms:
					for q in qubits:
						pauli_circ.h(q)
			if self.noise_model:
				pauli_circ = self.noise_model.apply(pauli_circ)
			pauli_circuits.append(pauli_circ)

		return pauli_circuits

	def tomography(self):
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
		tomography_circuits = []
		pauli_strings = list(product(['X', 'Y', 'Z'], repeat=self.n))
		for m in pauli_strings:
			# Construct Pauli circuits
			circuit = self.ansatz.copy()
			for i, t in enumerate(m):
				if t == 'X':
					circuit.h(self.system_qubits[i])
				elif t == 'Y':
					circuit.si(self.system_qubits[i])
					circuit.h(self.system_qubits[i])
				if self.noise_model:
					circuit = self.noise_model.apply(circuit)
			tomography_circuits.append(circuit.make_bound_circuit(self.params_dict()))

		if isinstance(self.backend, LocalSimulator):
			results = [
				self.backend.run(circ, shots=self.shots).result()
				for circ in tomography_circuits
			]
		else:  # If not LocalSimulator, we use run_batch
			results = self.backend.run_batch(tomography_circuits, shots=self.shots).results()

		for result, m in zip(results, pauli_strings):
			# Compute rho and sigma
			for shot, n in result.measurement_counts.items():
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

		return rho, np.diag(sigma)

	def sampled_cost_fun(self, x):
		self.nfev += 1
		self.params = x

		def all_z_expectation(shot):
			return 2 * shot[self.n:].count('0') - self.n

		def xx_expectation(shot, q1, q2):
			return 1 if shot[q1] == shot[q2] else -1

		def entropy(p, shot):
			j = 0
			for b in shot[: self.n]:
				j = (j << 1) | int(b)
			p[j] += n

		energy = 0
		p = np.zeros(self.dimension)

		# We run the two Pauli circuits in one batch, to optimize the number of calls to the backend
		bound_circs = [
			pauli_circ.make_bound_circuit(self.params_dict())
			for pauli_circ in self.pauli_circuits
		]

		if isinstance(self.backend, LocalSimulator):
			results = [
				self.backend.run(circ, shots=self.shots).result()
				for circ in bound_circs
			]
		else:  # If not LocalSimulator, we use run_batch
			results = self.backend.run_batch(bound_circs, shots=self.shots).results()

		for result, (label, (coef, qubits)) in zip(results, self.commuting_terms.items()):
			counts = result.measurement_counts.items()
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
		self.hamiltonian_eigenvalues = gibbs.hamiltonian_eigenvalues
