import inspect
from collections import Counter
from typing import List, Tuple, Optional

import numpy as np
from mthree import M3Mitigation
from mthree.classes import QuasiCollection, QuasiDistribution
from mthree.utils import final_measurement_mapping
from qiskit import QuantumCircuit, transpile, Aer
from qiskit.algorithms import optimizers
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector, partial_trace
from qiskit_aer.noise import NoiseModel
from qiskit_experiments.library import StateTomography
from qiskit_ibm_runtime.program import UserMessenger, ProgramBackend
from scipy.special import xlogy

_optimizers = dict(inspect.getmembers(optimizers, inspect.isclass))


class GibbsIsing:
	def __init__(
			self,
			backend: ProgramBackend,
			user_messenger: UserMessenger,
			n: int = 2,
			J: float = 1.,
			h: float = 0.5,
			beta: float = 1.,
			ancilla_reps: Optional[int] = None,
			system_reps: Optional[int] = None,
			optimizer: Optional[str] = None,
			min_kwargs: Optional[dict] = None,
			shots: int = 1024,
			skip_transpilation: bool = False,
			use_measurement_mitigation: bool = False,
			noise_model: Optional[NoiseModel] = None
	) -> None:
		"""
		Class constructor
		:param backend: ProgramBackend
		:param user_messenger: UserMessenger
		:param n: number of qubits
		:param J: XX coefficient
		:param h: Z coefficient
		:param beta: inverse temperature beta
		:param ancilla_reps: number of ancilla PQC repetitions (layers)
		:param system_reps: number of system PQC repetitions (layers)
		:param optimizer: Qiskit optimizer as a string
		:param min_kwargs: kwargs for the optimizer
		:param shots: number of shots for each circuit evaluation
		:param skip_transpilation: whether to skip circuit transpilation or not
		:param use_measurement_mitigation: whether to use measurement mitigation or not
		:param noise_model: optional noise model
		"""
		# Hamiltonian and cost function setup
		self.n = n
		self.J = J
		self.h = h
		self.commuting_terms = self.ising_hamiltonian_commuting_terms(self.n, self.J, self.h)
		self.total_n = 2 * self.n
		self.dimension = 2 ** self.n
		self.beta = beta
		self.inverse_beta = 1. / self.beta
		if ancilla_reps is not None:
			self.ancilla_reps = ancilla_reps
		else:
			self.ancilla_reps = 1
		if system_reps is not None:
			self.system_reps = system_reps
		else:
			self.system_reps = self.n - 1
		self.skip_transpilation = skip_transpilation
		self.use_measurement_mitigation = use_measurement_mitigation
		# Ansatz
		self.ancilla_qubits = range(n)
		self.system_qubits = range(n, 2 * n)
		self.theta = self.theta_iter()
		self.ansatz = self.var_ansatz(self.n)
		self.pauli_circuits = self.generate_ising_measurement_circuits()
		self.num_pauli_circuits = len(self.pauli_circuits)
		# Minimizer kwargs
		if not min_kwargs:
			if optimizer:
				self.min_kwargs = dict()
			else:
				self.min_kwargs = dict(maxiter=100 * self.n)
		else:
			self.min_kwargs = min_kwargs
		# Optimizer
		if not optimizer:
			optimizer = 'SPSA'
		self.optimizer = _optimizers.get(optimizer)(**self.min_kwargs, callback=self.callback)
		# Bounds
		self.bounds = [(-np.pi, np.pi)] * len(self.ansatz.parameters)
		# Shots
		self.shots = shots
		self.total_shots = self.shots * self.num_pauli_circuits
		# Setup backend
		self.backend = backend
		# Noise model
		if noise_model:
			# noinspection PyDeprecation
			self.noise_model = NoiseModel.from_dict(noise_model)
			self.backend.set_options(noise_model=self.noise_model)
		else:
			self.noise_model = None
		# User messenger
		self.user_messenger = user_messenger
		# Transpilation
		self.transpilation_options = dict(optimization_level=3, layout_method='sabre', routing_method='sabre')
		if not self.skip_transpilation:
			self.pauli_circuits = transpile(self.pauli_circuits, self.backend, **self.transpilation_options)
		# Error mitigation
		if self.use_measurement_mitigation:
			self.mappings = final_measurement_mapping(self.pauli_circuits)
			self.mit = M3Mitigation(self.backend)
			self.mit.cals_from_system(self.mappings, shots=self.shots)
		# Logging
		self.iter = None
		self.nfev = None
		self.cost = None
		self.energy = None
		self.entropy = None
		self.params = None
		self.eigenvalues = None
		# Other
		self.options = None
		self.service = None
		self.sampler = None
		self.x0 = None
		self.rho = None
		self.sigma = None
		self.noiseless_rho = None
		self.noiseless_sigma = None
		self.noiseless_eigenvalues = None
		self.hamiltonian_eigenvalues = None
		self.noiseless_hamiltonian_eigenvalues = None

		self.user_messenger.publish(f"Initialized GibbsIsing object with n={self.n}, J={self.J}, h={self.h}, "
		                            f"beta={self.beta}")

	def run(self, x0: List[float] = None) -> dict:
		"""
		Main entry point of the class, to run the VQA
		:param x0: list of initial parameters
		:return: dictionary of results
		"""
		self.iter = 0
		self.nfev = 0
		if x0 is None:
			self.x0 = np.random.uniform(-np.pi, np.pi, self.ansatz.num_parameters)
		else:
			self.x0 = x0
		# Start optimization
		self.user_messenger.publish("Starting optimization")
		result = self.optimizer.minimize(fun=self.cost_fun, x0=self.x0, bounds=self.bounds)
		self.user_messenger.publish("Finished optimization")
		# Compute cost function at the last parameters
		self.params = result.x
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
		self.user_messenger.publish("Post-processed results")
		# Compile data
		data = dict(
			final=True,
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
			skip_transpilation=self.skip_transpilation,
			use_measurement_mitigation=self.use_measurement_mitigation,
			ansatz=self.ansatz,
			optimizer=self.optimizer.__class__.__name__,
			min_kwargs=self.min_kwargs,
			shots=self.shots,
			backend=self.backend.name
		)
		# Publish data
		self.user_messenger.publish("Publishing results")
		self.user_messenger.publish(data)
		# return data
		self.user_messenger.publish("Returning results")
		return data

	def cost_fun(self, x: List[float]) -> float:
		"""
		Free energy cost function
		:param x: list of parameters
		:return: cost in terms of the free energy
		"""
		self.nfev += 1
		self.params = x
		# Bind parameters to circuits
		bound_circuits = [circuit.bind_parameters(self.params) for circuit in self.pauli_circuits]
		# Submit the job and get the result counts back
		results = self.backend.run(bound_circuits, shots=self.shots).result().get_counts()
		# Apply error mitigation (get QuasiCollection from M3)
		if self.use_measurement_mitigation:
			results = self.mit.apply_correction(results, self.mappings)
		else:
			results = QuasiCollection([QuasiDistribution([(key, value / self.shots) for key, value in result.items()])
			                           for result in results])
		# Post-process results
		energy = 0
		p = Counter()
		for counts, (label, (coef, qubits)) in zip(results, self.commuting_terms.items()):
			# Evaluate energy and entropy
			for shot, n in counts.items():
				# Note that Qiskit returns in little-endian, and we read big-endian,
				# so the bit string needs to be reversed
				shot = shot[::-1]
				# Energy
				if label == 'Z':
					energy += self.all_z_expectation(shot[self.n:], self.n) * coef * n
				else:
					for q1, q2 in qubits:
						energy += self.xx_expectation(shot[self.n:], q1, q2) * coef * n
				# Entropy
				self.probabilities(p, shot[:self.n], n)
		# If we get a quasi distribution, get back the nearest probability distribution
		p = QuasiDistribution(p, shots=self.total_shots).nearest_probability_distribution()
		# Compute cost
		self.energy = energy
		self.eigenvalues = [i[1] for i in sorted(p.items())]  # Sort eigenvalues according to bit string
		self.entropy = self.entropy_fun(self.eigenvalues)
		self.cost = self.energy - self.inverse_beta * self.entropy

		return self.cost

	@staticmethod
	def all_z_expectation(shot: str, n: int) -> int:
		"""
		Expectation value of measuring the all-Z operator
		:param shot: shot bit string
		:param n: number of qubits
		:return: expectation value
		"""
		return 2 * shot.count('0') - n

	@staticmethod
	def xx_expectation(shot: str, q1: int, q2: int) -> int:
		"""
		Expectation of measuring XX
		:param shot: shot bit string
		:param q1: qubit 1
		:param q2: qubit 2
		:return: expectation value, 1 or -1
		"""
		return 1 if shot[q1] == shot[q2] else -1

	# Update list of probabilities given a count of bit strings
	def probabilities(self, p: Counter, shot: str, n: int) -> None:
		j = 0
		for b in shot:
			j = (j << 1) | int(b)
		p.update({j: n / self.num_pauli_circuits})

	@staticmethod
	def entropy_fun(p: Counter) -> float:
		"""
		Compute the entropy given a list of probabilities
		:param p:
		:return: entropy
		"""
		# noinspection PyCallingNonCallable
		return -np.sum([xlogy(i, i) for i in p])

	@staticmethod
	def theta_iter():
		"""
		Iterator for the PQC
		"""
		n = 0
		while True:
			yield Parameter(fr'$\theta_{n}$')
			n += 1

	def generate_ising_measurement_circuits(self) -> List[QuantumCircuit]:
		"""
		Takes commuting terms and produces the circuits required to measure expectation values
		:return: list of Pauli circuits with appended measurements
		"""
		pauli_circuits = []
		for label, (_, terms) in self.commuting_terms.items():
			pauli_circ = self.ansatz.copy()
			if label != 'Z':
				for q in self.system_qubits:
					pauli_circ.h(q)
			pauli_circ.measure_all()
			pauli_circuits.append(pauli_circ)

		return pauli_circuits

	@staticmethod
	def ising_hamiltonian_commuting_terms(n: int, J: float = 1., h: float = 0.5) -> dict:
		"""
		Generate dictionary of ising hamiltonian commuting terms
		:param n: number of qubits
		:param J: XX coefficient
		:param h: Z coefficient
		:return: dictionary of commuting terms
		"""
		terms = dict()
		if J != 0:
			if n == 2:
				terms.update(XX=[-J, [[0, 1]]])
			elif n > 2:
				terms.update(XX=[-J, [[i, (i + 1) % n] for i in range(n)]])
		if h != 0:
			terms.update(Z=[-h, [[i] for i in range(n)]])

		return terms

	def var_ansatz(self, n: int) -> QuantumCircuit:
		"""
        Generate the variational ansatz
		:param n: number of qubits
		:return: returns the PQC
		"""
		qc = QuantumCircuit(2 * n)
		UA = self.ancilla_unitary(n)
		US = self.system_unitary(n)
		qc.append(UA.to_instruction(), range(n))
		for i in range(n):
			qc.cx(i, i + n)
		qc.append(US.to_instruction(), range(n, 2 * n))

		return qc

	def ancilla_unitary(self, n: int) -> QuantumCircuit:
		"""
		Ancilla ansatz
		:param n: number of qubits
		:return: ancilla PQC
		"""
		qc = QuantumCircuit(n)
		for _ in range(self.ancilla_reps):
			for i in range(n):
				qc.ry(next(self.theta), i)
				if i > 0:
					qc.cx(i - 1, i)

		# Last one-qubit layer
		for i in range(n):
			qc.ry(next(self.theta), i)

		return qc

	def system_unitary(self, n: int) -> QuantumCircuit:
		"""
		System ansatz
		:param n: number of qubits
		:return: system PQC
		"""
		qc = QuantumCircuit(n)

		for _ in range(self.system_reps):
			for i in range(0, n - 1, 2):
				self.add_ising_gate(qc, i, i + 1)
			if n > 2:
				for i in range(1, n, 2):
					self.add_ising_gate(qc, i, (i + 1) % n)

		return qc

	def add_ising_gate(self, qc: QuantumCircuit, q1: int, q2: int) -> None:
		"""
		Add transpiled RP = R_yx.R_xy gate to a circuit
		:param qc: quantum circuit
		:param q1: qubit 1
		:param q2: qubit 2
		"""
		qc.h([q1, q2])
		qc.cx(q1, q2)
		qc.ry(next(self.theta), q2)
		qc.cx(q1, q2)
		qc.cx(q2, q1)
		qc.ry(next(self.theta), q1)
		qc.cx(q2, q1)
		qc.h([q1, q2])

	# noinspection PyUnusedLocal
	def callback(self, *args, **kwargs) -> None:
		"""
		Callback function to save intermediary results
		:param args: not used
		:param kwargs: not used
		"""
		if self.iter == 0:
			print('| iter | nfev | Cost | Energy | Entropy |')
		self.iter += 1
		print(f"| {self.iter} | {self.nfev} | {self.cost:.8f} | {self.energy:.8f} | {self.entropy:.8f} |")
		message = dict(
			final=False,
			iter=self.iter,
			nfev=self.nfev,
			cost=self.cost,
			energy=self.energy,
			entropy=self.entropy,
			params=self.params,
			eigenvalues=self.eigenvalues
		)
		self.user_messenger.publish(message)

	def statevector_tomography(self) -> Tuple[List[List[complex]], List[float]]:
		"""
		Perform statevector tomography
		:return: density matrix rho of system qubits and diagonal matrix sigma of ancilla qubits
		"""
		circuit = self.ansatz.bind_parameters(self.params)
		statevector = Statevector(circuit)
		rho = partial_trace(statevector, self.ancilla_qubits).data
		sigma = partial_trace(statevector, self.system_qubits).data.diagonal().real

		return rho, sigma

	def sampled_tomography(self) -> Tuple[List[List[complex]], List[float]]:
		"""
		Perform sampled tomography
		:return: density matrix rho of system qubits and diagonal matrix sigma of ancilla qubits
		"""
		# State tomography for rho
		rho_qst = StateTomography(self.ansatz.bind_parameters(self.params), measurement_qubits=self.system_qubits)
		rho_qst.set_transpile_options(**self.transpilation_options)
		rho_data = rho_qst.run(self.backend, shots=self.shots).block_for_results()
		rho = rho_data.analysis_results('state').value.data
		# State tomography for sigma (assuming it is diagonal)
		sigma_qst = StateTomography(self.ansatz.bind_parameters(self.params), measurement_qubits=self.ancilla_qubits)
		sigma_qst.set_transpile_options(**self.transpilation_options)
		sigma_data = sigma_qst.run(self.backend, shots=self.shots).block_for_results()
		sigma = sigma_data.analysis_results('state').value.data.diagonal().real

		return rho, sigma


# Don't annotate!!!
def main(
		backend=Aer.get_backend('aer_simulator'),
		user_messenger=UserMessenger(),
		n=2,
		J=1.,
		h=0.5,
		beta=1.,
		N=1,
		ancilla_reps=None,
		system_reps=None,
		x0=None,
		optimizer=None,
		min_kwargs=None,
		shots=1024,
		skip_transpilation=False,
		use_measurement_mitigation=False,
		noise_model=None
):
	"""
	Entrypoint for our qiskit_runtime Program.
	:param backend: ProgramBackend.
	:param user_messenger: UserMessenger.
	:param n: number of qubits.
	:param J: XX coefficient.
	:param h: Z coefficient.
	:param beta: inverse temperature beta (can be list), default is 1. Warning: do not set to 0, set at least 1e-10.
	:param N: repeated number of runs for each beta.
	:param ancilla_reps: number of ancilla PQC repetitions (layers), default is 1.
	:param system_reps: number of system PQC repetitions (layers), default is n - 1.
	:param x0: list of initial parameters, default is uniform random.
	:param optimizer: Qiskit optimizer as a string, default is SPSA.
	:param min_kwargs: kwargs for the optimizer.
	:param shots: number of shots for each circuit evaluation.
	:param skip_transpilation: whether to skip circuit transpilation or not.
	:param use_measurement_mitigation: whether to use measurement mitigation or not.
	:param noise_model: optional noise model.
	:return: list of dictionaries containing results.
	"""
	user_messenger.publish("Starting program")
	if not isinstance(beta, list):
		beta = [beta]
	multiple_results = []
	for b in beta:
		results = []
		for i in range(N):
			user_messenger.publish(f"Beta: {b}, run: {i}")
			gibbs = GibbsIsing(backend, user_messenger, n, J, h, b, ancilla_reps, system_reps, optimizer, min_kwargs,
			                   shots, skip_transpilation, use_measurement_mitigation, noise_model)
			result = gibbs.run(x0)

			results.append(result)
		# Add to our results
		multiple_results.append(results)

	user_messenger.publish("Returning program results")
	return multiple_results
