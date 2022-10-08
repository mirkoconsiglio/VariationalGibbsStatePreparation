import inspect
from collections import Counter

import numpy as np
from mthree import M3Mitigation
from mthree.classes import QuasiCollection, QuasiDistribution
from mthree.utils import final_measurement_mapping
from qiskit import QuantumCircuit, transpile
from qiskit.algorithms import optimizers
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector, partial_trace
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_experiments.library import StateTomography
from qiskit_ibm_runtime.program import UserMessenger
from scipy.special import xlogy

_optimizers = dict(inspect.getmembers(optimizers, inspect.isclass))


class GibbsIsing:
	def __init__(
			self,
			n=2,
			J=1.,
			h=0.5,
			beta=1.,
			ancilla_reps=None,
			system_reps=None,
			optimizer='SPSA',
			min_kwargs=None,
			shots=1024,
			backend=None,
			user_messenger=None,
			skip_transpilation=False,
			use_measurement_mitigation=False,
			noise_model=None
	):
		# Hamiltonian and cost function setup
		self.n = n
		self.J = J
		self.h = h
		self.commuting_terms = self.ising_hamiltonian_commuting_terms()
		self.total_n = 2 * self.n
		self.dimension = 2 ** self.n
		self.beta = beta
		self.inverse_beta = 1. / self.beta
		self.ancilla_reps = ancilla_reps if ancilla_reps else self.n - 1
		self.system_reps = system_reps if system_reps else self.n - 1
		self.skip_transpilation = skip_transpilation
		self.use_measurement_mitigation = use_measurement_mitigation
		# Ansatz
		self.ancilla_qubits = range(n)
		self.system_qubits = range(n, 2 * n)
		self.theta = self._theta()
		self.ansatz = self.var_ansatz(self.n)
		self.pauli_circuits = self.generate_measurement_circuits()
		self.num_pauli_circuits = len(self.pauli_circuits)
		# Minimizer kwargs
		if not min_kwargs:
			self.min_kwargs = dict()
		else:
			self.min_kwargs = min_kwargs
		# Optimizer
		self.optimizer = _optimizers[optimizer](**self.min_kwargs, callback=self.callback)
		# Bounds
		self.bounds = [(0, 2 * np.pi)] * len(self.ansatz.parameters)
		# Shots
		self.shots = shots
		self.total_shots = self.shots * self.num_pauli_circuits
		# Setup backend
		if not backend:
			self.backend = AerSimulator()
		else:
			self.backend = backend
		# Noise model
		if noise_model:
			# noinspection PyDeprecation
			self.noise_model = NoiseModel.from_dict(noise_model)
			self.backend.set_options(noise_model=self.noise_model)
		else:
			self.noise_model = None
		# User messenger
		if not user_messenger:
			self.user_messenger = UserMessenger()
		else:
			self.user_messenger = user_messenger
		# Transpilation
		self.transpilation_options = dict(optimization_level=3, layout_method='sabre', routing_method='sabre')
		# self.pauli_circuits[-1].decompose().draw('mpl', filename=f'figures/qiskit_circuit_{self.n}')
		if not self.skip_transpilation:
			self.pauli_circuits = transpile(self.pauli_circuits, self.backend, **self.transpilation_options)
		# self.pauli_circuits[-1].draw('mpl', filename=f'figures/transpiled_qiskit_circuit_{self.n}')
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

	def run(self, x0=None):
		self.iter = 0
		self.nfev = 0
		if x0 is None:
			self.x0 = np.random.uniform(0, 2 * np.pi, self.ansatz.num_parameters)
		else:
			self.x0 = x0
		# Start optimization
		print("| iter | nfev | Cost | Energy | Entropy |")
		result = self.optimizer.minimize(fun=self.cost_fun, x0=self.x0, bounds=self.bounds)
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
		# Compile data
		data = dict(
			n=self.n,
			J=self.J,
			h=self.h,
			beta=self.beta,
			ancilla_reps=self.ancilla_reps,
			system_reps=self.system_reps,
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
		self.user_messenger.publish(data)
		# return data
		return data

	def cost_fun(self, x):
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
				# so the shot string needs to be reversed
				shot = shot[::-1]
				# Energy
				if label == 'z':
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
	def all_z_expectation(shot, n):
		return 2 * shot.count('0') - n

	@staticmethod
	def xx_expectation(shot, q1, q2):
		return 1 if shot[q1] == shot[q2] else -1

	def probabilities(self, p, shot, n):
		j = 0
		for b in shot:
			j = (j << 1) | int(b)
		p.update({j: n / self.num_pauli_circuits})

	@staticmethod
	def entropy_fun(p):
		# noinspection PyCallingNonCallable
		return -np.sum([xlogy(i, i) for i in p])

	@staticmethod
	def _theta():
		n = 1
		while True:
			yield Parameter(f'θ{n}')
			n += 1

	def generate_measurement_circuits(self):
		pauli_circuits = []
		for label, (_, terms) in self.commuting_terms.items():
			pauli_circ = self.ansatz.copy()
			if label != 'z':
				for qubits in terms:
					for q in qubits:  # Hadamard gate
						pauli_circ.rz(np.pi / 2, q + self.n)
						pauli_circ.sx(q + self.n)
			pauli_circ.measure_all()
			pauli_circuits.append(pauli_circ)

		return pauli_circuits

	def ising_hamiltonian_commuting_terms(self):
		terms = dict()
		if self.J != 0:
			terms.update(ising_even_odd=[-self.J, [[i, i + 1] for i in range(0, self.n - 1, 2)]])
			if self.n > 2:
				if self.n % 2 == 0:
					terms.update(
						ising_odd_even=[-self.J, [[i, i + 1] for i in range(1, self.n - 2, 2)] + [[0, self.n - 1]]])
				else:
					terms.update(ising_odd_even=[-self.J, [[i, i + 1] for i in range(1, self.n - 2, 2)]])
					terms.update(ising_closed=[-self.J, [[0, self.n - 1]]])
		if self.h != 0:
			terms.update(z=[-self.h, list(range(self.n))])

		return terms

	def var_ansatz(self, n):
		qc = QuantumCircuit(2 * n)
		UA = self.ancilla_unitary(n)
		US = self.system_unitary(n)
		qc.append(UA.to_instruction(), range(n))
		for i in range(n):
			qc.cx(i, i + n)
		qc.append(US.to_instruction(), range(n, 2 * n))

		return qc

	def ancilla_unitary(self, n):  # Ancilla ansatz
		qc = QuantumCircuit(n)
		for _ in range(self.ancilla_reps):
			for i in range(n):
				self.add_ry_gate(qc, i)
				if i > 0:
					qc.cx(i - 1, i)

		# Last one-qubit layer
		for i in range(n):
			self.add_ry_gate(qc, i)

		return qc

	def add_ry_gate(self, qc, q):
		qc.sx(q)
		qc.rz(next(self.theta), q)
		qc.sx(q)
		qc.rz(np.pi, q)

	def system_unitary(self, n):  # System ansatz
		qc = QuantumCircuit(n)
		for _ in range(self.system_reps):
			for i in range(0, n - 1, 2):
				self.add_ising_gate(qc, i, i + 1)
			for i in range(1, n - 1, 2):
				self.add_ising_gate(qc, i, i + 1)

		return qc

	def add_ising_gate(self, qc, q1, q2):  # U = R_yx.R_xy
		qc.sx(q1)
		qc.sx(q2)
		qc.rz(3 * np.pi / 2, q1)
		qc.cx(q1, q2)
		qc.rz(next(self.theta), q2)
		qc.sx(q1)
		qc.sx(q2)
		qc.rz(next(self.theta), q1)
		qc.rz(np.pi, q2)
		qc.sx(q1)
		qc.rz(np.pi / 2, q1)
		qc.cx(q1, q2)
		qc.sx(q1)
		qc.rz(np.pi / 2, q2)

	# noinspection PyUnusedLocal
	def callback(self, *args, **kwargs):
		self.iter += 1
		print(f"| {self.iter} | {self.nfev} | {self.cost:.8f} | {self.energy:.8f} | {self.entropy:.8f} |")
		message = dict(
			iter=self.iter,
			nfev=self.nfev,
			cost=self.cost,
			energy=self.energy,
			entropy=self.entropy,
			params=self.params,
			eigenvalues=self.eigenvalues
		)
		self.user_messenger.publish(message)

	def statevector_tomography(self):
		circuit = self.ansatz.bind_parameters(self.params)
		statevector = Statevector(circuit)
		rho = partial_trace(statevector, self.ancilla_qubits).data.real
		sigma = partial_trace(statevector, self.system_qubits).data.diagonal().real

		return rho, sigma

	def sampled_tomography(self):
		# State tomography for rho
		rho_qst = StateTomography(self.ansatz.bind_parameters(self.params), measurement_qubits=self.system_qubits)
		rho_qst.set_transpile_options(**self.transpilation_options)
		# rho_qst.analysis.set_options(fitter='cvxpy_gaussian_lstsq')
		rho_data = rho_qst.run(self.backend, shots=8192).block_for_results()
		rho = rho_data.analysis_results('state').value.data
		# State tomography for sigma (assuming it is diagonal)
		sigma_qst = StateTomography(self.ansatz.bind_parameters(self.params), measurement_qubits=self.ancilla_qubits)
		sigma_qst.set_transpile_options(**self.transpilation_options)
		# sigma_qst.analysis.set_options(fitter='cvxpy_gaussian_lstsq')
		sigma_data = sigma_qst.run(self.backend, shots=8192).block_for_results()
		sigma = sigma_data.analysis_results('state').value.data.diagonal().real

		return rho, sigma


# The entrypoint for our qiskit_runtime Program
def main(
		backend=None,
		user_messenger=None,
		n=2,
		J=1.,
		h=0.5,
		beta=None,
		ancilla_reps=1,
		system_reps=1,
		x0=None,
		optimizer='SPSA',
		min_kwargs=None,
		shots=1024,
		use_measurement_mitigation=True,
		skip_transpilation=False,
		adiabatic_assistance=False,
		noise_model=None
):
	if beta is None:
		beta = [1e-8, 0.2, 0.5, 0.8, 1., 1.2, 2., 5.]
	elif not isinstance(beta, list):
		beta = [beta]
	results = []
	for _beta in beta:
		gibbs = GibbsIsing(n, J, h, _beta, ancilla_reps, system_reps, optimizer, min_kwargs, shots, backend,
		                   user_messenger, skip_transpilation, use_measurement_mitigation, noise_model)
		result = gibbs.run(x0)
		# Add to our results
		results.append(result)
		# Start optimization from parameters of previous beta
		if adiabatic_assistance:
			x0 = result['params']

	return results
