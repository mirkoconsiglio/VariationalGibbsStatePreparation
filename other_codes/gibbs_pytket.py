import numpy as np
from pytket import Circuit, Qubit
from pytket.extensions.qulacs import QulacsBackend
from qiskit.quantum_info import partial_trace
from sympy import Symbol

class Gibbs:
	def __init__(self, n, optimizer):
		self.nqubits = n
		self.optimizer = optimizer

		self.dimension = 2 ** self.nqubits
		self.ancilla_qubits = list(range(self.nqubits, 2 * self.nqubits))
		self.system_qubits = list(range(self.nqubits))
		self.ansatz = None
		self.num_params = None
		self.expec = [Qubit(i) for i in range(2 * self.nqubits - 1, -1, -1)]

		self.hamiltonian = None
		self.inverse_beta = None
		self.x0 = None
		self.bounds = None
		self.reps = None
		self.params = None
		self.eigvals = None
		self.energy = None
		self.entropy = None
		self.cost = None
		self.iter = None
		self.statevector = None
		self.result = None
		self.backend = None
		self.quantum_instance = None
		self.shots = None
		self.circuit = None
		self.circuit_state_fn = None

	def run(self, hamiltonian, beta, reps=None, x0=None, bounds=None, backend=None, quantum_instance=None, shots=None):
		self.hamiltonian = hamiltonian
		self.inverse_beta = 1 / beta
		self.reps = reps if reps else 2 * self.nqubits
		self.ansatz = self.var_ansatz()
		self.num_params = len(self.ansatz.free_symbols())
		self.x0 = x0 if x0 else np.random.uniform(-2 * np.pi, 2 * np.pi, self.num_params)
		self.bounds = bounds if bounds else [(-2 * np.pi, 2 * np.pi) for _ in range(self.num_params)]
		self.backend = backend if backend else QulacsBackend()
		self.shots = shots

		self.iter = 0
		print('Iter    Cost       Energy     Entropy')
		self.result = self.optimizer.minimize(fun=self.cost_fun, x0=self.x0, bounds=self.bounds)
		return GibbsResult(self)

	def callback(self):
		if self.iter % 100 == 0:
			print(f'{self.iter}', f'{self.cost:.8f}', f'{self.energy:.8f}', f'{self.entropy:.8f}')

	def cost_fun(self, x):
		self.iter += 1
		self.params = x
		self.circuit = self.ansatz.copy()
		self.circuit.symbol_substitution({sym: val for sym, val in zip(self.ansatz.free_symbols(), self.params)})
		self.statevector = self.backend.run_circuit(self.circuit).get_state()
		self.energy = self.energy_fun()
		self.entropy = self.entropy_fun()
		self.cost = self.energy - self.inverse_beta * self.entropy
		self.callback()
		return self.cost

	def energy_fun(self): # TODO: Sampled expectation
		return self.hamiltonian.state_expectation(self.statevector, self.expec).real

	def entropy_fun(self): # TODO: Sampled entropy
		if self.shots:
			counts = self.statevector.sample_counts(self.shots, self.ancilla_qubits)
			self.eigvals = [i / self.shots for i in counts.values()]
		else:
			self.eigvals = np.diagonal(partial_trace(self.statevector, self.system_qubits).data).real
		return -np.sum([i * np.log(i) for i in self.eigvals])

	def var_ansatz(self):
		qc = Circuit(2 * self.nqubits)
		def theta():
			symbol = Symbol(f'$\\theta_{{{theta.counter}}}$')
			theta.counter += 1
			return symbol
		theta.counter = 0

		# Unitary ancillas

		# Layers
		for r in range(self.reps):
			for i in range(self.nqubits):
				qc.Ry(theta(), i)
				if i > 0:
					qc.CX(i - 1, i)

		# Last one-qubit layer and connecting CX
		for i in range(self.nqubits):
			qc.Ry(theta(), i)
			qc.CX(i, i + self.nqubits)

		# Unitary system

		# Layers
		for r in range(self.reps):
			for i in range(self.nqubits, 2 * self.nqubits):
				qc.Ry(theta(), i)
				if i > self.nqubits:
					qc.CX(i - 1, i)

		# Last one-qubit layer
		for i in range(self.nqubits, 2 * self.nqubits):
			qc.Ry(theta(), i)

		qc.to_latex_file('figures/circuit.tex')

		return qc


class GibbsResult:
	def __init__(self, gibbs):
		self.result = gibbs.result
		self.optimal_parameters = gibbs.params
		self.unitary = gibbs.circuit
		self.cost = gibbs.cost
		self.gibbs_state = partial_trace(gibbs.statevector, gibbs.ancilla_qubits).data
		self.eigenvalues = np.sort(gibbs.eigvals)
		self.hamiltonian_eigenvalues = np.sort(gibbs.cost - gibbs.inverse_beta * np.log(gibbs.eigvals))
		self.energy = gibbs.energy
		self.entropy = gibbs.entropy
