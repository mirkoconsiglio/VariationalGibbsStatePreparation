import numpy as np
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.quantum_info import Statevector, partial_trace
from qiskit.circuit.library import RYGate


class Gibbs:
	def __init__(self, n, ansatz, optimizer):
		self.nqubits = n
		self.ansatz = ansatz
		self.optimizer = optimizer

		self.dimension = 2 ** self.nqubits
		self.ancilla_qubits = list(range(self.nqubits))
		self.system_qubits = list(range(self.nqubits, 2 * self.nqubits))
		self.qc = self.circuit()
		self.num_params = self.qc.num_parameters

		self.hamiltonian = None
		self.inverse_beta = None
		self.x0 = None
		self.bounds = None
		self.params = None
		self.eigvals = None
		self.energy = None
		self.entropy = None
		self.cost = None
		self.iter = None
		self.statevector = None
		self.result = None

	def run(self, hamiltonian, beta, x0=None, bounds=None):
		self.hamiltonian = hamiltonian
		self.inverse_beta = 1 / beta
		self.x0 = x0 if x0 else np.random.uniform(-2 * np.pi, 2 * np.pi, self.num_params)
		self.bounds = bounds if bounds else [(-2 * np.pi, 2 * np.pi) for _ in range(self.num_params)]

		self.iter = 0
		print('Iter    Cost       Energy     Entropy')
		self.result = self.optimizer.minimize(fun=self.cost_fun, x0=self.x0, bounds=self.bounds)
		return GibbsResult(self)

	@staticmethod
	def callback(i, cost, energy, entropy):
		if i % 100 == 0:
			print(f'{i}', f'{cost:.8f}', f'{energy:.8f}', f'{entropy:.8f}')

	def cost_fun(self, x):
		self.iter += 1
		self.params = x
		self.statevector = Statevector(self.qc.assign_parameters(self.params))
		self.energy = self.energy_fun()
		self.entropy = self.entropy_fun()
		self.cost = self.energy - self.inverse_beta * self.entropy
		self.callback(self.iter, self.cost, self.energy, self.entropy)
		return self.cost

	def energy_fun(self):
		return self.statevector.expectation_value(self.hamiltonian, self.system_qubits).real

	def entropy_fun(self):
		self.eigvals = self.eigvals_fun()
		return -np.sum([i * np.log(i) for i in self.eigvals])

	def eigvals_fun(self):
		alpha = self.params[:self.dimension - 1]
		eigvals = []
		for i in range(self.dimension):
			l = 1
			j = f'{i:0{self.nqubits}b}'
			for m, k in enumerate(j):
				p = 2 ** m - 1
				q = 0 if i == 0 or m == 0 else int(np.floor(i // 2 ** (self.nqubits - m)))
				if k == '0':
					l *= np.cos(alpha[p + q] / 2) ** 2
				else:
					l *= np.sin(alpha[p + q] / 2) ** 2
			eigvals.append(l)
		return eigvals

	def circuit(self):
		alpha = ParameterVector('α', self.dimension - 1)
		qc = QuantumCircuit(2 * self.nqubits)
		for i in range(self.dimension - 1):
			if i == 0:
				qc.ry(alpha[i], 0)
			else:
				j = int(np.floor(np.log2(i + 1)))
				k = f'{i - 2 ** j + 1:0{j}b}'[::-1]
				qc.append(RYGate(alpha[i]).control(j, ctrl_state=k), list(range(j + 1)))
		for i in range(self.nqubits):
			qc.cx(i, self.nqubits + i)
		qc.append(self.ansatz, range(self.nqubits, 2 * self.nqubits))
		return qc


class GibbsResult:
	def __init__(self, gibbs):
		self.result = gibbs.result
		self.optimal_parameters = gibbs.params
		self.unitary = gibbs.qc.assign_parameters(gibbs.params)
		self.cost = gibbs.cost
		self.gibbs_state = partial_trace(gibbs.statevector, gibbs.ancilla_qubits)
		self.eigenvalues = gibbs.eigvals
		self.energy = gibbs.energy
		self.entropy = gibbs.entropy
