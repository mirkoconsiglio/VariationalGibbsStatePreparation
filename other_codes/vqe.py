import numpy as np
from qiskit.quantum_info import Statevector

class VQE:
	def __init__(self, ansatz, optimizer):
		self.ansatz = ansatz
		self.optimizer = optimizer

		self.hamiltonian = None
		self.x0 = None
		self.iter = None
		self.cost = None
		self.result = None

	def run(self, hamiltonian, x0=None):
		self.hamiltonian = hamiltonian
		if x0 is None:
			x0 = np.random.uniform(-2 * np.pi, 2 * np.pi, self.ansatz.num_parameters)
		self.x0 = x0

		self.iter = 0

		self.result = self.optimizer.minimize(self.cost_fun, self.x0)

		return VQEResult(self)

	def cost_fun(self, x):
		self.iter += 1
		qc = self.ansatz.assign_parameters(x)
		self.cost = Statevector(qc).expectation_value(self.hamiltonian).real
		self.callback()
		return self.cost

	def callback(self):
		if self.iter % 100 == 0:
			print(f'{self.iter}', f'{self.cost:.8f}')


class VQEResult:
	def __init__(self, vqe):
		self.eigenvalue = vqe.result.fun
		self.eigenstate = Statevector(vqe.ansatz.assign_parameters(vqe.result.x)).data
		self.optimal_parameters = vqe.result.x
