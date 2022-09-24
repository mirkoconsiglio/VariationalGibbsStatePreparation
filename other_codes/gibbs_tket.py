import numpy as np
from pytket.circuit import Circuit, Qubit, BasisOrder
from pytket.extensions.qiskit import tk_to_qiskit, AerUnitaryBackend
from pytket.extensions.qulacs import QulacsBackend
from pytket.pauli import Pauli, QubitPauliString
from pytket.utils.operators import QubitPauliOperator
from qiskit.algorithms.optimizers import *
from qiskit.quantum_info import partial_trace
from scipy.special import xlogy
from sympy import Symbol


class Gibbs:
	def __init__(self, n):
		self.nqubits = n
		self.total_nqubits = 2 * self.nqubits
		self.dimension = 2 ** self.nqubits
		self.ancilla_qubits = list(range(self.nqubits))
		self.system_qubits = list(range(self.nqubits, self.total_nqubits))
		self.ansatz = None
		self.param_keys = None
		self.num_params = None
		self.num_ancilla_params = None
		self.num_system_params = None
		self.hamiltonian = None
		self.inverse_beta = None
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
		self.state = None
		self.result = None
		self.shots = None
		self.eigenvalues = None
		self.eigenvectors = None

		self.theta_n = None
		self.backend = QulacsBackend()
		self.cost_fun = None
		self.pauli_circuits = None
		self.coeffs = None
		self.constant_term = None
		self.operator = None
		self.commuting_terms = None
		self.pauli_sym = {'I': Pauli.I, 'X': Pauli.X, 'Y': Pauli.Y, 'Z': Pauli.Z}
		self.ncircuits = None
		self.total_shots = None

	def run(self, hamiltonian, beta, min_kwargs=None, x0=None, bounds=None, shots=None, ancilla_reps=None,
	        system_reps=None, commuting_terms=None):
		self.operator = hamiltonian
		self.hamiltonian = self.openfermion_to_tket_gibbs(self.operator)
		self.inverse_beta = 1 / beta
		self.ancilla_reps = ancilla_reps or self.nqubits - 1
		self.system_reps = system_reps or self.total_nqubits
		self.commuting_terms = commuting_terms
		self.ncircuits = len(self.commuting_terms) if self.commuting_terms else \
			len(self.operator.terms)
		self.param_keys = []
		self.ansatz = self.var_ansatz()
		self.x0 = x0 if x0 else np.random.uniform(0, 2, self.num_params)
		self.bounds = bounds if bounds else [(0, 2) for _ in range(self.num_params)]
		# Set up minimizer kwargs
		self.min_kwargs = min_kwargs if min_kwargs else dict()
		self.min_kwargs.update(callback=self.callback)
		self.shots = shots
		if self.shots:
			self.cost_fun = self.sampled_cost_fun
			self.pauli_circuits = self.generate_measurement_circuits()
			self.total_shots = self.shots * self.ncircuits
		else:
			self.cost_fun = self.statevector_cost_fun

		self.iter = 0
		self.nfev = 0
		print('| iter | nfev | Cost | Energy | Entropy |')
		self.result = POWELL(**self.min_kwargs).minimize(fun=self.cost_fun, x0=self.x0, bounds=self.bounds)
		# Update
		self.eigenvalues, self.eigenvectors = self.eigensystem()
		self.params = self.result.x
		self.cost = self.result.fun
		circuit = self.ansatz.copy()
		circuit.symbol_substitution({sym: val for sym, val in zip(self.param_keys, self.params)})
		self.state = self.backend.run_circuit(circuit).get_state()

		return GibbsResult(self)

	def openfermion_string_to_tket_string(self, paulis):
		qlist = []
		plist = []
		for q, p in paulis:
			qlist.append(Qubit(q + self.nqubits))
			plist.append(self.pauli_sym[p])
		return QubitPauliString(qlist, plist)

	def openfermion_to_tket_gibbs(self, operator):
		op = dict()
		for term, coef in operator.terms.items():
			op[self.openfermion_string_to_tket_string(term)] = coef
		return QubitPauliOperator(op)

	def theta(self):
		string = f'$\\theta_{{{self.theta_n}}}$'
		symbol = Symbol(string)
		self.param_keys.append(symbol)
		self.theta_n += 1
		return symbol

	def var_ansatz(self):
		qc = Circuit(self.total_nqubits)
		self.num_ancilla_params = self.nqubits * (self.ancilla_reps + 1)
		self.num_system_params = self.nqubits * (self.system_reps + 1)
		self.num_params = self.num_ancilla_params + self.num_system_params
		self.theta_n = 0
		# Unitary ancilla
		qc = self.ancilla_unitary(qc, self.ancilla_qubits)
		# Connecting CNOTs
		for i in range(self.nqubits):
			qc.CX(i, i + self.nqubits)
		# Unitary system
		qc = self.system_unitary(qc, self.system_qubits)
		self.num_params = len(qc.free_symbols())
		self.num_system_params = self.num_params - self.num_ancilla_params

		return qc

	def ancilla_unitary(self, qc=None, qubits=None):
		if not qc:
			qc = Circuit(self.nqubits)
		if not qubits:
			qubits = range(qc.n_qubits)

		# Layers
		for r in range(self.ancilla_reps):
			for i in qubits:
				qc.Ry(self.theta(), i)
				if i > qubits[0]:
					qc.CX(i - 1, i)

		# Last one-qubit layer
		for i in qubits:
			qc.Ry(self.theta(), i)

		return qc

	def system_unitary(self, qc=None, qubits=None):
		if not qc:
			qc = Circuit(self.nqubits)
		if not qubits:
			qubits = range(qc.n_qubits)

		# Layers
		for r in range(self.system_reps):
			for i in qubits:
				qc.Ry(self.theta(), i)
				if i > qubits[0]:
					qc.CX(i - 1, i)

		# Last one-qubit layer
		for i in qubits:
			qc.Ry(self.theta(), i)

		return qc

	def ancilla_unitary_params(self):
		return self.params[:self.num_ancilla_params]

	def system_unitary_params(self):
		return self.params[self.num_ancilla_params:]

	def ancilla_unitary_matrix(self):
		self.theta_n = 0
		U = self.ancilla_unitary()
		U.symbol_substitution({sym: val for sym, val in zip(self.param_keys[:self.num_ancilla_params],
		                                                    self.ancilla_unitary_params())})

		return AerUnitaryBackend().run_circuit(U).get_unitary()

	def system_unitary_matrix(self):
		self.theta_n = self.num_ancilla_params
		U = self.system_unitary()
		U.symbol_substitution({sym: val for sym, val in zip(self.param_keys[self.num_ancilla_params:],
		                                                    self.system_unitary_params())})

		return U.get_unitary()

	def eigensystem(self):
		eigvecs = self.system_unitary_matrix()
		eigvals = self.eigenvalues.copy()
		indices = np.argsort(eigvals)

		return eigvals[indices], eigvecs[indices].T

	def generate_measurement_circuits(self):
		pauli_circuits = []
		if self.commuting_terms:
			for label, terms in self.commuting_terms.items():
				if len(terms) > 0:
					pauli_circ = Circuit(self.total_nqubits)
					if label != 'z':
						for qubits in terms[1]:
							for q in qubits:
								pauli_circ.H(q)
					state_and_measure = self.ansatz.copy()
					state_and_measure.append(pauli_circ)
					state_and_measure.measure_all()
					pauli_circuits.append(state_and_measure)
		else:
			for p, _ in self.operator.terms.items():
				pauli_circ = Circuit(self.total_nqubits)
				for q, s in p:
					if s == 'Y':
						pauli_circ.Sdg(q)
						pauli_circ.H(q)
					elif s == 'X':
						pauli_circ.H(q)
				state_and_measure = self.ansatz.copy()
				state_and_measure.append(pauli_circ)
				state_and_measure.measure_all()
				pauli_circuits.append(state_and_measure)
		return pauli_circuits

	def callback(self, *args, **kwargs):
		self.iter += 1
		print(f'| {self.iter} | {self.nfev} | {self.cost:.8f} | {self.energy:.8f} | {self.entropy:.8f} |')

	def statevector_cost_fun(self, x):
		self.nfev += 1
		self.params = x
		circuit = self.ansatz.copy()
		circuit.symbol_substitution({sym: val for sym, val in zip(self.param_keys, self.params)})
		self.state = self.backend.run_circuit(circuit).get_state()
		self.energy = self.hamiltonian.state_expectation(self.state).real
		self.eigenvalues = np.sum(np.abs(np.reshape(self.state, (self.dimension, self.dimension))) ** 2, axis=1)
		self.entropy = -np.sum([xlogy(i, i) for i in self.eigenvalues])
		self.cost = self.energy - self.inverse_beta * self.entropy

		return self.cost

	def sampled_cost_fun(self, x):
		self.nfev += 1
		self.params = x

		circuits = []
		for pauli_circ in self.pauli_circuits:
			circuit = pauli_circ.copy()
			circuit.symbol_substitution({sym: val for sym, val in zip(self.param_keys, self.params)})
			circuits.append(circuit)
		handles = self.backend.process_circuits(circuits, n_shots=self.shots)
		results = self.backend.get_results(handles)

		energy = 0
		p = np.zeros(self.dimension)
		for result, (label, terms) in zip(results, self.commuting_terms.items()):
			for shot, n in result.get_counts().items():
				# Energy
				coef, qubits = terms
				if label == 'z':
					energy += (2 * shot[self.nqubits:].count(0) - self.nqubits) * coef * n
				else:
					for q1, q2 in qubits:
						if shot[q1] == shot[q2]:
							energy += coef * n
						else:
							energy -= coef * n
				# Entropy
				j = 0
				for b in shot[:self.nqubits]:
					j = (j << 1) | b
				p[j] += n

		self.energy = energy / self.shots
		self.eigenvalues = p / self.total_shots
		self.entropy = -np.sum([xlogy(i, i) for i in self.eigenvalues])
		self.cost = self.energy - self.inverse_beta * self.entropy

		return self.cost


class GibbsResult:  # TODO: Fix eigenvector ordering
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
		self.gibbs_state = partial_trace(gibbs.state, gibbs.system_qubits).data
		self.eigenvalues = gibbs.eigenvalues
		self.eigenvectors = gibbs.eigenvectors
		self.hamiltonian_eigenvalues = np.sort(gibbs.cost - gibbs.inverse_beta * np.log(self.eigenvalues))
