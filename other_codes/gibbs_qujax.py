import numpy as np
from pytket.circuit import Circuit, Qubit, BasisOrder
from pytket.extensions.qujax import tk_to_qujax
from pytket.pauli import Pauli, QubitPauliString
from qiskit.quantum_info import partial_trace
from scipy.special import xlogy

from jax import numpy as jnp, random, vmap, jit
from jax.lax import scan
from qujax import get_statetensor_to_expectation_func, sample_bitstrings
from optax import adam, apply_updates


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
		self.cost_fun = None
		self.pauli_circuits = None
		self.coeffs = None
		self.constant_term = None
		self.operator = None
		self.commuting_terms = None
		self.pauli_sym = {'I': Pauli.I, 'X': Pauli.X, 'Y': Pauli.Y, 'Z': Pauli.Z}
		self.ncircuits = None
		self.total_shots = None
		self.terms = None
		self.qubits = None
		self.coefs = None
		self.optimiser = None
		self.random_key = random.PRNGKey(0)
		self.init_key, self.train_key = random.split(self.random_key)
		self.param_to_st = None
		self.init_opt_state = None

	def run(self, hamiltonian, beta, min_kwargs=None, shots=None, ancilla_reps=None, system_reps=None,
	        commuting_terms=None):
		self.hamiltonian = hamiltonian
		self.terms, self.qubits, self.coefs = self.openfermion_to_qujax(self.hamiltonian)
		self.inverse_beta = 1 / beta
		self.ancilla_reps = ancilla_reps or self.nqubits - 1
		self.system_reps = system_reps or self.total_nqubits
		self.commuting_terms = commuting_terms
		self.ncircuits = len(self.commuting_terms) if self.commuting_terms else \
			len(self.operator.terms)
		self.ansatz = self.var_ansatz()

		# Invoke qujax
		self.param_to_st = tk_to_qujax(self.ansatz)
		st_to_expectation = get_statetensor_to_expectation_func(self.terms, self.qubits, self.coefs)

		# Shots
		self.shots = shots
		if self.shots:
			self.cost_fun = self.sampled_cost_fun
			self.pauli_circuits = self.generate_measurement_circuits()
			self.total_shots = self.shots * self.ncircuits
		else:
			self.cost_fun = self.statevector_cost_fun

		# Optimiser
		self.optimiser = adam(1e-3)
		self.x0 = random.uniform(self.init_key, shape=(self.num_params,), minval=0., maxval=2.)
		self.init_opt_state = self.optimiser.init(self.x0)
		# Run optimisation
		self.iter = 0
		self.nfev = 0
		print('| iter | nfev | Cost | Energy | Entropy |')
		result = scan(self.cost_fun, (self.x0, self.init_opt_state, self.train_key), jnp.arange(1000))
		# Update
		self.eigenvalues, self.eigenvectors = self.eigensystem()
		self.params = self.result.x
		self.cost = self.result.fun
		circuit = self.ansatz.copy()
		circuit.symbol_substitution({sym: val for sym, val in zip(self.param_keys, self.params)})
		self.state = self.backend.run_circuit(circuit).get_state()

		return GibbsResult(self)

	def openfermion_to_qujax(self, operator):
		pass

	def var_ansatz(self):
		qc = Circuit(self.total_nqubits)
		# Number of params
		self.num_ancilla_params = self.nqubits * (self.ancilla_reps + 1)
		self.num_system_params = self.nqubits * (self.system_reps + 1)
		self.num_params = self.num_ancilla_params + self.num_system_params
		self.theta_n = 0
		params = jnp.zeros((self.num_params,))
		# Unitary ancilla
		qc = self.ancilla_unitary(params, qc, self.ancilla_qubits)
		# Connecting CNOTs
		for i in range(self.nqubits):
			qc.CX(i, i + self.nqubits)
		# Unitary system
		qc = self.system_unitary(params, qc, self.system_qubits)

		return qc

	def ancilla_unitary(self, params, qc=None, qubits=None):
		if not qc:
			qc = Circuit(self.nqubits)
		if not qubits:
			qubits = range(qc.n_qubits)

		# Layers
		for r in range(self.ancilla_reps):
			for i in qubits:
				qc.Ry(params[self.theta_n], i)
				self.theta_n += 1
				if i > qubits[0]:
					qc.CX(i - 1, i)

		# Last one-qubit layer
		for i in qubits:
			qc.Ry(params[self.theta_n], i)
			self.theta_n += 1

		return qc

	def system_unitary(self, params, qc=None, qubits=None):
		if not qc:
			qc = Circuit(self.nqubits)
		if not qubits:
			qubits = range(qc.n_qubits)

		# Layers
		for r in range(self.system_reps):
			for i in qubits:
				qc.Ry(params[self.theta_n], i)
				self.theta_n += 1
				if i > qubits[0]:
					qc.CX(i - 1, i)

		# Last one-qubit layer
		for i in qubits:
			qc.Ry(params[self.theta_n], i)
			self.theta_n += 1

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
		return U.get_unitary()

	def system_unitary_matrix(self):
		self.theta_n = self.num_ancilla_params
		U = self.system_unitary()
		U.symbol_substitution({sym: val for sym, val in zip(self.param_keys[self.num_ancilla_params:],
		                                                    self.system_unitary_params())})
		return U.get_unitary()

	def eigensystem(self):
		self.eigenvectors = self.system_unitary_matrix()
		indices = np.argsort(self.eigenvalues)

		return self.eigenvalues[indices], self.eigenvectors[indices].T

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

	def param_to_statevector_cost_and_grad(self, params):
		random_keys = random.split(random_key, 2 * params.size + 1)
		cost_key = random_keys[0]
		grad_keys = random_keys[1:]

		st = self.param_to_st(params)
		sampled_bistrings = sample_bitstrings(cost_key, st, self.shots)
		sampled_cost = vmap(cost_of_bitstring)(sampled_bistrings).mean()

		def sample_grad_k(k, random_key_plus, random_key_minus):
			param_plus = params.at[k].set(params[k] + 0.5)
			st_plus = self.param_to_st(param_plus)
			sample_bitstrings_plus = sample_bitstrings(random_key_plus, st_plus, self.shots)

			param_minus = params.at[k].set(params[k] - 0.5)
			st_minus = self.param_to_st(param_minus)
			sample_bitstrings_minus = sample_bitstrings(random_key_minus, st_minus, self.shots)

			return (vmap(cost_of_bitstring)(sample_bitstrings_plus).mean() -
			        vmap(cost_of_bitstring)(sample_bitstrings_minus).mean()) / 2

		sampled_grad = vmap(sample_grad_k)(jnp.arange(params.size),
		                                   grad_keys[:params.size],
		                                   grad_keys[params.size:])

		return sampled_cost, sampled_grad

	def statevector_gd_iteration(self, params_and_opt_state, step):
		# Unpack args
		params, opt_state = params_and_opt_state
		# Compute cost and gradient
		cost, grad = self.param_to_statevector_cost_and_grad(params)

		updates, new_opt_state = self.optimiser.update(grad, opt_state, params)
		new_params = apply_updates(params, updates)

		return new_params, new_opt_state

	def param_to_sampled_cost_and_grad(self, params, random_key):
		random_keys = random.split(random_key, 2 * params.size + 1)
		cost_key = random_keys[0]
		grad_keys = random_keys[1:]

		st = self.param_to_st(params)
		sampled_bistrings = sample_bitstrings(cost_key, st, self.shots)
		sampled_cost = vmap(cost_of_bitstring)(sampled_bistrings).mean()

		def sample_grad_k(k, random_key_plus, random_key_minus):
			param_plus = params.at[k].set(params[k] + 0.5)
			st_plus = self.param_to_st(param_plus)
			sample_bitstrings_plus = sample_bitstrings(random_key_plus, st_plus, self.shots)

			param_minus = params.at[k].set(params[k] - 0.5)
			st_minus = self.param_to_st(param_minus)
			sample_bitstrings_minus = sample_bitstrings(random_key_minus, st_minus, self.shots)

			return (vmap(cost_of_bitstring)(sample_bitstrings_plus).mean() -
			        vmap(cost_of_bitstring)(sample_bitstrings_minus).mean()) / 2

		sampled_grad = vmap(sample_grad_k)(jnp.arange(params.size),
		                                   grad_keys[:params.size],
		                                   grad_keys[params.size:])

		return sampled_cost, sampled_grad

	def sampled_gd_iteration(self, params_and_opt_state_and_rk, step):
		# Unpack args
		params, opt_state, rk_in = params_and_opt_state_and_rk
		rk_out, rk_samp = random.split(rk_in)
		# Compute cost and gradient
		cost, grad = self.param_to_sampled_cost_and_grad(params, rk_samp)

		updates, new_opt_state = self.optimiser.update(grad, opt_state, params)
		new_params = apply_updates(params, updates)

		return new_params, new_opt_state, rk_out


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
