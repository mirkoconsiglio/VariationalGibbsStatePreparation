import numpy as np
import matplotlib.pyplot as plt
from main_qulacs import xx_hamiltonian, ExactResult
from gibbs_ising_qulacs import xx_Gibbs
from qulacs import QuantumState
from qulacs.state import inner_product
from matplotlib import rc

rc('font', **{'family': 'Times New Roman', 'sans-serif': ['Times New Roman'], 'size': 20})
rc('text', usetex=True)


# Parameters
n = 2  # number of qubits
J = 1
h = 0.5
h_quenched = 2 * h
beta = 1
shots = None  # Number of shots to sample
# Define Hamiltonian
H1 = xx_hamiltonian(n, J, h)
# Define minimizer kwargs
min_kwargs = dict()
# Run VQA
gibbs1 = xx_Gibbs(n)
result1 = gibbs1.run(H1, beta, min_kwargs, shots=shots)
U1 = gibbs1.system_unitary()
for i, param in enumerate(result1.system_params):
	U1.set_parameter(i, param)
e_n = result1.hamiltonian_eigenvalues
p_n = result1.eigenvalues

exact_result1 = ExactResult(H1, beta)
ex_e_n = exact_result1.hamiltonian_eigenvalues
ex_p_n = exact_result1.eigenvalues
eigvecs_n = exact_result1.eigenvectors

# Define Hamiltonian
H2 = xx_hamiltonian(n, J, h_quenched)
# Define minimizer kwargs
min_kwargs = dict()
# Run VQA
gibbs2 = xx_Gibbs(n)
result2 = gibbs2.run(H2, beta, min_kwargs, shots=shots)
U2 = gibbs2.system_unitary()
for i, param in enumerate(result2.system_params):
	U2.set_parameter(i, param)
e_m = result2.hamiltonian_eigenvalues

exact_result2 = ExactResult(H2, beta)
ex_e_m = exact_result2.hamiltonian_eigenvalues
eigvecs_m = exact_result2.eigenvectors

W_list = []
ex_W_list = []
p_mn = np.zeros((2 ** n, 2 ** n))
ex_p_mn = np.zeros((2 ** n, 2 ** n))
for i in range(2 ** n):
	for j in range(2 ** n):
		state1 = QuantumState(n)
		state1.set_computational_basis(j)
		U1.update_quantum_state(state1)

		state2 = QuantumState(n)
		state2.set_computational_basis(i)
		U2.update_quantum_state(state2)

		p_mn[i, j] = np.abs(inner_product(state2, state1)) ** 2
		W_list.append(e_m[i] - e_n[j])

		ex_p_mn[i, j] = np.abs(np.vdot(eigvecs_m, eigvecs_n))
		ex_W_list.append(ex_e_m[i] - ex_e_n[j])


fig, ax = plt.subplots(figsize=(12, 8))

pdf = [np.sum([p_mn[i, j] * p_n[j] * 1 if k - (e_m[i] - e_n[j]) == 0 else 0
       for j in range(2 ** n) for i in range(2 ** n)]) for k in W_list]
ax.scatter(W_list, pdf)

ex_pdf = [np.sum([ex_p_mn[i, j] * ex_p_n[j] * 1 if k - (ex_e_m[i] - ex_e_n[j]) == 0 else 0
          for j in range(2 ** n) for i in range(2 ** n)]) for k in ex_W_list]
ax.scatter(ex_W_list, ex_pdf)

fig.savefig(f'pdf.pdf')

plt.show()
