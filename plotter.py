import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from os import listdir
from os.path import isdir
from main_qulacs import ExactResult
import pickle
import re

rc('font', **{'family': 'Times New Roman', 'sans-serif': ['Times New Roman'], 'size': 20})
rc('text', usetex=True)


def plot_multiple_beta(directory):
	fig1, ax1 = plt.subplots(figsize=(12, 8))
	# ax1.set_title(r'Fidelity $F$ vs inverse temperature $\beta$')
	ax1.set_xlabel(r'$\beta$')
	ax1.set_ylabel(r'$F$')
	ax1.set_xscale('log')
	ax1.set_ylim([0.48, 1.02])
	ax1.grid(visible=True, which='major', axis='both')

	fig2, ax2 = plt.subplots(figsize=(12, 8))
	# ax2.set_title(r'Trace distance $D_{Tr}$ vs inverse temperature $\beta$')
	ax2.set_xlabel(r'$\beta$')
	ax2.set_ylabel(r'$D_\mathrm{Tr}$')
	ax2.set_xscale('log')
	ax2.grid(visible=True, which='major', axis='both')

	fig3, ax3 = plt.subplots(figsize=(12, 8))
	# ax3.set_title(r'Relative entropy $S$ vs inverse temperature $\beta$')
	ax3.set_xlabel(r'$\beta$')
	ax3.set_ylabel(r'$\mathcal{S}$')
	ax3.set_xscale('log')
	ax3.grid(visible=True, which='major', axis='both')

	for folder in filter(lambda x: isdir(f'{directory}/{x}'), listdir(directory)):
		with open(f'{directory}/{folder}/data.pkl', 'rb') as file:
			data = pickle.load(file)
		n = int(folder[-1])
		beta = [i['beta'] for i in data]
		fidelity = [i['fidelity'] for i in data]
		trace_distance = [i['trace_distance'] for i in data]
		relative_entropy = [i['relative_entropy'] for i in data]
		overlaps = [i['overlaps'] for i in data]

		fig, ax = plt.subplots(figsize=(12, 8))
		# ax.set_title(r'Fidelity, trace distance and relative entropy vs inverse temperature $\beta$')
		ax.set_xlabel(r'$\beta$')
		ax.set_ylabel(r'$Y$')
		ax.set_xscale('log')

		ax.scatter(beta, fidelity, label='Fidelity')
		ax.scatter(beta, trace_distance, label='Trace Distance')
		ax.scatter(beta, relative_entropy, label='Relative Entropy')
		ax.legend()
		fig.savefig(f'{directory}/{folder}/plot.pdf', dpi=600)

		fig, ax = plt.subplots(figsize=(12, 8))
		# ax.set_title(r'Overlaps $O$ of eigenstates vs inverse temperature $\beta$')
		ax.set_xlabel(r'$\beta$')
		ax.set_ylabel(r'$O$')
		ax.set_xscale('log')

		for i in np.transpose(overlaps):
			ax.scatter(beta, i)
		fig.savefig(f'{directory}/{folder}/plot_overlaps.pdf', dpi=600)

		ax1.scatter(beta, fidelity, label=f'{n}-qubits')
		ax2.scatter(beta, trace_distance, label=f'{n}-qubits')
		ax3.scatter(beta, relative_entropy, label=f'{n}-qubits')

	ax1.legend()
	fig1.savefig(f'{directory}/fidelity_plot.pdf', dpi=600, transparent=True)
	ax2.legend()
	fig2.savefig(f'{directory}/trace_distance_plot.pdf', dpi=600, transparent=True)
	ax3.legend()
	fig3.savefig(f'{directory}/relative_entropy_plot.pdf', dpi=600, transparent=True)

	plt.show()

def plot_var_layers(directory):
	fig, ax = plt.subplots(figsize=(12, 8))
	ax.set_xlabel(r'Layers')
	ax.set_ylabel(r'$F$')
	ax.grid(visible=True, which='major', axis='both')
	layers = []
	F = []
	exact_F = 0
	for folder in filter(lambda x: isdir(f'{directory}/{x}'), listdir(directory)):
		with open(f'{directory}/{folder}/data.pkl', 'rb') as file:
			data = pickle.load(file)

		layers.append(int(re.findall(r'\d+', folder)[0]))
		F.append(data['calculated_result'].cost)
		exact_F = data['exact_result'].cost

	ax.scatter(layers, F)
	ax.axhline(exact_F, ls='--')
	fig.savefig(f'{directory}/plot.pdf', dpi=600)

	plt.show()


if __name__ == '__main__':
	plot_multiple_beta('statevector_h_0.5')
