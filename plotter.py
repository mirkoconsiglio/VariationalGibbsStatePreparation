import json
from os import listdir

import matplotlib.pyplot as plt
from matplotlib import rc

rc('font', **{'family': 'Times New Roman', 'sans-serif': ['Times New Roman'], 'size': 20})
rc('text', usetex=True)


def plot_multiple_beta(folder):
	fig, ax = plt.subplots(figsize=(12, 8))
	ax.set_xlabel(r'$\beta$')
	ax.set_ylabel(r'$Y$')
	ax.grid(visible=True, which='major', axis='both')

	beta = []
	fidelity = []
	trace_distance = []
	relative_entropy = []
	for file in filter(lambda x: x.endswith('.json'), listdir(folder)):
		with open(f'{folder}/{file}', 'r') as f:
			data = json.load(f)
		beta.append(data['beta'])
		fidelity.append(data['metrics']['calculated_fidelity'])
		trace_distance.append(data['metrics']['calculated_trace_distance'])
		relative_entropy.append(data['metrics']['calculated_relative_entropy'])

	ax.scatter(beta, fidelity, label='Fidelity')
	ax.scatter(beta, trace_distance, label='Trace Distance')
	ax.scatter(beta, relative_entropy, label='Relative Entropy')
	ax.legend()
	fig.savefig(f'{folder}/plot.pdf', dpi=600)

	plt.show()

if __name__ == '__main__':
	plot_multiple_beta('qiskit_runtime/jobs/ibmq_jakarta/ccu9nluhs7ee9ds5stsg')
