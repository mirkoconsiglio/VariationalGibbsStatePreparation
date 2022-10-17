import json
from os import listdir
from os.path import isdir

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc, cm
from matplotlib.colors import Normalize, LinearSegmentedColormap
from qiskit_ibm_runtime import RuntimeDecoder

rc('font', **{'family': 'Times New Roman', 'sans-serif': ['Times New Roman'], 'size': 12})
rc('text', usetex=True)


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, cycle=0., name='shiftedcmap'):
	cdict = {
		'red': [],
		'green': [],
		'blue': [],
		'alpha': []
	}

	# regular index to compute the colors
	reg_index = (np.linspace(start, stop, 257) + cycle) % 1

	# shifted index to match the data
	shift_index = np.hstack([
		np.linspace(0.0, midpoint, 128, endpoint=False),
		np.linspace(midpoint, 1.0, 129, endpoint=True)
	])

	for ri, si in zip(reg_index, shift_index):
		r, g, b, a = cmap(ri)

		cdict['red'].append((si, r, r))
		cdict['green'].append((si, g, g))
		cdict['blue'].append((si, b, b))
		cdict['alpha'].append((si, a, a))

	newcmap = LinearSegmentedColormap(name, cdict)

	return newcmap


def mapper(vmin=0, vmax=1, cmap='hsv'):
	norm = Normalize(vmin, vmax)
	return cm.ScalarMappable(norm=norm, cmap=cmap)


def complex_to_size_and_color(data, cmap, vmin=0, vmax=1):
	size = abs(data).flatten()
	angle = np.angle(data).flatten()
	m = mapper(vmin, vmax, cmap)
	color = m.to_rgba(angle)

	return size, color


def set_bar3d(ax, xpos, ypos, zpos, dx, dy, dz, color):
	ax.view_init(elev=30, azim=-40)
	ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=color)
	ax.set_xticks(range(4), ['00', '01', '10', '11'])
	ax.set_yticks(range(4), ['00', '01', '10', '11'])
	ax.set_zticks([0.0, 0.2, 0.4, 0.6, 0.8], [0.0, 0.2, 0.4, 0.6, 0.8])
	ax.tick_params(axis='both', pad=-2)
	ax.set_zlim(0, 0.8)
	ax.xaxis.labelpad = 100


# noinspection PyUnusedLocal
def rad_fn(x, pos=None):
	if x >= 0:
		n = int((x / np.pi) * 2.0 + 0.25)
	else:
		n = int((x / np.pi) * 2.0 - 0.25)

	if n == 0:
		return '0'
	elif n == 1:
		return r'$\pi/2$'
	elif n == 2:
		return r'$\pi$'
	elif n == -1:
		return r'$-\pi/2$'
	elif n == -2:
		return r'$-\pi$'
	elif n % 2 == 0:
		return fr'${n // 2}\pi$'
	else:
		return fr'${n}\pi/2$'


def plot_density(folder, cmap):
	fig = plt.figure(figsize=(6, 9))
	gs = fig.add_gridspec(3, 2)
	axes = np.array([fig.add_subplot(gs[j, i], projection='3d') for j in range(3) for i in range(2)]).reshape(3, 2)

	numOfCols = 4
	numOfRows = 4

	xpos = np.arange(0, numOfCols, 1)
	ypos = np.arange(0, numOfRows, 1)
	xpos, ypos = np.meshgrid(xpos + 0.5, ypos + 0.5)

	xpos = xpos.flatten()
	ypos = ypos.flatten()
	zpos = np.zeros(numOfCols * numOfRows)

	dx = np.ones(numOfRows * numOfCols) * 0.8
	dy = np.ones(numOfCols * numOfRows) * 0.8

	i = 0
	for file in filter(lambda x: x.endswith('.json'), listdir(folder)):
		with open(f'{folder}/{file}', 'r') as f:
			data = json.load(f, cls=RuntimeDecoder)

		beta = data['metadata']['beta']

		if beta != 1e-8 and beta != 1. and beta != 5.:
			continue

		ax = axes[i, 0]
		exact_rho = np.asarray(data['exact_result']['gibbs_state'])
		dz, color = complex_to_size_and_color(exact_rho, cmap, -np.pi, np.pi)
		set_bar3d(ax, xpos, ypos, zpos, dx, dy, dz, color)
		if i == 0:
			ax.text2D(0.45, 0.95, r'$\rho_\textrm{\small{Gibbs}}$', transform=ax.transAxes, fontsize=20)

		ax = axes[i, 1]
		calculated_rho = np.asarray(data['calculated_result']['rho'])
		dz, color = complex_to_size_and_color(calculated_rho, cmap, -np.pi, np.pi)
		set_bar3d(ax, xpos, ypos, zpos, dx, dy, dz, color)
		ax.text2D(-0.17, 0.85, fr'$\beta = {int(beta)}$', transform=ax.transAxes, fontsize=20)
		if i == 0:
			ax.text2D(0.46, 0.95, r'$\rho_\textrm{\small{Calc}}$', transform=ax.transAxes, fontsize=20)

		i += 1

	fig.colorbar(mapper(-np.pi, np.pi, cmap), ax=axes[2, :], shrink=0.6, orientation='horizontal',
	             ticks=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi], format=rad_fn)

	fig.subplots_adjust(left=0.05, bottom=0.17, right=0.95, top=1, wspace=0.1, hspace=0.05)

	fig.savefig(f'Bar3D.pdf', bbox_inches='tight')

	# plt.subplot_tool()
	plt.show()


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
		beta.append(data['metadata']['beta'])
		fidelity.append(data['metrics']['calculated_fidelity'])
		trace_distance.append(data['metrics']['calculated_trace_distance'])
		relative_entropy.append(data['metrics']['calculated_relative_entropy'])

	ax.scatter(beta, fidelity, label='Fidelity')
	ax.scatter(beta, trace_distance, label='Trace Distance')
	ax.scatter(beta, relative_entropy, label='Relative Entropy')
	ax.legend()
	fig.savefig(f'{folder}/plot.pdf', dpi=600)

	plt.show()


def plot_multiple_fidelity(directory):
	fig, ax = plt.subplots(figsize=(12, 8))
	ax.set_xlabel(r'$\beta$')
	ax.set_ylabel(r'$F$')
	ax.grid(visible=True, which='major', axis='both')

	for folder in filter(lambda x: isdir(f'{directory}/{x}'), listdir(directory)):
		n = None
		beta = []
		fidelity = []
		for file in filter(lambda x: x.endswith('.json'), listdir(f'{directory}/{folder}')):
			with open(f'{directory}/{folder}/{file}', 'r') as f:
				data = json.load(f)
			n = data['metadata']['n']
			beta.append(data['metadata']['beta'])
			fidelity.append(data['metrics']['calculated_fidelity'])

		ax.scatter(beta, fidelity, label=f'{n}-qubits')

	ax.legend()
	fig.savefig(f'{directory}/fidelity_plot.pdf', dpi=600)

	plt.show()


if __name__ == '__main__':
	# plot_multiple_beta('qulacs/data/exact_n_2_J_1.00_h_0.50')
	plot_multiple_fidelity('qulacs/data')
# plot_density('qiskit_runtime/jobs/ibmq_jakarta/ccvfuro9ujl45d6clog0', shiftedColorMap(cc.cm.CET_C8s, cycle=0.5))
