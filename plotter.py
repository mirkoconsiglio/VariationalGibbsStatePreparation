import json
from os import listdir
from os.path import isdir

# noinspection PyUnresolvedReferences
import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc, cm
from matplotlib.colors import Normalize, LinearSegmentedColormap
from qiskit_ibm_runtime import RuntimeDecoder

rc('font', **{'family': 'Times New Roman', 'sans-serif': ['Times New Roman'], 'size': 18})
rc('text', usetex=True)

scale = 0.6


def forward(x):
	p = 0.6
	return np.array([i ** p if i >= 0 else -((-i) ** p) for i in x])


def reverse(x):
	p = 1 / scale
	return np.array([i ** p if i >= 0 else -((-i) ** p) for i in x])


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, cycle=0., name='shiftedcmap'):
	cdict = {
		'red': [],
		'green': [],
		'blue': [],
		'alpha': []
	}

	# regular index to compute the colors
	reg_index = (np.linspace(start, stop, 257) + cycle) % 1

	# shifted index to match the data_h_0.5
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
	ax.set_xticks(range(1, 5), ['00', '01', '10', '11'])
	ax.set_yticks(range(1, 5), ['00', '01', '10', '11'])
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


def plot_density(folder, cmap=shiftedColorMap(cc.cm.CET_C8s, cycle=0.5), reverse=False, show=True):
	if reverse:
		shape = (2, 3)
	else:
		shape = (3, 2)
	fig = plt.figure(figsize=tuple([3 * i for i in shape[::-1]]))
	gs = fig.add_gridspec(*shape)
	axes = np.array([fig.add_subplot(gs[j, i], projection='3d') for j in range(shape[0])
	                 for i in range(shape[1])]).reshape(*shape)

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

		if reverse:
			ax = axes[1, i]
		else:
			ax = axes[i, 0]
		exact_rho = np.asarray(data['exact_result']['gibbs_state'])
		dz, color = complex_to_size_and_color(exact_rho, cmap, -np.pi, np.pi)
		set_bar3d(ax, xpos, ypos, zpos, dx, dy, dz, color)
		if i == 0 or reverse:
			if reverse:
				x, y = 0.45, 0.95
			else:
				x, y = 0.45, 0.95
			ax.text2D(x, y, r'$\rho_\textrm{\small{Gibbs}}$', transform=ax.transAxes, fontsize=20)
		if reverse:
			ax.text2D(-0.05, 0.94, fr'$\beta = {int(beta)}$', transform=ax.transAxes, fontsize=20)
		else:
			ax.text2D(-0.17, 0.85, fr'$\beta = {int(beta)}$', transform=ax.transAxes, fontsize=20)

		if reverse:
			ax = axes[0, i]
		else:
			ax = axes[i, 1]
		calculated_rho = np.asarray(data['calculated_result']['rho'])
		dz, color = complex_to_size_and_color(calculated_rho, cmap, -np.pi, np.pi)
		set_bar3d(ax, xpos, ypos, zpos, dx, dy, dz, color)
		if i == 0 or reverse:
			if reverse:
				x, y = 0.46, 0.95
			else:
				x, y = 0.46, 0.95
			ax.text2D(x, y, r'$\rho_\textrm{\small{Calc}}$', transform=ax.transAxes, fontsize=20)

		i += 1

	fig.colorbar(mapper(-np.pi, np.pi, cmap), ax=axes[shape[0] - 1, :], shrink=0.6, orientation='horizontal',
	             ticks=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi], format=rad_fn, label='phase')

	if reverse:
		d = dict(left=0.05, bottom=0.2, right=0.95, top=1, wspace=0.1, hspace=0.1)
	else:
		d = dict(left=0.05, bottom=0.17, right=0.95, top=1, wspace=0.1, hspace=0.05)
	fig.subplots_adjust(**d)

	fig.savefig(f'Bar3D.pdf', bbox_inches='tight')

	if show:
		plt.show()


def plot_result_min_avg_max(folder, show=True):
	fig, ax = plt.subplots(figsize=(12, 8))
	ax.set_xscale('function', functions=(forward, reverse))
	ax.set_xlabel(r'$\beta$')
	ax.set_ylabel(r'$F$')
	ax.set_xlim(-0.02, 5.3)
	ax.grid(visible=True, which='both', axis='both')

	beta = []
	min_fidelity = []
	avg_fidelity = []
	max_fidelity = []
	for file in filter(lambda x: x.endswith('.json'), listdir(folder)):
		with open(f'{folder}/{file}', 'r') as f:
			data = json.load(f)
		beta.append(data['metadata']['beta'])

		fidelity = data['metrics']['noiseless_fidelity']

		min_fidelity.append(np.min(fidelity))
		avg_fidelity.append(np.average(fidelity))
		max_fidelity.append(np.max(fidelity))

	ax.scatter(beta, min_fidelity, label=f'minimum')
	ax.scatter(beta, avg_fidelity, label=f'average')
	ax.scatter(beta, max_fidelity, label=f'maximum')

	ax.legend()

	fig.savefig(f'{folder}/fidelity_plot.pdf', dpi=600, transparent=True)

	if show:
		plt.show()


def plot_multiple_results_max(directory, show=True):
	fig1, ax1 = plt.subplots(figsize=(12, 8))
	ax1.set_xscale('function', functions=(forward, reverse))
	ax1.set_xlabel(r'$\beta$')
	ax1.set_ylabel(r'$F$')
	ax1.set_xlim(-0.02, 5.3)
	ax1.set_xticks([0, 0.2, 0.5, 1, 2, 3, 4, 5])
	ax1.grid(visible=True, which='both', axis='both')

	fig2, ax2 = plt.subplots(figsize=(12, 8))
	ax2.set_xscale('function', functions=(forward, reverse))
	ax2.set_xlabel(r'$\beta$')
	ax2.set_ylabel(r'Purity Difference')
	ax2.set_xlim(-0.02, 5.3)
	ax2.set_xticks([0, 0.2, 0.5, 1, 2, 3, 4, 5])
	ax2.grid(visible=True, which='both', axis='both')

	fig3, ax3 = plt.subplots(figsize=(12, 8))
	ax3.set_xscale('function', functions=(forward, reverse))
	ax3.set_xlabel(r'$\beta$')
	ax3.set_ylabel(r'$D_\textrm{KL}(P || Q)$')
	ax3.set_xlim(-0.02, 5.3)
	ax3.set_xticks([0, 0.2, 0.5, 1, 2, 3, 4, 5])
	ax3.grid(visible=True, which='both', axis='both')

	fig4, ax4 = plt.subplots(figsize=(12, 8))
	ax4.set_xscale('function', functions=(forward, reverse))
	ax4.set_xlabel(r'$\beta$')
	ax4.set_ylabel(r'$F$')
	ax4.set_xlim(-0.02, 5.3)
	ax4.set_xticks([0, 0.2, 0.5, 1, 2, 3, 4, 5])
	ax4.grid(visible=True, which='both', axis='both')

	fig5, ax5 = plt.subplots(figsize=(12, 8))
	ax5.set_xscale('function', functions=(forward, reverse))
	ax5.set_xlabel(r'$\beta$')
	ax5.set_ylabel(r'Purity Difference')
	ax5.set_xlim(-0.02, 5.3)
	ax5.set_xticks([0, 0.2, 0.5, 1, 2, 3, 4, 5])
	ax5.grid(visible=True, which='both', axis='both')

	fig6, ax6 = plt.subplots(figsize=(12, 8))
	ax6.set_xscale('function', functions=(forward, reverse))
	ax6.set_xlabel(r'$\beta$')
	ax6.set_ylabel(r'$D_\textrm{KL}(P || Q)$')
	ax6.set_xlim(-0.02, 5.3)
	ax6.set_xticks([0, 0.2, 0.5, 1, 2, 3, 4, 5])
	ax6.grid(visible=True, which='both', axis='both')

	markers = ['o', 's', 'd', '^']

	h = None
	for m, folder in zip(markers, filter(lambda x: isdir(f'{directory}/{x}'), listdir(directory))):
		n = None
		beta = []

		fidelity = []
		fiderr = []
		purity = []
		purerr = []
		kldivergence = []
		klerr = []

		noiseless_fidelity = []
		noiseless_purity = []
		noiseless_kldivergence = []

		for file in filter(lambda x: x.endswith('.json'), listdir(f'{directory}/{folder}')):
			with open(f'{directory}/{folder}/{file}', 'r') as f:
				data = json.load(f)
			n = data['metadata']['n']
			h = data['metadata']['h']
			std_err = 1. / np.sqrt(data['metadata']['shots'])

			beta.append(data['metadata']['beta'])
			fidelity.append(np.max(data['metrics']['calculated_fidelity']))
			fiderr.append(std_err)
			purity.append(
				np.min(np.abs(np.asarray(data['metrics']['calculated_purity']) - data['metrics']['exact_purity'])))
			purerr.append(std_err)
			kldivergence.append(np.min(data['metrics']['calculated_kullback_leibler_divergence']))
			klerr.append(std_err)

			noiseless_fidelity.append(np.max(data['metrics']['noiseless_fidelity']))
			noiseless_purity.append(
				np.min(np.abs(np.asarray(data['metrics']['noiseless_purity']) - data['metrics']['exact_purity'])))
			noiseless_kldivergence.append(np.min(data['metrics']['noiseless_kullback_leibler_divergence']))

		ax1.errorbar(beta, fidelity, yerr=fiderr, capsize=5, marker=m, label=f'$n={n}$')
		ax2.errorbar(beta, purity, yerr=purerr, capsize=5, marker=m, label=f'$n={n}$')
		ax3.errorbar(beta, kldivergence, yerr=klerr, capsize=5, marker=m, label=f'$n={n}$')

		ax4.plot(beta, noiseless_fidelity, marker=m, label=f'$n={n}$')
		ax5.plot(beta, noiseless_purity, marker=m, label=f'$n={n}$')
		ax6.plot(beta, noiseless_kldivergence, marker=m, label=f'$n={n}$')

	ax1.legend()
	ax2.legend()
	ax3.legend()

	ax4.legend()
	ax5.legend()
	ax6.legend()

	fig1.savefig(f'{directory}/fidelity_plot_{h:.2f}.pdf', bbox_inches='tight')
	fig1.savefig(f'{directory}/fidelity_plot_{h:.2f}.png', dpi=600, transparent=True, bbox_inches='tight')

	fig2.savefig(f'{directory}/purity_plot_{h:.2f}.pdf', bbox_inches='tight')
	fig2.savefig(f'{directory}/purity_plot_{h:.2f}.png', dpi=600, transparent=True, bbox_inches='tight')

	fig3.savefig(f'{directory}/relative_entropy_plot_{h:.2f}.pdf', bbox_inches='tight')
	fig3.savefig(f'{directory}/relative_entropy_plot_{h:.2f}.png', dpi=600, transparent=True, bbox_inches='tight')

	fig4.savefig(f'{directory}/noiseless_fidelity_plot_{h:.2f}.pdf', bbox_inches='tight')
	fig4.savefig(f'{directory}/noiseless_fidelity_plot_{h:.2f}.png', dpi=600, transparent=True, bbox_inches='tight')

	fig5.savefig(f'{directory}/noiseless_purity_plot_{h:.2f}.pdf', bbox_inches='tight')
	fig5.savefig(f'{directory}/noiseless_purity_plot_{h:.2f}.png', dpi=600, transparent=True, bbox_inches='tight')

	fig6.savefig(f'{directory}/noiseless_relative_entropy_plot_{h:.2f}.pdf', bbox_inches='tight')
	fig6.savefig(f'{directory}/noiseless_relative_entropy_plot_{h:.2f}.png', dpi=600, transparent=True,
	             bbox_inches='tight')

	if show:
		plt.show()


def plot_3multiple_results_max(directories, show=True):
	fig, axes = plt.subplots(1, 3, figsize=(18, 6))
	for i, ax in enumerate(axes):
		ax.set_xscale('function', functions=(forward, reverse))
		ax.set_xlabel(r'$\beta$')
		if i == 0:
			ax.set_ylabel(r'$F$')
		ax.set_xlim(-0.02, 5.3)
		ax.set_xticks([0, 0.2, 0.5, 1, 2, 3, 4, 5])
		ax.grid(visible=True, which='both', axis='both')

	markers = ['o', 's', 'd', '^']

	for ax, directory in zip(axes, directories):
		for m, folder in zip(markers, filter(lambda x: isdir(f'{directory}/{x}'), listdir(directory))):
			n = None
			beta = []
			fidelity = []
			for file in filter(lambda x: x.endswith('.json'), listdir(f'{directory}/{folder}')):
				with open(f'{directory}/{folder}/{file}', 'r') as f:
					data = json.load(f)
				n = data['metadata']['n']
				beta.append(data['metadata']['beta'])
				fidelity.append(data['metrics']['noiseless_fidelity'])

			ax.plot(beta, fidelity, marker=m, label=f'$n={n}$')

	for h, ax in zip(['0.5', '1.0', '1.5'], axes):
		ax.legend(title=fr'$h = {h}$')

	fig.savefig('fidelity_plot_3.pdf', bbox_inches='tight')
	fig.savefig('fidelity_plot_3.png', dpi=600, transparent=True, bbox_inches='tight')

	if show:
		plt.show()


if __name__ == '__main__':
	plot_multiple_results_max('qiskit_runtime/old_jobs/ibm_nairobi')
# plot_3multiple_results_max(['qulacs/data_h_0.5', 'qulacs/data_h_1.0', 'qulacs/data_h_1.5'])
# plot_density('qiskit_runtime/old_jobs/ibmq_jakarta_old/ccvfuro9ujl45d6clog0')
