# Variational Gibbs State Preparation

Code accompanying the paper: [_Variational Gibbs State Preparation on NISQ devices_](https://arxiv.org/abs/2303.11276).

Run statevector/shot-based/noisy/real device simulations for preparing the Gibbs state of transverse field Ising model.

## Statevector Simulations

Statevector and shot-based classical simulations are carried out
using [Qulacs](https://github.com/qulacs/qulacs). [`main_qulacs.py`](qulacs/main_qulacs.py) is the starting point where:

- `n` is the number of qubits
- `J` is the interaction strength of the Ising model
- `h` is the magnetic field strength of the Ising model
- `beta` is the inverse temperature
- `shots` is the number of shots for shot-based simulations
- `seed` to set the seed
- `ancilla_reps` for selecting the number of layers for the ancilla unitary
- `system_reps` for selecting the number of layers for the system unitary
- `commuting_terms` is a boolean specifying whether to group commuting terms in the Hamiltonian for measurements
- `N` represents the number of runs of the local optimizer
- `optimizer` for choosing the optimizer,
  from [Qiskit's library](https://qiskit.org/documentation/stubs/qiskit.algorithms.optimizers.html)
- `min_kwargs` are the keyword arguments for the optimizer

Will automatically print and plot the results after finishing the simulations

## Noisy and Real Device Simulations

Noisy and real device simulations are carried out using [Qiskit](https://qiskit.org/). `test_qiskit_program.py` is for
testing the program on your local machine with similar inputs as the Qulacs code. Also accepts a `noise_model` by
supplying either: a dictionary of a
Qiskit [`NoiseModel`](https://qiskit.org/documentation/stubs/qiskit_aer.noise.NoiseModel.html#qiskit_aer.noise.NoiseModel);
or else a string with the name of the IBM quantum computer (requires an authenticated IBM account on your device).

[`upload_qiskit_program.py`](/qiskit_runtime/upload_qiskit_program.py) uploads a Qiskit Runtime Program to your account,
which can then be run using [`vgsp_launcher.py`](/qiskit_runtime/vgsp_launcher.py), with similar inputs as the Qulacs
code. [`job_results.py`](qiskit_runtime/job_results.py) retrieves completed jobs and prints/plots the results.
