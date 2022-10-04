from qiskit_ibm_runtime import QiskitRuntimeService

if __name__ == '__main__':
	# Program metadata
	meta = dict(
		name="vgsp_ising",
		description="Variational Gibbs State Preparation (VGSP) for the Ising model",
		max_execution_time=100000,
		spec=dict()
	)
	# Input Parameters
	meta["spec"]["parameters"] = {
		"$schema": "https://json-schema.org/draft/2019-09/schema",
		"properties": {
			"backend": {
				"description": "What backend to use, required.",
				"type": "string",
				"default": None
			},
			"user_messenger": {
				"description": "What type of user messenger to use to retrieve"
				               "interim data, defaults to UserMessenger().",
				"type": "integer",
				"default": None
			},
			"n": {
				"description": "number of qubits of the Ising model.",
				"type": "integer",
				"default": 2
			},
			"J": {
				"description": "Coupling constant of the Ising model.",
				"type": "float",
				"default": 1.
			},
			"h": {
				"description": "Magnetic field strength of the Ising model.",
				"type": "float",
				"default": 0.5
			},
			"beta": {
				"description": "Inverse temperature beta of the thermal state you want"
				               "to determine, can be a list of betas. Default is"
				               "[1e-8, 0.2, 0.5, 0.8, 1., 1.2, 2., 5.].",
				"type": "float | list[float]",
				"default": None
			},
			"ancilla_reps": {
				"description": "Number of layer repetitions for the ancilla unitary.",
				"type": "integer",
				"default": 1
			},
			"system-reps": {
				"description": "Number of layer repetitions for the system unitary.",
				"type": "integer",
				"default": 1
			},
			"x0": {
				"description": "Initial vector of parameters. This is a numpy array, "
				               "default is random parameters between 0 and 2π.",
				"type": "array",
				"default": None
			},
			"optimizer": {
				"description": "Classical optimizer to use, default is SPSA().",
				"type": "Qiskit Optimizer",
				"default": None
			},
			"shots": {
				"description": "The number of shots used for each circuit evaluation.",
				"type": "integer",
				"default": 1024
			},
			"use_measurement_mitigation": {
				"description": "Use measurement mitigation using the M3 package, default is True.",
				"type": "boolean",
				"default": True,
			},
			"skip_transpilation": {
				"description": "Skip backend transpilation, default is False.",
				"type": "boolean",
				"default": False,
			},
			"adiabatic_assistance": {
				"description": "Whether to use adiabatic assistance, that is the"
				               "parameters of the previous beta are used for the"
				               "next beta, default is False.",
				"type": "boolean",
				"default": False,
			}
		}
	}
	# Output results
	meta["spec"]["return_values"] = {
		"$schema": "https://json-schema.org/draft/2019-09/schema",
		"description": "Final result containing list of parameters, cost, energy, entropy, eigenvalues, "
		               "iterations and function evaluations, among others.",
		"type": "array",
	}
	# Interim results
	meta["spec"]["interim_results"] = {
		"$schema": "https://json-schema.org/draft/2019-09/schema",
		"description": "List of results such as parameters, cost, energy, entropy, eigenvalues, "
		               "iterations and function evaluations.",
		"type": "array",
	}

	service = QiskitRuntimeService()

	service.update_program(program_id='vgsp-ising-xVBvWpWqyV', data='vgsp_ising_program.py')
	print("Program updated")

# program_id = service.upload_program(data='vgsp_ising_program.py', metadata=meta)
# print("Program uploaded")
# print(f"Program ID: {program_id}")
