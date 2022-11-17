from qiskit_ibm_runtime import QiskitRuntimeService

if __name__ == '__main__':
	# Program metadata
	meta = dict(
		name="vgsp_ising",
		description="Variational Gibbs State Preparation (VGSP) for the Ising model",
		max_execution_time=86400,  # 24 hours
		spec=dict()
	)
	# Input Parameters
	meta["spec"]["parameters"] = {
		"$schema": "https://json-schema.org/draft/2019-09/schema",
		"properties": {
			"backend": {
				"description": "What backend to use, required.",
				"type": "str"
			},
			"user_messenger": {
				"description": "What type of user messenger to use to retrieve interim data, defaults to "
							   "UserMessenger.",
				"type": "UserMessenger"
			},
			"n": {
				"description": "number of qubits of the Ising model.",
				"type": "int",
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
				"description": "Inverse temperature beta of the thermal state you want to determine, can be a list of "
							   "betas, default is 1.",
				"type": "Union[float, list[float]]",
				"default": None
			},
			"N": {
				"description": "Number of repeated runs to perform for each beta.",
				"type": "int",
				"default": 1
			},
			"ancilla_reps": {
				"description": "Number of layer repetitions for the ancilla unitary.",
				"type": "Optional[int]",
				"default": 1
			},
			"system-reps": {
				"description": "Number of layer repetitions for the system unitary.",
				"type": "Optional[int]",
				"default": 1
			},
			"x0": {
				"description": "Initial vector of parameters. This is a numpy array, default is random parameters "
							   "between -π and π.",
				"type": "Optional[list[float]]",
				"default": None
			},
			"optimizer": {
				"description": "Qiskit optimizer to use, defaults to SPSA.",
				"type": "Optional[str]",
				"default": None
			},
			"min_kwargs": {
				"description": "kwargs for the optimizer.",
				"type": "Optional[dict]",
				"default": None
			},
			"shots": {
				"description": "The number of shots used for each circuit evaluation.",
				"type": "int",
				"default": 1024
			},
			"use_measurement_mitigation": {
				"description": "Use measurement mitigation using the M3 package, default is True.",
				"type": "bool",
				"default": True,
			},
			"skip_transpilation": {
				"description": "Skip backend transpilation, default is False.",
				"type": "bool",
				"default": False,
			},
			"noise_model": {
				"description": "Optional noise model: when a string get noise model of backend; or else directly "
							   "supply noise model dictionary",
				"type": "Optional[Union[str, dict]]",
				"default": None,
			},
			"credentials": {
				"description": "dictionary of credentials when using noise model supplied by string",
				"type": "Optional[dict]",
				"default": None,
			}
		}
	}
	# Output results
	meta["spec"]["return_values"] = {
		"$schema": "https://json-schema.org/draft/2019-09/schema",
		"description": "Final result containing dicts of parameters, cost, energy, entropy, eigenvalues, iterations "
					   "and function evaluations, among others.",
		"type": "array",
	}
	# Interim results
	meta["spec"]["interim_results"] = {
		"$schema": "https://json-schema.org/draft/2019-09/schema",
		"description": "dict of results such as parameters, cost, energy, entropy, eigenvalues, iterations and "
					   "function evaluations.",
		"type": "dict",
	}
	
	service = QiskitRuntimeService()
	
	service.update_program(program_id='vgsp-ising-qGq4q73MaV', data='vgsp_ising_program.py')
	print("Program updated")

# program_id = service.upload_program(data='vgsp_ising_program.py', metadata=meta)
# print("Program uploaded")
# print(f"Program ID: {program_id}")
