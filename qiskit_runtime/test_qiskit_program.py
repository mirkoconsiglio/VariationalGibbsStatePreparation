from qiskit.providers.aer import AerSimulator

from job_results import print_results
from vgsp_ising_program import main

backend = AerSimulator()
results = main(backend=backend, beta=1., adiabatic_assistance=True)
print_results(results)
