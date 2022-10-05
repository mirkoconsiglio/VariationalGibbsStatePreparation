from qiskit import IBMQ
from qiskit.providers.aer import AerSimulator

from job_results import print_results
from vgsp_ising_program import main

backend = AerSimulator.from_backend(IBMQ.load_account().get_backend('ibmq_jakarta'))
results = main(backend=backend)
print_results(results)
