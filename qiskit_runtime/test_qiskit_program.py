from qiskit import IBMQ
from qiskit_aer.noise import NoiseModel

from job_results import print_results
from vgsp_ising_program import main

results = main(noise_model=NoiseModel.from_backend(IBMQ.load_account().get_backend('ibmq_jakarta')).to_dict())
print_results(results)
