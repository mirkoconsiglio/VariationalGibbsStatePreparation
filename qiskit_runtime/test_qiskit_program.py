from qiskit import IBMQ
from qiskit_aer.noise import NoiseModel

from gibbs_functions import print_multiple_results
from vgsp_ising_program import main

noise_model = NoiseModel.from_backend(IBMQ.load_account().get_backend('ibmq_jakarta')).to_dict()

results = main(N=2, beta=[1., 2.])

print_multiple_results(results)
