from qiskit import IBMQ
from qiskit_aer.noise import NoiseModel

from gibbs_functions import print_multiple_results
from plotter import plot_result_min_avg_max
from vgsp_ising_program import main

noise_model = NoiseModel.from_backend(IBMQ.load_account().get_backend('ibmq_jakarta')).to_dict()

results = main(n=2, N=1, beta=[1.], noise_model=noise_model)

print_multiple_results(results, output_folder='test_job')

plot_result_min_avg_max('test_job')
