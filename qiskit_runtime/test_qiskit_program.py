from gibbs_functions import print_multiple_results
from plotter import plot_result_min_avg_max
from vgsp_ising_program import main

noise_model = 'ibmq_jakarta'

results = main(n=2, N=1, beta=[1.], noise_model=noise_model)

folder = 'test_job'

print_multiple_results(results, output_folder=folder)

plot_result_min_avg_max(folder)
