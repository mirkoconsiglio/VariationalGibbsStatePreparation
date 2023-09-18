from argparse import ArgumentParser

from gibbs_functions import print_multiple_results
from vgsp_ising_program import main

if __name__ == '__main__':
	parser = ArgumentParser(description='VGSP')
	parser.add_argument('--n', default=2, type=int)
	parser.add_argument('--N', default=1, type=int)
	parser.add_argument('--beta', default=1., type=float, nargs='+')
	parser.add_argument('--h', default=0.5, type=float)
	parser.add_argument('--shots', default=10000, type=int)
	parser.add_argument('--noise_model', default='ibm_nairobi', type=str)
	parser.add_argument('--folder', default='test_job', type=str)
	args = vars(parser.parse_args())
	folder = args.pop('folder')

	results = main(**args)

	print_multiple_results(results, output_folder=folder, zip_file=False, hamiltonian='Ising')
