from argparse import ArgumentParser

from gibbs_functions import print_multiple_results
from vgsp_ising_program import main

if __name__ == '__main__':
	parser = ArgumentParser(description='VGSP')
	parser.add_argument('--n', default=3, type=int)
	parser.add_argument('--N', default=1, type=int)
	parser.add_argument('--beta', default=0.5, type=float, nargs='+')
	parser.add_argument('--noise_model', default='ibmq_guadalupe', type=str)
	parser.add_argument('--folder', default='test_job', type=str)
	args = vars(parser.parse_args())
	folder = args.pop('folder')

	results = main(**args)

	print_multiple_results(results, output_folder=folder, zip_file=False)
