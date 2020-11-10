import sys
import argparse

from analyzers.calibration import CalinaAnalyzer
from analyzers.noise import NoiseAnalyzer
from calibration_dataset import Tell1Dataset, SingleRunDataset


def run(AnalyzerClass, DatasetClass,args):
	if args.run_list:
		paths = DatasetClass.get_filepaths_from_runlist(args.source)
	else:
		paths = DatasetClass.get_filepaths_from_dir(args.source)
	paths = list(reversed(sorted(paths)))
	agg = AnalyzerClass(
		paths,
		args.destination,
		args.force_recalc,
		args.progress_bar,
		do_each_calibration=args.do_calibration,
		do_each_sensor=args.do_sensor,
		do_each_link=args.do_link
	)
	agg.go()


def main(args):
	if args.noise:
		AnalyzerClass = NoiseAnalyzer
		DatasetClass = SingleRunDataset
	else:
		AnalyzerClass = CalinaAnalyzer
		DatasetClass = Tell1Dataset
	run(AnalyzerClass, DatasetClass, args)
	return 0


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description=""""
		Calculates the outlierness for calibration of VeLo.
		By default it will not recalculate existing callibrations (those that exist in 'by_day' table in database.
		""")
	parser.add_argument('source', type=str, help='path to calibration database dump directory or runlist')
	parser.add_argument('--noise', dest='noise', action='store_true', default=False,
						help='run for noise data')
	parser.add_argument('-d', dest='destination', type=str, help='path to database', default='database.db')
	parser.add_argument('--run-list', dest='run_list', action='store_true',
						help='source is path to RunList.txt (ending with \'RunList.txt\')'
						)
	parser.add_argument('--progress-bar', dest='progress_bar', action='store_true', default=False,
						help='whether to display pymc3\'s progress bar)'
						)
	parser.add_argument('--force_recalculation', dest='force_recalc', action='store_true', default=False,
						help='force recalculation of the callibrations that already exists in calina database'
						)
	parser.add_argument('--do_calibration', dest='do_calibration', action='store_true', default=False,
						help='do calculation of outlierness per celibration (R and phi)'
						)
	parser.add_argument('--do_sensor', dest='do_sensor', action='store_true', default=False,
						help='do calculation of outlierness per each sensor'
						)
	parser.add_argument('--do_link', dest='do_link', action='store_true', default=False,
						help='do calculation of outlierness per each link (WARNING! MIGHT BE TIME CONSUMING)'
						)
	args = parser.parse_args()


	sys.exit(main(args))
