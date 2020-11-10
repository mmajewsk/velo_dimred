"""
This scripts is not for reuse.
Its only goal is to show some data directly from database.

MM
"""

import argparse

import sys

from analyzers.noise import NoiseAnalyzer

sys.path.append('..')

import os
from analyzers.calibration import CalinaAnalyzer
from calibration_dataset import SingleRunDataset
from processing import remove_anomalies


class NoiseAnalyzerOld(CalinaAnalyzer):

	def __init__(self, *args, **kwargs):
		CalinaAnalyzer.__init__(self,*args, **kwargs)
		self.task_name = 'noise'

	def go(self):
		self.force_recalc=True
		CalinaAnalyzer.DayClass.create_stats_tables(self.db_path)
		for noise_path in self.paths:
			nsr = SingleRunDataset(noise_path)
			df = nsr.process()
			df = remove_anomalies(df, cut_val=None)
			for sensor_type in ['R', 'phi']:
				df_sensor = df[df.sensor_type==sensor_type]
				params = self.params[sensor_type]
				self.work_params['by_day'] = dict(draws=800, tune=200)
				#self.each_calibration(df_sensor, params)
				self.each_sensor(df_sensor, params)
				#self.each_link(df_sensor, params)


if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('path', type=str, help='path to files')
	parser.add_argument('--start', type=int, help='start run number', default=0)
	parser.add_argument('--stop', type=int, help='stop run number', default=sys.maxsize)
	args = parser.parse_args()
	dir_path = args.path
	paths = os.listdir(dir_path)
	name_filter = lambda x: (args.start < int(x.split("_")[0]) < args.stop)
	#name_filter = lambda x: (0 < int(x.split("_")[0]) < sys.maxsize)
	paths = list(filter(name_filter, paths))
	full_paths = [os.path.join(dir_path, path) for path in paths]
	na = NoiseAnalyzer(full_paths, 'databasex2.db', force_recalc=True, verbose=True, do_each_calibration=True)
	na.go()





