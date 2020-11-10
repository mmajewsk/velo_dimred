import logging
import os


os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

try:
	from matplotlib.cbook import boxplot_stats
except ImportError:
	from tools.cbook_snippet import boxplot_stats

logging.basicConfig(format='%(asctime)s [%(levelname)s]= %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import warnings
import numpy as np
from calibration_dataset import CalinaLazyDataset, sensor_mapping, Tell1Dataset
from calina import HandlerByDay, HandlerBySensor,\
	HandlerByLink, DBAdapter, ScoreTask, StatsTask,\
	SensorNotFoundException
from calina.implementations import CalibHandlerTuple
from outlierness import read_parameters, Outlierness


class CalinaAnalyzer:
	DayClass = HandlerByDay
	SensorClass = HandlerBySensor
	LinkClass = HandlerByLink


	def __init__(self,
				 paths : list,
				 destination: str,
				 force_recalc: bool,
				 verbose: bool,
				 do_each_calibration: bool = False,
				 do_each_sensor: bool = False,
				 do_each_link: bool = False
				 ):
		self.paths = paths
		self.destination = destination
		self.force_recalc = force_recalc
		self.verbose = verbose
		self.params = read_parameters()
		self.db_path = r'sqlite:///' + self.destination
		self.db_adapter = DBAdapter(self.db_path)
		self.do_each_calibration = do_each_calibration
		self.do_each_sensor = do_each_sensor
		self.do_each_link = do_each_link
		self.multiprocessing = False
		self.task_name = 'calib'
		self.work_params ={
			'by_day': dict(draws=8000, tune=2000),
			'by_sensor': dict(draws=800, tune=200),
			'by_link': dict(draws=80, tune=20)
		}

	def score_and_stat(self, out, handler):
		task = ScoreTask(out, handler, db_adapter=self.db_adapter, task_name=self.task_name, recalculate=self.force_recalc)
		task_type = handler.task_type
		score_data = task.work(**self.work_params[task_type])
		if score_data is None:
			score_data = task.existing
			score_data['score'] = np.fromstring(score_data['score'])
		stat_task = StatsTask.from_ScoreTask(task)
		stat_task.work(score_data, **self.work_params[handler.task_type])

	def each_calibration(self, ds, outlierness_params, sensor_type):
		ds2 = CalinaLazyDataset(ds.dataset, ds.sensor_type, dict(cut_val=50))
		handler = self.DayClass(ds2)
		new_handler = CalibHandlerTuple.from_data_handler(handler)
		new_handler.set_key_parameters(dict(sensor_type=sensor_type))
		out = Outlierness(verbose=self.verbose, **outlierness_params)
		self.score_and_stat(out, new_handler)

	def each_sensor(self, ds, outlierness_params):
		out = Outlierness(verbose=self.verbose, **outlierness_params)
		handler = self.SensorClass(ds)
		new_handler = CalibHandlerTuple.from_data_handler(handler)
		for iterator in new_handler.iterator():
			sensor_number = iterator['sensor_number']
			sensor_type = sensor_mapping[int(sensor_number[1:])]
			if sensor_type == ds.sensor_type:
				try:
					if np.any(new_handler.mask_values):
						mask = np.logical_not(new_handler.mask_values)
						out2 = out[mask]
					else:
						out2 = out[:]
					self.score_and_stat(out2, new_handler)
				except SensorNotFoundException as e:
					warn_msg = '{} Skipping calculations for {} sensor.'.format(e,e.sensor_number)
					warnings.warn(warn_msg)


	def each_link(self, ds, outlierness_params, link_size=31):
		raise NotImplemented
		out = Outlierness(verbose=self.verbose, **outlierness_params)
		handler = self.LinkClass(ds, values_start=9, link_size=link_size, link_end=1984)
		new_handler = CalibHandlerTuple.from_data_handler(handler)
		for i, keys in enumerate(new_handler.iterator()):
			sensor_number = keys['sensor_number']
			if sensor_mapping[int(sensor_number[1:])] == ds.sensor_type:
				link_range = range(keys['link_start'], keys['link_end'])
				link_outlierness = out[link_range]
				sensor_type = sensor_mapping[int(sensor_number[1:])]
				if sensor_type == ds.sensor_type:
					try:
						#@TODO this is not fixed
						if np.any(new_handler.mask_values):
							mask = np.logical_not(new_handler.mask_values)
							out2 = out[mask]
						else:
							out2 = out[:]
						self.score_and_stat(link_outlierness, new_handler)
					except SensorNotFoundException as e:
						warn_msg = '{} Skipping calculations for {} sensor.'.format(e, e.sensor_number)
						warnings.warn(warn_msg)



	def go(self):
		StatsTask.create_stats_tables(self.db_adapter)
		for calibration in self.paths:
			t1d = Tell1Dataset([calibration])
			for sensor_type in ['R', 'phi']:
				ds = CalinaLazyDataset(t1d, sensor_type)
				outlierness_params = self.params[sensor_type]
				if self.do_each_calibration:
					self.each_calibration(ds, outlierness_params, sensor_type)
				if self.do_each_sensor:
					self.each_sensor(ds, outlierness_params)
				if self.do_each_link:
					self.each_link(ds, outlierness_params)