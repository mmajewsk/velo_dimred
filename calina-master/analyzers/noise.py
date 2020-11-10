from analyzers.calibration import CalinaAnalyzer
from calibration_dataset import SingleRunDataset
from calina import StatsTask
from calina.implementations import NoiseHandlerTuple
from outlierness import Outlierness



class NoiseAnalyzer(CalinaAnalyzer):

	def __init__(self, *args, **kwargs):
		CalinaAnalyzer.__init__(self, *args, **kwargs)
		self.task_name = 'noise'

	def each_calibration(self, ds, outlierness_params, sensor_type):
		handler = self.DayClass(ds)
		new_handler = NoiseHandlerTuple.from_data_handler(handler)
		new_handler.set_key_parameters(dict(sensor_type=sensor_type))
		out = Outlierness(verbose=self.verbose, **outlierness_params)
		self.score_and_stat(out, new_handler)

	def each_sensor(self, ds, outlierness_params):
		raise NotImplemented

	def each_link(self, ds, outlierness_params, link_size=31):
		raise NotImplemented

	def go(self):
		StatsTask.create_stats_tables(self.db_adapter)
		for noise_path in self.paths:
			for sensor_type in ['R', 'phi']:
				run_dataset = SingleRunDataset(noise_path, sensor_type)
				outlierness_params = self.params[sensor_type]
				if self.do_each_calibration:
					self.each_calibration(run_dataset, outlierness_params, sensor_type)
				if self.do_each_sensor:
					self.each_sensor(run_dataset, outlierness_params)
				if self.do_each_link:
					self.each_link(run_dataset, outlierness_params)