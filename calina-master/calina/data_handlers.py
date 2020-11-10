from abc import ABC, abstractmethod
import pandas as pd
from calibration_dataset import DatasetTree, LazyDataset
from tools.misc import merge_two_dicts
import numpy as np

class SensorNotFoundException(ValueError):

	def __init__(self, sensor_number):
		ValueError.__init__(self, 'Sensor {} was not found in processed dataset'.format(sensor_number))
		self.sensor_number = sensor_number

class HandlerInterface(ABC):

	@classmethod
	def class_key_iterator(cls, *args):
		"""
		:return: Iterator,which returns dict
		"""
		pass

	@property
	@abstractmethod
	def task_type(self):
		pass

	@abstractmethod
	def key_parameters(self)->dict:
		pass

	@abstractmethod
	def sane_check_data(self):
		pass

	@abstractmethod
	def set_key_parameters(self, params: dict):
		pass


class DataHandler(HandlerInterface, ABC):

	def __init__(self, dataset: LazyDataset = None):
		self.dataset = dataset
		self._data_slice = None
		self._key_parameters = None
		self.values_start = 0


	def key_parameters(self)->dict:
		if self._key_parameters is None:
			raise ValueError('You must set key_parameter value after each use.')
		return self._key_parameters

	def set_key_parameters(self, params: dict):
		if not isinstance(params, dict):
			raise ValueError('Only dict type allowed.')
		self._key_parameters = params

	@property
	def data_slice(self):
		if self.dataset.df is None:
			self.dataset.read()
		return self.dataset.process()

	@staticmethod
	def get_values(obj, start):
		return obj.iloc[:, start:].values

	@property
	def data_values(self) -> pd.DataFrame:
		return DataHandler.get_values(self.data_slice, self.values_start)

	def iterator(self):
		for k_dict in self.class_key_iterator():
			self._key_parameters = k_dict
			yield k_dict


class HandlerByDay(DataHandler):
	_key_list = None


	@property
	def task_type(self):
		return 'by_day'

	@classmethod
	def class_key_iterator(cls):
		if cls._key_list is None:
			cls._key_list = []
			for sensor_type in DatasetTree.tree['sensor']['sensor_type']:
				cls._key_list.append(dict(sensor_type=sensor_type))
		return iter(cls._key_list)


	def sane_check_data(self):
		assert len(self.data_slice.sensor_type.unique()) == 1, "Only one type of sensor allowed"


class HandlerBySensor(DataHandler):
	_key_list = None

	@property
	def task_type(self):
		return 'by_sensor'

	@classmethod
	def class_key_iterator(cls):
		if cls._key_list is None:
			cls._key_list = []
			for sensor_number in DatasetTree.tree['sensor']['sensor_number']:
				cls._key_list.append(dict(sensor_number=sensor_number))
		return iter(cls._key_list)

	@property
	def mask(self):
		if self.dataset.mask is None:
			self.dataset.read()
		return self.dataset.mask[self.dataset.mask['sensor_number']==self.key_parameters()['sensor_number']]

	@property
	def mask_values(self):
		return self.get_values(self.mask, self.values_start)[0]

	@property
	def data_values(self):
		values = DataHandler.data_values.fget(self)
		if np.any(self.mask_values):
			return values[:,np.logical_not(self.mask_values)]
		else:
			return values

	@property
	def data_slice(self):
		slice = DataHandler.data_slice.fget(self)
		param_sensor_number = self.key_parameters()['sensor_number']
		if param_sensor_number not in slice['sensor_number'].unique():
			raise SensorNotFoundException(param_sensor_number)
		self._data_slice = slice[ slice['sensor_number'] == param_sensor_number ]
		return self._data_slice

	def sane_check_data(self):
		assert self.data_slice.shape[0] == 1, "Only single rows allowed"


class HandlerByLink(HandlerBySensor):
	_dict_key_list = {}

	def __init__(self,
				 dataset: LazyDataset,
				 values_start,
				 link_end,
				 link_size):
		DataHandler.__init__(self, dataset)
		self.link_size = 0
		self.values_start = values_start
		self.link_size = link_size
		self.link_end = link_end
		self._prev_key_params = None
		self.descriptor_range = range(self.values_start)

	@classmethod
	def class_key_iterator(cls, link_end, link_size):
		indeces = link_end, link_size
		if cls._dict_key_list.get(indeces,None) is None:
			cls._dict_key_list[indeces] = []
			for sensor_number in HandlerBySensor.class_key_iterator():
				for i in range(0, link_end - link_size, link_size):
					cls._dict_key_list[indeces].append(
						dict(
							sensor_number=sensor_number['sensor_number'],
							link_start=i,
							link_end=i+link_size,
						)
					)
		return iter(cls._dict_key_list[indeces])

	def iterator(self):
		for k_dict in self.class_key_iterator(self.link_end, self.link_size):
			self._key_parameters = k_dict
			yield k_dict

	@property
	def task_type(self):
		return 'by_link'

	@property
	def data_slice(self):
		if self._key_parameters != self._prev_key_params:
			original_slice = HandlerBySensor.data_slice.fget(self)
			key_params = self._key_parameters
			link_start, link_end = key_params['link_start'], key_params['link_end']
			link_start, link_end = self.values_start + link_start, self.values_start + link_end
			link_data_range = range(link_start,link_end)
			data_slice_range = list(self.descriptor_range)+list(link_data_range)
			self._data_slice = original_slice.iloc[:, data_slice_range]
			self._prev_key_params = self._key_parameters
		return self._data_slice

	def sane_check_data(self):
		assert self.data_slice.shape[0] == 1, "Only single rows allowed"
		actual_shape = self.data_values.shape[1]
		link_size_message = "Data does not fit link size, expected {}, received {}".format(self.link_size, actual_shape)
		assert  actual_shape == self.link_size, link_size_message


class DayKeyHandler(DataHandler):

	@staticmethod
	def get_day_as_string(dataset: LazyDataset) -> str:
		timestamp = dataset.filename()
		return timestamp

	def key_parameters(self):
		return dict(day=self.get_day_as_string(self.dataset))

	def sane_check_data(self):

		assert len(self.data_slice.datetime.unique()) == 1, "Only single day callibration allowed"

	@property
	def task_type(self):
		return 'day_key'


class HandlerTuple(HandlerInterface):
	def __init__(self, data_handler_tuple: tuple):
		self.data_handler_tuple = data_handler_tuple

	def key_parameters(self):
		keys = [handler.key_parameters() for handler in self.data_handler_tuple]
		return merge_two_dicts(*keys)

	def sane_check_data(self):
		assert len(self.data_handler_tuple) == 2
		for handler in self.data_handler_tuple:
			handler.sane_check_data()

	def set_key_parameters(self, params: dict):
		pass


class BiasedHandlerTuple(DataHandler, HandlerTuple):
	FirstHandlerClass = DayKeyHandler

	@classmethod
	def from_data_handler(cls, handler: DataHandler, values_start=None):
		handler.values_start = values_start
		dkh = cls.FirstHandlerClass(handler.dataset)
		dkh._data_slice = handler._data_slice
		handler_tuple = (dkh, handler)
		obj = BiasedHandlerTuple(handler_tuple)
		obj.values_start = values_start
		return obj

	def __init__(self, handler_tuple):
		DataHandler.__init__(self, handler_tuple[1].dataset)
		HandlerTuple.__init__(self, handler_tuple)

	def set_key_parameters(self, params: dict):
		self.data_handler_tuple[1].set_key_parameters(params)

	def key_parameters(self):
		return HandlerTuple.key_parameters(self)

	def iterator(self):
		return self.data_handler_tuple[1].iterator()

	@property
	def task_type(self):
		return self.data_handler_tuple[1].task_type

	@property
	def data_slice(self):
		return self.data_handler_tuple[1].data_slice

	@property
	def data_values(self):
		return self.data_handler_tuple[1].data_values

	@property
	def mask_values(self):
		return self.data_handler_tuple[1].mask_values

	@property
	def mask(self):
		return self.data_handler_tuple[1].mask