import unittest
from typing import Iterable
from unittest.mock import Mock, MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd

from calibration_dataset import DatasetTree
from calina import DataHandler, HandlerByDay, HandlerBySensor, HandlerByLink, DayKeyHandler, HandlerTuple


class MockDataHandler(DataHandler):

	@classmethod
	def key_iterator(cls, *args):
		pass

	@property
	def task_type(self):
		pass

	def sane_check_data(self):
		pass


class TestDataHandler(unittest.TestCase):

	def setUp(self):
		self.dataset_mock = Mock()
		self.test_object = MockDataHandler(self.dataset_mock)

	def test_init(self):
		self.assertEqual(self.test_object.dataset,self.dataset_mock)
		self.assertEqual(self.test_object._data_slice, None)
		self.assertEqual(self.test_object._key_parameters, None)

	def test_key_parameters(self):
		with self.assertRaises(ValueError):
			self.test_object.key_parameters()
		t_d = {'pramname': 'paramval'}
		self.test_object._key_parameters = {'pramname':'paramval'}
		result = self.test_object.key_parameters()
		self.assertEqual(t_d, result)
		self.test_object._key_parameters = None
		with self.assertRaises(ValueError):
			self.test_object.key_parameters()

	def test_set_key_parameters(self):
		self.test_object.set_key_parameters(dict(val='val'))
		self.assertEqual(self.test_object._key_parameters,dict(val='val'))

	def test_data_slice(self):
		self.test_object.dataset.read = MagicMock()
		self.test_object.dataset.df = None
		self.test_object.dataset.process = MagicMock(return_value='data_slice')
		result = self.test_object.data_slice
		self.assertEqual(result, 'data_slice')
		self.test_object.dataset.process.assert_called_once()
		self.test_object.dataset.read.assert_called_once()

		self.test_object.dataset.df = 'value'
		self.test_object.dataset.result = MagicMock(return_value='data_slice2')
		result = self.test_object.data_slice
		self.assertEqual(result, 'data_slice')


class TestHandlerByDay(unittest.TestCase):

	def setUp(self):
		self.dataset_mock = Mock()
		self.test_object = HandlerByDay(self.dataset_mock)

	def test_task_type(self):
		self.assertEqual(self.test_object.task_type, 'by_day')

	def test_key_iterator(self):
		self.assertEqual(self.test_object._key_list, None)
		result = HandlerByDay.class_key_iterator()
		self.assertIsInstance(result, Iterable)
		result_list = list(result)
		expected = [dict(sensor_type=stype) for stype in DatasetTree.tree['sensor']['sensor_type']]
		self.assertListEqual(result_list, expected)

	def test_sane_check_data(self):
		self.test_object.data_slice.sensor_type.unique = MagicMock(return_value=[1,2])
		with self.assertRaises(AssertionError):
			self.test_object.sane_check_data()
		self.test_object.data_slice.sensor_type.unique.assert_called_once()
		self.test_object.data_slice.sensor_type.unique = MagicMock(return_value=[1])
		self.test_object.sane_check_data()


class TestHandlerBySensor(unittest.TestCase):

	def setUp(self):
		self.dataset_mock = Mock()
		self.test_object = HandlerBySensor(self.dataset_mock)

	def test_task_type(self):
		self.assertEqual(self.test_object.task_type, 'by_sensor')

	def test_key_iterator(self):
		self.assertEqual(self.test_object._key_list, None)
		result = HandlerBySensor.class_key_iterator()
		self.assertIsInstance(result, Iterable)
		result_list = list(result)
		expected = [dict(sensor_number=snumber) for snumber in DatasetTree.tree['sensor']['sensor_number']]
		self.assertListEqual(result_list, expected)

	def test_sane_check_data(self):
		with patch('calina.data_handlers.HandlerBySensor.data_slice',
								 new=PropertyMock(return_value=np.array([[1,2],[3,4]]))):
			with self.assertRaises(AssertionError):
				self.test_object.sane_check_data()
		with patch('calina.data_handlers.HandlerBySensor.data_slice',
								 new=PropertyMock(return_value=np.array([[3, 4],]))):
			self.test_object.sane_check_data()


class TestHandlerByLink(unittest.TestCase):

	def setUp(self):
		self.dataset_mock = Mock()
		self.range = range(2)
		self.dataset_mock.range = self.range
		self.test_object = HandlerByLink(self.dataset_mock, values_start=0, link_end=9, link_size=3)
		self.data = pd.DataFrame([[1, 32, 3,6], [0, 4, 55, 7]])

	def test_init(self):
		self.assertEqual(self.test_object.dataset,self.dataset_mock)
		self.assertEqual(self.test_object._data_slice, None)
		self.assertEqual(self.test_object._key_parameters, None)
		self.assertEqual(self.test_object.dataset.range, self.range)

	def test_task_type(self):
		self.assertEqual(self.test_object.task_type, 'by_link')

	def test_key_iterator(self):
		with patch('calina.data_handlers.HandlerBySensor.class_key_iterator',
				   new=MagicMock(return_value=iter([{'sensor_number':'#1'},{'sensor_number':'#2'}]))):
			self.assertEqual(self.test_object._dict_key_list.get((9,3), None), None)
			result = HandlerByLink.class_key_iterator(9,3)
			self.assertIsInstance(result, Iterable)
			result_list = list(result)
			HandlerBySensor.class_key_iterator.assert_called_once()
			expected = []
			for snumber in ['#1', '#2']:
				expected += [dict(sensor_number=snumber, link_start=i, link_end=i+3) for i in range(0,6,3)]
			self.assertListEqual(result_list, expected)
		with patch('calina.data_handlers.HandlerBySensor.class_key_iterator',
								 new=MagicMock(return_value=iter(['#1', '#2']))):
			result = HandlerByLink.class_key_iterator(9,3)
			self.assertFalse(HandlerBySensor.class_key_iterator.called)
			self.assertEqual(list(result), HandlerByLink._dict_key_list[(9,3)])


	def test_sane_check_data(self):
		tmpdata = self.data.copy()
		with patch('calina.data_handlers.HandlerByLink.data_slice', new=PropertyMock(return_value=tmpdata.iloc[0:1, :1+1])):
			with patch('calina.data_handlers.HandlerByLink.data_values',new=PropertyMock(return_value=np.array([[0,1,2],]))):
				self.test_object.sane_check_data()
		with patch('calina.data_handlers.HandlerByLink.data_slice', new=PropertyMock(return_value=tmpdata.iloc[0:1, :1+1])):
			with patch('calina.data_handlers.HandlerByLink.data_values',
					   new=PropertyMock(return_value=np.array([[0, 2,4,3], ]))):
				with self.assertRaises(AssertionError):
					self.test_object.sane_check_data()
		with patch('calina.data_handlers.HandlerByLink.data_slice', new=PropertyMock(return_value=tmpdata.iloc[0:2, :1+2])):
			with patch('calina.data_handlers.HandlerByLink.data_values',
					   new=PropertyMock(return_value=np.array([[0, 2], ]))):
				with self.assertRaises(AssertionError):
					self.test_object.sane_check_data()


class TestDayKeyHandler(unittest.TestCase):
	def setUp(self):
		self.dataset_mock = Mock()
		self.test_object = DayKeyHandler(self.dataset_mock)

	def test_get_day_as_string(self):
		self.dataset_mock.filename = MagicMock(return_value='dmock')
		result = DayKeyHandler.get_day_as_string(self.dataset_mock)
		self.assertEqual(result, 'dmock')
		self.dataset_mock.filename.assert_called_once()

	def test_key_parameters(self):
		self.test_object.get_day_as_string = MagicMock(return_value='gdas')
		result = self.test_object.key_parameters()
		self.assertEqual(result, dict(day='gdas'))
		self.test_object.get_day_as_string.assert_called_with(self.test_object.dataset)

	def test_task_type(self):
		self.assertEqual(self.test_object.task_type, 'day_key')


class MockHandlerTuple(HandlerTuple):

	def task_type(self):
		pass


class TestHandlerTuple(unittest.TestCase):

	def setUp(self):
		self.a = MagicMock()
		self.b = MagicMock()
		self.a.key_parameters = MagicMock(return_value=dict(a='a'))
		self.b.key_parameters = MagicMock(return_value=dict(b='b'))
		self.a.sane_check_data = MagicMock()
		self.b.sane_check_data = MagicMock()
		self.tuple = (self.a, self.b)
		self.test_object = MockHandlerTuple(self.tuple)

	def test_init(self):
		obj = MockHandlerTuple(('a', 'b'))
		self.assertEqual(obj.data_handler_tuple, ('a','b'))

	def test_key_parameters(self):
		result = self.test_object.key_parameters()
		self.assertEqual(result, dict(a='a', b='b'))
		self.test_object.data_handler_tuple[0].key_parameters.assert_called_once()
		self.test_object.data_handler_tuple[1].key_parameters.assert_called_once()

	def test_sane_check_data(self):
		self.test_object.sane_check_data()
		self.test_object.data_handler_tuple[0].sane_check_data.assert_called_once()
		self.test_object.data_handler_tuple[1].sane_check_data.assert_called_once()
		with self.assertRaises(AssertionError):
			self.test_object.data_handler_tuple = (1,3,4,2,3)
			self.test_object.sane_check_data()