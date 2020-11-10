from analyzers.calibration import CalinaAnalyzer
import unittest
from unittest.mock import Mock, patch, MagicMock
import os
import dataset
from outlierness import read_parameters, Outlierness
from calibration_dataset import Tell1Dataset, CalinaLazyDataset
import numpy as np

class MockOutlierness(Outlierness):

	def score(self, x, *args, **kwargs):

		assert self.centers.shape[0]==x.shape[1]
		return np.random.rand(1,9)

	def score_array(self, *args, **kwargs):
		return self.score(*args, **kwargs)

class TestCalinaAnalyzer(unittest.TestCase):
	db_path='functional_test.db'

	@classmethod
	def setUpClass(cls):
		try:
			if os.path.isfile(cls.db_path):
				os.remove(cls.db_path)
		except PermissionError:
			pass

	@classmethod
	def tearDownClass(cls):
		try:
			if os.path.isfile(cls.db_path):
				os.remove(cls.db_path)
		except PermissionError:
			pass

	def setUp(self):
		self.test_db_path = self.db_path
		self.params = read_parameters()
		self.sensor_type = 'R'
		calibration = 'test_data/calibration/2012_08_02-17_00_00.csv'
		self.out = MockOutlierness(verbose=True, **self.params[self.sensor_type])
		self.t1d = Tell1Dataset([calibration])
		self.lds = CalinaLazyDataset(self.t1d, self.sensor_type)
		self.lds.read()
		self.test_object = CalinaAnalyzer([calibration], self.test_db_path, force_recalc=True, verbose=True)

	@patch('analyzers.calibration.Outlierness', new=MockOutlierness)
	@patch('calina.tasks.datetime')
	def test_by_day(self, datetime):
		datetime.now = MagicMock(return_value='2018-06-21,00:40:27.800805')
		self.test_object.each_calibration(self.lds, self.test_object.params[self.sensor_type], sensor_type=self.sensor_type)
		expected = {'id':1,'day':'2012_08_02-17_00_00','sensor_type':'R','calculated':'2018-06-21,00:40:27.800805'}
		with\
				dataset.connect(self.test_object.db_path) as db:
			for i in db['calib_by_day'].all():
				for k in i:
					if k =='score':
						self.assertEqual((9,),np.fromstring(i[k]).shape)
					else:
						self.assertEqual(expected[k],i[k])


	@patch('analyzers.calibration.Outlierness', new=MockOutlierness)
	@patch('calina.tasks.datetime')
	def test_by_sensor(self, datetime):
		datetime.now = MagicMock(return_value='2018-06-21,00:40:27.800805')
		self.test_object.SensorClass.sensor_type = self.sensor_type
		self.test_object.each_sensor(self.lds, self.test_object.params[self.sensor_type])
		data_maker = lambda x : {'day': '2012_08_02-17_00_00', 'sensor_number': x['sensor_number'], 'calculated': '2018-06-21,00:40:27.800805'}
		expected = {x['sensor_number']:data_maker(x) for x in self.test_object.SensorClass.class_key_iterator()}
		with dataset.connect(self.test_object.db_path) as db:
			for i in db['calib_by_sensor'].all():
				for k in i:
					sn = i['sensor_number']
					if k =='score':
						self.assertEqual((9,),np.fromstring(i[k]).shape)
					elif k=='id':
						self.assertIsInstance(i[k], int)
					else:
						self.assertEqual(expected[sn][k], i[k])

	@patch('analyzers.calibration.Outlierness', new=MockOutlierness)
	@patch('calina.tasks.datetime')
	def test_by_link(self, datetime):
		datetime.now = MagicMock(return_value='2018-06-21,00:40:27.800805')
		self.test_object.SensorClass.sensor_type = self.sensor_type
		self.test_object.each_link(self.lds, self.test_object.params[self.sensor_type])
		data_maker = lambda x : {'day': '2012_08_02-17_00_00', 'sensor_number': x['sensor_number'], 'calculated': '2018-06-21,00:40:27.800805'}
		expected = {(x['sensor_number'],x['link_start'], x['link_end']):data_maker(x) for x in self.test_object.LinkClass.class_key_iterator(1984,31)}
		link_limit = 0
		with dataset.connect(self.test_object.db_path) as db:
			for i in db['calib_by_link'].all():
				link_limit +=1
				if link_limit>=10:
					break
				for k in i:
					sn = i['sensor_number']
					start = i['link_start']
					end = i['link_end']
					if k =='score':
						self.assertEqual((9,),np.fromstring(i[k]).shape)
					elif k=='id':
						self.assertIsInstance(i[k], int)
					elif k=='link_start' or k=='link_end':
						continue
					else:
						self.assertEqual(expected[(sn, start, end)][k], i[k])
