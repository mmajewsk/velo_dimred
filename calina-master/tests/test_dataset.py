import unittest
from unittest.mock import MagicMock, patch, Mock
from calibration_dataset import LazyDataset, CalinaLazyDataset, Tell1Dataset, SingleRunDataset
import pandas as pd
import os

module_path = os.path.abspath(__file__)
dirpath = os.path.dirname(module_path)


class TestTell1Dataset(unittest.TestCase):

	def test_from_run_list(self):
		run_list_path = os.path.join(dirpath, r"test_data/calibration/RunList.txt")
		td = Tell1Dataset.from_run_list(run_list_path)
		testlist = [
			'{}\\test_data/calibration\\2012_04_06-17_30_00.csv'.format(dirpath),
			'{}\\test_data/calibration\\2012_08_02-17_00_00.csv'.format(dirpath),
			'{}\\test_data/calibration\\2016_05_31-16_34_12.csv'.format(dirpath),
			'{}\\test_data/calibration\\2017_07_18-13_30_00.csv'.format(dirpath)
		]
		self.assertEqual(testlist, td.data_files)

	def test_from_directory(self):
		test_dir_path = os.path.join(dirpath,'test_data/calibration')
		td = Tell1Dataset.from_directory(test_dir_path)
		testlist = [
			'{}\\test_data/calibration\\2012_04_06-17_30_00.csv'.format(dirpath),
			'{}\\test_data/calibration\\2012_08_02-17_00_00.csv'.format(dirpath),
			'{}\\test_data/calibration\\2016_05_31-16_34_12.csv'.format(dirpath),
			'{}\\test_data/calibration\\2017_07_18-13_30_00.csv'.format(dirpath)
		]
		self.assertEqual(testlist, td.data_files)

	def test_join_data(self):
		run_list_path = os.path.join(dirpath, r"test_data/calibration/RunList.txt")
		td = Tell1Dataset.from_run_list(run_list_path)
		df = td.join_data()
		self.assertIsInstance(df, pd.DataFrame)
		self.assertEqual(df.shape[1],2307)
		columns = ['type', 'sensor', 'datetime'] + ['channel{}'.format(i) for i in range(2304)]
		self.assertListEqual(df.columns.tolist(), columns)



class MockLazyDataset(LazyDataset):
	def filename(self):
		pass

	def read(self):
		pass

	def _process(self):
		pass


class TestLazyDataset(unittest.TestCase):

	def setUp(self):
		dsobj = [1, 2, 3]
		self.test_obj = MockLazyDataset(dataset_object=dsobj)

	def test_init(self):
		dsobj = [1,2,3]
		obj = MockLazyDataset(dataset_object=dsobj)
		self.assertEqual(obj.dataset,dsobj)
		self.assertEqual(obj.df, None)
		self.assertEqual(obj.result, None)

	def test_process(self):
		self.test_obj._process = MagicMock(return_value='process')
		self.test_obj.result = None
		self.assertEqual(self.test_obj.process(), 'process')
		self.assertTrue(self.test_obj._process.called)
		self.test_obj._process = MagicMock(return_value='process')
		self.test_obj.result = [3,2,1]
		self.assertEqual(self.test_obj.process(), [3,2,1])
		self.assertFalse(self.test_obj._process.called)

class TestSingleRunDataset(unittest.TestCase):
	def setUp(self):
		self.data_path = "{}\\test_data\\noise\\206955_noise_run_dump.csv".format(dirpath)
		self.srd = SingleRunDataset(self.data_path)

	def test_init(self):
		columns =  ['type', 'run_number' ,'sensor_number', 'start', 'end'] + SingleRunDataset.channels
		self.assertEqual(columns, self.srd.columns)
		self.assertEqual(self.srd.path, self.data_path)

	def test_process(self):
		#@TODO this is only basic test
		self.assertEqual(self.srd.result, None)
		self.srd.read()
		result = self.srd.process()
		self.assertIsInstance(result, pd.DataFrame)




class TestCalinaLazyDataset(unittest.TestCase):

	def setUp(self):
		self.t1d = MagicMock()
		self.sensor_type = 'phi'
		self.fake_path = 'calina\\tests\\test_data/calibration\\2017_07_18-13_30_00.csv'
		self.test_obj = CalinaLazyDataset(self.t1d, self.sensor_type)

	def test_init(self):
		test_obj = CalinaLazyDataset(self.t1d, self.sensor_type)
		self.assertEqual(test_obj.sensor_type, self.sensor_type)
		self.assertEqual(test_obj.dataset, self.t1d)

	def test_filename(self, ):
		self.test_obj.dataset.data_files = [self.fake_path]
		self.assertEqual(self.test_obj.filename(),'2017_07_18-13_30_00')
		self.test_obj.dataset.data_files = [self.fake_path, 'dddd']
		with self.assertRaises(AssertionError):
			self.test_obj.filename()

	@patch('calibration_dataset.Tell1Dataset')
	def test_read(self, T1DMock):
		self.test_obj.dataset = T1DMock()
		dfmock = Mock()
		dfmock.df = [6,5,4]
		self.test_obj.dataset.dfh = {self.sensor_type: dfmock}
		self.test_obj.read()
		self.assertEqual(self.test_obj.df, dfmock.df)

	@patch('calibration_dataset.remove_anomalies')
	def test__process(self, r_a):
		r_a.return_value = 'test'
		self.test_obj.df = [3,2,4]
		self.assertEqual(self.test_obj._process(),'test')
		r_a.assert_called_with(self.test_obj.df, cut_val=None)