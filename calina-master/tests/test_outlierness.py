import unittest
from unittest.mock import MagicMock
import numpy as np
from calibration_dataset import Tell1Dataset, remove_anomalies
from outlierness import read_parameters, Outlierness, MultiTrace

class TestOutlierness(unittest.TestCase):

	def test_paramaters_read(self):
		params = read_parameters()
		self.assertIsInstance(params, dict)
		self.assertEqual(['R','phi'], list(params.keys()))
		for k in params:
			self.assertCountEqual(['centers','sds','k','m'], list(params[k].keys()))

	def test_init(self):
		params = read_parameters()
		out = Outlierness(**params['R'])
		self.assertTrue((out.centers==params['R']['centers']).all())
		self.assertTrue((out.sds==params['R']['sds']).all())
		self.assertTrue((out.k==params['R']['k']).all())
		self.assertTrue((out.m==params['R']['m']).all())
		m = params['R'].pop('m')
		m = m[:2]
		with self.assertRaises(AssertionError):
			out2 = Outlierness(m=m, **params['R'])

	def test_params_dict(self):
		params = read_parameters()
		out = Outlierness(**params['R'])
		self.assertTrue((out.params_dict['centers'] == params['R']['centers']).all())
		self.assertTrue((out.params_dict['sds'] == params['R']['sds']).all())
		self.assertTrue((out.params_dict['k'] == params['R']['k']).all())
		self.assertTrue((out.params_dict['m'] == params['R']['m']).all())

	def test_get_slice(self):
		params = read_parameters()
		out = Outlierness(**params['R'])
		out_slice = out[10:25]
		self.assertTrue((out_slice.centers == params['R']['centers'][10:25]).all())
		self.assertTrue((out_slice.sds == params['R']['sds'][10:25]).all())
		self.assertTrue((out_slice.k == params['R']['k'][10:25]).all())
		self.assertTrue((out_slice.m == params['R']['m'][10:25]).all())


	def test_score_array(self):
		params = read_parameters()
		score_args = dict(draws=10, tune=10)
		out = Outlierness(**params['R'])
		out.score = MagicMock(return_value=dict(x='test'))
		output = out.score_array([1,1,1], **score_args)
		self.assertEqual(output, 'test')
		out.score.assert_called_with([1,1,1], **score_args)

	def test_score(self):
		#!ATTTENTION! this test is data-dependant
		sensor_type = 'R'
		calibration = 'test_data/calibration/2012_08_02-17_00_00.csv'
		params = read_parameters()
		score_args = dict(draws=10, tune=10)
		out = Outlierness(verbose=True, **params[sensor_type])
		t1d = Tell1Dataset([calibration])
		data_slice = remove_anomalies(t1d.dfh[sensor_type].df)
		traces = out.score(data_slice.iloc[:,9:], **score_args)
		self.assertIsInstance(traces, MultiTrace)
		traces_arr = traces['x']
		mean_trace = np.mean(traces_arr)
		self.assertTrue(1.<mean_trace<4.)
