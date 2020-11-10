import sys

sys.path.append('..')
import unittest
from calina.basic import BaseTask, do_if_calculate
from calina import BiasedHandlerTuple
from unittest.mock import MagicMock, Mock, patch, PropertyMock
from calina.tasks import StatsTask, ScoreTask


class MockBaseTask(BaseTask):

	def do_task(self, **kwargs):
		pass

	def work(self, *args, **kwargs):
		pass

	def create_data_dict(self, *args, **kwargs):
		pass


class TestBaseTask(unittest.TestCase):

	def setUp(self):
		self.outlierness = 'a'
		self.db_adapter = Mock()
		self.task_name = 'task_name'
		self.recalculate = False
		self.data_handler = Mock()
		self.test_object = MockBaseTask(self.outlierness, self.data_handler, self.db_adapter, self.task_name, self.recalculate)

	def test_init(self):
		self.assertEqual(self.test_object.outlierness, self.outlierness)
		self.assertEqual(self.test_object.db_adapter, self.db_adapter)
		self.assertEqual(self.test_object.task_name, self.task_name)
		self.assertEqual(self.test_object._recalculate, self.recalculate)
		self.assertEqual(self.test_object.data_handler, self.data_handler)
		self.assertEqual(self.test_object.table_name, None)

	def test_existing(self):
		self.test_object.db_adapter.get_single = MagicMock(return_value=None)
		self.test_object.table_name = 'table_name'
		self.test_object.data_handler.key_parameters = MagicMock(return_value='kparams')
		self.assertIsNone(self.test_object.existing)
		self.test_object.db_adapter.get_single.assert_called_with('table_name','kparams')
		self.test_object.db_adapter.get_single = MagicMock(return_value={})
		self.test_object.table_name = 'table_name'
		self.test_object.data_handler.key_parameters = MagicMock(return_value='kparams')
		self.assertEqual(self.test_object.existing, {})
		self.test_object.db_adapter.get_single.assert_called_with('table_name', 'kparams')

	def test_calculate(self):
		with patch('tests.MockBaseTask.existing', new=PropertyMock(return_value=None)):
			self.test_object._recalculate = False
			result = self.test_object.calculate
			self.assertEqual(result, True)
		with patch('tests.MockBaseTask.existing', new=PropertyMock(return_value=None)):
			self.test_object._recalculate = True
			result = self.test_object.calculate
			self.assertEqual(result, True)
		with patch('tests.MockBaseTask.existing', new=PropertyMock(return_value={})):
			self.test_object._recalculate = False
			result = self.test_object.calculate
			self.assertEqual(result, False)
		with patch('tests.MockBaseTask.existing', new=PropertyMock(return_value={})):
			self.test_object._recalculate = True
			result = self.test_object.calculate
			self.assertEqual(result, True)

	def test_decorator_do_if_calculate(self):
		def func_mock(self1, *args, **kwargs):
			return 'test'
		decorated = do_if_calculate(func_mock)
		obj = Mock()
		obj.calculate = True
		result = decorated(obj)
		self.assertEqual(result, 'test')
		obj.calculate = False
		result = decorated(obj)
		self.assertEqual(result, None)


class TestScoreTask(unittest.TestCase):

	def setUp(self):
		self.outlierness = Mock()
		self.db_adapter = Mock()
		self.task_name = 'task_name'
		self.recalculate = False
		self.data_handler = Mock()
		self.data_handler.task_type = 'a'
		self.test_object = ScoreTask(self.outlierness, self.data_handler, self.db_adapter, self.task_name, self.recalculate)

	def test_init(self):
		self.assertEqual(self.test_object.outlierness, self.outlierness)
		self.assertEqual(self.test_object.db_adapter, self.db_adapter)
		self.assertEqual(self.test_object.task_name, self.task_name)
		self.assertEqual(self.test_object._recalculate, self.recalculate)
		self.assertEqual(self.test_object.data_handler, self.data_handler)
		self.assertEqual(self.test_object.data_handler, self.data_handler)
		self.assertEqual(self.test_object.table_name, 'task_name_a')

	@patch('calina.tasks.datetime')
	def test_create_data_dict(self, datetime):
		score = [1,2,3]
		datetime.now = MagicMock(return_value='sometime')
		key_params = dict(a='a')
		result = self.test_object.create_data_dict(score, key_params)
		expected = dict(a='a', score=score, calculated='sometime')
		self.assertEqual(result, expected)

	def test_do_task(self):
		score = [1, 2, 3]
		self.outlierness.score_array = MagicMock(return_value=score)
		self.data_handler.data_values = 'vals'
		self.data_handler.key_parameters =MagicMock(return_value=dict(k='k'))
		kwargs = dict(args='ddd')
		self.test_object.create_data_dict = MagicMock(return_value='score_data')
		result = self.test_object.do_task(**kwargs)
		self.assertEqual(result, 'score_data')
		self.outlierness.score_array.assert_called_with('vals', **kwargs)
		self.test_object.create_data_dict.assert_called_with(score=score, key_params=dict(k='k'))

	def test_work(self):
		kwargs = dict(args='ddd')
		with patch('calina.tasks.ScoreTask.calculate', new=PropertyMock(return_value=False)):
			result = self.test_object.work(**kwargs)
			self.assertIsNone(result)
		with patch('calina.tasks.ScoreTask.calculate', new=PropertyMock(return_value=True)):
			self.test_object.do_task = MagicMock(return_value=dict())
			self.test_object.data_handler.key_parameters = MagicMock(return_value=dict(a='a'))
			self.test_object.db_adapter.save = MagicMock(return_value=1)
			result = self.test_object.work(**kwargs)
			self.assertEqual(result, dict(id=1))
			self.test_object.do_task.assert_called_with(**kwargs)
			self.test_object.data_handler.key_parameters.assert_called_once()
			self.test_object.db_adapter.save.assert_called_with(self.test_object.table_name, dict(id=1), ['a'])



class TestStatTask(unittest.TestCase):

	def setUp(self):
		self.outlierness = Mock()
		self.db_adapter = Mock()
		self.task_name = 'task_name'
		self.recalculate = False
		self.data_handler = Mock()
		self.data_handler.task_type = 'a'
		self.score_task = ScoreTask(self.outlierness, self.data_handler, self.db_adapter, self.task_name, self.recalculate)
		self.test_object = StatsTask.from_ScoreTask(self.score_task)

	def test_from_ScoreTask(self):
		self.assertEqual(self.test_object.outlierness, self.outlierness)
		self.assertEqual(self.test_object.db_adapter, self.db_adapter)
		self.assertEqual(self.test_object.task_name, self.task_name)
		self.assertEqual(self.test_object._recalculate, self.recalculate)
		self.assertEqual(self.test_object.data_handler, self.data_handler)
		self.assertEqual(self.test_object.table_name, 'stats_task_name_a')

	@patch('calina.tasks.boxplot_stats')
	def test_serializable_boxplot_stats(self, boxplot_mock):
		boxplot_mock.return_value = ['stats']
		result = self.test_object.serializable_boxplot_stats('data')
		self.assertEqual(result, 'stats')
		boxplot_mock.assert_called_with('data',whis='range')

	def test_create_data_dict(self):
		score=[12,3,4]
		self.test_object.serializable_boxplot_stats = MagicMock(return_value='test')
		result = self.test_object.create_data_dict(score)
		self.assertEqual(result, 'test')
		self.test_object.serializable_boxplot_stats.assert_called_with(score)

	def test_do_task(self):
		normal_score = [1, 2, 3]
		self.data_handler.sane_check_data = MagicMock()
		kwargs = dict(args='ddd')
		self.test_object.create_data_dict = MagicMock(return_value='score_data')
		result = self.test_object.do_task(normal_score, **kwargs)
		self.assertEqual(result, 'score_data')
		self.data_handler.sane_check_data.assert_called_once()
		self.test_object.create_data_dict.assert_called_with(normal_score)


	def test_work(self):
		with patch('calina.tasks.StatsTask.calculate', new=PropertyMock(return_value=True)):
			kwargs = dict(args='ddd')
			self.test_object.do_task = MagicMock(return_value=dict(c='c'))
			self.test_object.db_adapter.save = MagicMock(return_value=1)
			score_data = dict(score=[1,2,3], id=1)
			result = self.test_object.work(score_data, **kwargs)
			stats=  dict(score_id=1, c='c')
			self.assertEqual(result, stats)
			self.test_object.do_task.assert_called_with(score_data['score'], **kwargs)
			self.test_object.db_adapter.save.assert_called_with(self.test_object.table_name, stats, ['score_id'])

class TestCalibHandlerTuple(unittest.TestCase):

	def setUp(self):
		self.data_handler = Mock()
		self.data_handler.dataset = Mock()
		self.a = MagicMock()
		self.a.create_key_dict = MagicMock(return_value=dict(a='a'))
		self.a.sane_check_data = MagicMock()
		self.test_object = BiasedHandlerTuple.from_data_handler(self.data_handler)


	@patch('calina.data_handlers.DataHandler.__init__')
	@patch('calina.data_handlers.HandlerTuple.__init__')
	def test_init(self, ht, dh):
		#dh = MagicMock()
		#ht= MagicMock()
		a = Mock()
		a.dataset = 'asm'
		t = ('smth',a)
		obj = BiasedHandlerTuple(t)
		dh.assert_called_with(obj, 'asm')
		ht.assert_called_with(obj, t)

	@patch('calina.data_handlers.DayKeyHandler')
	def test_from_data_handler(self, dkh):
		dkh.return_value = Mock()
		self.assertIsInstance(self.test_object, BiasedHandlerTuple)

	def test_set_key_parameters(self):
		params = dict(f='f')
		self.data_handler.set_key_parameters = MagicMock()
		self.test_object.set_key_parameters(params)
		self.data_handler.set_key_parameters.assert_called_with(params)


if __name__ == '__main__':
	unittest.main()
