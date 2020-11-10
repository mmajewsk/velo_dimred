from abc import ABC, abstractmethod
from calina.database_adapter import DBAdapter
from calina.data_handlers import DataHandler
from outlierness import Outlierness


class BaseTask(ABC):
	def __init__(self,
				 outlierness: Outlierness,
				 data_handler: DataHandler,
				 db_adapter: DBAdapter,
				 task_name: str,
				 recalculate: bool = False
				 ):
		self.outlierness = outlierness
		self.data_handler = data_handler
		self.db_adapter = db_adapter
		self._recalculate = recalculate
		self.task_name = task_name
		self.table_name = None
		self.db_adapter = db_adapter


	@abstractmethod
	def do_task(self, *args, **kwargs):
		pass

	@abstractmethod
	def work(self, *args, **kwargs):
		pass

	@property
	def existing(self):
		return self.db_adapter.get_single(self.table_name, self.data_handler.key_parameters())

	@property
	def calculate(self):
		return self._recalculate or (self.existing is None)

	@abstractmethod
	def create_data_dict(self, *args, **kwargs):
		pass


def log_task(_logger):
	def real_decorator(func):
		def func_wrapper(self, *args, **kwargs):
			_logger.debug(
			   "Calculating [{}] for task: [{}] for key parameters:{}".format(self.data_handler.task_type, self.task_name,
																			  str(self.data_handler.key_parameters())))
			return func(self, *args, **kwargs)
		return func_wrapper
	return real_decorator


def log_skip_task(_logger):
	def real_decorator(func):
		def func_wrapper(self, *args, **kwargs):
			result = func(self, *args, **kwargs)
			if result is None:
				_logger.debug(
				"SKIPPING [{}] for task: [{}] for key parameters:{}".format(self.data_handler.task_type, self.task_name,
															str(self.data_handler.key_parameters())))
			return result
		return func_wrapper
	return real_decorator


def do_if_calculate(func):
	def func_wrapper(self, *args, **kwargs):
		if self.calculate:
			return func(self, *args, **kwargs)
		else:
			return None
	return func_wrapper