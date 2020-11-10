import json
import numpy as np
import pymc3 as pm
from pymc3.backends.base import MultiTrace
import pandas as pd
import os
from abc import ABC, abstractmethod

module_path = os.path.abspath(__file__)
dirpath = os.path.dirname(module_path)
parameters_path = os.path.join(dirpath,'model_parameters','all_parameters.json')

def read_parameters():
	assert os.path.isfile(parameters_path), "Parameter file not found"
	with open(parameters_path, 'r') as f:
		return json.load(f)

class OutliernessBase(ABC):
	def __init__(self, verbose: bool = False):
		self.verbose = verbose

	@abstractmethod
	def __getitem__(self, item):
		"""
		:param slice:
		:return: the slice of the parameters
		"""
		pass

	@abstractmethod
	def score_array(self, x, *args, **kwargs) -> np.ndarray:
		"""
		return the scores as np.ndarray and pass all arguments to score
		"""
		pass

	@abstractmethod
	def score(self, x: np.ndarray,**kwargs):
		"""
		Return the score in any type
		:param x: parameters to score
		:param kwargs: arguments passed on to the learned machine
		"""
		pass

class Outlierness(OutliernessBase):

	def __init__(self, centers: np.ndarray, sds: np.ndarray, k: np.ndarray, m: np.ndarray, verbose=False):
		"""
		Parameters are used in following distribution:
		 >>> pm.Normal('obs', mu=centers + k * outlierness, sd=self.sds + self.m * outlierness)
		:param centers: np.ndarray of shape (n,)
		:param sds: np.ndarray of shape (n,)
		:param k: np.ndarray of shape (n,)
		:param m: np.ndarray of shape (n,)
		"""
		OutliernessBase.__init__(self, verbose)
		centers = Outlierness._list_to_array(centers)
		sds = Outlierness._list_to_array(sds)
		k = Outlierness._list_to_array(k)
		m = Outlierness._list_to_array(m)
		assert all([centers.shape == el.shape for el in [sds, k, m]]), "Shapes of the parameters must be equal"
		self.centers = centers
		self.sds = sds
		self.k = k
		self.m = m

	@staticmethod
	def _list_to_array(l):
		return l if type(l) == np.ndarray else np.array(l)

	@property
	def params_dict(self):
		return dict(centers=self.centers, sds=self.sds, k=self.k, m=self.m)

	def __getitem__(self, key):
		if isinstance(key, slice) or isinstance(key, range) or (isinstance(key,np.ndarray) and key.shape==self.k.shape):
			return type(self)(verbose=self.verbose, **{k: v[key] for k, v in self.params_dict.items()})

	def score_array(self, x, *args, **kwargs) -> np.ndarray:
		"""
		The same as :func:`~outlierness.Outlierness.score` with as_array=True
		"""
		return self.score(x, *args, **kwargs)['x']

	def score(self, x: np.ndarray, draws: int = 800, tune:int =200, **kwargs) -> MultiTrace:
		"""
		:param x: np.ndarray of shape (n,m), m dimensions is irrelevant
		:param draws: passed into pm.sample()
		:param tune: passed into pm.sample()
		:param kwargs: passed into pm.sample()
		"""
		kwargs['progressbar'] = kwargs.get('verbose', self.verbose)
		with pm.Model() as model:
			outlierness = pm.Uniform('x', 0, 20)
			observations = pm.Normal('obs', mu=self.centers + self.k * outlierness, sd=self.sds + self.m * outlierness, observed=x)
			step = pm.Metropolis(vars=[outlierness])
			trace = pm.sample(draws=draws, tune=tune, step=step, **kwargs)
		return trace
