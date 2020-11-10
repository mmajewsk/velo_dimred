import numpy as np
import os
import pandas as pd
import re
from abc import ABC, abstractmethod
from processing import remove_anomalies

"""
This code is ugly, as all data cleaning is.

MM
"""

module_path = os.path.abspath(__file__)
dirpath = os.path.dirname(module_path)
mapping_csv_path = os.path.join(dirpath,'module_mapping.csv')

assert os.path.isfile(mapping_csv_path), "Mapping file not found"

def get_module_map():
	return pd.read_csv(mapping_csv_path, sep=' ')

def create_sensor_to_type():
	module_map = get_module_map()
	sensor_number_to_type = {}
	for i, data in module_map.iterrows():
		if not pd.isnull(data['sensor_number']):
			sensor_number_to_type[int(data['sensor_number'])] = data['sensor_type']
	return sensor_number_to_type

def split_to_variables(df):

	df_pedestals = df[df['type'] == 'pedestal']
	dfp = df_pedestals.dropna(axis=1, how='all')
	dfl = df[df['type'] == 'low_threshold']
	dfh = df[df['type'] == 'hit_threshold']
	return dfp, dfl, dfh

sensor_mapping = create_sensor_to_type()
list_of_modules = np.array(list(range(38))+ [39,  40,  41,  48])
list_of_modules = ["mod_{}".format(i) for i in list_of_modules]
def create_slot_labels():
	labels = []
	for part in range(1,26):
		for i in "LR":
			labels.append("VL{:02d}{}".format(part,i))
	return labels
slot_labels = np.array(['PU01L', 'PU01R', 'PU02L', 'PU02R']+create_slot_labels(),dtype=object)
sensor_numbers= [ '#{}'.format(i) for i in np.array(list(range(42))+list(range(64,106))+list(range(128,132)))]

def find_dummy_channels(dataset : pd.DataFrame):
	blocked = (dataset==127.0)
	trully_dummy = []
	for channel in blocked:
		column = blocked[channel]
		if column.all():
			trully_dummy.append(channel)
	return trully_dummy

def remove_dummy_channels(data : pd.DataFrame):
	dummy_channels = find_dummy_channels(data.iloc[:, 9:])
	channels = data.drop(dummy_channels, axis=1)
	#assert channels.iloc[:,9:].shape[1] == 2048, "Number of channelsis not equal to 2048, found {}".format(channels.iloc[:,9:].shape[1])
	columns = ["channel{}".format(i) for i in range(channels.iloc[:,9:].shape[1])]
	channels.columns = channels.columns[:9].tolist()+columns
	return channels


class DatasetTree:
	tree = {
		'type':
			[
				'pedestal',
				'low_threshold',
				'hit_threshold',
			],
		'sensor': {
			'mod_nr': list_of_modules,
			'slot_label': slot_labels,
			'mod_type': ['PU', 'VELO_phi', 'VELO_R', 'VELO_Rx', 'VELO_phix'],
			'sensor_type': ['R', 'phi'],
			'sensor_number': sensor_numbers,
		}

	}
	module_map = get_module_map()
	def  __init__(self, df, tree, module_map):
		self.df = df.copy()
		self.tree = tree
		self.module_map = module_map.copy()

	@staticmethod
	def from_df(df):
		return DatasetTree(df, DatasetTree.tree, DatasetTree.module_map)

	@staticmethod
	def from_tell1dataset_df(df):
		res = df.merge(DatasetTree.module_map, left_on='sensor', right_on='sensor_number', how='outer')
		res = res[~res['sensor_number'].isnull()]
		res.loc[:, 'sensor_number'] = res['sensor_number'].apply(lambda x: '#' + str(int(x)))
		cols = res.columns.tolist()
		cols = cols[:3]+cols[-6:]+cols[3:-6]
		df = res[cols]
		return DatasetTree.from_df(df)


	def __match_type(self, item):
		type_list = [(item in v) for v in self.tree['type']]
		matches = sum(type_list)
		if matches==1:
			type_ind = type_list.index()
			type = self.tree['type'][type_ind]
			return DatasetTree(self.df[self.df['type']==type], self.tree, self.module_map)
		else:
			return None

	def __match_sensor(self, item):
		matches = []
		for k, v in self.tree['sensor'].items():
			if item in v:
				matches.append((k,item))
		if len(matches)==1:
			k,v = matches[0]
			return DatasetTree(self.df[self.df[k]==v], self.tree, self.module_map)

	def __getitem__(self, item):
		if isinstance(item,str):
			match = self.__match_type(item)
			if match is not None:
				return match
			match = self.__match_sensor(item)
			if match is not None:
				return match
		return self.df[item]

	def __repr__(self):
		return self.df.__repr__()

class Datasetfolder:
	filename_regex_format = ''

	@staticmethod
	def extract_name(x):
		return os.path.basename(x)

	@classmethod
	def filter_filepath(cls, filepaths: list) -> list:
		regex = re.compile(cls.filename_regex_format)
		condition = lambda x: regex.match(cls.extract_name(x))
		selected_files = filter(condition, filepaths)
		return list(selected_files)

	@classmethod
	def get_filepaths_from_dir(cls, dirpath:str) -> list:
		file_list = os.listdir(dirpath)
		file_list = [os.path.join(dirpath, filename) for filename in file_list]
		selected_files = cls.filter_filepath(file_list)
		return selected_files

	@classmethod
	def get_filepaths_from_runlist(cls, runlist_path):
		runlist = None
		with open(runlist_path, 'r') as f:
			runlist = f.read()
		runlist = runlist.split('\n')
		runlist_dir = os.path.dirname(runlist_path)
		filepaths = [os.path.join(runlist_dir, filename + '.csv') for filename in runlist]
		selected_files = cls.filter_filepath(filepaths)
		return selected_files

class Tell1Dataset(Datasetfolder):
	dataset_channels = ['channel{}'.format(i) for i in range(2304)]
	dataset_columns = ['type', 'sensor'] + dataset_channels
	filename_format = '%Y_%m_%d-%H_%M_%S'
	filename_regex_format = r'\d{4}_\d{2}_\d{2}-\d{2}_\d{2}_\d{2}.csv'


	def __init__(self, data_files: list):
		"""

		:param data_files: List of strings with the csv files with calibration data.
		"""
		self.data_files = data_files
		self._df=None
		self.groups = {
			'pedestal':None,
			'hit_threshold':None,
			'low_threshold':None,
			'link_mask':None,
		}


	@staticmethod
	def from_directory(dirpath: str) -> 'Tell1Dataset':
		selected_files = Tell1Dataset.get_filepaths_from_dir(dirpath)
		return Tell1Dataset(selected_files)

	@staticmethod
	def from_run_list(runlist_path: str) -> 'Tell1Dataset':
		selected_files = Tell1Dataset.get_filepaths_from_runlist(runlist_path)
		return Tell1Dataset(selected_files)


	@staticmethod
	def remove_dummy_threshold_channels(df):
		df_r = remove_dummy_channels(df[df['sensor_type'] == 'R'])
		df_phi = remove_dummy_channels(df[df['sensor_type'] == 'phi'])
		df = pd.merge(df_r, df_phi, left_index=True, right_index=True)
		return df

	def check_if_splitted(self):
		if not np.all([v for v in self.groups.values()]):
			variables = {type: data for type, data in self.df.df.groupby('type')}
			for type,data in variables.items():
				self.groups[type] = DatasetTree(data, self.df.tree, self.df.module_map)
			self.dfhr = remove_dummy_channels(self.groups['hit_threshold']['R'].df)
			self.dfhphi = remove_dummy_channels(self.groups['hit_threshold']['phi'].df)
			dfh = pd.concat([self.dfhr, self.dfhphi])
			self.groups['hit_threshold'].df = dfh
			self.dflr = remove_dummy_channels(self.groups['low_threshold']['R'].df)
			self.dflphi = remove_dummy_channels(self.groups['low_threshold']['phi'].df)
			dfl = pd.concat([self.dflr, self.dflphi])
			self.groups['low_threshold'].df = dfl
			self.groups['link_mask'].df = self.groups['link_mask'].df.dropna(axis='columns', how='all')

	def remove_dummy(self, _df):
		df = _df.df
		self.tomerge = []
		for type in DatasetTree.tree['type']:
			parameter = df[df['type'] == type]
			parameter = Tell1Dataset.remove_dummy_threshold_channels(parameter)
			self.tomerge.append(parameter.copy())
		_df.df = df
		return _df

	@property
	def df(self):
		if self._df is None:
			self._df = self.join_data()
			self._df = DatasetTree.from_tell1dataset_df(self._df)
			self._df = self.remove_dummy(self._df)
		return self._df

	@property
	def dfp(self):
		self.check_if_splitted()
		return self.groups['pedestal']

	@property
	def dfl(self):
		self.check_if_splitted()
		return self.groups['low_threshold']

	@property
	def dfh(self):
		self.check_if_splitted()
		return self.groups['hit_threshold']

	@property
	def dfm(self):
		self.check_if_splitted()
		return self.groups['link_mask']

	def read(self):
		self.check_if_splitted()

	def join_data(self) -> pd.DataFrame:
		columns = Tell1Dataset.dataset_columns
		df_list = []
		for csv_path in self.data_files:
			df = pd.read_csv(csv_path,sep=' ',names=columns)
			filename = os.path.basename(csv_path)
			time, _ = os.path.splitext(filename)
			df['sensor'] = df['sensor'].apply(lambda x: int(x.split('VeloTELL1Board')[1]))
			df.insert(1, 'time', time)
			df_list.append(df.copy())
		df = pd.concat(df_list)
		df = df.sort_values(['time','sensor'],ascending=[1,1])
		df = df.reset_index(drop=True)
		df['datetime'] = pd.to_datetime(df['time'], format=self.filename_format)
		columns = columns[:2] + ['datetime'] + columns[2:]
		df = df[columns]
		return df


class LazyDataset(ABC):

	def __init__(self, dataset_object):
		self.dataset = dataset_object
		self.df=None
		self.result = None


	@property
	@abstractmethod
	def filename(self):
		pass

	@abstractmethod
	def read(self):
		pass

	def process(self, force=False):
		if self.result is None or force:
			self.result = self._process()
		return self.result

	@abstractmethod
	def _process(self):
		pass

class SingleRunDataset(LazyDataset, Datasetfolder):
	datetime_string_format = "%Y-%m-%dT%H:%M:%S"
	datetime_regex_format = '\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}'
	filename_regex_format = r'\d+_{}_{}_\S+_run_dump.csv'.format(datetime_regex_format, datetime_regex_format)
	channels = ['channel{}'.format(channel) for channel in range(2048)]

	def __init__(self, path, sensor_type=None):
		LazyDataset.__init__(self, self)
		self.path = path
		self.sensor_type = sensor_type
		self.columns = ['type', 'run_number' ,'sensor_number', 'start', 'end'] + self.channels

	def filename(self):
		return self.extract_name(self.path)

	def add_sensor_type(self):
		formatter = lambda x: sensor_mapping[int(x[1:])]
		return self.df['sensor_number'].apply(formatter)

	def format_sensor_number(self):
		prefix = 'VeloTELL1Board'
		cut_point = len(prefix)
		formatter = lambda x: '#{}'.format(x[cut_point:])
		return self.df['sensor_number'].apply(formatter)

	def rearrange_columns(self):
		columns = self.df.columns
		self.columns = list([columns[-1]]) + list(columns[:-1])
		return self.df[self.columns]

	def read(self):
		self.df = pd.read_csv(self.path, names=self.columns, sep=' ')
		self.df['sensor_number'] = self.format_sensor_number()
		self.df['sensor_type'] = self.add_sensor_type()
		self.df['start'] = pd.to_datetime(self.df['start'], format=self.datetime_string_format)
		self.df['end'] = pd.to_datetime(self.df['end'], format=self.datetime_string_format)
		self.df = self.rearrange_columns()
		if self.sensor_type is not None:
			self.df = self.df[self.df.sensor_type == self.sensor_type]

	def _process(self):
		return remove_anomalies(self.df, cut_val=None)


class CalinaLazyDataset(LazyDataset):

	def __init__(self, t1d: Tell1Dataset, sensor_type: str, anomalies_kwrgs:dict=None):
		if anomalies_kwrgs is None:
			self.anomalies_kwrgs = {'cut_val': None}
		else:
			self.anomalies_kwrgs = anomalies_kwrgs
		self.sensor_type = sensor_type
		LazyDataset.__init__(self, t1d)
		self.mask = None

	def filename(self):
		assert len(self.dataset.data_files) == 1
		filename = Tell1Dataset.extract_name(self.dataset.data_files[0])
		name = os.path.splitext(filename)[0]
		return name

	def read(self):
		self.df = self.dataset.dfh[self.sensor_type].df
		self.mask = remove_anomalies(self.dataset.dfm[self.sensor_type].df, cut_val=None)

	def _process(self):
		return remove_anomalies(self.df, **self.anomalies_kwrgs)



	

