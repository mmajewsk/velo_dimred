import pandas as pd

BADCALS = {
		'2011-03-07':1.,
		'2012-08-02':3.,
		'2012-07-30':10.,
		'2012-08-01':10.
}

header_crosstalk_list = [0, 32, 64, 96, 159, 191, 223, 255, 287, 319, 351, 383, 415, 447, 479, 511, 543, 575, 607, 639, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992, 1024, 1056, 1088, 1120, 1183, 1215, 1247, 1279, 1311, 1343, 1375, 1407, 1439, 1471, 1503, 1535, 1567, 1599, 1631, 1663, 1664, 1696, 1728, 1760, 1792, 1824, 1856, 1888, 1920, 1952, 1984, 2016]

def remove_anomalies(
		sensor_data: pd.DataFrame,
		cut_val: int =50,
		channel_list: list = header_crosstalk_list,
		descriptors: int = 9) -> pd.DataFrame:
	"""
	Removes header cross talk, anomalies (value greater than cut_val), and returns clean dataframe.
	:param dataset: pd.DataFrame instance to clear
	:param cut_val: treshold value for anomalies (cut above that value), None == no cut
	:param channel_list: list of int channels to remove
	"""
	header_columns = ["channel{}".format(c) for c in channel_list]
	no_header_crosstalk = sensor_data.drop(labels=header_columns,axis='columns')
	no_anomalies = no_header_crosstalk.copy()
	if cut_val != None:
		cut_mask= (no_header_crosstalk.iloc[:, descriptors:] >= cut_val)
		negative_sensor_readout_mask = (cut_mask.sum(axis=1) == 0)
		no_hc_no_anomalies = no_header_crosstalk[negative_sensor_readout_mask]
		no_anomalies = no_hc_no_anomalies.copy()
	return no_anomalies


def create_x_feature(dataset, badcals):
	for dt in dataset.datetime.unique():
		key = str(dt)[:10]
		val = badcals.get(key, 0.)
		dataset.loc[dataset.datetime==dt,'x'] = val
	return dataset

def create_basic_dataset(dataset, sensor_type, badcals):
	sensor_data = dataset.dfh[sensor_type].df
	no_hc_no_anomalies = remove_anomalies(sensor_data, sensor_type)
	dataset = create_x_feature(no_hc_no_anomalies, badcals)
	return dataset

def create_negative(basic_dataset, badcals):
	bad_calibration = badcals.keys()
	negative_feature = basic_dataset[basic_dataset['datetime'].isin(bad_calibration)]
	return negative_feature

def create_positive(basic_dataset, badcals):
	bad_calibration = badcals.keys()
	positive = basic_dataset[~basic_dataset['datetime'].isin(bad_calibration)]
	del positive['x']
	return positive
