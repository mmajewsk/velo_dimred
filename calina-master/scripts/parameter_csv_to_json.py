import pandas as pd
import json
import os

def read_parameters(path_to_file):
	return list(pd.read_csv(path_to_file).values.T[0])

if __name__=="__main__":
	parameters = {}
	dirpath = r'C:\repositories\Phd\LHCb\calint\model_parameters'
	for sensor_type in ['R','phi']:
		parameters[sensor_type] = {}
		for parameter_name in ['centers','k','m','sds']:
			filepath = os.path.join(dirpath, "{}_mean{}.csv".format(sensor_type, parameter_name))
			parameters[sensor_type][parameter_name] = read_parameters(filepath)
	with open(r'..\model_parameters\all_parameters.json', 'w') as f:
		json.dump(parameters, f)
