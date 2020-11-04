from calibration_dataset import Tell1Dataset
import os


class MyDS(Tell1Dataset):
	filename_format = '%Y-%m-%d'
	filename_regex_format = r'\d{4}-\d{2}-\d{2}.csv'


if __name__ == "__main__":
    datapath = "../../data/calibrations/"
    data_list = MyDS.get_filepaths_from_dir(datapath)
    mds = MyDS(data_list)
    print(mds.df())
