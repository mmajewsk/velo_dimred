import sys

sys.path.append('..')
import os
from analyzers.calibration import CalinaAnalyzer
from pathlib import Path
from calibration_dataset import Tell1Dataset

if __name__	=='__main__':
	chosen = ['2017-10-02.csv', '2017-10-10.csv', '2017-10-12.csv']
	calibrations2 = Path(r'C:\repositories\Phd\LHCb\velo_analysis\data\Tell1CalibCsv')
	chosen_files = [calibrations2 / file for file in chosen]
	agg = CalinaAnalyzer(chosen_files, 'database.db', False, True, do_each_link=True)
	agg.go()