import datetime
import numpy as np
import dataset
import matplotlib.pyplot as plt

if __name__=='__main__':
	with dataset.connect('sqlite:///database.db') as noisedb:
		stats = noisedb['stats_noise_by_day'].all()

	stats = [dict(stat) for stat in stats]
	r_stats = []
	date_format = '%Y-%m-%dT%H:%M:%S'
	for i, stat in enumerate(stats):
		stats[i]['fliers'] = np.fromstring(stat['fliers'])
		with dataset.connect('sqlite:///database.db') as noisedb:
			noise = noisedb['noise_by_day'].find_one(id=stat['score_id'])
		if noise['sensor_type']=='R':
			_ = stat.pop('id')
			stats[i]['label'] = datetime.datetime.strptime(noise['start'], date_format)
			r_stats.append(stats[i])
	fig, axe = plt.subplots(4,1)
	axe[0].bxp(r_stats[:25])
	axe[1].bxp(r_stats[25:50])
	#axe[2].bxp(r_stats[50:75])
	fig.autofmt_xdate()
	plt.show()
