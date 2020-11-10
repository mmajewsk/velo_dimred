import dataset
import pandas as pd
import argparse

def main(source, run_list_source, destination):
	source_db_path = r"sqlite:///{}".format(source)
	destination_db = r"sqlite:///{}".format(destination)
	df = pd.read_csv(run_list_source, header=None)
	translate_dict = {v[0][:10].replace('_', '-'): v[0] for v in df.values}
	with dataset.connect(source_db_path) as db:
		tables = list(db.tables)
	for table in tables:
		if 'stats' in table:
			continue
		stats_table = 'stats_{}'.format(table)
		with dataset.connect(source_db_path) as db:
			all_data = db[table].all()
			with dataset.connect(destination_db) as db_dest:
				for values in all_data:
					day = values['day']
					values['day'] = translate_dict[day]
					score_id = values.pop('id')
					new_score_id = db_dest[table].insert(values)
					stat_data = db[stats_table].find_one(score_id=score_id)
					stat_id = stat_data.pop('id')
					stat_data['score_id'] = new_score_id
					db_dest[stats_table].insert(stat_data)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description='Convert calculation date in database to %Y_%m_%d-%H_%M_%S format')
	parser.add_argument('source', type=str, help='source database')
	parser.add_argument('run_list', type=str, help='path to RunList.txt which contains proper dates')
	parser.add_argument('destination', type=str, help='path to new database with corrected format')
	args = parser.parse_args()
	main(args.source, args.run_list, args.destination)
