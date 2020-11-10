import dataset

class DBAdapter:

	def __init__(self, db_path):
		self.db_path = db_path

	def get_single(self, table, keys):
		with dataset.connect(self.db_path) as db:
			element = db[table].find_one(**keys)
			return element

	def check_if_exists(self, table, keys):
			element = self.get_single(table, keys)
			return element != None

	def save(self, table, data, keys):
		with dataset.connect(self.db_path) as db:
			pk = db[table].upsert(data, keys)
		return pk