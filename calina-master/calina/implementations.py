import time

from calibration_dataset import LazyDataset, Tell1Dataset, SingleRunDataset
from calina import DayKeyHandler, BiasedHandlerTuple, DataHandler


class RunKeyHandler(DayKeyHandler):

    @staticmethod
    def get_id_start_end_as_string(dataset: LazyDataset):
        # accepted filename format example:
        # '165052_2015-10-07T07:42:02_2015-10-07T08:42:07_noise_run_dump.csv'
        name = dataset.filename()
        splited = name.split("_")
        runid = int(splited[0])
        start, end = tuple(map(lambda x : time.strptime(x, SingleRunDataset.datetime_string_format), splited[1:3]))
        start, end = tuple(map(lambda x: time.strftime(Tell1Dataset.filename_format, x), (start,end)))
        return runid, start, end

    def key_parameters(self):
        id, start, end = RunKeyHandler.get_id_start_end_as_string(self.dataset)
        return dict(runid=id, start=start, end=end)

    def sane_check_data(self):
        assert len(self.data_slice.start.unique())==1, "Only single run allowed"

    @property
    def task_type(self):
        return 'start_end_key'


class NoiseHandlerTuple(BiasedHandlerTuple):
    FirstHandlerClass = RunKeyHandler

    @classmethod
    def from_data_handler(cls, handler: DataHandler):
        return super(NoiseHandlerTuple, cls).from_data_handler(handler, values_start=6)


class CalibHandlerTuple(BiasedHandlerTuple):

    @staticmethod
    def from_data_handler(handler: DataHandler):
        #@TODO rhis stuff 16:39 11.09.2018
        return BiasedHandlerTuple.from_data_handler(handler, values_start=9)