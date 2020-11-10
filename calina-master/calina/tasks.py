import logging
import dataset

from calina import log_task, log_skip_task, do_if_calculate

try:
    from matplotlib.cbook import boxplot_stats
except ImportError:
    from tools.cbook_snippet import boxplot_stats
from calina.basic import BaseTask
from calina.database_adapter import DBAdapter
from outlierness import Outlierness
from calina.data_handlers import DataHandler
from tools.misc import merge_two_dicts
from datetime import datetime as datetime

logging.basicConfig(format='%(asctime)s [%(levelname)s]= %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ScoreTask(BaseTask):
    def __init__(self,
                 outlierness: Outlierness,
                 data_handler: DataHandler,
                 db_adapter: DBAdapter,
                 task_name: str,
                 recalculate: bool = False,
                 ):
        BaseTask.__init__(self, outlierness, data_handler,
                          db_adapter, task_name, recalculate=recalculate)
        self.table_name = "{}_{}".format(
            task_name, self.data_handler.task_type)

    def create_data_dict(self, score, key_params):
        score_info = dict(
            score=score,
            calculated=str(datetime.now()),
        )
        return merge_two_dicts(key_params, score_info)

    @log_task(logger)
    def do_task(self, **kwargs):
        self.data_handler.sane_check_data()
        score_data = self.data_handler.data_values
        score = self.outlierness.score_array(score_data, **kwargs)
        score_data = self.create_data_dict(
            score=score, key_params=self.data_handler.key_parameters())
        return score_data

    @log_skip_task(logger)
    @do_if_calculate
    def work(self, *args, **kwargs):
        score_data = self.do_task(**kwargs)
        pk_score = self.db_adapter.save(self.table_name, score_data, list(
            self.data_handler.key_parameters().keys()))
        score_data['id'] = pk_score
        return score_data


class StatsTask(BaseTask):

    @staticmethod
    def from_ScoreTask(task):
        obj = StatsTask(task.outlierness, task.data_handler,
                        task.db_adapter, task.task_name, recalculate=task._recalculate)
        obj.table_name = 'stats_' + task.table_name
        return obj

    @staticmethod
    def serializable_boxplot_stats(data):
        stats = boxplot_stats(data, whis='range')[0]
        return stats

    def create_data_dict(self, score):
        return self.serializable_boxplot_stats(score)

    @staticmethod
    def create_stats_tables(db_adapter: DBAdapter):
        # This might not be needed
        with dataset.connect(db_adapter.db_path) as db:
            db.create_table('stats_by_day', primary_id='score_id')
            db.create_table('stats_by_sensor', primary_id='score_id')
            db.create_table('stats_by_link', primary_id='score_id')

    def do_task(self, normal_score, **kwargs):
        self.data_handler.sane_check_data()
        bxp_stats = self.create_data_dict(normal_score)
        return bxp_stats

    @log_skip_task(logger)
    @do_if_calculate
    def work(self, score_data, **kwargs):
        normal_score, normal_score_pk = score_data['score'], score_data['id']
        stats = self.do_task(normal_score, **kwargs)
        stats['score_id'] = normal_score_pk
        self.db_adapter.save(self.table_name, stats, ['score_id'])
        return stats
