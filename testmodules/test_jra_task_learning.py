from testmodules.test_base_jra_task_learning import TestBaseTaskLearning
from modules.jra_sk_model import JRASkModel
import scripts.jra_race_raptype as rc_rap

class TestJraRcRapTaskLearning(TestBaseTaskLearning):
    clean_flag = True
    obj_column_list = ['RAP_TYPE', 'TRACK_BIAS_ZENGO', 'TRACK_BIAS_UCHISOTO', 'PRED_PACE']
    target = 'RAP_TYPE'

    def setUp(self):
        """ テスト実施前に必要な処理を記載する。呼び出しクラスやフォルダの指定等 """
        model_version = 'jra_rc_raptype'
        self.intermediate_folder = self.dict_path + 'intermediate/' + model_version + '_' + self.mode + '/' + self.model_name + '/'
        self.skmodel = rc_rap.SkModel(self.model_name, model_version, self.start_date, self.end_date, self.mock_flag, self.test_flag, self.mode)
        self._proc_check_folder()

