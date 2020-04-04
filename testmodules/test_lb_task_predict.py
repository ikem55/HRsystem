from testmodules.test_base_task_predict import TestBaseTaskPredict
from modules.lb_sk_model import LBSkModel
from scripts.lb_v1 import SkModel as LBv1SkModel
from scripts.lb_v2 import SkModel as LBv2SkModel
from scripts.lb_v3 import SkModel as LBv3SkModel

class TestLBTaskPredict(TestBaseTaskPredict):
    clean_flag = True

    def setUp(self):
        """ テスト実施前に必要な処理を記載する。呼び出しクラスやフォルダの指定等 """
        model_version = 'lb'
        table_name = '地方競馬レース馬'
        self.intermediate_folder = self.dict_path + 'intermediate/' + model_version + '_' + self.mode + '/' + self.model_name + '/'
        self.skmodel = LBSkModel(self.model_name, model_version, self.start_date, self.end_date, self.mock_flag, self.test_flag, self.mode)
        table_name = table_name + "_test"
        self.skmodel.set_table_name(table_name)
        self._proc_check_folder()


class TestLBv1TaskPredict(TestBaseTaskPredict):
    clean_flag = True

    def setUp(self):
        """ テスト実施前に必要な処理を記載する。呼び出しクラスやフォルダの指定等 """
        model_version = 'lb_v1'
        table_name = '地方競馬レース馬'
        self.intermediate_folder = self.dict_path + 'intermediate/' + model_version + '_' + self.mode + '/' + self.model_name + '/'
        self.skmodel = LBv1SkModel(self.model_name, model_version, self.start_date, self.end_date, self.mock_flag, self.test_flag, self.mode)
        table_name = table_name + "_test"
        self.skmodel.set_table_name(table_name)
        self._proc_check_folder()

class TestLBv2TaskPredict(TestBaseTaskPredict):
    clean_flag = True

    def setUp(self):
        """ テスト実施前に必要な処理を記載する。呼び出しクラスやフォルダの指定等 """
        model_version = 'lb_v2'
        table_name = '地方競馬レース馬'
        self.intermediate_folder = self.dict_path + 'intermediate/' + model_version + '_' + self.mode + '/' + self.model_name + '/'
        self.skmodel = LBv2SkModel(self.model_name, model_version, self.start_date, self.end_date, self.mock_flag, self.test_flag, self.mode)
        table_name = table_name + "_test"
        self.skmodel.set_table_name(table_name)
        self._proc_check_folder()

class TestLBv3TaskPredict(TestBaseTaskPredict):
    clean_flag = True

    def setUp(self):
        """ テスト実施前に必要な処理を記載する。呼び出しクラスやフォルダの指定等 """
        model_version = 'lb_v3'
        table_name = '地方競馬レース馬'
        self.intermediate_folder = self.dict_path + 'intermediate/' + model_version + '_' + self.mode + '/' + self.model_name + '/'
        self.skmodel = LBv3SkModel(self.model_name, model_version, self.start_date, self.end_date, self.mock_flag, self.test_flag, self.mode)
        table_name = table_name + "_test"
        self.skmodel.set_table_name(table_name)
        self._proc_check_folder()