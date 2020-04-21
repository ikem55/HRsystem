from testmodules.test_base_task_learning import TestBaseTaskLearning
from modules.lb_sk_model import LBSkModel
from scripts.lb_v1 import SkModel as LBv1SkModel
from scripts.lb_v2 import SkModel as LBv2SkModel
from scripts.lb_v3 import SkModel as LBv3SkModel
from scripts.lb_v4 import SkModel as LBv4SkModel
from scripts.lbr_v1 import SkModel as LBRv1SkModel

class TestLBTaskLearning(TestBaseTaskLearning):
    clean_flag = True

    def setUp(self):
        """ テスト実施前に必要な処理を記載する。呼び出しクラスやフォルダの指定等 """
        model_version = 'lb'
        table_name = '地方競馬レース馬'
        self.intermediate_folder = self.dict_path + 'intermediate/' + model_version + '_' + self.mode + '/' + self.model_name + '/'
        self.skmodel = LBSkModel(self.model_name, model_version, self.start_date, self.end_date, self.mock_flag, self.test_flag, self.mode)
        table_name = table_name + "_test"
        self.skmodel.set_test_table(table_name)
        self._proc_check_folder()


class TestLBv1TaskLearning(TestBaseTaskLearning):
    clean_flag = True

    def setUp(self):
        """ テスト実施前に必要な処理を記載する。呼び出しクラスやフォルダの指定等 """
        model_version = 'lb_v1'
        table_name = '地方競馬レース馬V1'
        self.intermediate_folder = self.dict_path + 'intermediate/' + model_version + '_' + self.mode + '/' + self.model_name + '/'
        self.skmodel = LBv1SkModel(self.model_name, model_version, self.start_date, self.end_date, self.mock_flag, self.test_flag, self.mode)
        table_name = table_name + "_test"
        self.skmodel.set_test_table(table_name)
        self._proc_check_folder()

class TestLBv2TaskLearning(TestBaseTaskLearning):
    clean_flag = True

    def setUp(self):
        """ テスト実施前に必要な処理を記載する。呼び出しクラスやフォルダの指定等 """
        model_version = 'lb_v2'
        table_name = '地方競馬レース馬V2'
        self.intermediate_folder = self.dict_path + 'intermediate/' + model_version + '_' + self.mode + '/' +self. model_name + '/'
        self.skmodel = LBv2SkModel(self.model_name, model_version, self.start_date, self.end_date, self.mock_flag, self.test_flag, self.mode)
        table_name = table_name + "_test"
        self.skmodel.set_test_table(table_name)
        self._proc_check_folder()

class TestLBv3TaskLearning(TestBaseTaskLearning):
    clean_flag = True

    def setUp(self):
        """ テスト実施前に必要な処理を記載する。呼び出しクラスやフォルダの指定等 """
        model_version = 'lb_v3'
        table_name = '地方競馬レース馬V3'
        self.intermediate_folder = self.dict_path + 'intermediate/' + model_version + '_' + self.mode + '/' +self. model_name + '/'
        self.skmodel = LBv3SkModel(self.model_name, model_version, self.start_date, self.end_date, self.mock_flag, self.test_flag, self.mode)
        table_name = table_name + "_test"
        self.skmodel.set_test_table(table_name)
        self._proc_check_folder()

class TestLBv4TaskLearning(TestBaseTaskLearning):
    clean_flag = True
    model_name = 'raceuma_lgm'
    target = "１着"
    obj_column_list = ['１着', '２着', '３着']
    obj_column_list_tr = ['１着_tr', '２着_tr', '３着_tr']

    def setUp(self):
        """ テスト実施前に必要な処理を記載する。呼び出しクラスやフォルダの指定等 """
        model_version = 'lb_v4'
        table_name = '地方競馬レース馬V4'
        self.intermediate_folder = self.dict_path + 'intermediate/' + model_version + '_' + self.mode + '/' +self. model_name + '/'
        self.skmodel = LBv4SkModel(self.model_name, model_version, self.start_date, self.end_date, self.mock_flag, self.test_flag, self.mode)
        table_name = table_name + "_test"
        self.skmodel.set_test_table(table_name)
        self._proc_check_folder()

    def test_20_check_learning_df(self):
        """ 学習に利用するデータフレームのテスト """
        import sys
        import pickle
        import modules.util as mu

        print("--  " + sys._getframe().f_code.co_name + " start --")
        file_name = self.intermediate_folder + 'learning_' + self.cls_val + '_' + self.val + '.pkl'

        with open(file_name, 'rb') as f:
            df = pickle.load(f)
            self.skmodel.proc.set_target_flag(self.target)
            self.skmodel.proc.set_learning_data(df, self.target)
            self.skmodel.proc.divide_learning_data()
            X_train = self.skmodel.proc.X_train
            mu.check_df(X_train)

class TestLBRv1SkModelLearning(TestBaseTaskLearning):
    clean_flag = True
    model_name = 'race_lgm'
    target = "UMAREN_ARE"
    obj_column_list = ['UMAREN_ARE', 'UMATAN_ARE', 'SANRENPUKU_ARE']
    obj_column_list_tr = ['UMAREN_ARE_tr', 'UMATAN_ARE_tr', 'SANRENPUKU_ARE_tr']

    def setUp(self):
        """ テスト実施前に必要な処理を記載する。呼び出しクラスやフォルダの指定等 """
        model_version = 'lbr_v1'
        table_name = '地方競馬レースV1'
        self.intermediate_folder = self.dict_path + 'intermediate/' + model_version + '_' + self.mode + '/' +self. model_name + '/'
        self.skmodel = LBRv1SkModel(self.model_name, model_version, self.start_date, self.end_date, self.mock_flag, self.test_flag, self.mode)
        table_name = table_name + "_test"
        self.skmodel.set_test_table(table_name)
        self._proc_check_folder()

    def test_20_check_learning_df(self):
        """ 学習に利用するデータフレームのテスト """
        import sys
        import pickle
        import modules.util as mu

        print("--  " + sys._getframe().f_code.co_name + " start --")
        file_name = self.intermediate_folder + 'learning_' + self.cls_val + '_' + self.val + '.pkl'

        with open(file_name, 'rb') as f:
            df = pickle.load(f)
            self.skmodel.proc.set_target_flag(self.target)
            self.skmodel.proc.set_learning_data(df, self.target)
            self.skmodel.proc.divide_learning_data()
            self.skmodel.proc.load_learning_target_encoding()
            X_train = self.skmodel.proc.X_train
            mu.check_df(X_train)