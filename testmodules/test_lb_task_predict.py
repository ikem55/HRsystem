from testmodules.test_base_task_predict import TestBaseTaskPredict
from modules.lb_sk_model import LBSkModel
from scripts.lb_v1 import SkModel as LBv1SkModel
from scripts.lb_v2 import SkModel as LBv2SkModel
from scripts.lb_v3 import SkModel as LBv3SkModel
from scripts.lb_v4 import SkModel as LBv4SkModel
from scripts.lbr_v1 import SkModel as LBRv1SkModel

class TestLBTaskPredict(TestBaseTaskPredict):
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


class TestLBv1TaskPredict(TestBaseTaskPredict):
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

class TestLBv2TaskPredict(TestBaseTaskPredict):
    clean_flag = True

    def setUp(self):
        """ テスト実施前に必要な処理を記載する。呼び出しクラスやフォルダの指定等 """
        model_version = 'lb_v2'
        table_name = '地方競馬レース馬V2'
        self.intermediate_folder = self.dict_path + 'intermediate/' + model_version + '_' + self.mode + '/' + self.model_name + '/'
        self.skmodel = LBv2SkModel(self.model_name, model_version, self.start_date, self.end_date, self.mock_flag, self.test_flag, self.mode)
        table_name = table_name + "_test"
        self.skmodel.set_test_table(table_name)
        self._proc_check_folder()

class TestLBv3TaskPredict(TestBaseTaskPredict):
    clean_flag = True

    def setUp(self):
        """ テスト実施前に必要な処理を記載する。呼び出しクラスやフォルダの指定等 """
        model_version = 'lb_v3'
        table_name = '地方競馬レース馬V3'
        self.intermediate_folder = self.dict_path + 'intermediate/' + model_version + '_' + self.mode + '/' + self.model_name + '/'
        self.skmodel = LBv3SkModel(self.model_name, model_version, self.start_date, self.end_date, self.mock_flag, self.test_flag, self.mode)
        table_name = table_name + "_test"
        self.skmodel.set_test_table(table_name)
        self._proc_check_folder()

class TestLBv4TaskPredict(TestBaseTaskPredict):
    clean_flag = True
    model_name = 'raceuma_lgm'
    target = "１着"
    index_list = ["RACE_KEY", "NENGAPPI"]
    obj_column_list = ['１着', '２着', '３着']
    obj_column_list_tr = ['１着_tr', '２着_tr', '３着_tr']
    import_list = ["RACE_KEY", "pred", "prob", "target", "target_date"]

    def setUp(self):
        """ テスト実施前に必要な処理を記載する。呼び出しクラスやフォルダの指定等 """
        model_version = 'lb_v4'
        table_name = '地方競馬レース馬V4'
        self.intermediate_folder = self.dict_path + 'intermediate/' + model_version + '_' + self.mode + '/' + self.model_name + '/'
        self.skmodel = LBv4SkModel(self.model_name, model_version, self.start_date, self.end_date, self.mock_flag, self.test_flag, self.mode)
        table_name = table_name + "_test"
        self.skmodel.set_test_table(table_name)
        self._proc_check_folder()

    def test_11_proc_predict_sk_model(self):
        """ 予測データの作成ができることを確認 """
        import sys
        import pandas as pd
        import modules.util as mu

        print("--  " + sys._getframe().f_code.co_name + " start --")
        predict_file_name = self.intermediate_folder + mu.convert_date_to_str(self.end_date) + '_exp_data.pkl'
        exp_data = pd.read_pickle(predict_file_name)
        print(exp_data.shape)
        all_pred_df = pd.DataFrame()
        class_list = self.skmodel.class_list
        for cls_val in class_list:
            val_list = self.skmodel.get_val_list(exp_data, cls_val)
            for val in val_list:
                # 対象の競馬場のデータを取得する
                filter_df = self.skmodel.get_filter_df(exp_data, cls_val, val)
                # 予測を実施
                pred_df = self.skmodel.proc_predict_sk_model(filter_df, cls_val, val)
                all_pred_df = pd.concat([all_pred_df, pred_df])
                break
        import_df = self.skmodel.create_import_data(all_pred_df)
        self.skmodel.eval_pred_data(import_df)
        # not empty check
        self.assertFalse(len(import_df.index) == 0)
        # 必要な列があるかチェック
        contain_columns_set = set(self.import_list)
        contain_check = self.proc_test_contain_columns_check(import_df, contain_columns_set)
        self.assertTrue(contain_check)

class TestLBRv1TaskPredict(TestBaseTaskPredict):
    clean_flag = True
    model_name = 'race_lgm'
    target = "UMAREN_ARE"
    index_list = ["RACE_KEY", "NENGAPPI"]
    obj_column_list = ['UMAREN_ARE', 'UMATAN_ARE', 'SANRENPUKU_ARE']
    obj_column_list_tr = ['UMAREN_ARE_tr', 'UMATAN_ARE_tr', 'SANRENPUKU_ARE_tr']
    import_list = ["RACE_KEY", "pred", "prob", "target", "target_date"]

    def setUp(self):
        """ テスト実施前に必要な処理を記載する。呼び出しクラスやフォルダの指定等 """
        model_version = 'lbr_v1'
        table_name = '地方競馬レースV1'
        self.intermediate_folder = self.dict_path + 'intermediate/' + model_version + '_' + self.mode + '/' + self.model_name + '/'
        self.skmodel = LBRv1SkModel(self.model_name, model_version, self.start_date, self.end_date, self.mock_flag, self.test_flag, self.mode)
        table_name = table_name + "_test"
        self.skmodel.set_test_table(table_name)
        self._proc_check_folder()