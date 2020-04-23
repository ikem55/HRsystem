from modules.base_sk_model import BaseSkModel
from testmodules.test_base_common import TestBaseCommon
import modules.util as mu

import my_config as mc
import os
import pandas as pd
import sys
import shutil

class TestBaseTaskPredict(TestBaseCommon):
    """ Learning処理を実施できることを確認するためのテスト """
    start_date = '2018/02/01'
    end_date = '2018/02/01'
    mode = 'predict'
    model_name = 'raceuma_ens'
    mock_flag = False
    test_flag = True
    dict_path = mc.return_base_path(test_flag)
    clean_flag = False
    index_list = ["RACE_KEY", "UMABAN", "NENGAPPI"]
    import_list = ["RACE_KEY", "UMABAN", "pred", "prob", "predict_std", "predict_rank", "target", "target_date"]

    def setUp(self):
        """ テスト実施前に必要な処理を記載する。呼び出しクラスやフォルダの指定等 """
        model_version = 'base'
        table_name = '地方競馬レース馬'
        self.intermediate_folder = self.dict_path + 'intermediate/' + model_version + '_' + self.mode + '/' + self.model_name + '/'
        self.skmodel = BaseSkModel(self.model_name, model_version, self.start_date, self.end_date, self.mock_flag, self.test_flag, self.mode)
        table_name = table_name + "_test"
        self.skmodel.set_test_table(table_name)
        self._proc_check_folder()

    def test_00_preprocess(self):
        """ テストを実施する前の前処理（フォルダのクリーンとか） """
        print("--  " + sys._getframe().f_code.co_name + " start --")
        intermediate_folder = self.intermediate_folder
        if self.clean_flag:
            shutil.rmtree(intermediate_folder)

    def test_01_create_predict_data(self):
        """ predict_dfを問題なく作成できることを確認 """
        print("--  " + sys._getframe().f_code.co_name + " start --")
        predict_file_name = self.intermediate_folder + mu.convert_date_to_str(self.end_date) + '_exp_data.pkl'
        if not os.path.exists(predict_file_name):
            df = self.skmodel.create_predict_data()
            # not empty check
            self.assertFalse(len(df.index) == 0)
            # columns check
            # 分類軸用の列があるか確認
            contain_columns_set = set(self.skmodel.class_list + self.index_list)
            contain_check = self.proc_test_contain_columns_check(df, contain_columns_set)
            self.assertTrue(contain_check)
            # データ区分等不要な項目がないか確認
            contain_not_columns_set = set(["WIN_FLAG", "JIKU_FLAG", "ANA_FLAG", "確定着順"])
            not_contain_check = self.proc_test_not_contain_columns_check(df, contain_not_columns_set)
            self.assertTrue(not_contain_check)
            # value check

            # 後続処理のためにデータを保存
            df.to_pickle(predict_file_name)

    def test_11_proc_predict_sk_model(self):
        """ 予測データの作成ができることを確認 """
        print("--  " + sys._getframe().f_code.co_name + " start --")
        predict_file_name = self.intermediate_folder + mu.convert_date_to_str(self.end_date) + '_exp_data.pkl'
        exp_data = pd.read_pickle(predict_file_name)
        all_pred_df = pd.DataFrame()
        class_list = self.skmodel.class_list
        for cls_val in class_list:
            val_list = self.skmodel.get_val_list(exp_data, cls_val)
            for val in val_list:
                # 対象の競馬場のデータを取得する
                filter_df = self.skmodel.get_filter_df(exp_data, cls_val, val)
                # 予測を実施
                check_df = filter_df.dropna()
                if not check_df.empty:
                    pred_df = self.skmodel.proc_predict_sk_model(filter_df, cls_val, val)
                    all_pred_df = pd.concat([all_pred_df, pred_df])
                break
        all_pred_df.dropna(inplace=True)
        import_df = self.skmodel.create_import_data(all_pred_df)
        # self.skmodel.eval_pred_data(import_df)
        # not empty check
        self.assertFalse(len(import_df.index) == 0)
        # 必要な列があるかチェック
        contain_columns_set = set(self.import_list)
        contain_check = self.proc_test_contain_columns_check(import_df, contain_columns_set)
        self.assertTrue(contain_check)
