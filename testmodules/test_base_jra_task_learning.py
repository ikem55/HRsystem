from modules.base_sk_model import BaseSkModel
from testmodules.test_base_common import TestBaseCommon

import my_config as mc
import modules.util as mu
import os
import pickle
import sys
import shutil

class TestBaseTaskLearning(TestBaseCommon):
    """ Learning処理を実施できることを確認するためのテスト """
    start_date = '2018/01/01'
    end_date = '2018/01/11'
    mode = 'learning'
    model_name = 'race_lgm'
    mock_flag = False
    test_flag = True
    dict_path = mc.return_base_path(test_flag)
    clean_flag = False
    target = "WIN_FLAG"
    obj_column_list = ["WIN_FLAG", "JIKU_FLAG", "ANA_FLAG"]
    obj_column_list_tr = ["WIN_FLAG_tr", "JIKU_FLAG_tr", "ANA_FLAG_tr"]

    def setUp(self):
        """ テスト実施前に必要な処理を記載する。呼び出しクラスやフォルダの指定等 """
        model_version = 'base'
        self.intermediate_folder = self.dict_path + 'intermediate/' + model_version + '_' + self.mode + '/' + self.model_name + '/'
        self.skmodel = BaseSkModel(self.model_name, model_version, self.start_date, self.end_date, self.mock_flag, self.test_flag, self.mode)
        self._proc_check_folder()

    def test_00_preprocess(self):
        """ テストを実施する前の前処理（フォルダのクリーンとか） """
        print("--  " + sys._getframe().f_code.co_name + " start --")
        model_folder = self.skmodel.model_folder
        dict_folder = self.skmodel.dict_folder
        intermediate_folder = self.intermediate_folder
        if self.clean_flag:
            shutil.rmtree(model_folder)
            shutil.rmtree(dict_folder)
            shutil.rmtree(intermediate_folder)


    def test_01_create_learning_data(self):
        """ learning_dfを問題なく作成できることを確認 """
        print("--  " + sys._getframe().f_code.co_name + " start --")
        if not os.path.exists(self.intermediate_folder + '_learning_df.pkl'):
            self.skmodel.create_learning_data()
            df = self.skmodel.learning_df
            # not empty check
            self.assertFalse(len(df.index) == 0)
            # columns check
            # 分類軸用の列があるか確認
            contain_columns_set = set(self.skmodel.class_list)
            contain_check = self.proc_test_contain_columns_check(df, contain_columns_set)
            self.assertTrue(contain_check)
            # データ区分等不要な項目がないか確認
            contain_not_columns_set = set(['データ区分_x'])
            not_contain_check = self.proc_test_not_contain_columns_check(df, contain_not_columns_set)
            self.assertTrue(not_contain_check)
            # value check

            # 後続処理のためにデータを保存
            df.to_pickle(self.intermediate_folder + '_learning.pkl')


    def test_11_create_feature_select_data(self):
        """ 特徴量作成処理を問題なくできることを確認。test_01の結果を使いたい。すでに作成に成功している場合はスキップ """
        print("--  " + sys._getframe().f_code.co_name + " start --")
        dict_list = os.listdir(self.skmodel.dict_folder)
        check_dict = self.obj_column_list_tr
        check_flag = False
        for check in check_dict:
            check_list = [s for s in dict_list if check in s]
            if len(check_list) == 0:
                check_flag = True
        if check_flag:
            file_name = self.intermediate_folder + "_learning_df.pkl"
            with open(file_name, 'rb') as f:
                learning_df = pickle.load(f)
                self.skmodel.create_featrue_select_data(learning_df)

    def test_21_proc_learning_sk_model(self):
        """ 学習モデルの作成が問題なくできることを確認。test_02の結果を使いたい"""
        print("--  " + sys._getframe().f_code.co_name + " start --")
        self.create_folder()
        te_p = self.intermediate_folder
        with open(self.intermediate_folder + '_learning.pkl', 'rb') as f:
            df = pickle.load(f)
            # 学習を実施
            print(self.target)
            self.skmodel.proc.learning_sk_model(df, self.target)
