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
    end_date = '2018/01/31'
    mode = 'learning'
    model_name = 'raceuma_ens'
    mock_flag = False
    test_flag = True
    dict_path = mc.return_base_path(test_flag)
    clean_flag = False

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
        if not os.path.exists(self.intermediate_folder + '_learning_all.pkl'):
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
            save_learning_df = self.skmodel.get_all_learning_df_for_save()
            save_learning_df.to_pickle(self.intermediate_folder + '_learning_all.pkl')
            df.to_pickle(self.intermediate_folder + '_learning.pkl')

    def test_02_check_dimension(self):
        """ 分類軸毎にデータを分割できることを確認。test_01の結果を使いたい """
        print("--  " + sys._getframe().f_code.co_name + " start --")
        file_name = self.intermediate_folder + "_learning.pkl"
        contain_columns_set = set(["WIN_FLAG", "JIKU_FLAG", "ANA_FLAG"])
        contain_not_columns_set = set(["データ区分_x"])
        df_list = os.listdir(self.intermediate_folder)
        with open(file_name, 'rb') as f:
            df = pickle.load(f)
            class_list = self.skmodel.class_list
            for cls_val in class_list:
                check_list = [s for s in df_list if "learning_" + cls_val in s]
                if len(check_list) == 0:
                    val_list = self.skmodel.get_val_list(df, cls_val)
                    # val_listが空でないことを確認
                    self.assertFalse(len(val_list) == 0)
                    val_list.to_pickle(self.intermediate_folder + cls_val + "_list.pkl")
                    for val in val_list:
                        filter_df = self.skmodel.get_filter_df(df, cls_val, val)
                        # filter_dfが空でないことを確認
                        self.assertFalse(len(filter_df.index) == 0)
                        # 必要な項目がちゃんとあるか確認
                        contain_check = self.proc_test_contain_columns_check(filter_df, contain_columns_set)
                        self.assertTrue(contain_check)
                        # 不要な項目がないか確認
                        not_contain_check = self.proc_test_not_contain_columns_check(df, contain_not_columns_set)
                        self.assertTrue(not_contain_check)
                        filter_df.to_pickle(self.intermediate_folder + "learning_" + cls_val + "_" + val + ".pkl")
                        break

    def test_11_create_feature_select_data(self):
        """ 特徴量作成処理を問題なくできることを確認。test_01の結果を使いたい。すでに作成に成功している場合はスキップ """
        print("--  " + sys._getframe().f_code.co_name + " start --")
        dict_list = os.listdir(self.skmodel.dict_folder)
        check_dict = ["WIN_FLAG_tr", "JIKU_FLAG_tr", "ANA_FLAG_tr"]
        check_flag = False
        for check in check_dict:
            check_list = [s for s in dict_list if check in s]
            if len(check_list) == 0:
                check_flag = True
        if check_flag:
            file_name = self.intermediate_folder + "_learning_all.pkl"
            with open(file_name, 'rb') as f:
                learning_df = pickle.load(f)
                self.skmodel.create_featrue_select_data(learning_df)

    def test_21_proc_learning_sk_model(self):
        """ 学習モデルの作成が問題なくできることを確認。test_02の結果を使いたい"""
        print("--  " + sys._getframe().f_code.co_name + " start --")
        ### 途中から実行できるようにしたいがファイル処理を考えないといけない。
        self.create_folder()
        te_p = self.intermediate_folder
        model_third_folder = self.skmodel.ens_folder_path + self.model_name +'/third/'
        class_list = self.skmodel.class_list
        for cls_val in class_list:
            print(cls_val)
            file_name = self.intermediate_folder + cls_val + "_list.pkl"
            created_model_list = [s for s in os.listdir(self.skmodel.model_folder + 'third/') if cls_val in s]
            with open(file_name, 'rb') as f:
                val_list = pickle.load(f)
                tr_list = [s for s in os.listdir(te_p) if cls_val in s]
                for val in val_list:
                    print(val)
                    created_model_list_val = [s for s in created_model_list if val in s]
                    print(created_model_list_val)
                    if len(created_model_list_val) == len(self.skmodel.obj_column_list):
                        print("-----------------------------\r\n --- skip create learning model -- \r\n")
                    else:
                        data_file_name = [s for s in tr_list if val in s]
                        print(data_file_name)
                        with open(self.intermediate_folder + data_file_name[0], 'rb') as f:
                            df = pickle.load(f)
                            # 学習を実施
                            check_df = df.dropna()
                            if not check_df.empty:
                                self.skmodel.proc_learning_sk_model(df, cls_val, val)
                    break
