import luigi
from luigi.mock import MockTarget
from modules.base_slack import OperationSlack

import os
import shutil
import pickle
from luigi.util import requires
from datetime import datetime as dt


class Sub_get_learning_data(luigi.Task):
    # 学習に必要なデータを作成する処理。競馬場ごと（各学習用）と全レコード（特徴量作成用）のデータを作成する
    task_namespace = 'base_learning'
    start_date = luigi.Parameter()
    end_date = luigi.Parameter()
    skmodel = luigi.Parameter()
    intermediate_folder = luigi.Parameter()

    def run(self):
        # SkModelを読んで学習データを作成する。すべてのデータを作成後、競馬場毎のデータを作成する
        print("----" + __class__.__name__ + ": run")
        slack = OperationSlack()
        slack.post_slack_text(dt.now().strftime("%Y/%m/%d %H:%M:%S") + " start Sub_get_learning_data job:" + self.skmodel.version_str)
        with self.output().open("w") as target:
            print("------ learning_dfを作成")
            self.skmodel.create_learning_data()
            print("------ 分類軸用の列を削除")
            save_learning_df = self.skmodel.get_all_learning_df_for_save()
            print("------ 学習用の全データを保存")
            save_learning_df.to_pickle(self.intermediate_folder + '_learning_all.pkl')

            print("------ 分類軸毎の学習処理を開始")
            class_list = self.skmodel.class_list
            for cls_val in class_list:
                print("------ " + cls_val + "毎のデータを抽出して保存")
                val_list = self.skmodel.get_val_list(self.skmodel.learning_df, cls_val)
                val_list.to_pickle(self.intermediate_folder + cls_val + "_list.pkl")
                for val in val_list:
                    filter_df = self.skmodel.get_filter_df(self.skmodel.learning_df, cls_val, val)
                    print(filter_df.shape)
                    filter_df.to_pickle(self.intermediate_folder + "learning_" + cls_val + "_" + val + ".pkl")

            slack.post_slack_text(
                dt.now().strftime("%Y/%m/%d %H:%M:%S") + " finish Sub_get_learning_data job:" + self.skmodel.version_str)
            print(__class__.__name__ + " says: task finished".format(task=self.__class__.__name__))

    def output(self):
        # データ作成処理済みフラグを作成する
        return luigi.LocalTarget(format=luigi.format.Nop, path=self.intermediate_folder + __class__.__name__)

@requires(Sub_get_learning_data)
class Sub_create_feature_select_data(luigi.Task):
    # 特徴量を作成するための処理。target encodingやborutaによる特徴選択の処理を実行
    task_namespace = 'base_learning'

    def requires(self):
        # 前提条件：各学習データが作成されていること
        print("---" + __class__.__name__+ " : requires")
        return Sub_get_learning_data()

    def run(self):
        # 特徴量作成処理を実施。learningの全データ分を取得してSkModel特徴作成処理を実行する
        print("---" + __class__.__name__ + ": run")
        slack = OperationSlack()
        slack.post_slack_text(dt.now().strftime("%Y/%m/%d %H:%M:%S") + " start Sub_create_feature_select_data job:" + self.skmodel.version_str)
        with self.output().open("w") as target:
            file_name = self.intermediate_folder + "_learning_all.pkl"
            with open(file_name, 'rb') as f:
                learning_df = pickle.load(f)
                self.skmodel.create_featrue_select_data(learning_df)
            slack.post_slack_text(dt.now().strftime(
                "%Y/%m/%d %H:%M:%S") + " finish Sub_create_feature_select_data job:" + self.skmodel.version_str)
            print(__class__.__name__ + " says: task finished".format(task=self.__class__.__name__))

    def output(self):
        # 処理済みフラグファイルを作成する
        return luigi.LocalTarget(format=luigi.format.Nop, path=self.intermediate_folder + __class__.__name__)



@requires(Sub_create_feature_select_data)
class End_baoz_learning(luigi.Task):
    # BAOZ用の予測モデルを作成する
    # Sub_get_learning_data -> End_baoz_learningの順で実行する
    task_namespace = 'base_learning'

    def requires(self):
        # 学習用のデータを取得する
        print("---" + __class__.__name__+ " : requires")
        return Sub_create_feature_select_data()

    def run(self):
        # 目的変数、場コード毎に学習を実施し、学習モデルを作成して中間フォルダに格納する
        print("---" + __class__.__name__ + ": run")
        slack = OperationSlack()
        slack.post_slack_text(dt.now().strftime("%Y/%m/%d %H:%M:%S") + " start End_baoz_learning job:" + self.skmodel.version_str)
        self.create_folder()
        with self.output().open("w") as target:
            print("------ 分類軸毎の学習モデルを作成")
            class_list = self.skmodel.class_list
            for cls_val in class_list:
                print("------ " + cls_val + "毎のデータを抽出して処理を実施")
                file_name = self.intermediate_folder + cls_val + "_list.pkl"
                created_model_list = [s for s in os.listdir(self.skmodel.model_folder + 'third/') if cls_val in s]
                with open(file_name, 'rb') as f:
                    val_list = pickle.load(f)
                    for val in val_list:
                        # 対象の競馬場のデータを取得する
                        print(" cls_val:" + cls_val + " val:" + val)
                        created_model_list_val = [s for s in created_model_list if val in s]
                        print(created_model_list_val)
                        if len(created_model_list_val) == len(self.skmodel.obj_column_list):
                            print("\r\n ----- skip create model ---- \r\n")
                        else:
                            file_name = self.intermediate_folder + "learning_" + cls_val + "_" + val + ".pkl"
                            with open(file_name, 'rb') as f:
                                df = pickle.load(f)
                                # 学習を実施
                                # check_df = df.dropna()
                                # print(df.shape)
                                # print(check_df.shape)
                                # if not check_df.empty:
                                self.skmodel.proc_learning_sk_model(df, cls_val, val)
            slack.post_slack_text(dt.now().strftime("%Y/%m/%d %H:%M:%S") +
                " finish End_baoz_learning job:" + self.skmodel.version_str)
            print(__class__.__name__ + " says: task finished".format(task=self.__class__.__name__))


    def create_folder(self):
        for folder in ['first/train/', 'first/test/', 'second/train/', 'second/test/', 'third/train/', 'third/test/', 'third/param/']:
            int_folder = self.intermediate_folder + folder
            if not os.path.exists(int_folder):
                os.makedirs(int_folder)
            model_folder = self.skmodel.model_folder + folder
            if not os.path.exists(model_folder):
                os.makedirs(model_folder)

    def output(self):
        # 学習は何度も繰り返せるようにMockのoutputを返す
        return MockTarget("output")

