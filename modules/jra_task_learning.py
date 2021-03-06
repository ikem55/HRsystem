import luigi
from luigi.mock import MockTarget
from modules.base_slack import OperationSlack
import modules.util as mu
import pickle
from luigi.util import requires
from datetime import datetime as dt


class Sub_get_learning_data(luigi.Task):
    # 学習に必要なデータを作成する処理。競馬場ごと（各学習用）と全レコード（特徴量作成用）のデータを作成する
    task_namespace = 'base_learning'
    start_date = luigi.Parameter()
    end_date = luigi.Parameter()
    skmodel = luigi.Parameter()
    test_flag = luigi.Parameter()
    intermediate_folder = luigi.Parameter()

    def run(self):
        # SkModelを読んで学習データを作成する。すべてのデータを作成後、競馬場毎のデータを作成する
        print("----" + __class__.__name__ + ": run")
        mu.create_folder(self.intermediate_folder)
        if not self.test_flag:
            slack = OperationSlack()
            slack.post_slack_text(dt.now().strftime("%Y/%m/%d %H:%M:%S") + " start Sub_get_learning_data job:" + self.skmodel.version_str)
        with self.output().open("w") as target:
            print("------ learning_dfを作成")
            self.skmodel.create_learning_data()
            print("------ 学習用のデータを保存")
            self.skmodel.learning_df.to_pickle(self.intermediate_folder + '_learning_df.pkl')
            if not self.test_flag:
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
        if not self.test_flag:
            slack = OperationSlack()
            slack.post_slack_text(dt.now().strftime("%Y/%m/%d %H:%M:%S") + " start Sub_create_feature_select_data job:" + self.skmodel.version_str)
        with self.output().open("w") as target:
            file_name = self.intermediate_folder + "_learning_df.pkl"
            with open(file_name, 'rb') as f:
                learning_df = pickle.load(f)
                self.skmodel.create_featrue_select_data(learning_df)
            if not self.test_flag:
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
        # 目的変数毎に学習を実施し、学習モデルを作成して中間フォルダに格納する
        print("---" + __class__.__name__ + ": run")
        if not self.test_flag:
            slack = OperationSlack()
            slack.post_slack_text(dt.now().strftime("%Y/%m/%d %H:%M:%S") + " start End_baoz_learning job:" + self.skmodel.version_str)

        file_name = self.intermediate_folder + "_learning_df.pkl"
        with open(file_name, 'rb') as f:
            df = pickle.load(f)
            # 学習を実施
            self.skmodel.proc_learning_sk_model(df)

        if not self.test_flag:
            slack.post_slack_text(dt.now().strftime("%Y/%m/%d %H:%M:%S") +
                " finish End_baoz_learning job:" + self.skmodel.version_str)
        print(__class__.__name__ + " says: task finished".format(task=self.__class__.__name__))


    def output(self):
        # 学習は何度も繰り返せるようにMockのoutputを返す
        return MockTarget("output")

