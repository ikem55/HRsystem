import luigi
from luigi.util import requires

import modules.util as mu
from modules.base_slack import OperationSlack

import pandas as pd
from datetime import datetime as dt

class Sub_get_exp_data(luigi.Task):
    """
    ScikitLearnのモデルで説明変数となるデータを生成するタスク。baoz_intermediateフォルダにデータが格納される
    """
    task_namespace = 'base_predict'
    start_date = luigi.Parameter()
    end_date = luigi.Parameter()
    skmodel = luigi.Parameter()
    intermediate_folder = luigi.Parameter()
    export_mode = luigi.Parameter()

    def run(self):
        """
        渡されたexp_data_nameに基づいてSK_DATA_MODELから説明変数のデータを取得する処理を実施。pickelファイル形式でデータを保存
        """
        print("----" + __class__.__name__ + ": run")
        slack = OperationSlack()
        slack.post_slack_text(dt.now().strftime("%Y/%m/%d %H:%M:%S") + " start predict job:" + self.skmodel.version_str)
        with self.output().open("w") as target:
            print("------ モデル毎に予測データが違うので指定してデータ作成を実行")
            predict_df = self.skmodel.create_predict_data()
            print(predict_df.shape)
            predict_df.to_pickle(self.intermediate_folder + mu.convert_date_to_str(self.end_date) + '_exp_data.pkl')
            print(__class__.__name__ + " says: task finished".format(task=self.__class__.__name__))

    def output(self):
        """
        :return: MockのOutputを返す
        """
#        return MockTarget("output")
        return luigi.LocalTarget(format=luigi.format.Nop, path=self.intermediate_folder + mu.convert_date_to_str(self.end_date) + "_" + __class__.__name__ + "_" + self.skmodel.model_name)


@requires(Sub_get_exp_data)
class End_baoz_predict(luigi.Task):
    """
    ScikitLearnのモデルを元に予測値を計算するLuigiタスク
    """
    task_namespace = 'base_predict'

    def requires(self):
        """
        | 処理対象日ごとにSub_predict_dataを呼び出して予測データを作成する
        """
        print("---" + __class__.__name__+ " : requires")
        return Sub_get_exp_data()

    def run(self):
        """
        | 処理の最後
        """
        print("---" + __class__.__name__ + ": run")
        with self.output().open("w") as target:
            exp_data = pd.read_pickle(self.intermediate_folder + mu.convert_date_to_str(self.end_date) + '_exp_data.pkl')
            print("------ 分類軸毎の学習モデルを作成")
            all_pred_df = pd.DataFrame()
            class_list = self.skmodel.class_list
            for cls_val in class_list:
                print("------ " + cls_val + "毎のデータを抽出して処理を実施")
                val_list = self.skmodel.get_val_list(exp_data, cls_val)
                for val in val_list:
                    # 対象の競馬場のデータを取得する
                    print("=============== cls_val:" + cls_val + " val:" + val + " ===========================")
                    filter_df = self.skmodel.get_filter_df(exp_data, cls_val, val)
                    # 予測を実施
                    check_df = filter_df.dropna()
                    print(filter_df.shape)
                    print(check_df.shape)
                    if not check_df.empty:
                        pred_df = self.skmodel.proc_predict_sk_model(filter_df, cls_val, val)
                        all_pred_df = pd.concat([all_pred_df, pred_df])
            print(all_pred_df.head())
            all_pred_df.dropna(inplace=True)
            print(all_pred_df.head())
            import_df = self.skmodel.create_import_data(all_pred_df)
            if self.export_mode:
                print("export data")
                import_df.to_pickle(self.intermediate_folder + 'export_data.pkl')
            else:
                self.skmodel.import_data(import_df)
            slack = OperationSlack()
            slack.post_slack_text(dt.now().strftime("%Y/%m/%d %H:%M:%S") +
                " finish predict job:" + self.skmodel.version_str)
            print(__class__.__name__ + " says: task finished".format(task=self.__class__.__name__))

    def output(self):
        """
        :return: MockのOutputを返す
        """
#        return MockTarget("output")
        return luigi.LocalTarget(format=luigi.format.Nop, path=self.intermediate_folder + mu.convert_date_to_str(self.end_date) + "_" + __class__.__name__ + "_" + self.skmodel.model_name)
