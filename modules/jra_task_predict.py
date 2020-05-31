import luigi
from luigi.util import requires

import modules.util as mu
from modules.base_slack import OperationSlack
import modules.util as mu

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
    test_flag = luigi.Parameter()
    intermediate_folder = luigi.Parameter()
    export_mode = luigi.Parameter()

    def run(self):
        """
        渡されたexp_data_nameに基づいてSK_DATA_MODELから説明変数のデータを取得する処理を実施。pickelファイル形式でデータを保存
        """
        print("----" + __class__.__name__ + ": run")
        mu.create_folder(self.intermediate_folder)
        if not self.test_flag:
            slack = OperationSlack()
            slack.post_slack_text(dt.now().strftime("%Y/%m/%d %H:%M:%S") + " start predict job:" + self.skmodel.version_str)
        with self.output().open("w") as target:
            predict_df = self.skmodel.create_predict_data()
            print("Sub_get_exp_data run: predict_df", predict_df.shape)
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
            # 予測を実施して予測結果ファイルを作成
            all_df = self.skmodel.proc_predict_sk_model(exp_data)
            pd.set_option('display.max_columns', 3000)
            pd.set_option('display.max_rows', 3000)
            print(all_df.head(20))
            if self.test_flag:
                print("精度チェック")
                self.skmodel.eval_pred_data(all_df)
            else:
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
