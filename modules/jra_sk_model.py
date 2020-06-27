from modules.base_sk_model import BaseSkModel
from modules.jra_sk_proc import JRASkProc
from modules.jra_extract import JRAExtract
import pandas as pd
import modules.util as mu
import os
import glob
from datetime import datetime as dt
from datetime import timedelta

class JRASkModel(BaseSkModel):
    """
    地方競馬の機械学習モデルを定義
    """
    version_str = 'jra'
    model_path = ""
    pred_folder = ""

    """ 予測結果を格納するフォルダ """

    def _get_skproc_object(self, version_str, start_date, end_date, model_name, mock_flag, test_flag):
        proc = JRASkProc(version_str, start_date, end_date, model_name, mock_flag, test_flag, self.obj_column_list)
        return proc

    def _set_folder_path(self, mode):
        self.model_path = self.dict_path + 'model/' + self.version_str + '/'
        self.dict_folder = self.dict_path + 'dict/' + self.version_str + '/'
        self.pred_folder = self.dict_path + 'pred/' + self.version_str + '/'
        mu.create_folder(self.model_path)
        mu.create_folder(self.dict_folder)
        mu.create_folder(self.pred_folder)

    def proc_learning_sk_model(self, df):
        """  説明変数ごとに、指定された場所の学習を行う

        :param dataframe df: dataframe
        :param str basho: str
        """
        for target in self.obj_column_list:
            print(target)
            self.proc.learning_sk_model(df, target)


    def proc_predict_sk_model(self, df):
        """ predictする処理をまとめたもの。指定されたbashoのターゲットフラグ事の予測値を作成して連結したものをdataframeとして返す

        :param dataframe df: dataframe
        :param str val: str
        :return: dataframe
        """
        all_df = pd.DataFrame()
        if not df.empty:
            for target in self.obj_column_list:
                pred_df = self.proc._predict_sk_model(df, target)
                print(pred_df.shape)
                if not pred_df.empty:
                    pred_df["target"] = target
                    pred_df["model_name"] = self.model_name
                    date_list = sorted(pred_df["target_date"].drop_duplicates().tolist())
                    for date in date_list:
                        target_df = pred_df[pred_df["target_date"] == date]
                        target_df = target_df.sort_values(["RACE_KEY", "target", "predict_rank"])
                        if len(target_df["RACE_KEY"].drop_duplicates().tolist()) <= 10:
                            print("数が少ないのでスキップ")
                        else:
                            target_df.to_pickle(self.pred_folder + target + "_" + date + ".pickle")
                    all_df = pd.concat([all_df, pred_df]).round(3)
        return all_df

    def eval_pred_data(self, df):
        """ 予測されたデータの精度をチェック """
        self.proc._set_target_variables()
        result_df = self.proc.result_df
        for target in self.obj_column_list:
            print(target)
            target_df = df[df["target"] == target]
            check_df = self._eval_check_df(result_df, target_df, target)
            avg_rate = check_df["的中"].mean()
            print(round(avg_rate*100, 1))

    def _eval_check_df(self, result_df, target_df, target):
        target_df = target_df.query("predict_rank == 1")
        temp_df = result_df[["RACE_KEY", "UMABAN", target]].rename(columns={target: "result"})
        check_df = pd.merge(target_df, temp_df, on=["RACE_KEY", "UMABAN"])
        check_df.loc[:, "的中"] = check_df["result"].apply(lambda x: 1 if x == 1 else 0)
        return check_df

    @classmethod
    def get_recent_day(cls, base_start_date, pred_folder):
        file_list = glob.glob(pred_folder + "/*.pickle")
        file_df = pd.DataFrame({"filename": file_list})
        if file_df.empty:
            return base_start_date
        else:
            file_df.loc[:, "target_date"] = file_df["filename"].str[-15:-7]
            max_date = file_df["target_date"].max()
            start_date = (dt.strptime(max_date, '%Y%m%d') + timedelta(days=1)).strftime('%Y/%m/%d')
            return start_date
"""
    def del_create_import_data(self, all_df): #いらない？
        all_df.dropna(inplace=True)
        grouped_all_df = all_df.groupby(["RACE_KEY", "UMABAN", "target"], as_index=False).mean()
        date_df = all_df[["RACE_KEY", "target_date"]].drop_duplicates()
        temp_grouped_df = pd.merge(grouped_all_df, date_df, on="RACE_KEY")
        grouped_df = self._calc_grouped_data(temp_grouped_df)
        import_df = grouped_df[["RACE_KEY", "UMABAN", "pred", "prob", "predict_std", "predict_rank", "target", "target_date"]].round(3)
        print(import_df)
        return import_df
"""
