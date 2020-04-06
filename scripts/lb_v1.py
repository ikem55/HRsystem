from modules.lb_extract import LBExtract
from modules.lb_transform import LBTransform
from modules.lb_load import LBLoad
from modules.lb_sk_model import LBSkModel
from modules.lb_sk_proc import LBSkProc
import modules.util as mu
import my_config as mc

import luigi
from modules.base_task_learning import End_baoz_learning
from modules.base_task_predict import End_baoz_predict

from factor_analyzer import FactorAnalyzer
from datetime import datetime as dt
from datetime import timedelta
import sys
import pandas as pd
import numpy as np
import os
from distutils.util import strtobool

basedir = os.path.dirname(__file__)[:-8]
print(basedir)
sys.path.append(basedir)

# 呼び出し方
# python lb_v1.py learning True True
# ====================================== パラメータ　要変更 =====================================================

MODEL_VERSION = 'lb_v1'
MODEL_NAME = 'raceuma_ens'
TABLE_NAME = '地方競馬レース馬V1'

# ============================================================================================================

CONN_STR = (
    r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
    r'DBQ=C:\BaoZ\DB\MasterDB\MyDB.MDB;'
)

# ====================================== クラス　要変更 =========================================================

class Ext(LBExtract):
    pass

class Tf(LBTransform):
    def factory_analyze_raceuma_result_df(self, race_df, input_raceuma_df, dict_folder):
        """ RaceUmaの因子分析を行うためのデータを取得 """
        print("factory_analyze_raceuma_result_df")
        temp_df = pd.merge(input_raceuma_df, race_df, on="競走コード")
        X = temp_df[
            ['競走コード', '馬番', 'タイム指数', '単勝オッズ', '着差', '先行率', 'ペース偏差値', 'コーナー順位4', '距離増減', '斤量比', '追込率', '平均タイム',
             "cos_day", "sin_day", "距離", "頭数", "非根幹", "上り係数", "逃げ勝ち", "内勝ち", "外勝ち", "短縮勝ち", "延長勝ち", "人気勝ち"]]

        mmsc_columns = ["頭数"]
        mmsc_dict_name = "sc_fa_race_mmsc"
        stdsc_columns = ["距離"]
        stdsc_dict_name = "sc_fa_race_stdsc"
        X = mu.scale_df_for_fa(X, mmsc_columns, mmsc_dict_name, stdsc_columns, stdsc_dict_name, dict_folder)

        X_fact = X.drop(["競走コード", "馬番"], axis=1)

        X_fact = X_fact.replace(np.inf, np.nan).fillna(X_fact.median()).fillna(0)
        X_fact.iloc[0] = X_fact.iloc[0] + 0.000001

        dict_name = "fa_raceuma_result_df"
        filename = dict_folder + dict_name + '.pkl'
        if os.path.exists(filename):
            fa = mu.load_dict(dict_name, dict_folder)
        else:
            fa = FactorAnalyzer(n_factors=3, rotation='promax', impute='drop')
            fa.fit(X_fact)
            mu.save_dict(fa, dict_name, dict_folder)

        fa_np = fa.transform(X_fact)
        fa_df = pd.DataFrame(fa_np, columns=["fa_1", "fa_2", "fa_3"])
        fa_df = pd.concat([X[["競走コード", "馬番"]], fa_df], axis=1)
        X_fact = pd.merge(input_raceuma_df, fa_df, on=["競走コード", "馬番"])
        return X_fact

class Ld(LBLoad):

    def _get_extract_object(self, start_date, end_date, mock_flag):
        """ 利用するExtクラスを指定する """
        ext = Ext(start_date, end_date, mock_flag)
        return ext

    def _get_transform_object(self, start_date, end_date):
        """ 利用するTransformクラスを指定する """
        tf = Tf(start_date, end_date)
        return tf

    def _proc_scale_df_for_fa(self, raceuma_df):
        mmsc_columns = ["コーナー順位4", "距離増減", "斤量比", "単勝人気", "確定着順", "騎手ランキング", "調教師ランキング", "負担重量"]
        mmsc_dict_name = "sc_fa_mmsc"
        stdsc_columns = ["タイム指数", "単勝オッズ", "着差", "ペース偏差値", "平均タイム", "休養週数", "馬体重", "馬体重増減", "デフォルト得点", "得点V1",
                         "得点V2", "得点V3"]
        stdsc_dict_name = "sc_fa_stdsc"
        raceuma_df = mu.scale_df_for_fa(raceuma_df, mmsc_columns, mmsc_dict_name, stdsc_columns, stdsc_dict_name,self.dict_folder)
        return raceuma_df

    def _get_prev_df(self, num, race_result_df, raceuma_result_df):
        """ numで指定した過去走のデータを取得して、raceuma_base_df,race_base_dfにセットする

        :param int num: number(過去１走前の場合は1)
        """
        prev_race_key = "近走競走コード" + str(num)
        prev_umaban = "近走馬番" + str(num)
        raceuma_base_df = self.raceuma_df[["競走コード", "馬番", prev_race_key, prev_umaban]]
        temp_prev_raceuma_df = raceuma_result_df.rename(columns={"競走コード": prev_race_key, "馬番": prev_umaban})
        this_raceuma_df = pd.merge(raceuma_base_df, temp_prev_raceuma_df, on=[prev_race_key, prev_umaban])
        this_raceuma_df = this_raceuma_df.rename(columns={"競走コード_x": "競走コード", "馬番_x": "馬番"}).drop([prev_race_key, prev_umaban], axis=1)

        race_base_df = raceuma_base_df[["競走コード", "馬番", prev_race_key]]
        temp_prev_race_df = race_result_df.rename(columns={"競走コード": prev_race_key})
        this_race_df = pd.merge(race_base_df, temp_prev_race_df, on=prev_race_key)
        this_race_df = this_race_df.rename(columns={"競走コード_x": "競走コード"}).drop(prev_race_key, axis=1)
        this_race_df.drop(["データ区分"], axis=1, inplace=True)
        this_raceuma_df.drop(["データ区分"], axis=1, inplace=True)
        merged_df = pd.merge(this_race_df, this_raceuma_df, on=["競走コード", "馬番"])
        merged_df = self.tf.normalize_prev_merged_df(merged_df)
        merged_df = merged_df.drop(["年月日", "月日", "距離", "天候コード", "血統登録番号", "所属", "転厩", "後３ハロン","タイム"], axis=1)
        return merged_df


class SkProc(LBSkProc):
    """
    地方競馬の機械学習処理プロセスを取りまとめたクラス。
    """
    def _get_load_object(self, version_str, start_date, end_date, mock_flag, test_flag):
        ld = Ld(version_str, start_date, end_date, mock_flag, test_flag)
        return ld


    def _create_feature(self):
        """ マージしたデータから特徴量を生成する """
        self.base_df.loc[:, "継続騎乗"] = (self.base_df["騎手名"] == self.base_df["騎手名_1"]).astype(int)
        self.base_df.loc[:, "同場騎手"] = (self.base_df["騎手所属場コード"] == self.base_df["場コード"]).astype(int)
        self.base_df.loc[:, "同所属場"] = (self.base_df["調教師所属場コード"] == self.base_df["場コード"]).astype(int)
        self.base_df.loc[:, "同所属騎手"] = (self.base_df["騎手所属場コード"] == self.base_df["調教師所属場コード"]).astype(int)
        self.base_df.loc[:, "同主催者"] = (self.base_df["主催者コード"] == self.base_df["主催者コード_1"]).astype(int)
        self.base_df.loc[:, "同場コード"] = (self.base_df["場コード"] == self.base_df["場コード_1"]).astype(int)
        self.base_df.loc[:, "同根幹"] = (self.base_df["非根幹"] == self.base_df["非根幹_1"]).astype(int)
        self.base_df.loc[:, "同距離グループ"] = (self.base_df["距離グループ"] == self.base_df["距離グループ_1"]).astype(int)
        self.base_df.loc[:, "頭数差"] = self.base_df["頭数"] - self.base_df["頭数_1"]
        self.base_df.loc[:, "前走位置取り変化"] = self.base_df["コーナー順位4_2"] -self. base_df["コーナー順位4_1"]
        self.base_df.loc[:, "休み明け"] = self.base_df["休養週数"].apply(lambda x: True if x >= 20 else False)
        self.base_df.loc[:, "前走凡走"] = self.base_df.apply(lambda x: 1 if (x["単勝人気_1"] < 0.3 and x["確定着順_1"] > 0.3) else 0, axis=1)
        self.base_df.loc[:, "前走激走"] = self.base_df.apply(lambda x: 1 if (x["単勝人気_1"] > 0.5 and x["確定着順_1"] < 0.3) else 0, axis=1)
        self.base_df.loc[:, "前走逃げそびれ"] = self.base_df.apply(lambda x: 1 if (x["予想展開"] == 1 and x["コーナー順位4_1"] > 0.5) else 0, axis=1)
        self.base_df.drop(
            ["非根幹_1", "非根幹_2", "主催者コード_1", "主催者コード_2", "場コード_1", "場コード_2", "距離グループ_1", "距離グループ_2",
             "頭数_1", "頭数_2"],
            axis=1, inplace=True)

class SkModel(LBSkModel):
    class_list = ['競走種別コード', '場コード']
    table_name = TABLE_NAME

    def _get_skproc_object(self, version_str, start_date, end_date, model_name, mock_flag, test_flag):
        proc = SkProc(version_str, start_date, end_date, model_name, mock_flag, test_flag, self.obj_column_list)
        return proc


# ============================================================================================================

if __name__ == "__main__":
    args = sys.argv
    print("------------- start luigi tasks ----------------")
    print(args)
    print("mode：" + args[1])  # learning or predict
    print("mock flag：" + args[2])  # True or False
    print("test mode：" + args[3])  # True or False
    mode = args[1]
    mock_flag = strtobool(args[2])
    test_flag = strtobool(args[3])
    dict_path = mc.return_base_path(test_flag)
    INTERMEDIATE_FOLDER = dict_path + 'intermediate/' + MODEL_VERSION + '_' + args[1] + '/' + MODEL_NAME + '/'
    print("intermediate_folder:" + INTERMEDIATE_FOLDER)

    if mode == "learning":
        if test_flag:
            print("Test mode")
            start_date = '2018/01/01'
            end_date = '2018/01/31'
        else:
            start_date = '2015/01/01'
            end_date = '2018/12/31'
        if mock_flag:
            print("use mock data")
        print("MODE:learning mock_flag:" + str(args[2]) + "  start_date:" + start_date + " end_date:" + end_date)

        sk_model = SkModel(MODEL_NAME, MODEL_VERSION, start_date, end_date, mock_flag, test_flag, mode)

        luigi.build([End_baoz_learning(start_date=start_date, end_date=end_date, skmodel=sk_model,
                                       intermediate_folder=INTERMEDIATE_FOLDER)], local_scheduler=True)

    elif mode == "predict":
        if test_flag:
            print("Test mode")
            start_date = '2018/01/10'
            end_date = '2018/01/11'
        else:
            base_start_date = '2019/01/01'
            start_date = SkModel.get_recent_day(base_start_date)
            end_date = (dt.now() + timedelta(days=0)).strftime('%Y/%m/%d')
            if start_date > end_date:
                print("change start_date")
                start_date = end_date
        if mock_flag:
            print("use mock data")
        print("MODE:predict mock_flag:" + str(args[2]) + "  start_date:" + start_date + " end_date:" + end_date)

        sk_model = SkModel(MODEL_NAME, MODEL_VERSION, start_date, end_date, mock_flag, test_flag, mode)
        if test_flag:
            print("set test table")
            table_name = TABLE_NAME + "_test"
            sk_model.set_test_table(table_name)
            sk_model.create_mydb_table(table_name)

        luigi.build([End_baoz_predict(start_date=start_date, end_date=end_date, skmodel=sk_model,
                                      intermediate_folder=INTERMEDIATE_FOLDER, export_mode=False)], local_scheduler=True)

    else:
        print("error")