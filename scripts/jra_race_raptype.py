from modules.jra_extract import JRAExtract
from modules.jra_transform import JRATransform
from modules.jra_load import JRALoad
from modules.jra_sk_model import JRASkModel
from modules.jra_sk_proc import JRASkProc
import modules.util as mu
import my_config as mc

import luigi
from modules.jra_task_learning import End_baoz_learning
from modules.jra_task_predict import End_baoz_predict

from datetime import datetime as dt
from datetime import timedelta
import sys
import pandas as pd
import numpy as np
import scipy
import os
from distutils.util import strtobool
from sklearn.model_selection import train_test_split
import optuna.integration.lightgbm as lgb

import featuretools as ft
import pickle


basedir = os.path.dirname(__file__)[:-8]
print(basedir)
sys.path.append(basedir)

# 呼び出し方
# python jra_race_raptype.py learning True True
# ====================================== パラメータ　要変更 =====================================================
# ラップタイプの予測を行う。LightGBMを使う

MODEL_VERSION = 'jra_rc_raptype'
MODEL_NAME = 'race_lgm'

# ====================================== クラス　要変更 =========================================================

class Ext(JRAExtract):
    pass

class Tf(JRATransform):
    pass

class Ld(JRALoad):
    def _get_extract_object(self, start_date, end_date, mock_flag):
        """ 利用するExtクラスを指定する """
        ext = Ext(start_date, end_date, mock_flag)
        return ext

    def _get_transform_object(self, start_date, end_date):
        """ 利用するTransformクラスを指定する """
        tf = Tf(start_date, end_date)
        return tf


    def set_result_df(self):
        """ result_dfを作成するための処理。result_dfに処理がされたデータをセットする """
        result_race_df = self.ext.get_race_table_base()[["RACE_KEY", "RAP_TYPE", "TRACK_BIAS_ZENGO", "TRACK_BIAS_UCHISOTO"]]
        result_raceuma_df = self.ext.get_raceuma_table_base().query("着順 == 1")[["RACE_KEY", "レースペース流れ"]].drop_duplicates()
        return pd.merge(result_race_df, result_raceuma_df, on="RACE_KEY")

class SkProc(JRASkProc):
    """
    地方競馬の機械学習処理プロセスを取りまとめたクラス。
    """
    index_list = ["RACE_KEY", "target_date"]
    # LightGBM のハイパーパラメータ
    obj_column_list = ['RAP_TYPE', 'TRACK_BIAS_ZENGO', 'TRACK_BIAS_UCHISOTO', 'PRED_PACE']
    lgbm_params = {
        'RAP_TYPE':{'objective': 'multiclass', 'metric': 'multi_logloss', 'num_class': 10},
        'TRACK_BIAS_ZENGO':{'objective': 'multiclass', 'metric': 'multi_logloss', 'num_class': 5},
        'TRACK_BIAS_UCHISOTO':{'objective': 'multiclass', 'metric': 'multi_logloss', 'num_class': 5},
        'PRED_PACE':{'objective': 'multiclass', 'metric': 'multi_logloss', 'num_class': 10},
                   }

    def _get_load_object(self, version_str, start_date, end_date, mock_flag, test_flag):
        ld = Ld(version_str, start_date, end_date, mock_flag, test_flag)
        return ld

    def _create_feature(self):
        """ マージしたデータから特徴量を生成する """
        print("_create_feature")
        raceuma_df = self.base_df[["RACE_KEY", "UMABAN", "脚質", "距離適性", "父馬産駒連対平均距離", "母父馬産駒連対平均距離", "IDM", "テン指数",
                                   "ペース指数", "上がり指数", "位置指数", "ＩＤＭ結果_1", "テン指数結果_1", "上がり指数結果_1", "ペース指数結果_1", "レースＰ指数結果_1",
                                   "先行率_1", "追込率_1", "fa_1_1", "fa_2_1", "fa_3_1", "fa_4_1", "fa_5_1"]]
        raceuma_df.loc[:, "RACE_UMA_KEY"] = raceuma_df["RACE_KEY"] + raceuma_df["UMABAN"]
        raceuma_df.drop("UMABAN", axis=1, inplace=True)
        # https://qiita.com/daigomiyoshi/items/d6799cc70b2c1d901fb5
        es = ft.EntitySet(id="race")
        es.entity_from_dataframe(entity_id='race', dataframe=self.ld.race_df.drop("target_date", axis=1), index="RACE_KEY")
        es.entity_from_dataframe(entity_id='raceuma', dataframe=raceuma_df, index="RACE_UMA_KEY")
        relationship = ft.Relationship(es['race']["RACE_KEY"], es['raceuma']["RACE_KEY"])
        es = es.add_relationship(relationship)
        print(es)
        # 集約関数
        aggregation_list = ['min', 'max', 'mean', 'skew', 'percent_true']
        transform_list = []
        # run dfs
        print("un dfs")
        feature_matrix, features_dfs = ft.dfs(entityset=es, target_entity='race', agg_primitives=aggregation_list,
                                              trans_primitives=transform_list, max_depth=2)
        print("_create_feature: feature_matrix", feature_matrix.shape)

        # 予想１番人気のデータを取得
        ninki_df = self.base_df.query("基準人気順位==1")[["RACE_KEY", "脚質", "上昇度", "激走指数", "蹄コード", "見習い区分", "展開記号", "騎手期待単勝率", "騎手期待３着内率",
                                                    "距離フラグ", "クラスフラグ", "転厩フラグ", "去勢フラグ", "乗替フラグ", "放牧先ランク", "厩舎ランク", "調教量評価", "仕上指数変化", "調教評価",
                                                    "IDM", "騎手指数", "情報指数", "総合指数", "人気指数", "調教指数", "厩舎指数", "テン指数", "ペース指数", "上がり指数", "位置指数", "追切指数",
                                                    "仕上指数", "ＩＤＭ結果_1", "ＩＤＭ結果_2", "fa_1_1", "fa_2_1", "fa_3_1", "fa_4_1", "fa_5_1",
                                                    "ru_cluster_1", "course_cluster_1"]].groupby("RACE_KEY").first().reset_index().add_prefix("人気_").rename(columns={"人気_RACE_KEY":"RACE_KEY"})
        print(ninki_df.shape)
        # 逃げ予想馬のデータを取得
        nige_df = self.base_df.query("展開記号=='1'")[["RACE_KEY", "脚質", "距離適性", "上昇度", "激走指数", "蹄コード", "見習い区分", "距離フラグ", "クラスフラグ", "乗替フラグ", "IDM", "騎手指数",
                                                    "テン指数", "ペース指数", "上がり指数", "位置指数", "斤量_1", "テン指数結果_1", "上がり指数結果_1", "ペース指数結果_1", "レースＰ指数結果_1",
                                                    "RAP_TYPE_1", "距離", "距離_1", "先行率_1", "ＩＤＭ結果_1","fa_1_1", "fa_2_1", "fa_3_1", "fa_4_1", "fa_5_1", "ru_cluster_1",
                                                    "course_cluster_1"]].add_prefix("逃げ_").rename(columns={"逃げ_RACE_KEY":"RACE_KEY"})
        nige_df.loc[:, "逃げ_距離増減"] = nige_df["逃げ_距離"] - nige_df["逃げ_距離_1"]
        nige_df.drop(["逃げ_距離", "逃げ_距離_1"], axis=1, inplace=True)
        nige_ddf = nige_df.groupby("RACE_KEY")
        nige_df2 = nige_df.loc[nige_ddf["逃げ_テン指数"].idxmax(),: ]
        self.base_df = pd.merge(feature_matrix, nige_df2, on="RACE_KEY", how="left")
        self.base_df = pd.merge(self.base_df, ninki_df, on="RACE_KEY")
        self.base_df = pd.merge(self.base_df, self.ld.race_df[["RACE_KEY", "target_date"]], on="RACE_KEY")

    def _drop_columns_base_df(self):
        self.base_df.drop(['NENGAPPI', 'COURSE_KEY', '発走時間'], axis=1, inplace=True)

    def proc_create_learning_data(self):
        self._proc_create_base_df()
        self._drop_unnecessary_columns()
        self._set_target_variables()
        self.base_df = self.base_df.fillna(self.base_df.median())
        learning_df = pd.merge(self.base_df, self.result_df, on ="RACE_KEY")
        return learning_df

    def proc_create_predict_data(self):
        self._proc_create_base_df()
        self._drop_unnecessary_columns()
        self.base_df = self.base_df.fillna(self.base_df.median())
        return self.base_df

    def _set_target_variables(self):
        self.result_df = self.ld.set_result_df()
        self.result_df['RAP_TYPE'] = self.result_df['RAP_TYPE'].apply(lambda x: mu.encode_rap_type(x))
        self.result_df['TRACK_BIAS_ZENGO'] = self.result_df['TRACK_BIAS_ZENGO'].apply(lambda x: mu._encode_zengo_bias(x))
        self.result_df['TRACK_BIAS_UCHISOTO'] = self.result_df['TRACK_BIAS_UCHISOTO'].apply(lambda x: mu._calc_uchisoto_bias(x))
        self.result_df['PRED_PACE'] = self.result_df['レースペース流れ'].apply(lambda x: mu._encode_race_pace(x))
        self.result_df.drop("レースペース流れ", axis=1, inplace=True)


    def load_learning_target_encoding(self):
        pass

    def _set_predict_target_encoding(self, df):
        return df

    def _sub_create_pred_df(self, temp_df, y_pred):
        pred_df = pd.DataFrame(y_pred, columns=range(y_pred.shape[1]))
        base_df = pd.DataFrame({"RACE_KEY": temp_df["RACE_KEY"], "target_date": temp_df["target_date"]})
        pred_df = pd.concat([base_df, pred_df], axis=1)
        pred_df = pred_df.set_index(["RACE_KEY", "target_date"])
        pred_df = pred_df.stack().reset_index().rename(columns={"level_2": "CLASS", 0: "prob"})
        pred_df = self._calc_grouped_data(pred_df)
        return pred_df

    def _calc_grouped_data(self, df):
        """ 与えられたdataframe(予測値）に対して偏差化とランク化を行ったdataframeを返す

        :param dataframe df: dataframe
        :return: dataframe
        """
        grouped = df.groupby("RACE_KEY")
        grouped_df = grouped.describe()['prob'].reset_index()
        merge_df = pd.merge(df, grouped_df, on="RACE_KEY")
        merge_df['predict_std'] = (
            merge_df['prob'] - merge_df['mean']) / merge_df['std'] * 10 + 50
        df['predict_rank'] = grouped['prob'].rank("dense", ascending=False)
        merge_df = pd.merge(merge_df, df[["RACE_KEY", "CLASS", "predict_rank"]], on=["RACE_KEY", "CLASS"])
        return_df = merge_df[['RACE_KEY', 'CLASS', 'target_date', 'prob', 'predict_std', 'predict_rank']]
        return return_df

class SkModel(JRASkModel):
    obj_column_list = ['RAP_TYPE', 'TRACK_BIAS_ZENGO', 'TRACK_BIAS_UCHISOTO', 'PRED_PACE']

    def _get_skproc_object(self, version_str, start_date, end_date, model_name, mock_flag, test_flag):
        proc = SkProc(version_str, start_date, end_date, model_name, mock_flag, test_flag, self.obj_column_list)
        return proc

    def create_featrue_select_data(self, learning_df):
        pass

    def _eval_check_df(self, result_df, target_df, target):
        target_df = target_df.query("predict_rank == 1")
        temp_df = result_df[["RACE_KEY", target]].rename(columns={target: "result"})
        check_df = pd.merge(target_df, temp_df, on="RACE_KEY")
        check_df.loc[:, "的中"] = check_df.apply(lambda x: 1 if x["CLASS"] == int(x["result"]) else 0, axis=1)
        return check_df


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
            end_date = '2018/12/31'
        else:
            start_date = '2012/01/01'
            end_date = '2018/12/31'
        if mock_flag:
            print("use mock data")
        print("MODE:learning mock_flag: " + str(args[2]) + "  start_date:" + start_date + " end_date:" + end_date)

        sk_model = SkModel(MODEL_NAME, MODEL_VERSION, start_date, end_date, mock_flag, test_flag, mode)

        luigi.build([End_baoz_learning(start_date=start_date, end_date=end_date, skmodel=sk_model, test_flag=test_flag,
                                       intermediate_folder=INTERMEDIATE_FOLDER)], local_scheduler=True)

    elif mode == "predict":
        if test_flag:
            print("Test mode")
            start_date = '2019/01/01'
            end_date = '2019/01/31'
        else:
            base_start_date = '2019/01/01'
            start_date = SkModel.get_recent_day(base_start_date)
            end_date = (dt.now() + timedelta(days=0)).strftime('%Y/%m/%d')
            if start_date > end_date:
                print("change start_date")
                start_date = end_date

        print("MODE:predict mock_flag:" + str(args[2]) + "  start_date:" + start_date + " end_date:" + end_date)

        sk_model = SkModel(MODEL_NAME, MODEL_VERSION, start_date, end_date, mock_flag, test_flag, mode)

        luigi.build([End_baoz_predict(start_date=start_date, end_date=end_date, skmodel=sk_model,test_flag=test_flag,
                                      intermediate_folder=INTERMEDIATE_FOLDER, export_mode=False)], local_scheduler=True)

    else:
        print("error")