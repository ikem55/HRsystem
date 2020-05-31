from modules.jra_extract import JRAExtract
from modules.jra_transform import JRATransform
from modules.jra_load import JRALoad
from modules.jra_sk_model import JRASkModel
from modules.jra_sk_proc import JRASkProc
import my_config as mc
import modules.util as mu

import luigi
from modules.jra_task_learning import End_baoz_learning
from modules.jra_task_predict import End_baoz_predict

from datetime import datetime as dt
from datetime import timedelta
import sys
import pandas as pd
import numpy as np
import pickle
import os
from distutils.util import strtobool



# 呼び出し方
# python jra_raceuma_mark.py learning True True
# ====================================== パラメータ　要変更 =====================================================
# 逃げ馬、上がり最速馬を予測する（レース馬単位)

MODEL_VERSION = 'jra_ru_mark'
MODEL_NAME = 'raceuma_lgm'

# ====================================== クラス　要変更 =========================================================

class Ext(JRAExtract):
    pass

class Tf(JRATransform):


    def create_feature_raceuma_result_df(self, raceuma_df):
        """  raceuma_dfの特徴量を作成する。馬番→馬番グループを作成して列を追加する。

        :param dataframe raceuma_df:
        :return: dataframe
        """
        temp_raceuma_df = raceuma_df.copy()
        temp_raceuma_df.loc[:, "非根幹"] = temp_raceuma_df["距離"].apply(lambda x: 0 if x % 400 == 0 else 1)
        temp_raceuma_df.loc[:, "距離グループ"] = temp_raceuma_df["距離"] // 400
        temp_raceuma_df.loc[:, "先行率"] = (temp_raceuma_df["コーナー順位４"] / temp_raceuma_df["頭数"])
        temp_raceuma_df.loc[:, "人気率"] = (temp_raceuma_df["確定単勝人気順位"] / temp_raceuma_df["頭数"])
        temp_raceuma_df.loc[:, "着順率"] = (temp_raceuma_df["着順"] / temp_raceuma_df["頭数"])
        temp_raceuma_df.loc[:, "追込率"] = (temp_raceuma_df["コーナー順位４"] - temp_raceuma_df["着順"]) / temp_raceuma_df["頭数"]
        temp_raceuma_df.loc[:, "平均タイム"] = temp_raceuma_df["タイム"] / temp_raceuma_df["距離"] * 200
        temp_raceuma_df.loc[:, "平坦コース"] = temp_raceuma_df["場コード"].apply(lambda x: 1 if x in ('01', '02', '03', '04', '08', '10') else 0)
        temp_raceuma_df.loc[:, "急坂コース"] = temp_raceuma_df["場コード"].apply(lambda x: 1 if x in ('06', '07', '09') else 0)
        temp_raceuma_df.loc[:, "好走"] = temp_raceuma_df["ru_cluster"].apply(lambda x: 1 if x in (1, 4, 7) else (-1 if x in (2, 3) else 0))
        temp_raceuma_df.loc[:, "内枠"] = temp_raceuma_df.apply(lambda x: 1 if int(x["UMABAN"]) / x["頭数"] <= 0.5 else 0, axis=1)
        temp_raceuma_df.loc[:, "外枠"] = temp_raceuma_df.apply(lambda x: 1 if int(x["UMABAN"]) / x["頭数"] > 0.5 else 0, axis=1)
        temp_raceuma_df.loc[:, "中山"] = temp_raceuma_df["場コード"].apply(lambda x: 1 if x == '06' else 0)
        temp_raceuma_df.loc[:, "東京"] = temp_raceuma_df["場コード"].apply(lambda x: 1 if x == '05' else 0)
        temp_raceuma_df.loc[:, "上がりかかる"] = temp_raceuma_df["後３Ｆタイム"].apply(lambda x: 1 if x >= 355 else 0)
        temp_raceuma_df.loc[:, "上がり速い"] = temp_raceuma_df["後３Ｆタイム"].apply(lambda x: 1 if x <= 335 else 0)
        temp_raceuma_df.loc[:, "ダート道悪"] = temp_raceuma_df.apply(lambda x: 1 if x["芝ダ障害コード"] == '2' and int(x["馬場状態"]) >= 30 else 0, axis=1)
        temp_raceuma_df.loc[:, "前崩れレース"] = temp_raceuma_df["TRACK_BIAS_ZENGO"].apply(lambda x: 1 if x >= 2 else 0)
        temp_raceuma_df.loc[:, "先行惜敗"] = temp_raceuma_df.apply(lambda x: 1 if x["着順"] in (2,3,4,5) and x["コーナー順位４"] in (1,2,3,4) else 0, axis=1)
        temp_raceuma_df.loc[:, "前残りレース"] = temp_raceuma_df["TRACK_BIAS_ZENGO"].apply(lambda x: 1 if x <= -2 else 0)
        temp_raceuma_df.loc[:, "差し損ね"] = temp_raceuma_df["ru_cluster"].apply(lambda x: 1 if x in (1, 6) else 0)
        temp_raceuma_df.loc[:, "上がり幅小さいレース"] = temp_raceuma_df["上がり指数結果_std"].apply(lambda x: 1 if x <= 2 else 0)
        temp_raceuma_df.loc[:, "内枠砂被り"] = temp_raceuma_df.apply(lambda x: 1 if x["芝ダ障害コード"] == '2' and x["内枠"] == 1 and x["着順"] > 5 else 0, axis =1)
        return temp_raceuma_df


    def drop_columns_raceuma_result_df(self, raceuma_df):
        """ 過去レースで不要な項目を削除する """
        raceuma_df = raceuma_df.drop(["NENGAPPI", "馬名", "レース名", "レース名略称", "騎手名", "調教師名", "1(2)着馬名", "パドックコメント", "脚元コメント",
                                      "素点", "馬場差", "ペース", "出遅", "位置取", "不利", "前不利", "中不利", "後不利", "レース", "コース取り", "上昇度コード",
                                      "クラスコード", "馬体コード", "気配コード", "確定複勝オッズ下", "10時単勝オッズ", "10時複勝オッズ",
                                      "天候コード", "本賞金", "収得賞金", "レース馬コメント", "KAISAI_KEY", "ハロンタイム０１", "ハロンタイム０２", "ハロンタイム０３",
                                      "ハロンタイム０４", "ハロンタイム０５", "ハロンタイム０６", "ハロンタイム０７", "ハロンタイム０８", "ハロンタイム０９", "ハロンタイム１０",
                                      "ハロンタイム１１", "ハロンタイム１２", "ハロンタイム１３", "ハロンタイム１４", "ハロンタイム１５", "ハロンタイム１６", "ハロンタイム１７",
                                      "ハロンタイム１８", "１コーナー", "２コーナー", "３コーナー", "４コーナー", "１角１", "１角２", "１角３", "２角１", "２角２", "２角３",
                                      "向正１", "向正２", "向正３", "３角１", "３角２", "３角３", "４角０", "４角１", "４角２", "４角３", "４角４", "直線０", "直線１",
                                      "直線２", "直線３", "直線４", "１着算入賞金", "１ハロン平均_mean", "ＩＤＭ結果_mean", "テン指数結果_mean", "上がり指数結果_mean",
                                      "ペース指数結果_mean", "前３Ｆタイム_mean", "後３Ｆタイム_mean", "コーナー順位１_mean", "コーナー順位２_mean", "コーナー順位３_mean",
                                      "コーナー順位４_mean", "前３Ｆ先頭差_mean", "後３Ｆ先頭差_mean", "追走力_mean", "追上力_mean", "後傾指数_mean", "１ハロン平均_std",
                                      "上がり指数結果_std", "ペース指数結果_std", "COURSE_KEY",
                                      "馬具(その他)コメント", "レースコメント", "異常区分", "血統登録番号", "単勝", "複勝", "馬体重増減", "KYOSO_RESULT_KEY"], axis=1)
        return raceuma_df


class Ld(JRALoad):
    def _get_extract_object(self, start_date, end_date, mock_flag):
        """ 利用するExtクラスを指定する """
        ext = Ext(start_date, end_date, mock_flag)
        return ext

    def _get_transform_object(self, start_date, end_date):
        """ 利用するTransformクラスを指定する """
        tf = Tf(start_date, end_date)
        return tf

    def _proc_horse_df(self, horse_base_df):
        horse_df = self.tf.drop_columns_horse_df(horse_base_df)
        # horse_df = self.tf.encode_horse_df(horse_df, self.dict_folder)
        return horse_df.copy()

    def _proc_prev_df(self, raceuma_5_prev_df):
        """  prev_dfを作成するための処理。prev1_raceuma_df,prev2_raceuma_dfに処理がされたデータをセットする。過去２走のデータと過去走を集計したデータをセットする  """
        raceuma_5_prev_df = self.tf.cluster_course_df(raceuma_5_prev_df, self.dict_path)
        raceuma_5_prev_df = self.tf.cluster_raceuma_result_df(raceuma_5_prev_df, self.dict_path)
        raceuma_5_prev_df = self.tf.factory_analyze_race_result_df(raceuma_5_prev_df, self.dict_path)
        self.prev5_raceuma_df = self._get_prev_df(5, raceuma_5_prev_df, "")
        self.prev5_raceuma_df.rename(columns=lambda x: x + "_5", inplace=True)
        self.prev5_raceuma_df.rename(columns={"RACE_KEY_5": "RACE_KEY", "UMABAN_5": "UMABAN", "target_date_5": "target_date"}, inplace=True)
        self.prev4_raceuma_df = self._get_prev_df(4, raceuma_5_prev_df, "")
        self.prev4_raceuma_df.rename(columns=lambda x: x + "_4", inplace=True)
        self.prev4_raceuma_df.rename(columns={"RACE_KEY_4": "RACE_KEY", "UMABAN_4": "UMABAN", "target_date_4": "target_date"}, inplace=True)
        self.prev3_raceuma_df = self._get_prev_df(3, raceuma_5_prev_df, "")
        self.prev3_raceuma_df.rename(columns=lambda x: x + "_3", inplace=True)
        self.prev3_raceuma_df.rename(columns={"RACE_KEY_3": "RACE_KEY", "UMABAN_3": "UMABAN", "target_date_3": "target_date"}, inplace=True)
        self.prev2_raceuma_df = self._get_prev_df(2, raceuma_5_prev_df, "")
        self.prev2_raceuma_df.rename(columns=lambda x: x + "_2", inplace=True)
        self.prev2_raceuma_df.rename(columns={"RACE_KEY_2": "RACE_KEY", "UMABAN_2": "UMABAN", "target_date_2": "target_date"}, inplace=True)
        self.prev1_raceuma_df = self._get_prev_df(1, raceuma_5_prev_df, "")
        self.prev1_raceuma_df.rename(columns=lambda x: x + "_1", inplace=True)
        self.prev1_raceuma_df.rename(columns={"RACE_KEY_1": "RACE_KEY", "UMABAN_1": "UMABAN", "target_date_1": "target_date"}, inplace=True)
        self.prev_feature_raceuma_df = self._get_prev_feature_df(raceuma_5_prev_df)

    def _get_prev_feature_df(self, raceuma_5_prev_df):
        max_columns = ['血統登録番号', 'target_date', 'fa_1', 'fa_2', 'fa_3', 'fa_4', 'fa_5', 'ＩＤＭ結果', 'テン指数結果', '上がり指数結果', 'ペース指数結果']
        min_columns = ['血統登録番号', 'target_date', 'fa_4', 'テン指数結果順位', '上がり指数結果順位', 'ペース指数結果順位']
        max_score_df = raceuma_5_prev_df[max_columns].groupby(['血統登録番号', 'target_date']).max().add_prefix("max_").reset_index()
        min_score_df = raceuma_5_prev_df[min_columns].groupby(['血統登録番号', 'target_date']).min().add_prefix("min_").reset_index()
        feature_df = pd.merge(max_score_df, min_score_df, on=["血統登録番号", "target_date"])
        race_df = self.race_df[["RACE_KEY", "course_cluster"]].copy()
        raceuma_df = self.raceuma_df[["RACE_KEY", "UMABAN", "血統登録番号", "target_date"]].copy()
        raceuma_df = pd.merge(race_df, raceuma_df, on="RACE_KEY")
        filtered_df = pd.merge(raceuma_df, raceuma_5_prev_df.drop(["RACE_KEY", "UMABAN"], axis=1), on=["血統登録番号", "target_date", "course_cluster"])[["RACE_KEY", "UMABAN", "ru_cluster"]]
        filtered_df_c1 = filtered_df.query("ru_cluster == '1'").groupby(["RACE_KEY", "UMABAN"]).count().reset_index()
        filtered_df_c1.columns = ["RACE_KEY", "UMABAN", "c1_cnt"]
        filtered_df_c2 = filtered_df.query("ru_cluster == '2'").groupby(["RACE_KEY", "UMABAN"]).count().reset_index()
        filtered_df_c2.columns = ["RACE_KEY", "UMABAN", "c2_cnt"]
        filtered_df_c3 = filtered_df.query("ru_cluster == '3'").groupby(["RACE_KEY", "UMABAN"]).count().reset_index()
        filtered_df_c3.columns = ["RACE_KEY", "UMABAN", "c3_cnt"]
        filtered_df_c4 = filtered_df.query("ru_cluster == '4'").groupby(["RACE_KEY", "UMABAN"]).count().reset_index()
        filtered_df_c4.columns = ["RACE_KEY", "UMABAN", "c4_cnt"]
        filtered_df_c7 = filtered_df.query("ru_cluster == '7'").groupby(["RACE_KEY", "UMABAN"]).count().reset_index()
        filtered_df_c7.columns = ["RACE_KEY", "UMABAN", "c7_cnt"]
        raceuma_df = pd.merge(raceuma_df, filtered_df_c1, on=["RACE_KEY", "UMABAN"], how="left")
        raceuma_df = pd.merge(raceuma_df, filtered_df_c2, on=["RACE_KEY", "UMABAN"], how="left")
        raceuma_df = pd.merge(raceuma_df, filtered_df_c3, on=["RACE_KEY", "UMABAN"], how="left")
        raceuma_df = pd.merge(raceuma_df, filtered_df_c4, on=["RACE_KEY", "UMABAN"], how="left")
        raceuma_df = pd.merge(raceuma_df, filtered_df_c7, on=["RACE_KEY", "UMABAN"], how="left")
        raceuma_df = raceuma_df.fillna(0)
        raceuma_df = pd.merge(raceuma_df, feature_df, on=["血統登録番号", "target_date"], how="left").drop(["course_cluster", "血統登録番号", "target_date"], axis=1)
        print(raceuma_df.head(30))
        return raceuma_df

    def set_result_df(self):
        """ result_dfを作成するための処理。result_dfに処理がされたデータをセットする """
        return self.ext.get_raceuma_table_base()[["RACE_KEY", "UMABAN", "着順", "複勝"]]

class SkProc(JRASkProc):
    """
    地方競馬の機械学習処理プロセスを取りまとめたクラス。
    """
    index_list = ["RACE_KEY", "UMABAN", "target_date"]
    # LightGBM のハイパーパラメータ
    obj_column_list = ['WIN_FLAG', 'JIKU_FLAG', 'ANA_FLAG']
    lgbm_params = {
        'WIN_FLAG':{'objective': 'binary'},
        'JIKU_FLAG':{'objective': 'binary'},
        'ANA_FLAG':{'objective': 'binary'},
                   }

    def _get_load_object(self, version_str, start_date, end_date, mock_flag, test_flag):
        ld = Ld(version_str, start_date, end_date, mock_flag, test_flag)
        return ld

    def _merge_df(self):
        self.base_df = pd.merge(self.ld.race_df, self.ld.raceuma_df, on=["RACE_KEY", "target_date", "NENGAPPI"])
        self.base_df = pd.merge(self.base_df, self.ld.horse_df, on=["血統登録番号", "target_date"])
        self.base_df = pd.merge(self.base_df, self.ld.prev1_raceuma_df, on=["RACE_KEY", "UMABAN", "target_date"], how='left')
        self.base_df = pd.merge(self.base_df, self.ld.prev2_raceuma_df, on=["RACE_KEY", "UMABAN", "target_date"], how='left')
        self.base_df = pd.merge(self.base_df, self.ld.prev3_raceuma_df, on=["RACE_KEY", "UMABAN", "target_date"], how='left')
        self.base_df = pd.merge(self.base_df, self.ld.prev4_raceuma_df, on=["RACE_KEY", "UMABAN", "target_date"], how='left')
        self.base_df = pd.merge(self.base_df, self.ld.prev5_raceuma_df, on=["RACE_KEY", "UMABAN", "target_date"], how='left')
        self.base_df = pd.merge(self.base_df, self.ld.prev_feature_raceuma_df, on=["RACE_KEY", "UMABAN"], how='left')

    def _create_feature(self):
        """ マージしたデータから特徴量を生成する """
        self.base_df.fillna(({'平坦コース_1':0, '平坦コース_2':0, '平坦コース_3':0, '平坦コース_4':0, '平坦コース_5':0, '急坂コース_1':0, '急坂コース_2':0, '急坂コース_3':0, '急坂コース_4':0, '急坂コース_5':0, '好走_1':0, '好走_2':0, '好走_3':0, '好走_4':0, '好走_5':0,
                              '内枠_1': 0, '内枠_2': 0, '内枠_3': 0, '内枠_4': 0, '内枠_5': 0, '外枠_1': 0, '外枠_2': 0, '外枠_3': 0, '外枠_4': 0, '外枠_5': 0, '上がりかかる_1': 0, '上がりかかる_2': 0, '上がりかかる_3': 0, '上がりかかる_4': 0, '上がりかかる_5': 0,
                              '上がり速い_1': 0, '上がり速い_2': 0, '上がり速い_3': 0, '上がり速い_4': 0, '上がり速い_5': 0, 'ダート道悪_1': 0, 'ダート道悪_2': 0, 'ダート道悪_3': 0, 'ダート道悪_4': 0, 'ダート道悪_5': 0,
                              '前崩れレース_1': 0, '前崩れレース_2': 0, '前崩れレース_3': 0, '前崩れレース_4': 0, '前崩れレース_5': 0, '先行惜敗_1': 0, '先行惜敗_2': 0, '先行惜敗_3': 0, '先行惜敗_4': 0, '先行惜敗_5': 0,
                              '前残りレース_1': 0, '前残りレース_2': 0, '前残りレース_3': 0, '前残りレース_4': 0, '前残りレース_5': 0, '差し損ね_1': 0, '差し損ね_2': 0, '差し損ね_3': 0, '差し損ね_4': 0, '差し損ね_5': 0,
                              '中山_1': 0, '中山_2': 0, '中山_3': 0, '中山_4': 0, '中山_5': 0, '東京_1': 0, '東京_2': 0, '東京_3': 0, '東京_4': 0, '東京_5': 0,
                              '上がり幅小さいレース_1': 0, '上がり幅小さいレース_2': 0, '上がり幅小さいレース_3': 0, '上がり幅小さいレース_4': 0, '上がり幅小さいレース_5': 0}), inplace=True)
        self.base_df.loc[:, "継続騎乗"] = (self.base_df["騎手コード"] == self.base_df["騎手コード_1"]).astype(int)
        self.base_df.loc[:, "距離増減"] = self.base_df["距離"] - self.base_df["距離_1"]
        self.base_df.loc[:, "同根幹"] = (self.base_df["非根幹"] == self.base_df["非根幹_1"]).astype(int)
        self.base_df.loc[:, "同距離グループ"] = (self.base_df["距離グループ"] == self.base_df["距離グループ_1"]).astype(int)
        self.base_df.loc[:, "前走凡走"] = self.base_df.apply(lambda x: 1 if (x["人気率_1"] < 0.3 and x["着順率_1"] > 0.5) else 0, axis=1)
        self.base_df.loc[:, "前走激走"] = self.base_df.apply(lambda x: 1 if (x["人気率_1"] > 0.5 and x["着順率_1"] < 0.3) else 0, axis=1)
        self.base_df.loc[:, "前走逃げそびれ"] = self.base_df.apply(lambda x: 1 if (x["展開記号"] == '1' and x["先行率_1"] > 0.5) else 0, axis=1)
        self.base_df.loc[:, "前走加速ラップ"] = self.base_df["ラップ差１ハロン_1"].apply(lambda x: 1 if x > 0 else 0)
        self.base_df.loc[:, "平坦"] = self.base_df.apply(lambda x: x["平坦コース_1"] * x["好走_1"] + x["平坦コース_2"] * x["好走_2"] + x["平坦コース_3"] * x["好走_3"] + x["平坦コース_4"] * x["好走_4"] + x["平坦コース_5"] * x["好走_5"], axis=1)
        self.base_df.loc[:, "急坂"] = self.base_df.apply(lambda x: x["急坂コース_1"] * x["好走_1"] + x["急坂コース_2"] * x["好走_2"] + x["急坂コース_3"] * x["好走_3"] + x["急坂コース_4"] * x["好走_4"] + x["急坂コース_5"] * x["好走_5"], axis=1)
        self.base_df.loc[:, "内枠"] = self.base_df.apply(lambda x: x["内枠_1"] * x["好走_1"] + x["内枠_2"] * x["好走_2"] + x["内枠_3"] * x["好走_3"] + x["内枠_4"] * x["好走_4"] + x["内枠_5"] * x["好走_5"], axis=1)
        self.base_df.loc[:, "外枠"] = self.base_df.apply(lambda x: x["外枠_1"] * x["好走_1"] + x["外枠_2"] * x["好走_2"] + x["外枠_3"] * x["好走_3"] + x["外枠_4"] * x["好走_4"] + x["外枠_5"] * x["好走_5"], axis=1)
        self.base_df.loc[:, "枠"] = self.base_df.apply(lambda x: x["内枠"] if int(x["UMABAN"]) / x["頭数"] <= 0.5 else x["外枠"], axis=1)
        self.base_df.loc[:, "中山"] = self.base_df.apply(lambda x: x["中山_1"] * x["好走_1"] + x["中山_2"] * x["好走_2"] + x["中山_3"] * x["好走_3"] + x["中山_4"] * x["好走_4"] + x["中山_5"] * x["好走_5"], axis=1)
        self.base_df.loc[:, "東京"] = self.base_df.apply(lambda x: x["東京_1"] * x["好走_1"] + x["東京_2"] * x["好走_2"] + x["東京_3"] * x["好走_3"] + x["東京_4"] * x["好走_4"] + x["東京_5"] * x["好走_5"], axis=1)
        self.base_df.loc[:, "上がり遅"] = self.base_df.apply(lambda x: x["上がりかかる_1"] * x["好走_1"] + x["上がりかかる_2"] * x["好走_2"] + x["上がりかかる_3"] * x["好走_3"] + x["上がりかかる_4"] * x["好走_4"] + x["上がりかかる_5"] * x["好走_5"], axis=1)
        self.base_df.loc[:, "上がり速"] = self.base_df.apply(lambda x: x["上がり速い_1"] * x["好走_1"] + x["上がり速い_2"] * x["好走_2"] + x["上がり速い_3"] * x["好走_3"] + x["上がり速い_4"] * x["好走_4"] + x["上がり速い_5"] * x["好走_5"], axis=1)
        self.base_df.loc[:, "ダート道悪"] = self.base_df.apply(lambda x: x["ダート道悪_1"] * x["好走_1"] + x["ダート道悪_2"] * x["好走_2"] + x["ダート道悪_3"] * x["好走_3"] + x["ダート道悪_4"] * x["好走_4"] + x["ダート道悪_5"] * x["好走_5"], axis=1)
        self.base_df.loc[:, "突然バテた馬の距離短縮"] = self.base_df.apply(lambda x: 1 if x["ru_cluster_1"] == 5 and x["距離増減"] <= -200 else 0, axis=1)
        self.base_df.loc[:, "短距離からの延長"] = self.base_df.apply(lambda x: 1 if x["距離_1"] <= 1600 and x["距離増減"] >= 200 and x["ru_cluster_1"] in (1, 6) else 0, axis=1)
        self.base_df.loc[:, "中距離からの延長"] = self.base_df.apply(lambda x: 1 if x["距離_1"] > 1600 and x["距離増減"] >= 200 and x["ru_cluster_1"] in (0, 6) else 0, axis=1)
        self.base_df.loc[:, "前崩れレースで先行惜敗"] = self.base_df.apply(lambda x: x["前崩れレース_1"] * x["先行惜敗_1"] + x["前崩れレース_2"] * x["先行惜敗_2"] + x["前崩れレース_3"] * x["先行惜敗_3"] + x["前崩れレース_4"] * x["先行惜敗_4"] + x["前崩れレース_5"] * x["先行惜敗_5"], axis=1)
        self.base_df.loc[:, "前残りレースで差し損ね"] = self.base_df.apply(lambda x: x["前残りレース_1"] * x["差し損ね_1"] + x["前残りレース_2"] * x["差し損ね_2"] + x["前残りレース_3"] * x["差し損ね_3"] + x["前残りレース_4"] * x["差し損ね_4"] + x["前残りレース_5"] * x["差し損ね_5"], axis=1)
        self.base_df.loc[:, "上がり幅小さいレースで差し損ね"] = self.base_df.apply(lambda x: x["上がり幅小さいレース_1"] * x["差し損ね_1"] + x["上がり幅小さいレース_2"] * x["差し損ね_2"] + x["上がり幅小さいレース_3"] * x["差し損ね_3"] + x["上がり幅小さいレース_4"] * x["差し損ね_4"] + x["上がり幅小さいレース_5"] * x["差し損ね_5"], axis=1)
        self.base_df.loc[:, "ダート短距離血統１"] = self.base_df["父馬名"].apply(lambda x: 1 if x in ('サウスヴィグラス', 'エンパイアメーカー', 'エーピーインディ', 'カジノドライヴ', 'パイロ') else 0)
        self.base_df.loc[:, "内枠短縮"] = self.base_df.apply(lambda x: 1 if int(x["UMABAN"]) / x["頭数"] <= 0.3 and x["距離増減"] <= -200 else 0, axis=1)
        self.base_df.loc[:, "外枠短縮"] = self.base_df.apply(lambda x: 1 if int(x["UMABAN"]) / x["頭数"] >= 0.7 and x["距離増減"] <= -200 else 0, axis=1)
        self.base_df.loc[:, "内枠延長"] = self.base_df.apply(lambda x: 1 if int(x["UMABAN"]) / x["頭数"] <= 0.3 and x["距離増減"] >= 200 else 0, axis=1)
        self.base_df.loc[:, "外枠延長"] = self.base_df.apply(lambda x: 1 if int(x["UMABAN"]) / x["頭数"] >= 0.7 and x["距離増減"] >= 200 else 0, axis=1)
        self.base_df.loc[:, "延長得意父"] = self.base_df["父馬名"].apply(lambda x:1 if x in ('ウォーエンブレム', 'オレハマッテルゼ', 'キャプテンスティーヴ', 'コンデュイット', 'スズカマンボ', 'チーフベアハート', 'チチカステナンゴ', 'ディープスカイ', 'ネオユニヴァース', 'ハーツクライ', 'ハービンジャー', 'フォーティナイナーズサン', 'マリエンバード', 'メイショウサムソン', 'ワークフォース') else 0 )
        self.base_df.loc[:, "延長得意母父"] = self.base_df["母父馬名"].apply(lambda x: 1 if x in ('アサティス', 'エルコレドール', 'エルコンドルパサー', 'オジジアン', 'クロコルージュ', 'コマンダーインチーフ', 'スキャターザゴールド', 'フォレストリー', 'フサイチペガサス', 'ホワイトマズル', 'マーケトリー', 'ミシル', 'モンズン', 'メジロマックイーン') else 0)
        self.base_df.loc[:, "砂被り苦手父"] = self.base_df["父馬名"].apply(lambda x: 1 if x in ('アドマイヤオーラ', 'アドマイヤマックス', 'コンデュイット', 'ステイゴールド', 'タイキシャトル', 'ダノンシャンティ', 'ナカヤマフェスタ', 'ハービンジャー', 'ファルブラヴ', 'マイネルラヴ', 'ローエングリン') else 0)
        self.base_df.loc[:, "砂被り苦手母父"] = self.base_df["母父馬名"].apply(lambda x: 1 if x in ('アンバーシャダイ', 'エリシオ', 'カーリアン', 'サッカーボーイ', 'タマモクロス', 'ニホンピロヴィナー', 'メジロライアン', 'ロックオブジブランルタル', 'ロドリゴデトリアーノ') else 0)
        self.base_df.loc[:, "逆ショッカー"] = self.base_df.apply(lambda x: 1 if x["距離増減"] <= -200 and x["コーナー順位３_1"] >= 5 and x["道中順位"] <= 8 else 0, axis=1)
        self.base_df.loc[:, "前走砂被り外枠"] = self.base_df.apply(lambda x: 1 if x["芝ダ障害コード_1"] == '2' and x["内枠砂被り_1"] == 1 else 0, axis=1)
        self.base_df.loc[:, "母父サンデー短縮"] = self.base_df.apply(lambda x: 1 if x["母父馬名"] == "サンデーサイレンス" and x["距離増減"] <= -200 else 0, axis=1)
        self.base_df.loc[:, "ボールドルーラー系"] = self.base_df.apply(lambda x: 1 if x["父系統コード"] == '1305' or x["母父系統コード"] == '1305' else 0, axis=1)
        self.base_df.loc[:, "ダーレーパイロ"] = self.base_df.apply(lambda x: 1 if x["生産者名"] == "ダーレー・ジャパン・ファーム" and x["父馬名"] == "パイロ" else 0, axis=1)
        self.base_df.loc[:, "ダ1200ｍ好走父米国型"] = self.base_df.apply(lambda x: 1 if x["ダート短距離血統１"] == 1 and x["芝ダ障害コード_1"] == '2' and x["距離_1"] <= 1200 and x["着順_1"] in (1,2,3) else 0, axis=1)
        self.base_df.loc[:, "ダマスカス系"] = self.base_df.apply(lambda x: 1 if x["父系統コード"] == '1701' or x["母父系統コード"] == '1701' else 0, axis=1)
        self.base_df.loc[:, "中山ダ１８００血統"] = self.base_df.apply(lambda x: 1 if x["母父系統コード"] != "1206" and x["父馬名"] in ('キングカメハメハ', 'ロージズインメイ', 'アイルハヴアナザー') else 0, axis=1)
        self.base_df.loc[:, "ナスルーラ系"] = self.base_df.apply(lambda x: 1 if x["父系統コード"] == '1301' or x["母父系統コード"] == '1301' else 0, axis=1)
        self.base_df.loc[:, "ダート系サンデー"] = self.base_df["父馬名"].apply(lambda x: 1 if x in ('ゴールドアリュール', 'キンシャサノキセキ') else 0)
        self.base_df.loc[:, "マイル芝血統"] = self.base_df["父馬名"].apply(lambda x:1 if x in ('ディープインパクト', 'キングカメハメハ', 'ハーツクライ', 'ステイゴールド') else 0)
        self.base_df.loc[:, "ノーザンＦ"] = self.base_df["生産者名"].apply(lambda x: 1 if x=="ノーザンファーム" else 0)
        self.base_df.loc[:, "ニジンスキー系"] = self.base_df.apply(lambda x: 1 if x["父系統コード"] == '1102' or x["父馬名"] == 'アドマイヤムーン' else 0, axis=1)
        self.base_df.loc[:, "Ｐサンデー系"] = self.base_df["父馬名"].apply(
            lambda x: 1 if x in ('ステイゴールド', 'マツリダゴッホ', 'アグネスタキオン', 'キンシャサノキセキ') else 0)
        self.base_df.loc[:, "母父トニービン"] = self.base_df["母父馬名"].apply(lambda x:1 if x =="トニービン" else 0)
        self.base_df.loc[:, "グレイソヴリン系"] = self.base_df.apply(lambda x: 1 if x["父系統コード"] == '1302' or x["母父系統コード"] == '1302' else 0, axis=1)
        self.base_df.loc[:, "マッチェム系"] = self.base_df.apply(lambda x: 1 if x["父系統コード"] in ('2101', '2102', '2103', '2104', '2105') or x["母父系統コード"] == '2101' else 0, axis=1)
        self.base_df.loc[:, "父ルーラーシップ"] = self.base_df["父馬名"].apply(lambda x:1 if x == "ルーラーシップ" else 0)
        self.base_df.loc[:, "父ネオユニ短縮"] = self.base_df.apply(lambda x:1 if x["父馬名"] == "ネオユニヴァース" and x["距離増減"] <= -200 else 0, axis=1)
        self.base_df.loc[:, "父ロードカナロア"] = self.base_df["父馬名"].apply(lambda x:1 if x == "ロードカナロア" else 0)
        self.base_df.loc[:, "母父ディープ"] = self.base_df["母父馬名"].apply(lambda x:1 if x =="ディープインパクト" else 0)
        self.base_df.loc[:, "母父キンカメ"] = self.base_df["母父馬名"].apply(lambda x:1 if x =="キングカメハメハ" else 0)


    def _drop_columns_base_df(self):
        self.base_df.drop(["場名", "NENGAPPI", "発走時間", "COURSE_KEY", "血統登録番号", "ZENSO1_KYOSO_RESULT", "ZENSO2_KYOSO_RESULT", "ZENSO3_KYOSO_RESULT", "ZENSO4_KYOSO_RESULT", "ZENSO5_KYOSO_RESULT",
                           "ZENSO1_RACE_KEY", "ZENSO2_RACE_KEY", "ZENSO3_RACE_KEY", "ZENSO4_RACE_KEY", "ZENSO5_RACE_KEY", "入厩年月日", "父馬名", "母父馬名", "生産者名", "産地名", "登録抹消フラグ", "参考前走", "入厩何日前",
                           "距離_2", "芝ダ障害コード_2", "右左_2", "内外_2", "馬場状態_2", "種別_2", "条件_2", "記号_2", "重量_2", "グレード_2", "頭数_2", "着順_2",
                           "タイム_2", "確定単勝オッズ_2", "確定単勝人気順位_2", "レースペース_2", "馬ペース_2", "コーナー順位１_2", "コーナー順位２_2", "コーナー順位３_2",
                           "コーナー順位４_2", "コース_2", "レースペース流れ_2", "馬ペース流れ_2", "４角コース取り_2", "IDM_2", "ペースアップ位置_2", "ラスト５ハロン_2",
                           "ラスト４ハロン_2", "ラスト３ハロン_2", "ラスト２ハロン_2", "ラップ差４ハロン_2", "ラップ差３ハロン_2", "ラップ差２ハロン_2", "ラップ差１ハロン_2",
                           "連続何日目_2", "芝種類_2", "草丈_2", "転圧_2", "凍結防止剤_2", "中間降水量_2", "ハロン数_2",
                           "前３Ｆタイム_2", "後３Ｆタイム_2", "前３Ｆ先頭差_2", "後３Ｆ先頭差_2", "騎手コード_2", "調教師コード_2", "レース脚質_2",
                           "ラスト１ハロン_2", "RAP_TYPE_2",
                           "芝_2", "外_2", "重_2", "軽_2", "場コード_2", "course_cluster_2", "非根幹_2", "距離グループ_2",
                           "平坦コース_2", "急坂コース_2", "好走_2", "内枠_2", "外枠_2", "中山_2", "東京_2", "上がりかかる_2", "上がり速い_2",
                           "ダート道悪_2", "前崩れレース_2", "先行惜敗_2", "前残りレース_2", "差し損ね_2", "上がり幅小さいレース_2", "内枠砂被り_2",

                           "距離_3", "芝ダ障害コード_3", "右左_3", "内外_3", "馬場状態_3", "種別_3", "条件_3", "記号_3", "重量_3", "グレード_3", "頭数_3", "着順_3",
                           "タイム_3", "確定単勝オッズ_3", "確定単勝人気順位_3", "レースペース_3", "馬ペース_3", "コーナー順位１_3", "コーナー順位２_3", "コーナー順位３_3",
                           "コーナー順位４_3", "コース_3", "レースペース流れ_3", "馬ペース流れ_3", "４角コース取り_3", "IDM_3", "ペースアップ位置_3", "ラスト５ハロン_3",
                           "ラスト４ハロン_3", "ラスト３ハロン_3", "ラスト２ハロン_3", "ラップ差４ハロン_3", "ラップ差３ハロン_3", "ラップ差２ハロン_3", "ラップ差１ハロン_3",
                           "連続何日目_3", "芝種類_3", "草丈_3", "転圧_3", "凍結防止剤_3", "中間降水量_3", "ハロン数_3",
                           "前３Ｆタイム_3", "後３Ｆタイム_3", "前３Ｆ先頭差_3", "後３Ｆ先頭差_3", "騎手コード_3", "調教師コード_3", "レース脚質_3",
                           "ラスト１ハロン_3", "RAP_TYPE_3",
                           "芝_3", "外_3", "重_3", "軽_3", "場コード_3", "course_cluster_3", "非根幹_3", "距離グループ_3",
                           "平坦コース_3", "急坂コース_3", "好走_3", "内枠_3", "外枠_3", "中山_3", "東京_3", "上がりかかる_3", "上がり速い_3",
                           "ダート道悪_3", "前崩れレース_3", "先行惜敗_3", "前残りレース_3", "差し損ね_3", "上がり幅小さいレース_3", "内枠砂被り_3",

                           "距離_4", "芝ダ障害コード_4", "右左_4", "内外_4", "馬場状態_4", "種別_4", "条件_4", "記号_4", "重量_4", "グレード_4", "頭数_4", "着順_4",
                           "タイム_4", "確定単勝オッズ_4", "確定単勝人気順位_4", "レースペース_4", "馬ペース_4", "コーナー順位１_4", "コーナー順位２_4", "コーナー順位３_4",
                           "コーナー順位４_4", "コース_4", "レースペース流れ_4", "馬ペース流れ_4", "４角コース取り_4", "IDM_4", "ペースアップ位置_4", "ラスト５ハロン_4",
                           "ラスト４ハロン_4", "ラスト３ハロン_4", "ラスト２ハロン_4", "ラップ差４ハロン_4", "ラップ差３ハロン_4", "ラップ差２ハロン_4", "ラップ差１ハロン_4",
                           "連続何日目_4", "芝種類_4", "草丈_4", "転圧_4", "凍結防止剤_4", "中間降水量_4", "ハロン数_4",
                           "前３Ｆタイム_4", "後３Ｆタイム_4", "前３Ｆ先頭差_4", "後３Ｆ先頭差_4", "騎手コード_4", "調教師コード_4", "レース脚質_4",
                           "ラスト１ハロン_4", "RAP_TYPE_4",
                           "芝_4", "外_4", "重_4", "軽_4", "場コード_4", "course_cluster_4", "非根幹_4", "距離グループ_4",
                           "平坦コース_4", "急坂コース_4", "好走_4", "内枠_4", "外枠_4", "中山_4", "東京_4", "上がりかかる_4", "上がり速い_4",
                           "ダート道悪_4", "前崩れレース_4", "先行惜敗_4", "前残りレース_4", "差し損ね_4", "上がり幅小さいレース_4", "内枠砂被り_4",

                           "距離_5", "芝ダ障害コード_5", "右左_5", "内外_5", "馬場状態_5", "種別_5", "条件_5", "記号_5", "重量_5", "グレード_5", "頭数_5", "着順_5",
                           "タイム_5", "確定単勝オッズ_5", "確定単勝人気順位_5", "レースペース_5", "馬ペース_5", "コーナー順位１_5", "コーナー順位２_5", "コーナー順位３_5",
                           "コーナー順位４_5", "コース_5", "レースペース流れ_5", "馬ペース流れ_5", "４角コース取り_5", "IDM_5", "ペースアップ位置_5", "ラスト５ハロン_5",
                           "ラスト４ハロン_5", "ラスト３ハロン_5", "ラスト２ハロン_5", "ラップ差４ハロン_5", "ラップ差３ハロン_5", "ラップ差２ハロン_5", "ラップ差１ハロン_5",
                           "連続何日目_5", "芝種類_5", "草丈_5", "転圧_5", "凍結防止剤_5", "中間降水量_5", "ハロン数_5",

                           "前３Ｆタイム_5", "後３Ｆタイム_5", "前３Ｆ先頭差_5", "後３Ｆ先頭差_5", "騎手コード_5", "調教師コード_5", "レース脚質_5", "ラスト１ハロン_5", "RAP_TYPE_5",
                           "芝_5", "外_5", "重_5", "軽_5", "場コード_5", "course_cluster_5", "非根幹_5", "距離グループ_5",
                           "平坦コース_5", "急坂コース_5", "好走_5", "内枠_5", "外枠_5", "中山_5", "東京_5", "上がりかかる_5", "上がり速い_5",
                           "ダート道悪_5", "前崩れレース_5", "先行惜敗_5", "前残りレース_5", "差し損ね_5", "上がり幅小さいレース_5", "内枠砂被り_5"
                           ], axis=1, inplace=True)

    def _set_label_list(self, df):
        """ label_listの値にわたされたdataframeのデータ型がobjectのカラムのリストをセットする。TargetEncodingを行わないカラムを除く

        :param dataframe df: dataframe
        """
        label_list = df.select_dtypes(include=object).columns.tolist()
        except_list = ["距離", "芝ダ障害コード", "右左", "内外", "種別", "記号", "重量", "グレード", "コース", "開催区分", "曜日", "天候コード", "芝馬場状態コード", "芝馬場状態内", "芝馬場状態中", "芝馬場状態外",
                       "直線馬場差最内", "直線馬場差内", "直線馬場差中", "直線馬場差外", "直線馬場差大外", "ダ馬場状態コード", "ダ馬場状態内", "ダ馬場状態中", "ダ馬場状態外", "芝種類", "転圧", "凍結防止剤", "場コード",
                       "距離グループ", "UMABAN", "RACE_KEY", "target_date", "グレード_1", "コース_1", "ペースアップ位置_1", "レースペース流れ_1", "右左_1", "参考前走騎手コード", "芝種類_1", "種別_1", "重量_1",
                       "条件クラス"]
        self.label_list = [i for i in label_list if i not in except_list]

    def _set_target_variables(self):
        self.result_df = self.ld.set_result_df()
        self._create_target_variable_win()
        self._create_target_variable_jiku()
        self._create_target_variable_ana()
        self.result_df.drop(["着順", "複勝"], axis=1, inplace=True)

    def _create_target_variable_win(self):
        """  WIN_FLAGをセットする。着順が１着 """
        self.result_df['WIN_FLAG'] = self.result_df['着順'].apply(lambda x: 1 if x == 1 else 0)

    def _create_target_variable_jiku(self):
        """  JIKU_FLAGをセットする。着順が１，２着 """
        self.result_df['JIKU_FLAG'] = self.result_df['着順'].apply(lambda x: 1 if x in (1,2) else 0)

    def _create_target_variable_ana(self):
        """  ANA_FLAGをセットする。複勝５００円以上を１とする """
        self.result_df['ANA_FLAG'] = self.result_df['複勝'].apply(lambda x: 1 if x >= 500 else 0)

    def _sub_create_pred_df(self, temp_df, y_pred):
        pred_df = pd.DataFrame(
            {"RACE_KEY": temp_df["RACE_KEY"], "UMABAN": temp_df["UMABAN"], "target_date": temp_df["target_date"],
             "prob": y_pred})
        pred_df = self._calc_grouped_data(pred_df)
        #pred_df.loc[:, "pred"] = pred_df.apply(lambda x: 1 if x["prob"] >= 0.5 else 0, axis=1)
        return pred_df


class SkModel(JRASkModel):
    obj_column_list = ['WIN_FLAG', 'JIKU_FLAG', 'ANA_FLAG']

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

    pd.set_option('display.max_rows', 300)

    if mode == "learning":
        if test_flag:
            print("Test mode")
            start_date = '2012/01/01'
            end_date = '2013/12/31'
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
        if mock_flag:
            print("use mock data")
        print("MODE:predict mock_flag:" + str(args[2]) + "  start_date:" + start_date + " end_date:" + end_date)

        sk_model = SkModel(MODEL_NAME, MODEL_VERSION, start_date, end_date, mock_flag, test_flag, mode)


        luigi.build([End_baoz_predict(start_date=start_date, end_date=end_date, skmodel=sk_model, test_flag=test_flag,
                                      intermediate_folder=INTERMEDIATE_FOLDER, export_mode=False)], local_scheduler=True)

    else:
        print("error")