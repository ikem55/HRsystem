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

from datetime import datetime as dt
from datetime import timedelta
import sys
import pandas as pd
import numpy as np
import os
from distutils.util import strtobool

from sklearn.model_selection import train_test_split
import optuna.integration.lightgbm as lgb
import lightgbm as lgb_original
import featuretools as ft
import pickle

import optuna.integration.lightgbm as lgb
import warnings
warnings.simplefilter('ignore')

basedir = os.path.dirname(__file__)[:-8]
print(basedir)
sys.path.append(basedir)

# 呼び出し方
# python lb_v5.py learning True True
# ====================================== パラメータ　要変更 =====================================================
# Modelv5 LightGBMを使って馬ごとのフラグの確率を計算する。木モデルなのでNull許容できるはず
MODEL_VERSION = 'lb_v5'
MODEL_NAME = 'raceuma_lgm'
TABLE_NAME = '地方競馬レース馬V5'

# ============================================================================================================

CONN_STR = (
    r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
    r'DBQ=C:\BaoZ\DB\MasterDB\MyDB.MDB;'
)

# ====================================== クラス　要変更 =========================================================

class Ext(LBExtract):
    pass

class Tf(LBTransform):
    def create_feature_race_df(self, race_df):
        """ 特徴となる値を作成する。ナイター、季節、非根幹、距離グループ、頭数グループを作成して列として付与する。

        :param dataframe race_df:
        :return: dataframe
        """
        temp_race_df = race_df.copy()
        temp_race_df.loc[:, 'ナイター'] = race_df['発走時刻'].apply(lambda x: True if x.hour >= 17 else False)
        temp_race_df.loc[:, '季節'] = (race_df['月日'].apply(lambda x: x.month) - 1) // 3
        temp_race_df['季節'].astype('str')
        temp_race_df.loc[:, "非根幹"] = race_df["距離"].apply(lambda x: True if x % 400 == 0 else False)
        temp_race_df.loc[:, "距離グループ"] = race_df["距離"] // 400
        temp_race_df.loc[:, "頭数グループ"] = race_df["頭数"] // 5
        temp_race_df.loc[:, "コース"] = race_df["場コード"].astype(str) + race_df["トラックコード"].astype(str)
        return temp_race_df

    def choose_race_result_column(self, race_df):
        """ レースデータから必要な列に絞り込む。列は '発走時刻', '月日', '競走コード', '距離'', '頭数', 'ペース', 'トラックコード', '後３ハロン'

        :param dataframe race_df:
        :return: dataframe
        """
        temp_race_df = race_df[
            ['主催者コード', '発走時刻', '月日', '競走コード', '距離', '場コード', '頭数', 'ペース', 'トラックコード', '後３ハロン']]
        return temp_race_df

    def create_feature_race_result_df(self, race_df, race_winner_df):
        """  race_dfの結果データから特徴量を作成して列を追加する。どのような馬が勝ったか（逃げ、内、外、短縮延長、人気等）を作成

        :param dataframe race_df:
        :return: dataframe
        """
        print("create_feature_race_result_df")
        temp_race_df = race_df.copy()
        winner_df = race_winner_df.copy()
        temp_merge_df = pd.merge(race_df, winner_df, on="競走コード")
        temp_merge_df.loc[:, "上り係数"] = temp_merge_df.apply(
            lambda x: 1 if x["後３ハロン"] == 0 else (x["後３ハロン"] / 600) / (x["タイム"] / x["距離"]), axis=1)
        temp_merge_df.loc[:, "逃げ勝ち"] = temp_merge_df["コーナー順位4"].apply(lambda x: True if x == 1 else False)
        temp_merge_df.loc[:, "内勝ち"] = temp_merge_df["枠番"].apply(lambda x: True if x in (1, 2, 3) else False)
        temp_merge_df.loc[:, "外勝ち"] = temp_merge_df["枠番"].apply(lambda x: True if x in (6, 7, 8) else False)
        temp_merge_df.loc[:, "短縮勝ち"] = temp_merge_df["距離増減"].apply(lambda x: True if x < 0 else False)
        temp_merge_df.loc[:, "延長勝ち"] = temp_merge_df["距離増減"].apply(lambda x: True if x > 0 else False)
        temp_merge_df.loc[:, "人気勝ち"] = temp_merge_df["単勝人気"].apply(lambda x: True if x == 1 else False)
        merge_df = pd.merge(temp_race_df,
                            temp_merge_df[["競走コード", "上り係数", "逃げ勝ち", "内勝ち", "外勝ち", "短縮勝ち", "延長勝ち", "人気勝ち"]],
                            on="競走コード")
        return merge_df

    def drop_columns_raceuma_df(self, raceuma_df):
        return raceuma_df.drop(["データ作成年月日", "予想タイム指数", "予想オッズ", "血統距離評価", "血統トラック評価", "血統成長力評価",
                                           "血統総合評価", "血統距離評価B", "血統トラック評価B", "血統成長力評価B", "血統総合評価B", "先行指数", "騎手コード", "騎手評価",
                                           "調教師評価", "枠順評価", "脚質評価", "調教師コード", "前走着順", "前走人気", "前走着差", "前走トラック種別コード", "前走馬体重",
                                           "タイム指数上昇係数", "タイム指数回帰推定値", "タイム指数回帰標準偏差", "前走休養週数", "騎手ランキング", "調教師ランキング",
                                           "得点V1順位", "得点V2順位", "デフォルト得点順位", "得点V3順位"], axis=1)

    def encode_raceuma_before_df(self, raceuma_df, dict_folder):
        """  列をエンコードする処理。騎手名、所属、転厩をラベルエンコーディングして値を置き換える。learning_modeがTrueの場合は辞書生成がされる。

        :param dataframe raceuma_df:
        :return: dataframe
        """
        temp_raceuma_df = raceuma_df.copy()
        temp_raceuma_df = self.choose_upper_n_count(temp_raceuma_df, "騎手名", 100, dict_folder)
        temp_raceuma_df.loc[:, '騎手名'] = mu.label_encoding(raceuma_df['騎手名'], '騎手名', dict_folder)
        temp_raceuma_df = self.choose_upper_n_count(temp_raceuma_df, "調教師名", 100, dict_folder)
        temp_raceuma_df.loc[:, '調教師名'] = mu.label_encoding(raceuma_df['調教師名'], '調教師名', dict_folder)
        temp_raceuma_df.loc[:, '所属'] = mu.label_encoding(raceuma_df['所属'], '所属', dict_folder)
        temp_raceuma_df.loc[:, '転厩'] = mu.label_encoding(raceuma_df['転厩'], '転厩', dict_folder)
        temp_raceuma_df.loc[:, '予想展開'] = temp_raceuma_df["予想展開"].astype(str)
        return temp_raceuma_df.copy()

    def create_feature_raceuma_df(self, raceuma_df):
        """  raceuma_dfの特徴量を作成する。

        :param dataframe raceuma_df:
        :return: dataframe
        """
        temp_raceuma_df = raceuma_df.copy()
        temp_raceuma_df.loc[:, "馬番グループ"] = raceuma_df["馬番"] // 4
        temp_raceuma_df.loc[:, "予想タイム指数順位"] = raceuma_df["予想タイム指数順位"].apply(lambda x: 1 if x == 0 else 1 / x)
        temp_raceuma_df.loc[:, "休養週数"] = raceuma_df["休養週数"].apply(lambda x: 1 if x == 0 else 1 / x)
        temp_raceuma_df.loc[:, "休養後出走回数"] = raceuma_df["休養後出走回数"].apply(lambda x: 5 if x >= 5 else x)
        temp_raceuma_df.loc[:, "予想人気"] = raceuma_df["予想人気"].apply(lambda x: 1 if x == 0 else 1 / x)
        temp_raceuma_df.loc[:, "先行指数順位"] = raceuma_df["先行指数順位"].apply(lambda x: 1 if x == 0 else 1 / x)
        temp_raceuma_df.loc[:, "キャリア"] = raceuma_df["キャリア"].apply(lambda x: 10 if x >= 10 else x)
        temp_raceuma_df.loc[:, "馬齢"] = raceuma_df["馬齢"].apply(lambda x: 7 if x >= 7 else x)
        temp_raceuma_df.loc[:, "距離増減"] = raceuma_df["距離増減"] // 200
        return temp_raceuma_df

    def choose_raceuma_result_column(self, raceuma_df):
        """  レース馬データから必要な列に絞り込む。

        :param dataframe raceuma_df:
        :return: dataframe
        """
        temp_raceuma_df = raceuma_df[
            ['競走コード', '馬番', '枠番', '年月日', '血統登録番号', 'タイム指数', '単勝オッズ', '単勝人気', '確定着順', '着差', '休養週数', '先行率', 'タイム', '予想展開',
             'ペース偏差値', '展開コード', 'クラス変動', '騎手所属場コード', '騎手名', 'テン乗り', '負担重量', '馬体重', 'コーナー順位3', 'コーナー順位4', '距離増減', '調教師所属場コード',
             '斤量比', '上がりタイム']].copy()
        return temp_raceuma_df

    def encode_raceuma_result_df(self, raceuma_df, dict_folder):
        """  列をエンコードする処理。騎手名、所属、転厩をラベルエンコーディングして値を置き換える。learning_modeがTrueの場合は辞書生成がされる。

        :param dataframe raceuma_df:
        :return: dataframe
        """
        temp_raceuma_df = raceuma_df.copy()
        temp_raceuma_df = self.choose_upper_n_count(temp_raceuma_df, "騎手名", 100, dict_folder)
        temp_raceuma_df.loc[:, '騎手名'] = mu.label_encoding(raceuma_df['騎手名'], '騎手名', dict_folder).astype(str)
        temp_raceuma_df.loc[:, '展開脚質'] = raceuma_df['展開コード'].astype(str).str[:1].astype(int)
        temp_raceuma_df.loc[:, '展開脚色'] = raceuma_df['展開コード'].astype(str).str[-1:].astype(int)
        return_df = temp_raceuma_df.drop('展開コード', axis=1)
        return return_df

    def normalize_raceuma_result_df(self, raceuma_df):
        """ レース馬の成績データのノーマライズ。あがり３ハロンとか """
        grouped_df = raceuma_df[['競走コード', '馬体重', '上がりタイム']].groupby('競走コード').agg(
            ['mean', 'std']).reset_index()
        grouped_df.columns = ['競走コード', '馬体重_mean', '馬体重_std', '上がりタイム_mean', '上がりタイム_std']
        merged_df = pd.merge(raceuma_df, grouped_df, on='競走コード')
        merged_df['馬体重偏差'] = merged_df.apply(lambda x: 50 if x['上がりタイム_std'] == 0 else (x['上がりタイム_mean'] - x['上がりタイム'])/ x['上がりタイム_std'] * 10 + 50 , axis=1)
        merged_df['上がりタイム偏差'] = merged_df.apply(lambda x: 50 if x['上がりタイム_std'] == 0 else (x['上がりタイム_mean'] - x['上がりタイム'])/ x['上がりタイム_std'] * 10 + 50 , axis=1)
        merged_df['上がりタイム順位'] = raceuma_df.groupby("競走コード")["上がりタイム"].rank(method='min')
        merged_df['上がりタイム順位'] = merged_df.apply(lambda x: np.nan if x["上がりタイム"] == 0 else x["上がりタイム順位"], axis=1)
        merged_df = merged_df[["競走コード", "馬番", "馬体重偏差", "上がりタイム偏差", "上がりタイム順位"]]
        return_df = pd.merge(raceuma_df, merged_df, on=["競走コード", "馬番"]).drop(["馬体重", "上がりタイム"], axis=1).rename(columns={"馬体重偏差": "馬体重", "上がりタイム偏差": "上がりタイム"})
        return return_df


    def create_feature_raceuma_result_df(self, race_df, raceuma_df):
        """  raceuma_dfの特徴量を作成する。

        :param dataframe raceuma_df:
        :return: dataframe
        """
        temp_raceuma_df = raceuma_df.copy()
        temp_merge_df = pd.merge(race_df, raceuma_df, on="競走コード")
        print("create_feature_raceuma_result_df: temp_merge_df", temp_merge_df.shape)
        temp_raceuma_df.loc[:, "同場騎手"] = (temp_merge_df["騎手所属場コード"] == temp_merge_df["場コード"]).astype(int)
        temp_raceuma_df.loc[:, "同所属場"] = (temp_merge_df["調教師所属場コード"] == temp_merge_df["場コード"]).astype(int)
        temp_raceuma_df.loc[:, "同所属騎手"] = (temp_merge_df["騎手所属場コード"] == temp_merge_df["調教師所属場コード"]).astype(int)
        temp_raceuma_df.loc[:, "追込率"] = (temp_merge_df["コーナー順位4"] - temp_merge_df["確定着順"]) / temp_merge_df["頭数"]
        temp_raceuma_df.loc[:, "平均タイム"] = temp_merge_df["タイム"] / temp_merge_df["距離"] * 200
        temp_raceuma_df.loc[:, "勝ち"] = temp_raceuma_df["確定着順"].apply(lambda x: True if x == 1 else False)
        temp_raceuma_df.loc[:, "１番人気"] = temp_raceuma_df["単勝人気"].apply(lambda x: True if x == 1 else False)
        temp_raceuma_df.loc[:, "３角先頭"] = temp_raceuma_df["コーナー順位3"].apply(lambda x: True if x == 1 else False)
        temp_raceuma_df.loc[:, "４角先頭"] = temp_raceuma_df["コーナー順位4"].apply(lambda x: True if x == 1 else False)
        temp_raceuma_df.loc[:, "上がり最速"] = temp_raceuma_df["上がりタイム順位"].apply(lambda x: True if x == 1 else False)
        temp_raceuma_df.loc[:, "休み明け"] = temp_raceuma_df["休養週数"].apply(lambda x: True if x >= 10 else False)
        temp_raceuma_df.loc[:, "連闘"] = temp_raceuma_df["休養週数"].apply(lambda x: True if x == 1 else False)
        temp_raceuma_df.loc[:, "大差負け"] = temp_raceuma_df["着差"].apply(lambda x: True if x >= 20 else False)
        temp_raceuma_df.loc[:, "凡走"] = temp_merge_df.apply(lambda x: True if x["確定着順"] - x["単勝人気"] > 5 else False, axis=1)
        temp_raceuma_df.loc[:, "好走"] = temp_merge_df["確定着順"].apply(lambda x: True if x <= 3 else False)
        temp_raceuma_df.loc[:, "激走"] = temp_merge_df.apply(lambda x: True if x["単勝人気"] - x["確定着順"] > 5 else False, axis=1)
        temp_raceuma_df.loc[:, "逃げそびれ"] = temp_merge_df.apply(lambda x: True if x["予想展開"] == 1 and - x["コーナー順位4"] > 3 else False, axis=1)

        return temp_raceuma_df.drop(["確定着順", "単勝人気", "コーナー順位3", "コーナー順位4", "休養週数", "着差"], axis=1).copy()

    def choose_horse_column(self, horse_df):
        """ 馬データから必要な列に絞り込む。対象は血統登録番号、繁殖登録番号１、繁殖登録番号５、東西所属コード、生産者コード、馬主コード

        :param dataframe raceuma_df:
        :return: dataframe
        """
        temp_horse_df = horse_df[['馬記号コード', '品種コード', '毛色コード', '血統登録番号', '繁殖登録番号1', '繁殖登録番号3', '繁殖登録番号5', '東西所属コード', '生産者コード', '馬主コード']]
        return temp_horse_df

    def encode_horse_df(self, horse_df, dict_folder):
        """  列をエンコードする処理。騎手名、所属、転厩をラベルエンコーディングして値を置き換える。learning_modeがTrueの場合は辞書生成がされる。

        :param dataframe horse_df:
        :return: dataframe
        """
        temp_horse_df = horse_df.copy()
        temp_horse_df = self.choose_upper_n_count(temp_horse_df, "繁殖登録番号1", 50, dict_folder)
        temp_horse_df.loc[:, '繁殖登録番号1'] = mu.label_encoding(horse_df['繁殖登録番号1'], '騎手名', dict_folder).astype(str)
        temp_horse_df = self.choose_upper_n_count(temp_horse_df, "繁殖登録番号3", 50, dict_folder)
        temp_horse_df.loc[:, '繁殖登録番号3'] = mu.label_encoding(horse_df['繁殖登録番号3'], '調教師名', dict_folder).astype(str)
        temp_horse_df = self.choose_upper_n_count(temp_horse_df, "繁殖登録番号5", 50, dict_folder)
        temp_horse_df.loc[:, '繁殖登録番号5'] = mu.label_encoding(horse_df['繁殖登録番号5'], '調教師名', dict_folder).astype(str)
        temp_horse_df = self.choose_upper_n_count(temp_horse_df, "生産者コード", 20, dict_folder)
        temp_horse_df.loc[:, '生産者コード'] = mu.label_encoding(horse_df['生産者コード'], '調教師名', dict_folder).astype(str)
        temp_horse_df = self.choose_upper_n_count(temp_horse_df, "馬主コード", 100, dict_folder)
        temp_horse_df.loc[:, '馬主コード'] = mu.label_encoding(horse_df['馬主コード'], '調教師名', dict_folder).astype(str)
        return temp_horse_df.copy()

    def factory_analyze_raceuma_result_df(self, race_df, input_raceuma_df, dict_folder):
        print("skip factory_analyze_raceuma_result_df")
        return input_raceuma_df

class Ld(LBLoad):
    def _get_extract_object(self, start_date, end_date, mock_flag):
        """ 利用するExtクラスを指定する """
        ext = Ext(start_date, end_date, mock_flag)
        return ext

    def _get_transform_object(self, start_date, end_date):
        """ 利用するTransformクラスを指定する """
        tf = Tf(start_date, end_date)
        return tf

    def _proc_race_df(self, race_base_df):
        race_df = self.tf.create_feature_race_df(race_base_df)
        return race_df.drop(["トラック種別コード", "競走番号", "場名"
                                        , "初出走頭数", "混合", "予想決着指数", "登録頭数", "回次", "日次"], axis=1)

    def set_prev_df(self):
        """  prev_dfを作成するための処理。prev1_raceuma_df,preV4_raceuma_dfに処理がされたデータをセットする。過去２走のデータをセットする  """
        print("set_prev_df")
        race_result_df, raceuma_result_df = self._get_prev_base_df(5)
        self.prev5_raceuma_df = self._get_prev_df(5, race_result_df, raceuma_result_df)
        self.prev5_raceuma_df.rename(columns=lambda x: x + "_5", inplace=True)
        self.prev5_raceuma_df.rename(columns={"競走コード_5": "競走コード", "馬番_5": "馬番"}, inplace=True)
        self.prev4_raceuma_df = self._get_prev_df(4, race_result_df, raceuma_result_df)
        self.prev4_raceuma_df.rename(columns=lambda x: x + "_4", inplace=True)
        self.prev4_raceuma_df.rename(columns={"競走コード_4": "競走コード", "馬番_4": "馬番"}, inplace=True)
        self.prev3_raceuma_df = self._get_prev_df(3, race_result_df, raceuma_result_df)
        self.prev3_raceuma_df.rename(columns=lambda x: x + "_3", inplace=True)
        self.prev3_raceuma_df.rename(columns={"競走コード_3": "競走コード", "馬番_3": "馬番"}, inplace=True)
        self.prev2_raceuma_df = self._get_prev_df(2, race_result_df, raceuma_result_df)
        self.prev2_raceuma_df.rename(columns=lambda x: x + "_2", inplace=True)
        self.prev2_raceuma_df.rename(columns={"競走コード_2": "競走コード", "馬番_2": "馬番"}, inplace=True)
        self.prev1_raceuma_df = self._get_prev_df(1, race_result_df, raceuma_result_df)
        self.prev1_raceuma_df.rename(columns=lambda x: x + "_1", inplace=True)
        self.prev1_raceuma_df.rename(columns={"競走コード_1": "競走コード", "馬番_1": "馬番"}, inplace=True)

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
        merged_df = pd.merge(this_race_df, this_raceuma_df, on=["競走コード", "馬番"])
        merged_df = merged_df.drop(['枠番',  'タイム指数', '単勝オッズ',
             "距離", "後３ハロン", "予想展開", "騎手所属場コード", "調教師所属場コード", 'タイム',
             "上がりタイム", "休み明け", "展開脚質", "展開脚色", "年月日", "月日", "距離", "血統登録番号"], axis=1)
        return merged_df

    def scale_df(self):
        print("skip scale_df")

    def _proc_scale_df_for_fa(self, raceuma_df):
        print("skip _proc_scale_df_for_fa")
        return raceuma_df

class SkProc(LBSkProc):
    """
    地方競馬の機械学習処理プロセスを取りまとめたクラス。
    """
    lgbm_params = {
        # 2値分類問題
        'objective': 'binary'
    }
    index_list = ["RACE_KEY", "NENGAPPI"]

    def _get_load_object(self, version_str, start_date, end_date, mock_flag, test_flag):
        ld = Ld(version_str, start_date, end_date, mock_flag, test_flag)
        return ld

    def learning_sk_model(self, df, cls_val, val, target):
        """ 指定された場所・ターゲットに対しての学習処理を行う

        :param dataframe df: dataframe
        :param str val: str
        :param str target: str
        """
        this_model_name = self.model_name + "_" + cls_val + '_' + val + '_' + target
        if os.path.exists(self.model_folder + this_model_name + '.pickle'):
            print("\r\n -- skip create learning model -- \r\n")
        else:
            self.set_target_flag(target)
            print("learning_sk_model: df", df.shape)
            if df.empty:
                print("--------- alert !!! no data")
            else:
                self.set_learning_data(df, target)
                self.divide_learning_data()
                if self.y_train.sum() == 0:
                    print("---- wrong data --- skip learning")
                else:
                    self.load_learning_target_encoding()
                    imp_features = self.learning_base_race_lgb(this_model_name)
                    self.x_df = self.x_df[imp_features]
                    self.divide_learning_data()
                    self._set_label_list(self.x_df) # 項目削除されているから再度ターゲットエンコーディングの対象リストを更新する
                    self.load_learning_target_encoding()
                    self.learning_race_lgb(this_model_name)


    def learning_base_race_lgb(self, this_model_name):
        # テスト用のデータを評価用と検証用に分ける
        X_eval, X_valid, y_eval, y_valid = train_test_split(self.X_test, self.y_test, random_state=42)

        # データセットを生成する
        lgb_train = lgb.Dataset(self.X_train, self.y_train)
        lgb_eval = lgb.Dataset(X_eval, y_eval, reference=lgb_train)

        # 上記のパラメータでモデルを学習する
        model = lgb_original.train(self.lgbm_params, lgb_train,
                          # モデルの評価用データを渡す
                          valid_sets=lgb_eval,
                          # 最大で 1000 ラウンドまで学習する
                          num_boost_round=50,
                          # 10 ラウンド経過しても性能が向上しないときは学習を打ち切る
                          early_stopping_rounds=10)

        # 特徴量の重要度を含むデータフレームを作成
        imp_df = pd.DataFrame()
        imp_df["feature"] = X_eval.columns
        imp_df["importance"] = model.feature_importance()
        print(imp_df)
        imp_df = imp_df.sort_values("importance")

        # 比較用のランダム化したモデルを学習する
        N_RUM = 10 #30くらいに数上げてよさそう
        null_imp_df = pd.DataFrame()
        for i in range(N_RUM):
            print(i)
            ram_lgb_train = lgb.Dataset(self.X_train, np.random.permutation(self.y_train))
            ram_lgb_eval = lgb.Dataset(X_eval, np.random.permutation(y_eval), reference=lgb_train)
            ram_model = lgb_original.train(self.lgbm_params, ram_lgb_train,
                              # モデルの評価用データを渡す
                              valid_sets=ram_lgb_eval,
                              # 最大で 1000 ラウンドまで学習する
                              num_boost_round=100,
                              # 10 ラウンド経過しても性能が向上しないときは学習を打ち切る
                              early_stopping_rounds=6)
            ram_imp_df = pd.DataFrame()
            ram_imp_df["feature"] = X_eval.columns
            ram_imp_df["importance"] = ram_model.feature_importance()
            ram_imp_df = ram_imp_df.sort_values("importance")
            ram_imp_df["run"] = i + 1
            null_imp_df = pd.concat([null_imp_df, ram_imp_df])

        # 閾値を設定
        THRESHOLD = 30

        # 閾値を超える特徴量を取得
        imp_features = []
        for feature in imp_df["feature"]:
            actual_value = imp_df.query(f"feature=='{feature}'")["importance"].values
            null_value = null_imp_df.query(f"feature=='{feature}'")["importance"].values
            percentage = (null_value < actual_value).sum() / null_value.size * 100
            if percentage >= THRESHOLD:
                imp_features.append(feature)

        print(len(imp_features))
        print(imp_features)

        self._save_learning_model(imp_features, this_model_name + "_feat_columns")
        return imp_features

    def learning_race_lgb(self, this_model_name):
        # テスト用のデータを評価用と検証用に分ける
        X_eval, X_valid, y_eval, y_valid = train_test_split(self.X_test, self.y_test, random_state=42)

        # データセットを生成する
        lgb_train = lgb.Dataset(self.X_train, self.y_train)
        lgb_eval = lgb.Dataset(X_eval, y_eval, reference=lgb_train)

        # 上記のパラメータでモデルを学習する
        model = lgb.train(self.lgbm_params, lgb_train,
                          # モデルの評価用データを渡す
                          valid_sets=lgb_eval,
                          # 最大で 1000 ラウンドまで学習する
                          num_boost_round=1000,
                          # 10 ラウンド経過しても性能が向上しないときは学習を打ち切る
                          early_stopping_rounds=10)

        self._save_learning_model(model, this_model_name)


    def _drop_unnecessary_columns(self):
        """ predictに不要な列を削除してpredict_dfを作成する。削除する列は血統登録番号、確定着順、タイム指数、単勝オッズ、単勝人気  """
        pass

    def _scale_df(self):
        pass


    def _merge_df(self):
        """  レース、レース馬、前走、過去走のデータを結合したdataframeをbase_dfにセットする。競走コードと馬番はRACE_KEY,UMABANに名前変更する  """
        print("merge_to_basedf")
        self.base_df = pd.merge(self.ld.race_df, self.ld.raceuma_df, on="競走コード")
        self.base_df = pd.merge(self.base_df, self.ld.horse_df, on="血統登録番号")
        self.base_df = pd.merge(self.base_df, self.ld.prev1_raceuma_df, on=["競走コード", "馬番"], how='left')
        self.base_df = pd.merge(self.base_df, self.ld.prev2_raceuma_df, on=["競走コード", "馬番"], how='left')
        self.base_df = pd.merge(self.base_df, self.ld.prev3_raceuma_df, on=["競走コード", "馬番"], how='left')
        self.base_df = pd.merge(self.base_df, self.ld.prev4_raceuma_df, on=["競走コード", "馬番"], how='left')
        self.base_df = pd.merge(self.base_df, self.ld.prev5_raceuma_df, on=["競走コード", "馬番"], how='left')

    def _create_feature(self,):
        """ 過去走と今回を比較した特徴量等、最終的な特徴良を生成する """
        print("_create_feature")
        self.base_df.loc[:, "継続騎乗"] = (self.base_df["騎手名"] == self.base_df["騎手名_1"])
        self.base_df.loc[:, "同騎手_1"] = (self.base_df["騎手名"] == self.base_df["騎手名_1"])
        self.base_df.loc[:, "同騎手_2"] = (self.base_df["騎手名"] == self.base_df["騎手名_2"])
        self.base_df.loc[:, "同騎手_3"] = (self.base_df["騎手名"] == self.base_df["騎手名_3"])
        self.base_df.loc[:, "同騎手_4"] = (self.base_df["騎手名"] == self.base_df["騎手名_4"])
        self.base_df.loc[:, "同騎手_5"] = (self.base_df["騎手名"] == self.base_df["騎手名_5"])
        self.base_df.loc[:, "同場騎手"] = (self.base_df["騎手所属場コード"] == self.base_df["場コード"])
        self.base_df.loc[:, "同所属場"] = (self.base_df["調教師所属場コード"] == self.base_df["場コード"])
        self.base_df.loc[:, "同所属騎手"] = (self.base_df["騎手所属場コード"] == self.base_df["調教師所属場コード"])
        self.base_df.loc[:, "同主催者"] = (self.base_df["主催者コード"] == self.base_df["主催者コード_1"])
        self.base_df.loc[:, "同場コード"] = (self.base_df["場コード"] == self.base_df["場コード_1"])
        self.base_df.loc[:, "同場_1"] = (self.base_df["場コード"] == self.base_df["場コード_1"])
        self.base_df.loc[:, "同場_2"] = (self.base_df["場コード"] == self.base_df["場コード_2"])
        self.base_df.loc[:, "同場_3"] = (self.base_df["場コード"] == self.base_df["場コード_3"])
        self.base_df.loc[:, "同場_4"] = (self.base_df["場コード"] == self.base_df["場コード_4"])
        self.base_df.loc[:, "同場_5"] = (self.base_df["場コード"] == self.base_df["場コード_5"])
        self.base_df.loc[:, "同距離グループ_1"] = (self.base_df["距離グループ"] == self.base_df["距離グループ_1"])
        self.base_df.loc[:, "同距離グループ_2"] = (self.base_df["距離グループ"] == self.base_df["距離グループ_2"])
        self.base_df.loc[:, "同距離グループ_3"] = (self.base_df["距離グループ"] == self.base_df["距離グループ_3"])
        self.base_df.loc[:, "同距離グループ_4"] = (self.base_df["距離グループ"] == self.base_df["距離グループ_4"])
        self.base_df.loc[:, "同距離グループ_5"] = (self.base_df["距離グループ"] == self.base_df["距離グループ_5"])
        self.base_df.loc[:, "同季節_1"] = (self.base_df["季節"] == self.base_df["季節_1"])
        self.base_df.loc[:, "同季節_2"] = (self.base_df["季節"] == self.base_df["季節_2"])
        self.base_df.loc[:, "同季節_3"] = (self.base_df["季節"] == self.base_df["季節_3"])
        self.base_df.loc[:, "同季節_4"] = (self.base_df["季節"] == self.base_df["季節_4"])
        self.base_df.loc[:, "同季節_5"] = (self.base_df["季節"] == self.base_df["季節_5"])
        self.base_df.loc[:, "負担重量_1"] = self.base_df["負担重量_1"] - self.base_df["負担重量"]
        self.base_df.loc[:, "負担重量_2"] = self.base_df["負担重量_2"] - self.base_df["負担重量"]
        self.base_df.loc[:, "負担重量_3"] = self.base_df["負担重量_3"] - self.base_df["負担重量"]
        self.base_df.loc[:, "負担重量_4"] = self.base_df["負担重量_4"] - self.base_df["負担重量"]
        self.base_df.loc[:, "負担重量_5"] = self.base_df["負担重量_5"] - self.base_df["負担重量"]
        self.base_df.loc[:, "頭数差"] = (self.base_df["頭数グループ"] - self.base_df["頭数グループ_1"])
        self.base_df.loc[:, "中央経験"] = self.base_df.apply(lambda x: True if (x["主催者コード_1"] == 1 or x["主催者コード_2"] == 1 or x["主催者コード_3"] == 1 or x["主催者コード_4"] == 1 or x["主催者コード_5"] == 1) else False, axis=1)
        self.base_df.loc[:, "休み明け"] = self.base_df["休養週数"].apply(lambda x: True if x >= 20 else False)
        self.base_df.loc[:, "覚醒"] = self.base_df.apply(lambda x: True if (x["激走_1"] + x["激走_2"] + x["激走_3"] + x["激走_4"] + x["激走_5"] >= 3) else False, axis=1)
        self.base_df.loc[:, "失速"] = self.base_df.apply(lambda x: True if (x["凡走_1"] + x["凡走_2"] + x["凡走_3"] + x["凡走_4"] + x["凡走_5"] >= 3) else False, axis=1)
        self.base_df.loc[:, "逃げそびれ凡走"] = self.base_df.apply(lambda x: True if (x["逃げそびれ_1"] * x["凡走_1"] + x["逃げそびれ_2"] * x["凡走_2"] + x["逃げそびれ_3"] * x["凡走_3"] + x["逃げそびれ_4"] * x["凡走_4"] + x["逃げそびれ_5"] * x["凡走_5"] >= 2) else False, axis=1)
        self.base_df.loc[:, "継続騎乗好走"] = self.base_df.apply(lambda x: True if x["継続騎乗"] * x["激走_1"] == 1 else False, axis=1)
        self.base_df.loc[:, "末脚安定"] = self.base_df.apply(lambda x: True if (x["上がりタイム順位_1"] + x["上がりタイム順位_2"] + x["上がりタイム順位_3"] <= 8 ) else False, axis=1)
        self.base_df.loc[:, "同騎手○"] = self.base_df.apply(lambda x: True if (x["同騎手_1"] * x["好走_1"] + x["同騎手_2"] * x["好走_2"] + x["同騎手_3"] * x["好走_3"] + x["同騎手_4"] * x["好走_4"] + x["同騎手_5"] * x["好走_5"]) >= 3 else False, axis=1)
        self.base_df.loc[:, "同騎手◎"] = self.base_df.apply(lambda x: True if (x["同騎手_1"] * x["激走_1"] + x["同騎手_2"] * x["激走_2"] + x["同騎手_3"] * x["激走_3"] + x["同騎手_4"] * x["激走_4"] + x["同騎手_5"] * x["激走_5"]) >= 3 else False, axis=1)
        self.base_df.loc[:, "同騎手逃げ"] = self.base_df.apply(lambda x: True if (x["同騎手_1"] * x["３角先頭_1"] + x["同騎手_2"] * x["３角先頭_2"] + x["同騎手_3"] * x["３角先頭_3"] + x["同騎手_4"] * x["３角先頭_4"] + x["同騎手_5"] * x["３角先頭_5"]) >= 2 else False, axis=1)
        self.base_df.loc[:, "同場○"] = self.base_df.apply(lambda x: True if (x["同場_1"] * x["好走_1"] + x["同場_2"] * x["好走_2"] + x["同場_3"] * x["好走_3"] + x["同場_4"] * x["好走_4"] + x["同場_5"] * x["好走_5"]) >= 3 else False, axis=1)
        self.base_df.loc[:, "同場◎"] = self.base_df.apply(lambda x: True if (x["同場_1"] * x["激走_1"] + x["同場_2"] * x["激走_2"] + x["同場_3"] * x["激走_3"] + x["同場_4"] * x["激走_4"] + x["同場_5"] * x["激走_5"]) >= 3 else False, axis=1)
        self.base_df.loc[:, "上がり最速数"] = self.base_df.apply(lambda x: True if (x["上がり最速_1"] + x["上がり最速_2"] + x["上がり最速_3"] + x["上がり最速_4"] + x["上がり最速_5"]) >= 3 else False, axis=1)
        self.base_df.loc[:, "逃げ好走"] = self.base_df.apply(lambda x: True if (x["４角先頭_1"] * x["好走_1"] + x["４角先頭_2"] * x["好走_2"] + x["４角先頭_3"] * x["好走_3"] + x["４角先頭_4"] * x["好走_4"] + x["４角先頭_5"] * x["好走_5"] >= 2) else False, axis=1)
        self.base_df.loc[:, "ムラっけ"] = self.base_df.apply(lambda x: True if (x["激走_1"] + x["激走_2"] + x["激走_3"] + x["激走_4"] + x["激走_5"] >= 2) and (x["大差負け_1"] + x["大差負け_2"] + x["大差負け_3"] + x["大差負け_4"] + x["大差負け_5"] >= 2)  else False, axis=1)
        self.base_df.loc[:, "連闘○"] = self.base_df.apply(lambda x: True if (x["連闘_1"] * x["好走_1"] + x["連闘_2"] * x["好走_2"] + x["連闘_3"] * x["好走_3"] + x["連闘_4"] * x["好走_4"] + x["連闘_5"] * x["好走_5"]) >= 3 else False, axis=1)
        self.base_df.loc[:, "連闘◎"] = self.base_df.apply(lambda x: True if (x["連闘_1"] * x["激走_1"] + x["連闘_2"] * x["激走_2"] + x["連闘_3"] * x["激走_3"] + x["連闘_4"] * x["激走_4"] + x["連闘_5"] * x["激走_5"]) >= 3 else False, axis=1)
        #self.base_df.loc[:, "ナイター○"] = self.base_df.apply(lambda x: 1 if (x["ナイター_1"] * x["好走_1"] + x["ナイター_2"] * x["好走_2"] + x["ナイター_3"] * x["好走_3"] + x["ナイター_4"] * x["好走_4"] + x["ナイター_5"] * x["好走_5"]) >= 3 else False, axis=1)
        #self.base_df.loc[:, "ナイター◎"] = self.base_df.apply(lambda x: 1 if (x["ナイター_1"] * x["激走_1"] + x["ナイター_2"] * x["激走_2"] + x["ナイター_3"] * x["激走_3"] + x["ナイター_4"] * x["激走_4"] + x["ナイター_5"] * x["激走_5"]) >= 3 else False, axis=1)
        self.base_df.loc[:, "同距離○"] = self.base_df.apply(lambda x: True if (x["同距離グループ_1"] * x["好走_1"] + x["同距離グループ_2"] * x["好走_2"] + x["同距離グループ_3"] * x["好走_3"] + x["同距離グループ_4"] * x["好走_4"] + x["同距離グループ_5"] * x["好走_5"]) >= 3 else False, axis=1)
        self.base_df.loc[:, "同距離◎"] = self.base_df.apply(lambda x: True if (x["同距離グループ_1"] * x["激走_1"] + x["同距離グループ_2"] * x["激走_2"] + x["同距離グループ_3"] * x["激走_3"] + x["同距離グループ_4"] * x["激走_4"] + x["同距離グループ_5"] * x["激走_5"]) >= 3 else False, axis=1)
        self.base_df.loc[:, "同季節○"] = self.base_df.apply(lambda x: True if (x["同季節_1"] * x["好走_1"] + x["同季節_2"] * x["好走_2"] + x["同季節_3"] * x["好走_3"] + x["同季節_4"] * x["好走_4"] + x["同季節_5"] * x["好走_5"]) >= 3 else False, axis=1)
        self.base_df.loc[:, "同季節◎"] = self.base_df.apply(lambda x: True if (x["同季節_1"] * x["激走_1"] + x["同季節_2"] * x["激走_2"] + x["同季節_3"] * x["激走_3"] + x["同季節_4"] * x["激走_4"] + x["同季節_5"] * x["激走_5"]) >= 3 else False, axis=1)
        self.base_df.loc[:, "同場騎手○"] = self.base_df.apply(lambda x: True if (x["同場騎手_1"] * x["好走_1"] + x["同場騎手_2"] * x["好走_2"] + x["同場騎手_3"] * x["好走_3"] + x["同場騎手_4"] * x["好走_4"] + x["同場騎手_5"] * x["好走_5"]) >= 3 else False, axis=1)
        self.base_df.loc[:, "同場騎手◎"] = self.base_df.apply(lambda x: True if (x["同場騎手_1"] * x["激走_1"] + x["同場騎手_2"] * x["激走_2"] + x["同場騎手_3"] * x["激走_3"] + x["同場騎手_4"] * x["激走_4"] + x["同場騎手_5"] * x["激走_5"]) >= 3 else False, axis=1)
        self.base_df.loc[:, "同所属場○"] = self.base_df.apply(lambda x: True if (x["同所属場_1"] * x["好走_1"] + x["同所属場_2"] * x["好走_2"] + x["同所属場_3"] * x["好走_3"] + x["同所属場_4"] * x["好走_4"] + x["同所属場_5"] * x["好走_5"]) >= 3 else False, axis=1)
        self.base_df.loc[:, "同所属場◎"] = self.base_df.apply(lambda x: True if (x["同所属場_1"] * x["激走_1"] + x["同所属場_2"] * x["激走_2"] + x["同所属場_3"] * x["激走_3"] + x["同所属場_4"] * x["激走_4"] + x["同所属場_5"] * x["激走_5"]) >= 3 else False, axis=1)
        self.base_df.loc[:, "同所属騎手○"] = self.base_df.apply(lambda x: True if (x["同所属騎手_1"] * x["好走_1"] + x["同所属騎手_2"] * x["好走_2"] + x["同所属騎手_3"] * x["好走_3"] + x["同所属騎手_4"] * x["好走_4"] + x["同所属騎手_5"] * x["好走_5"]) >= 3 else False, axis=1)
        self.base_df.loc[:, "同所属騎手◎"] = self.base_df.apply(lambda x: True if (x["同所属騎手_1"] * x["激走_1"] + x["同所属騎手_2"] * x["激走_2"] + x["同所属騎手_3"] * x["激走_3"] + x["同所属騎手_4"] * x["激走_4"] + x["同所属騎手_5"] * x["激走_5"]) >= 3 else False, axis=1)
        self.base_df.drop(
            ["発走時刻_1", "発走時刻_2", "発走時刻_3", "発走時刻_4", "発走時刻_5",
             "トラックコード_1", "トラックコード_2", "トラックコード_3", "トラックコード_4", "トラックコード_5",
             "ナイター_1", "ナイター_2", "ナイター_3", "ナイター_4", "ナイター_5",
             "距離グループ_1", "距離グループ_2", "距離グループ_3", "距離グループ_4", "距離グループ_5",
             "同季節_1", "同季節_2", "同季節_3", "同季節_4", "同季節_5",
             "連闘_1", "連闘_2", "連闘_3", "連闘_4", "連闘_5",
             "頭数グループ_1", "頭数グループ_2", "頭数グループ_3", "頭数グループ_4", "頭数グループ_5",
             "同騎手_1", "同騎手_2", "同騎手_3", "同騎手_4", "同騎手_5",
             "３角先頭_3", "３角先頭_4", "３角先頭_5",
             "４角先頭_3", "４角先頭_4", "４角先頭_5",
             "凡走_4", "凡走_5",
             "好走_4", "好走_5",
             "激走_4", "激走_5",
             "季節_1", "季節_2", "季節_3", "季節_4", "季節_5",
             "テン乗り_1", "テン乗り_2", "テン乗り_3", "テン乗り_4", "テン乗り_5",
             "騎手名_1", "騎手名_2", "騎手名_3", "騎手名_4", "騎手名_5",
             "勝ち_1", "勝ち_2", "勝ち_3", "勝ち_4", "勝ち_5",
             "大差負け_3", "大差負け_4", "大差負け_5",
             "同場騎手_1", "同場騎手_2", "同場騎手_3", "同場騎手_4", "同場騎手_5",
             "同所属場_1", "同所属場_2", "同所属場_3", "同所属場_4", "同所属場_5",
             "同所属騎手_1", "同所属騎手_2", "同所属騎手_3", "同所属騎手_4", "同所属騎手_5",
             "同距離グループ_1", "同距離グループ_2", "同距離グループ_3", "同距離グループ_4", "同距離グループ_5",
             "同場_1", "同場_2", "同場_3", "同場_4", "同場_5",
             "年月日", "近走競走コード1", "近走馬番1", "近走競走コード2", "近走馬番2", "近走競走コード3", "近走馬番3", "近走競走コード4", "近走馬番4",
             "近走競走コード5", "近走馬番5",
             "負担重量_1", "負担重量_2", "負担重量_3", "負担重量_4", "負担重量_5",
             "上がりタイム順位_3", "上がりタイム順位_4", "上がりタイム順位_5",
             "血統登録番号", "テン乗り", "性別コード",
             "ペース_1", "ペース_2", "ペース_3", "ペース_4", "ペース_5",
               '主催者コード_1', '場コード_1', 'コース_1', '主催者コード_2', '場コード_2', 'コース_2',
               '主催者コード_3', '場コード_3', 'コース_3', '主催者コード_4', '場コード_4', 'コース_4',
               '主催者コード_5', '場コード_5', 'コース_5'
             ],
        axis=1, inplace=True)
        self.base_df.loc[:, "競走馬コード"] = self.base_df["競走コード"].astype(str).str.cat(self.base_df["馬番"].astype(str))
        # https://qiita.com/daigomiyoshi/items/d6799cc70b2c1d901fb5
        es = ft.EntitySet(id="raceuma")
        ignore_list = ['騎手名', '調教師名', '所属', '転厩', '頭数', '枠番', '馬番', '予想勝ち指数', '距離グループ', '予想タイム指数順位',
                       '距離', '予想人気', '距離増減', '先行指数順位', '馬番グループ', '季節', '月日']
        es.entity_from_dataframe(entity_id='raceuma', dataframe=self.base_df.drop(ignore_list, axis=1), index="競走馬コード")
        es.normalize_entity(base_entity_id='raceuma', new_entity_id='race', index="競走コード")
        print(es)
        # 集約関数
        aggregation_list = ['max', 'median', 'percent_true']
        # run dfs
        print("_create_feature: base_df", self.base_df.shape)
        print("run dfs")
        feature_matrix, features_dfs = ft.dfs(entityset=es, target_entity='raceuma', agg_primitives=aggregation_list,
                                              trans_primitives=[], max_depth=2)
        self.base_df = pd.concat([feature_matrix.reset_index().drop("競走馬コード", axis=1), self.base_df[ignore_list]], axis=1)

    def _drop_columns_base_df(self):
        pass

    def _sub_distribute_predict_model(self, cls_val, val, target, temp_df):
        """ model_nameに応じたモデルを呼び出し予測を行う

        :param str val: 場所名
        :param str target: 目的変数名
        :param dataframe temp_df: 予測するデータフレーム
        :return dataframe: pred_df
        """
        this_model_name = self.model_name + "_" + cls_val + '_' + val + '_' + target
        pred_df = self.predict_race_lgm(this_model_name, temp_df)
        print(pred_df.head())
        return pred_df

    def predict_race_lgm(self, this_model_name, temp_df):
        print("======= this_model_name: " + this_model_name + " ==========")
        with open(self.model_folder + this_model_name + '_feat_columns.pickle', 'rb') as f:
            imp_features = pickle.load(f)
        temp_df = temp_df.replace(np.inf,np.nan).fillna(temp_df.replace(np.inf,np.nan).mean()).reset_index()
        exp_df = temp_df.drop(self.index_list, axis=1)
        exp_df = exp_df[imp_features].to_numpy()
        print("predict_race_lgm: exp_df", exp_df.shape)
        print(self.model_folder)
        if os.path.exists(self.model_folder + this_model_name + '.pickle'):
            with open(self.model_folder + this_model_name + '.pickle', 'rb') as f:
                model = pickle.load(f)
            y_pred = model.predict(exp_df)
            pred_df = pd.DataFrame(
                {"RACE_KEY": temp_df["RACE_KEY"], "UMABAN": temp_df["UMABAN"], "NENGAPPI": temp_df["NENGAPPI"],
                 "prob": y_pred})
            pred_df.loc[:, "pred"] = pred_df.apply(lambda x: 1 if x["prob"] >= 0.5 else 0, axis=1)
            return pred_df
        else:
            return pd.DataFrame()


class SkModel(LBSkModel):
    class_list = ['主催者コード']
    table_name = TABLE_NAME

    def _get_skproc_object(self, version_str, start_date, end_date, model_name, mock_flag, test_flag):
        proc = SkProc(version_str, start_date, end_date, model_name, mock_flag, test_flag, self.obj_column_list)
        return proc

    def proc_learning_sk_model(self, df, cls_val, val):
        """  説明変数ごとに、指定された場所の学習を行う

        :param dataframe df: dataframe
        :param str basho: str
        """
        if len(df.index) >= 30:
            for target in self.obj_column_list:
                print(target)
                self.proc.learning_sk_model(df, cls_val, val, target)
        else:
            print("---- 少数レコードのため学習スキップ -- " + str(len(df.index)))

    def create_import_data(self, all_df):
        """ 予測値の偏差と順位を追加して格納 """
        grouped_df = self._calc_grouped_data(all_df)
        import_df = grouped_df[["RACE_KEY", "UMABAN", "pred", "prob", "predict_std", "predict_rank", "target", "target_date"]].round(3)
        print(import_df)
        return import_df


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
            start_date = '2018/01/01'
            end_date = '2018/01/31'
        else:
            start_date = '2015/01/01'
            end_date = '2018/12/31'
        if mock_flag:
            print("use mock data")
        print("MODE:learning mock_flag: " + str(args[2]) + "  start_date:" + start_date + " end_date:" + end_date)

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