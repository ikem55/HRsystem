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

from sklearn.model_selection import train_test_split
import lightgbm as lgb_original
from sklearn.metrics import accuracy_score
import featuretools as ft
import pyodbc
import pickle

import optuna.integration.lightgbm as lgb
import warnings
warnings.simplefilter('ignore')

basedir = os.path.dirname(__file__)[:-8]
print(basedir)
sys.path.append(basedir)

# 呼び出し方
# python lb_v1.py learning True True
# ====================================== パラメータ　要変更 =====================================================
# Modelv4 LightGBMを使ってレースの１着、２着、３着、着外の確率を計算する。木モデルなのでNull許容できるはず
MODEL_VERSION = 'lb_v4'
MODEL_NAME = 'raceuma_lgm'
TABLE_NAME = '地方競馬レース馬V4'

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
        temp_race_df.loc[:, 'ナイター'] = race_df['発走時刻'].apply(lambda x: 1 if x.hour >= 17 else 0)
        temp_race_df.loc[:, '季節'] = (race_df['月日'].apply(lambda x: x.month) - 1) // 3
        temp_race_df['季節'].astype('str')
        temp_race_df.loc[:, "非根幹"] = race_df["距離"].apply(lambda x: 0 if x % 400 == 0 else 1)
        temp_race_df.loc[:, "距離グループ"] = race_df["距離"] // 400
        temp_race_df.loc[:, "頭数グループ"] = race_df["頭数"] // 5
        return temp_race_df

    def choose_race_result_column(self, race_df):
        """ レースデータから必要な列に絞り込む。列は '発走時刻', '月日', '競走コード', '距離'', '頭数', 'ペース', 'トラックコード', '後３ハロン'

        :param dataframe race_df:
        :return: dataframe
        """
        temp_race_df = race_df[
            ['発走時刻', '月日', '競走コード', '距離', '頭数', 'ペース', 'トラックコード', '後３ハロン']]
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
        temp_merge_df.loc[:, "逃げ勝ち"] = temp_merge_df["コーナー順位4"].apply(lambda x: 1 if x == 1 else 0)
        temp_merge_df.loc[:, "内勝ち"] = temp_merge_df["枠番"].apply(lambda x: 1 if x in (1, 2, 3) else 0)
        temp_merge_df.loc[:, "外勝ち"] = temp_merge_df["枠番"].apply(lambda x: 1 if x in (6, 7, 8) else 0)
        temp_merge_df.loc[:, "短縮勝ち"] = temp_merge_df["距離増減"].apply(lambda x: 1 if x < 0 else 0)
        temp_merge_df.loc[:, "延長勝ち"] = temp_merge_df["距離増減"].apply(lambda x: 1 if x > 0 else 0)
        temp_merge_df.loc[:, "人気勝ち"] = temp_merge_df["単勝人気"].apply(lambda x: 1 if x == 1 else 0)
        merge_df = pd.merge(temp_race_df,
                            temp_merge_df[["競走コード", "上り係数", "逃げ勝ち", "内勝ち", "外勝ち", "短縮勝ち", "延長勝ち", "人気勝ち"]],
                            on="競走コード")
        return merge_df

    def drop_columns_raceuma_df(self, raceuma_df):
        return raceuma_df.drop(["データ作成年月日", "血統距離評価", "血統トラック評価", "血統成長力評価", "騎手所属場コード", "調教師所属場コード", "見習区分", "馬齢",
                                           "血統総合評価", "血統距離評価B", "血統トラック評価B", "血統成長力評価B", "血統総合評価B", "騎手コード", "騎手評価",
                                           "調教師評価", "枠順評価", "脚質評価", "調教師コード", "前走着順", "前走人気", "前走着差", "前走トラック種別コード",
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
        return temp_raceuma_df.copy()

    def create_feature_raceuma_df(self, raceuma_df):
        """  raceuma_dfの特徴量を作成する。

        :param dataframe raceuma_df:
        :return: dataframe
        """
        temp_raceuma_df = raceuma_df.copy()
        temp_raceuma_df.loc[:, "予想タイム指数順位"] = raceuma_df["予想タイム指数順位"].apply(lambda x: 1 if x == 0 else 1 / x)
        temp_raceuma_df.loc[:, "休養週数"] = raceuma_df["休養週数"].apply(lambda x: 1 if x == 0 else 1 / x)
        temp_raceuma_df.loc[:, "休養後出走回数"] = raceuma_df["休養後出走回数"].apply(lambda x: 5 if x >= 5 else x)
        temp_raceuma_df.loc[:, "予想人気"] = raceuma_df["予想人気"].apply(lambda x: 1 if x == 0 else 1 / x)
        temp_raceuma_df.loc[:, "先行指数順位"] = raceuma_df["先行指数順位"].apply(lambda x: 1 if x == 0 else 1 / x)
        temp_raceuma_df.loc[:, "距離増減"] = raceuma_df["距離増減"] // 200
        return temp_raceuma_df

    def choose_raceuma_result_column(self, raceuma_df):
        """  レース馬データから必要な列に絞り込む。

        :param dataframe raceuma_df:
        :return: dataframe
        """
        temp_raceuma_df = raceuma_df[
            ['競走コード', '馬番', '枠番', '年月日', '血統登録番号', 'タイム指数', '単勝オッズ', '単勝人気', '確定着順', '着差', '休養週数', '先行率', 'タイム', '予想展開',
             'ペース偏差値', '展開コード', '騎手名', 'テン乗り', '負担重量', '馬体重', 'コーナー順位3', 'コーナー順位4', '距離増減',
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
        print(temp_merge_df.shape)
        temp_raceuma_df.loc[:, "追込率"] = (temp_merge_df["コーナー順位4"] - temp_merge_df["確定着順"]) / temp_merge_df["頭数"]
        temp_raceuma_df.loc[:, "平均タイム"] = temp_merge_df["タイム"] / temp_merge_df["距離"] * 200
        temp_raceuma_df.loc[:, "勝ち"] = temp_raceuma_df["確定着順"].apply(lambda x: 1 if x == 1 else 0)
        temp_raceuma_df.loc[:, "１番人気"] = temp_raceuma_df["単勝人気"].apply(lambda x: 1 if x == 1 else 0)
        temp_raceuma_df.loc[:, "３角先頭"] = temp_raceuma_df["コーナー順位3"].apply(lambda x: 1 if x == 1 else 0)
        temp_raceuma_df.loc[:, "４角先頭"] = temp_raceuma_df["コーナー順位4"].apply(lambda x: 1 if x == 1 else 0)
        temp_raceuma_df.loc[:, "休み明け"] = temp_raceuma_df["休養週数"].apply(lambda x: 1 if x >= 10 else 0)
        temp_raceuma_df.loc[:, "連闘"] = temp_raceuma_df["休養週数"].apply(lambda x: 1 if x == 1 else 0)
        temp_raceuma_df.loc[:, "大差負け"] = temp_raceuma_df["着差"].apply(lambda x: 1 if x >= 20 else 0)
        temp_raceuma_df.loc[:, "凡走"] = temp_merge_df.apply(lambda x: 1 if x["確定着順"] - x["単勝人気"] > 5 else 0, axis=1)
        temp_raceuma_df.loc[:, "好走"] = temp_merge_df["確定着順"].apply(lambda x: 1 if x <= 3 else 0)
        temp_raceuma_df.loc[:, "激走"] = temp_merge_df.apply(lambda x: 1 if x["単勝人気"] - x["確定着順"] > 5 else 0, axis=1)

        return temp_raceuma_df.drop(["確定着順", "単勝人気", "コーナー順位3", "コーナー順位4", "休養週数", "着差"], axis=1).copy()

    def factory_analyze_raceuma_result_df(self, race_df, input_raceuma_df, dict_folder):
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
        merged_df = merged_df.drop(['枠番',  'タイム指数', '単勝オッズ', '先行率', 'ペース偏差値', '距離増減', '斤量比', '追込率', '平均タイム',
             "距離", "頭数", "上り係数", "逃げ勝ち", "内勝ち", "外勝ち", "短縮勝ち", "延長勝ち", "人気勝ち", "１番人気",
             "後３ハロン", "予想展開", 'タイム', "非根幹", "季節", "ナイター",
             "上がりタイム", "休み明け", "年月日", "月日", "距離", "血統登録番号"], axis=1)
        return merged_df

    def scale_df(self):
        print("skip scale_df")

class SkProc(LBSkProc):
    """
    地方競馬の機械学習処理プロセスを取りまとめたクラス。
    """
    lgbm_params = {
        # 多値分類問題
        'objective': 'multiclass',
        'metric': 'multi_logloss',
#        'metric': {'multi_logloss', 'multi_error'},
        # クラス数は 17(0も含める）
        'num_class': 17,
    }
    index_list = ["RACE_KEY", "NENGAPPI"]

    def _get_load_object(self, version_str, start_date, end_date, mock_flag, test_flag):
        ld = Ld(version_str, start_date, end_date, mock_flag, test_flag)
        return ld

    def _set_ld_data(self):
        """  Loadオブジェクトにデータをセットする処理をまとめたもの。Race,Raceuma,Horse,Prevのデータフレームをセットする

        :param object ld: データロードオブジェクト(ex.LocalBaozLoad)
        """
        self.ld.set_race_df()  # データ取得
        self.ld.set_raceuma_df()
        self.ld.set_prev_df()

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
            print(df.shape)
            if df.empty:
                print("--------- alert !!! no data")
            else:
                self.set_learning_data(df, target)
                self.divide_learning_data()
                if self.y_train.sum() == 0:
                    print("---- wrong data --- skip learning")
                else:
                    self.learning_race_lgb(this_model_name)


    def learning_race_lgb(self, this_model_name):
        # テスト用のデータを評価用と検証用に分ける
        X_eval, X_valid, y_eval, y_valid = train_test_split(self.X_test, self.y_test, random_state=42)

        # データセットを生成する
        lgb_train = lgb.Dataset(self.X_train, self.y_train)
        lgb_eval = lgb.Dataset(X_eval, y_eval, reference=lgb_train)

        # 上記のパラメータでモデルを学習する
        best_params, history = {}, []
        model = lgb.train(self.lgbm_params, lgb_train, valid_sets=lgb_eval,
                          verbose_eval=False,
                          num_boost_round=100,
                          early_stopping_rounds=5,
                          best_params=best_params,
                          tuning_history=history)
        print("Bset Paramss:", best_params)
        print('Tuning history:', history)

        self._save_learning_model(model, this_model_name)

    def proc_create_learning_data(self):
        self._proc_create_base_df()
        self._drop_unnecessary_columns()
        self._set_target_variables()
        learning_df = pd.merge(self.base_df, self.result_df, on="RACE_KEY")
        return learning_df

    def _drop_unnecessary_columns(self):
        """ predictに不要な列を削除してpredict_dfを作成する。削除する列は血統登録番号、確定着順、タイム指数、単勝オッズ、単勝人気  """
        pass

    def _set_target_variables(self):
        self.ld.set_result_df()
        ck1_df = self._create_target_ck1()
        ck2_df = self._create_target_ck2()
        ck3_df = self._create_target_ck3()
        self.result_df = pd.merge(ck1_df, ck2_df, on="競走コード")
        self.result_df = pd.merge(self.result_df, ck3_df, on="競走コード").rename(columns={"競走コード": "RACE_KEY"})

    def _create_target_ck1(self):
        """  １着馬を目的変数にセット """
        ck1_df = self.ld.result_df.query("確定着順 == 1")[["競走コード", "馬番"]].rename(columns={"馬番": "１着"})
        return ck1_df

    def _create_target_ck2(self):
        """  １着馬を目的変数にセット """
        ck2_df = self.ld.result_df.query("確定着順 == 2")[["競走コード", "馬番"]].rename(columns={"馬番": "２着"})
        return ck2_df

    def _create_target_ck3(self):
        """  １着馬を目的変数にセット """
        ck3_df = self.ld.result_df.query("確定着順 == 3")[["競走コード", "馬番"]].rename(columns={"馬番": "３着"})
        return ck3_df

    def _proc_create_base_df(self):
        self._set_ld_data()
        self._merge_df()
        self._create_feature()
        self._drop_columns_base_df()
        self._flat_base_df()
        self.base_df = self._rename_key(self.base_df)

    def _flat_base_df(self):
        """ レース馬情報を１行に並べてレース情報をつなげたものをbase_dfとして再作成する """
        print("_flat_base_df")
        temp_df = self.base_df.drop(["季節", "枠番", "頭数グループ", "距離グループ", "休養後出走回数"], axis=1)
        temp_df = temp_df.astype({"馬番": str})
        temp_df = temp_df.set_index(["競走コード", "馬番"])
        temp_unstack_df = temp_df.unstack()
        unstack_columns = ["_".join(pair) for pair in temp_unstack_df.columns]
        temp_unstack_df.columns = unstack_columns
        columns_base = temp_df.columns.values.tolist()
        columns_list = []
        for i in range(1, 17):
            columns_list += [s + "_" + str(i) for s in columns_base]
        dif_list = columns_list.copy()
        for col in unstack_columns:
            try:
                dif_list.remove(col)
            except ValueError:
                continue
        for col in dif_list:
            temp_unstack_df[col] = np.NaN
        self.base_df = pd.merge(self.ld.race_df, temp_unstack_df, on ="競走コード")
        self.base_df = self.base_df.astype({"場コード": int, "競走種別コード": int})
        self.base_df.drop(['発走時刻', 'グレードコード', '競走条件コード', 'トラックコード'], axis=1, inplace= True)

    def f(self, a):
        a.index = [0 for i in range(len(a))]
        del a['ID']
        out = a[0:1]
        for i in range(1, len(a)):
            out = out.join(a[i:i + 1], rsuffix='{0}'.format(i))
        return out

    def _rename_key(self, df):
        """ キー名を競走コード→RACE_KEY、馬番→UMABANに変更 """
        return_df = df.rename(columns={"競走コード": "RACE_KEY", "月日": "NENGAPPI"})
        return return_df

    def _merge_df(self):
        """  レース、レース馬、前走、過去走のデータを結合したdataframeをbase_dfにセットする。最後にRaceumaをフラットにするのでRace系は最小限のみの結合で残す  """
        print("merge_to_basedf")
        temp_race_df = self.ld.race_df[["競走コード", "距離グループ", "季節", "頭数グループ"]]
        self.base_df = pd.merge(temp_race_df, self.ld.raceuma_df, on="競走コード")
        self.base_df = pd.merge(self.base_df, self.ld.prev1_raceuma_df, on=["競走コード", "馬番"], how='left')
        self.base_df = pd.merge(self.base_df, self.ld.prev2_raceuma_df, on=["競走コード", "馬番"], how='left')
        self.base_df = pd.merge(self.base_df, self.ld.prev3_raceuma_df, on=["競走コード", "馬番"], how='left')
        self.base_df = pd.merge(self.base_df, self.ld.prev4_raceuma_df, on=["競走コード", "馬番"], how='left')
        self.base_df = pd.merge(self.base_df, self.ld.prev5_raceuma_df, on=["競走コード", "馬番"], how='left')


    def _create_feature(self,):
        """ 過去走と今回を比較した特徴量等、最終的な特徴良を生成する """
        self.base_df.loc[:, "同距離グループ_1"] = (self.base_df["距離グループ"] == self.base_df["距離グループ_1"]).astype(int)
        self.base_df.loc[:, "同距離グループ_2"] = (self.base_df["距離グループ"] == self.base_df["距離グループ_2"]).astype(int)
        self.base_df.loc[:, "同距離グループ_3"] = (self.base_df["距離グループ"] == self.base_df["距離グループ_3"]).astype(int)
        self.base_df.loc[:, "同距離グループ_4"] = (self.base_df["距離グループ"] == self.base_df["距離グループ_4"]).astype(int)
        self.base_df.loc[:, "同距離グループ_5"] = (self.base_df["距離グループ"] == self.base_df["距離グループ_5"]).astype(int)
        self.base_df.loc[:, "負担重量_1"] = self.base_df["負担重量_1"] - self.base_df["負担重量"]
        self.base_df.loc[:, "負担重量_2"] = self.base_df["負担重量_2"] - self.base_df["負担重量"]
        self.base_df.loc[:, "負担重量_3"] = self.base_df["負担重量_3"] - self.base_df["負担重量"]
        self.base_df.loc[:, "負担重量_4"] = self.base_df["負担重量_4"] - self.base_df["負担重量"]
        self.base_df.loc[:, "負担重量_5"] = self.base_df["負担重量_5"] - self.base_df["負担重量"]
        self.base_df.loc[:, "頭数差"] = self.base_df["頭数グループ"] - self.base_df["頭数グループ_1"]
        self.base_df.loc[:, "末脚安定"] = self.base_df.apply(lambda x: 1 if (x["上がりタイム順位_1"] + x["上がりタイム順位_2"] + x["上がりタイム順位_3"] <= 8 ) else 0, axis=1)

    def _drop_columns_base_df(self):
        self.base_df.drop(
            ["発走時刻_1", "発走時刻_2", "発走時刻_3", "発走時刻_4", "発走時刻_5",
             "トラックコード_1", "トラックコード_2", "トラックコード_3", "トラックコード_4", "トラックコード_5",
             "距離グループ_1", "距離グループ_2", "距離グループ_3", "距離グループ_4", "距離グループ_5",
             "連闘_1", "連闘_2", "連闘_3", "連闘_4", "連闘_5",
             "頭数グループ_1", "頭数グループ_2", "頭数グループ_3", "頭数グループ_4", "頭数グループ_5",
             #"３角先頭_1", "３角先頭_2",
             "３角先頭_3", "３角先頭_4", "３角先頭_5",
             # "４角先頭_1", "４角先頭_2",
             "４角先頭_3", "４角先頭_4", "４角先頭_5",
             # "凡走_1", "凡走_2", "凡走_3",
            "凡走_4", "凡走_5",
             # "好走_1", "好走_2", "好走_3",
             "好走_4", "好走_5",
             # "激走_1", "激走_2", "激走_3",
             "激走_4", "激走_5",
             "テン乗り_1", "テン乗り_2", "テン乗り_3", "テン乗り_4", "テン乗り_5",
             "騎手名_1", "騎手名_2", "騎手名_3", "騎手名_4", "騎手名_5",
             "勝ち_1", "勝ち_2", "勝ち_3", "勝ち_4", "勝ち_5",
             # "大差負け_1", "大差負け_2",
             "大差負け_3", "大差負け_4", "大差負け_5",
             "同距離グループ_1", "同距離グループ_2", "同距離グループ_3", "同距離グループ_4", "同距離グループ_5",
             "年月日", "近走競走コード1", "近走馬番1", "近走競走コード2", "近走馬番2", "近走競走コード3", "近走馬番3", "近走競走コード4", "近走馬番4",
             "近走競走コード5", "近走馬番5",
             "負担重量_1", "負担重量_2", "負担重量_3", "負担重量_4", "負担重量_5",
             # "上がりタイム順位_1", "上がりタイム順位_2",
            "上がりタイム順位_3", "上がりタイム順位_4", "上がりタイム順位_5",
             "血統登録番号", "テン乗り", "性別コード",
             "ペース_1", "ペース_2", "ペース_3", "ペース_4", "ペース_5",
             ],
        axis=1, inplace=True)

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
        temp_df = temp_df.replace(np.inf,np.nan).fillna(temp_df.replace(np.inf,np.nan).mean()).reset_index()
        exp_df = temp_df.drop(self.index_list, axis=1).to_numpy()
        print(exp_df)
        print(self.model_folder)
        if os.path.exists(self.model_folder + this_model_name + '.pickle'):
            with open(self.model_folder + this_model_name + '.pickle', 'rb') as f:
                model = pickle.load(f)
            y_pred = model.predict(exp_df)
            print(y_pred)
            pred_df = pd.DataFrame(y_pred, columns=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
            base_df = pd.DataFrame({"RACE_KEY": temp_df["RACE_KEY"], "NENGAPPI": temp_df["NENGAPPI"]})
            pred_df = pd.concat([base_df, pred_df], axis=1)
            pred_df = pred_df.set_index(["RACE_KEY", "NENGAPPI"])
            pred_df = pred_df.stack().reset_index().rename(columns={"level_2":"UMABAN", 0: "prob"})
            pred_df = pred_df[pred_df["UMABAN"] != 0]
            pred_df.loc[:, "pred"] = 0
            return pred_df
        else:
            return pd.DataFrame()

    def _set_predict_target_encoding(self, df):
        return df

class SkModel(LBSkModel):
    class_list = ['主催者コード']
#    class_list = ['競走種別コード', '場コード', 'コース']
    table_name = TABLE_NAME
    obj_column_list = ['１着', '２着', '３着']

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
        print(all_df)
        #grouped_all_df = all_df.groupby(["RACE_KEY", "UMABAN", "target"], as_index=False).mean()
        #date_df = all_df[["RACE_KEY", "target_date"]].drop_duplicates()
        #temp_grouped_df = pd.merge(grouped_all_df, date_df, on="RACE_KEY")
        grouped_df = self.calc_grouped_data(all_df)
        import_df = grouped_df[["RACE_KEY", "UMABAN", "pred", "prob", "predict_std", "predict_rank", "target", "target_date"]].round(3)
        print(import_df)
        return import_df

    def create_featrue_select_data(self, learning_df):
        pass

    def eval_pred_data(self, df):
        """ 予測されたデータの精度をチェック """
        check_df = self.proc.create_eval_prd_data(df)
        for target in self.obj_column_list:
            print(target)
            target_df = check_df[check_df["target"] == target]
            target_df.loc[:, "的中"] = target_df.apply(lambda x: 1 if x["UMABAN"] == x[target] else 0, axis=1)
            target_df.loc[:, "惜しい"] = target_df.apply(lambda x: 1 if (x["UMABAN"] == x["１着"] or x["UMABAN"] == x["２着"] or x["UMABAN"] == x["３着"]) else 0, axis=1)
            hit_rate = target_df["的中"].mean()
            avg_rate = target_df["惜しい"].mean() - hit_rate
            print(f'的中率:{round(hit_rate*100, 1)} %  惜しい率:{round(avg_rate*100, 1)} %  馬券内率:{round((hit_rate + avg_rate)*100, 1)} %')

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