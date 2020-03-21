from modules.lb_extract import LBExtract
from modules.lb_transform import LBTransform
from modules.lb_load import LBLoad
from modules.lb_sk_model import LBSkModel
from modules.lb_sk_proc import LBSkProc
import modules.util as mu

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

MODEL_VERSION = 'lb_v2'
MODEL_NAME = 'raceuma_ens'
TABLE_NAME = '地方競馬レース馬V2'

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
        """ 特徴となる値を作成する。月日→月、距離→根幹、距離グループを作成して列として付与する。

        :param dataframe race_df:
        :return: dataframe
        """
        temp_race_df = race_df.copy()
        temp_race_df.loc[:, 'ナイター'] = race_df['発走時刻'].apply(lambda x: '1' if x.hour >= 17 else '0')
        temp_race_df.loc[:, '季節'] = (race_df['月日'].apply(lambda x: x.month) - 1) // 3
        temp_race_df['季節'].astype('str')
        temp_race_df.loc[:, "非根幹"] = race_df["距離"].apply(lambda x: 0 if x % 400 == 0 else 1)
        temp_race_df.loc[:, "距離グループ"] = race_df["距離"] // 400
        temp_race_df.loc[:, "頭数グループ"] = race_df["頭数"] // 5
        temp_race_df.loc[:, "コース"] = race_df["場コード"].astype(str) + race_df["トラックコード"].astype(str)
        return temp_race_df

    def choose_race_result_column(self, race_df):
        """ レースデータから必要な列に絞り込む。列はデータ区分、主催者コード、競走コード、月日、距離、場コード、頭数、予想勝ち指数、予想決着指数, 競走種別コード

        :param dataframe race_df:
        :return: dataframe
        """
        temp_race_df = race_df[
            ['主催者コード', '発走時刻', '月日', '競走コード', '距離', '場コード', '頭数', 'ペース', 'トラックコード', '後３ハロン']]
        return temp_race_df

    def create_feature_race_result_df(self, race_df, race_winner_df):
        """  race_ddfのデータから特徴量を作成して列を追加する。月日→月、距離→非根幹、距離グループを作成

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
        temp_raceuma_df.loc[:, '騎手名'] = mu.label_encoding(raceuma_df['騎手名'], '騎手名', dict_folder).astype(str)
        temp_raceuma_df = self.choose_upper_n_count(temp_raceuma_df, "調教師名", 100, dict_folder)
        temp_raceuma_df.loc[:, '調教師名'] = mu.label_encoding(raceuma_df['調教師名'], '調教師名', dict_folder).astype(str)
        temp_raceuma_df.loc[:, '所属'] = mu.label_encoding(raceuma_df['所属'], '所属', dict_folder).astype(str)
        temp_raceuma_df.loc[:, '転厩'] = mu.label_encoding(raceuma_df['転厩'], '転厩', dict_folder).astype(str)
        temp_raceuma_df.loc[:, '予想展開'] = temp_raceuma_df["予想展開"].astype(str)
        return temp_raceuma_df.copy()

    def create_feature_raceuma_df(self, raceuma_df):
        """  raceuma_dfの特徴量を作成する。馬番→馬番グループを作成して列を追加する。

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
        temp_raceuma_df.loc[:, '展開脚質'] = raceuma_df['展開コード'].astype(str).str[:1]
        temp_raceuma_df.loc[:, '展開脚色'] = raceuma_df['展開コード'].astype(str).str[-1:]
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
        """  raceuma_dfの特徴量を作成する。馬番→馬番グループを作成して列を追加する。

        :param dataframe raceuma_df:
        :return: dataframe
        """
        temp_raceuma_df = raceuma_df.copy()
        temp_merge_df = pd.merge(race_df, raceuma_df, on="競走コード")
        print(temp_merge_df.shape)
        temp_raceuma_df.loc[:, "同場騎手"] = (temp_merge_df["騎手所属場コード"] == temp_merge_df["場コード"]).astype(int)
        temp_raceuma_df.loc[:, "同所属場"] = (temp_merge_df["調教師所属場コード"] == temp_merge_df["場コード"]).astype(int)
        temp_raceuma_df.loc[:, "同所属騎手"] = (temp_merge_df["騎手所属場コード"] == temp_merge_df["調教師所属場コード"]).astype(int)
        temp_raceuma_df.loc[:, "追込率"] = (temp_merge_df["コーナー順位4"] - temp_merge_df["確定着順"]) / temp_merge_df["頭数"]
        temp_raceuma_df.loc[:, "平均タイム"] = temp_merge_df["タイム"] / temp_merge_df["距離"] * 200
        temp_raceuma_df.loc[:, "勝ち"] = temp_raceuma_df["確定着順"].apply(lambda x: 1 if x == 1 else 0)
        temp_raceuma_df.loc[:, "１番人気"] = temp_raceuma_df["単勝人気"].apply(lambda x: 1 if x == 1 else 0)
        temp_raceuma_df.loc[:, "３角先頭"] = temp_raceuma_df["コーナー順位3"].apply(lambda x: 1 if x == 1 else 0)
        temp_raceuma_df.loc[:, "４角先頭"] = temp_raceuma_df["コーナー順位4"].apply(lambda x: 1 if x == 1 else 0)
        temp_raceuma_df.loc[:, "上がり最速"] = temp_raceuma_df["上がりタイム順位"].apply(lambda x: 1 if x == 1 else 0)
        temp_raceuma_df.loc[:, "休み明け"] = temp_raceuma_df["休養週数"].apply(lambda x: 1 if x >= 10 else 0)
        temp_raceuma_df.loc[:, "連闘"] = temp_raceuma_df["休養週数"].apply(lambda x: 1 if x == 1 else 0)
        temp_raceuma_df.loc[:, "大差負け"] = temp_raceuma_df["着差"].apply(lambda x: 1 if x >= 20 else 0)
        temp_raceuma_df.loc[:, "凡走"] = temp_merge_df.apply(lambda x: 1 if x["確定着順"] - x["単勝人気"] > 5 else 0, axis=1)
        temp_raceuma_df.loc[:, "好走"] = temp_merge_df["確定着順"].apply(lambda x: 1 if x <= 3 else 0)
        temp_raceuma_df.loc[:, "激走"] = temp_merge_df.apply(lambda x: 1 if x["単勝人気"] - x["確定着順"] > 5 else 0, axis=1)
        temp_raceuma_df.loc[:, "逃げそびれ"] = temp_merge_df.apply(lambda x: 1 if x["予想展開"] == 1 and - x["コーナー順位4"] > 3 else 0, axis=1)

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
        """ RaceUmaの因子分析を行うためのデータを取得 """
        print("factory_analyze_raceuma_result_df")
        temp_df = pd.merge(input_raceuma_df, race_df, on="競走コード")
        X = temp_df[
            ['競走コード', '馬番', '枠番',  'タイム指数', '単勝オッズ', '先行率', 'ペース偏差値', '距離増減', '斤量比', '追込率', '平均タイム',
             "距離", "頭数", "非根幹", "上り係数", "逃げ勝ち", "内勝ち", "外勝ち", "短縮勝ち", "延長勝ち", "人気勝ち", "１番人気", "３角先頭",
             "４角先頭", "上がり最速", "上がりタイム", "連闘", "休み明け", "大差負け", "展開脚質", "展開脚色"]]

        mmsc_columns = ["頭数", "展開脚質", "展開脚色", "上がりタイム"]
        mmsc_dict_name = "sc_fa_race_mmsc"
        stdsc_columns = ["距離"]
        stdsc_dict_name = "sc_fa_race_stdsc"
        X = mu.scale_df_for_fa(X, mmsc_columns, mmsc_dict_name, stdsc_columns, stdsc_dict_name, dict_folder)

        X_fact = X.drop(["競走コード", "馬番"], axis=1).astype({'非根幹': int, '逃げ勝ち': int, '内勝ち': int, '外勝ち': int, '短縮勝ち': int, '延長勝ち': int,
                                                         '人気勝ち': int, '１番人気': int, '３角先頭': int, '４角先頭': int, '上がり最速': int, '休み明け': int,
                                                         '連闘': int, '大差負け': int})

        X_fact = X_fact.replace(np.inf, np.nan).fillna(X_fact.median()).fillna(0)
        X_fact.iloc[0] = X_fact.iloc[0] + 0.000001

        dict_name = "fa_raceuma_result_df"
        filename = dict_folder + dict_name + '.pkl'
        if os.path.exists(filename):
            fa = mu.load_dict(dict_name, dict_folder)
        else:
            fa = FactorAnalyzer(n_factors=5, rotation='promax', impute='drop')
            fa.fit(X_fact)
            mu.save_dict(fa, dict_name, dict_folder)

        fa_np = fa.transform(X_fact)
        fa_df = pd.DataFrame(fa_np, columns=["fa_1", "fa_2", "fa_3", "fa_4", "fa_5"])
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

    def _proc_race_df(self, race_base_df):
        race_df = self.tf.create_feature_race_df(race_base_df)
        return race_df.drop(["トラック種別コード", "競走番号", "場名"
                                        , "初出走頭数", "混合", "予想決着指数", "登録頭数", "回次", "日次"], axis=1)


    def _proc_horse_df(self, horse_base_df):
        horse_df = self.tf.choose_horse_column(horse_base_df)
        horse_df = self.tf.encode_horse_df(horse_df, self.dict_folder)
        return horse_df.copy()


    def set_prev_df(self):
        """  prev_dfを作成するための処理。prev1_raceuma_df,prev2_raceuma_dfに処理がされたデータをセットする。過去２走のデータをセットする  """
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
        # self._set_grouped_raceuma_prev_df(tf_prev)


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
             "後３ハロン", "予想展開", "騎手所属場コード", "調教師所属場コード", 'タイム',
             "上がりタイム", "休み明け", "展開脚質", "展開脚色", "年月日", "月日", "距離", "血統登録番号"], axis=1)
        return merged_df

    def scale_df(self):
        print("scale_df")
        mmsc_columns = ["距離", "頭数", "枠番", "休養後出走回数", "予想人気", "先行指数順位", "馬齢", "距離増減", "前走頭数"]
        mmsc_dict_name = "sc_base_mmsc"
        stdsc_columns = ["予想勝ち指数", "休養週数", "キャリア", "斤量比", "負担重量", "デフォルト得点", "得点V1", "得点V2", "得点V3"
            , "fa_1_1", "fa_2_1", "fa_3_1", "fa_4_1", "fa_5_1", "fa_1_2", "fa_2_2", "fa_3_2", "fa_4_2", "fa_5_2"
            , "fa_1_3", "fa_2_3", "fa_3_3", "fa_4_3", "fa_5_3", "fa_1_4", "fa_2_4", "fa_3_4", "fa_4_4", "fa_5_4"
            , "fa_1_5", "fa_2_5", "fa_3_5", "fa_4_5", "fa_5_5"]
        stdsc_dict_name = "sc_base_stdsc"

        self.base_df = mu.scale_df_for_fa(self.base_df, mmsc_columns, mmsc_dict_name, stdsc_columns, stdsc_dict_name, self.dict_folder)
        self.base_df.loc[:, "競走種別コード_h"] = self.base_df["競走種別コード"]
        self.base_df.loc[:, "場コード_h"] = self.base_df["場コード"]
        hash_track_columns = ["主催者コード", "競走種別コード_h", "場コード_h", "競走条件コード", "トラックコード"]
        hash_track_dict_name = "sc_base_hash_track"
        self.base_df = mu.hash_eoncoding(self.base_df, hash_track_columns, 10, hash_track_dict_name, self.dict_folder)

class SkProc(LBSkProc):
    """
    地方競馬の機械学習処理プロセスを取りまとめたクラス。
    """
    def _get_load_object(self, version_str, start_date, end_date, mock_flag, test_flag):
        ld = Ld(version_str, start_date, end_date, mock_flag, test_flag)
        return ld


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
        # self._left_join_base_df(self.grouped_raceuma_prev_df)
        #self.create_feature()
        #self.drop_base_columns()


    def create_feature(self):
        """ 最終的にマージされたDatabaseから特徴量を生成する """
        print(self.base_df.iloc[0])
        self.base_df = self.create_base_df_feature(self.base_df)
        self.base_df = self.drop_base_df(self.base_df)


    def _create_feature(self,):
        """ 過去走と今回を比較した特徴量等、最終的な特徴良を生成する """
        #        self.base_df.loc[:, "継続騎乗"] = self.base_df.apply(lambda x: 1 if x["騎手名"] == x["騎手名_1"] else 0 )
        # print(self.base_df.iloc[0])
        self.base_df.loc[:, "継続騎乗"] = (self.base_df["騎手名"] == self.base_df["騎手名_1"]).astype(int)
        self.base_df.loc[:, "同騎手_1"] = (self.base_df["騎手名"] == self.base_df["騎手名_1"]).astype(int)
        self.base_df.loc[:, "同騎手_2"] = (self.base_df["騎手名"] == self.base_df["騎手名_2"]).astype(int)
        self.base_df.loc[:, "同騎手_3"] = (self.base_df["騎手名"] == self.base_df["騎手名_3"]).astype(int)
        self.base_df.loc[:, "同騎手_4"] = (self.base_df["騎手名"] == self.base_df["騎手名_4"]).astype(int)
        self.base_df.loc[:, "同騎手_5"] = (self.base_df["騎手名"] == self.base_df["騎手名_5"]).astype(int)
        self.base_df.loc[:, "同場騎手"] = (self.base_df["騎手所属場コード"] == self.base_df["場コード"]).astype(int)
        self.base_df.loc[:, "同所属場"] = (self.base_df["調教師所属場コード"] == self.base_df["場コード"]).astype(int)
        self.base_df.loc[:, "同所属騎手"] = (self.base_df["騎手所属場コード"] == self.base_df["調教師所属場コード"]).astype(int)
        self.base_df.loc[:, "同主催者"] = (self.base_df["主催者コード"] == self.base_df["主催者コード_1"]).astype(int)
        self.base_df.loc[:, "同場コード"] = (self.base_df["場コード"] == self.base_df["場コード_1"]).astype(int)
        self.base_df.loc[:, "同場_1"] = (self.base_df["場コード"] == self.base_df["場コード_1"]).astype(int)
        self.base_df.loc[:, "同場_2"] = (self.base_df["場コード"] == self.base_df["場コード_2"]).astype(int)
        self.base_df.loc[:, "同場_3"] = (self.base_df["場コード"] == self.base_df["場コード_3"]).astype(int)
        self.base_df.loc[:, "同場_4"] = (self.base_df["場コード"] == self.base_df["場コード_4"]).astype(int)
        self.base_df.loc[:, "同場_5"] = (self.base_df["場コード"] == self.base_df["場コード_5"]).astype(int)
        self.base_df.loc[:, "同距離グループ_1"] = (self.base_df["距離グループ"] == self.base_df["距離グループ_1"]).astype(int)
        self.base_df.loc[:, "同距離グループ_2"] = (self.base_df["距離グループ"] == self.base_df["距離グループ_2"]).astype(int)
        self.base_df.loc[:, "同距離グループ_3"] = (self.base_df["距離グループ"] == self.base_df["距離グループ_3"]).astype(int)
        self.base_df.loc[:, "同距離グループ_4"] = (self.base_df["距離グループ"] == self.base_df["距離グループ_4"]).astype(int)
        self.base_df.loc[:, "同距離グループ_5"] = (self.base_df["距離グループ"] == self.base_df["距離グループ_5"]).astype(int)
        self.base_df.loc[:, "同季節_1"] = (self.base_df["季節"] == self.base_df["季節_1"]).astype(int)
        self.base_df.loc[:, "同季節_2"] = (self.base_df["季節"] == self.base_df["季節_2"]).astype(int)
        self.base_df.loc[:, "同季節_3"] = (self.base_df["季節"] == self.base_df["季節_3"]).astype(int)
        self.base_df.loc[:, "同季節_4"] = (self.base_df["季節"] == self.base_df["季節_4"]).astype(int)
        self.base_df.loc[:, "同季節_5"] = (self.base_df["季節"] == self.base_df["季節_5"]).astype(int)
        self.base_df.loc[:, "負担重量_1"] = self.base_df["負担重量_1"] - self.base_df["負担重量"]
        self.base_df.loc[:, "負担重量_2"] = self.base_df["負担重量_2"] - self.base_df["負担重量"]
        self.base_df.loc[:, "負担重量_3"] = self.base_df["負担重量_3"] - self.base_df["負担重量"]
        self.base_df.loc[:, "負担重量_4"] = self.base_df["負担重量_4"] - self.base_df["負担重量"]
        self.base_df.loc[:, "負担重量_5"] = self.base_df["負担重量_5"] - self.base_df["負担重量"]
        self.base_df.loc[:, "頭数差"] = self.base_df["頭数グループ"] - self.base_df["頭数グループ_1"]
        self.base_df.loc[:, "中央経験"] = self.base_df.apply(lambda x: 1 if (x["主催者コード_1"] == 1 or x["主催者コード_2"] == 1 or x["主催者コード_3"] == 1 or x["主催者コード_4"] == 1 or x["主催者コード_5"] == 1) else 0, axis=1)
        self.base_df.loc[:, "休み明け"] = self.base_df["休養週数"].apply(lambda x: True if x >= 20 else False)
        self.base_df.loc[:, "覚醒"] = self.base_df.apply(lambda x: 1 if (x["激走_1"] + x["激走_2"] + x["激走_3"] + x["激走_4"] + x["激走_5"] >= 3) else 0, axis=1)
        self.base_df.loc[:, "失速"] = self.base_df.apply(lambda x: 1 if (x["凡走_1"] + x["凡走_2"] + x["凡走_3"] + x["凡走_4"] + x["凡走_5"] >= 3) else 0, axis=1)
        self.base_df.loc[:, "逃げそびれ凡走"] = self.base_df.apply(lambda x: 1 if (x["逃げそびれ_1"] * x["凡走_1"] + x["逃げそびれ_2"] * x["凡走_2"] + x["逃げそびれ_3"] * x["凡走_3"] + x["逃げそびれ_4"] * x["凡走_4"] + x["逃げそびれ_5"] * x["凡走_5"] >= 2) else 0, axis=1)
        self.base_df.loc[:, "継続騎乗好走"] = self.base_df.apply(lambda x: 1 if x["継続騎乗"] * x["激走_1"] == 1 else 0, axis=1)
        self.base_df.loc[:, "末脚安定"] = self.base_df.apply(lambda x: 1 if (x["上がりタイム順位_1"] + x["上がりタイム順位_2"] + x["上がりタイム順位_3"] <= 8 ) else 0, axis=1)
        self.base_df.loc[:, "同騎手○"] = self.base_df.apply(lambda x: 1 if (x["同騎手_1"] * x["好走_1"] + x["同騎手_2"] * x["好走_2"] + x["同騎手_3"] * x["好走_3"] + x["同騎手_4"] * x["好走_4"] + x["同騎手_5"] * x["好走_5"]) >= 3 else 0, axis=1)
        self.base_df.loc[:, "同騎手◎"] = self.base_df.apply(lambda x: 1 if (x["同騎手_1"] * x["激走_1"] + x["同騎手_2"] * x["激走_2"] + x["同騎手_3"] * x["激走_3"] + x["同騎手_4"] * x["激走_4"] + x["同騎手_5"] * x["激走_5"]) >= 3 else 0, axis=1)
        self.base_df.loc[:, "同騎手逃げ"] = self.base_df.apply(lambda x: 1 if (x["同騎手_1"] * x["３角先頭_1"] + x["同騎手_2"] * x["３角先頭_2"] + x["同騎手_3"] * x["３角先頭_3"] + x["同騎手_4"] * x["３角先頭_4"] + x["同騎手_5"] * x["３角先頭_5"]) >= 2 else 0, axis=1)
        self.base_df.loc[:, "同場○"] = self.base_df.apply(lambda x: 1 if (x["同場_1"] * x["好走_1"] + x["同場_2"] * x["好走_2"] + x["同場_3"] * x["好走_3"] + x["同場_4"] * x["好走_4"] + x["同場_5"] * x["好走_5"]) >= 3 else 0, axis=1)
        self.base_df.loc[:, "同場◎"] = self.base_df.apply(lambda x: 1 if (x["同場_1"] * x["激走_1"] + x["同場_2"] * x["激走_2"] + x["同場_3"] * x["激走_3"] + x["同場_4"] * x["激走_4"] + x["同場_5"] * x["激走_5"]) >= 3 else 0, axis=1)
        self.base_df.loc[:, "上がり最速数"] = self.base_df.apply(lambda x: 1 if (x["上がり最速_1"] + x["上がり最速_2"] + x["上がり最速_3"] + x["上がり最速_4"] + x["上がり最速_5"]) >= 3 else 0, axis=1)
        self.base_df.loc[:, "逃げ好走"] = self.base_df.apply(lambda x: 1 if (x["４角先頭_1"] * x["好走_1"] + x["４角先頭_2"] * x["好走_2"] + x["４角先頭_3"] * x["好走_3"] + x["４角先頭_4"] * x["好走_4"] + x["４角先頭_5"] * x["好走_5"] >= 2) else 0, axis=1)
        self.base_df.loc[:, "ムラっけ"] = self.base_df.apply(lambda x: 1 if (x["激走_1"] + x["激走_2"] + x["激走_3"] + x["激走_4"] + x["激走_5"] >= 2) and (x["大差負け_1"] + x["大差負け_2"] + x["大差負け_3"] + x["大差負け_4"] + x["大差負け_5"] >= 2)  else 0, axis=1)
        self.base_df.loc[:, "連闘○"] = self.base_df.apply(lambda x: 1 if (x["連闘_1"] * x["好走_1"] + x["連闘_2"] * x["好走_2"] + x["連闘_3"] * x["好走_3"] + x["連闘_4"] * x["好走_4"] + x["連闘_5"] * x["好走_5"]) >= 3 else 0, axis=1)
        self.base_df.loc[:, "連闘◎"] = self.base_df.apply(lambda x: 1 if (x["連闘_1"] * x["激走_1"] + x["連闘_2"] * x["激走_2"] + x["連闘_3"] * x["激走_3"] + x["連闘_4"] * x["激走_4"] + x["連闘_5"] * x["激走_5"]) >= 3 else 0, axis=1)
        #self.base_df.loc[:, "ナイター○"] = self.base_df.apply(lambda x: 1 if (x["ナイター_1"] * x["好走_1"] + x["ナイター_2"] * x["好走_2"] + x["ナイター_3"] * x["好走_3"] + x["ナイター_4"] * x["好走_4"] + x["ナイター_5"] * x["好走_5"]) >= 3 else 0, axis=1)
        #self.base_df.loc[:, "ナイター◎"] = self.base_df.apply(lambda x: 1 if (x["ナイター_1"] * x["激走_1"] + x["ナイター_2"] * x["激走_2"] + x["ナイター_3"] * x["激走_3"] + x["ナイター_4"] * x["激走_4"] + x["ナイター_5"] * x["激走_5"]) >= 3 else 0, axis=1)
        self.base_df.loc[:, "同距離○"] = self.base_df.apply(lambda x: 1 if (x["同距離グループ_1"] * x["好走_1"] + x["同距離グループ_2"] * x["好走_2"] + x["同距離グループ_3"] * x["好走_3"] + x["同距離グループ_4"] * x["好走_4"] + x["同距離グループ_5"] * x["好走_5"]) >= 3 else 0, axis=1)
        self.base_df.loc[:, "同距離◎"] = self.base_df.apply(lambda x: 1 if (x["同距離グループ_1"] * x["激走_1"] + x["同距離グループ_2"] * x["激走_2"] + x["同距離グループ_3"] * x["激走_3"] + x["同距離グループ_4"] * x["激走_4"] + x["同距離グループ_5"] * x["激走_5"]) >= 3 else 0, axis=1)
        self.base_df.loc[:, "同季節○"] = self.base_df.apply(lambda x: 1 if (x["同季節_1"] * x["好走_1"] + x["同季節_2"] * x["好走_2"] + x["同季節_3"] * x["好走_3"] + x["同季節_4"] * x["好走_4"] + x["同季節_5"] * x["好走_5"]) >= 3 else 0, axis=1)
        self.base_df.loc[:, "同季節◎"] = self.base_df.apply(lambda x: 1 if (x["同季節_1"] * x["激走_1"] + x["同季節_2"] * x["激走_2"] + x["同季節_3"] * x["激走_3"] + x["同季節_4"] * x["激走_4"] + x["同季節_5"] * x["激走_5"]) >= 3 else 0, axis=1)
        self.base_df.loc[:, "同場騎手○"] = self.base_df.apply(lambda x: 1 if (x["同場騎手_1"] * x["好走_1"] + x["同場騎手_2"] * x["好走_2"] + x["同場騎手_3"] * x["好走_3"] + x["同場騎手_4"] * x["好走_4"] + x["同場騎手_5"] * x["好走_5"]) >= 3 else 0, axis=1)
        self.base_df.loc[:, "同場騎手◎"] = self.base_df.apply(lambda x: 1 if (x["同場騎手_1"] * x["激走_1"] + x["同場騎手_2"] * x["激走_2"] + x["同場騎手_3"] * x["激走_3"] + x["同場騎手_4"] * x["激走_4"] + x["同場騎手_5"] * x["激走_5"]) >= 3 else 0, axis=1)
        self.base_df.loc[:, "同所属場○"] = self.base_df.apply(lambda x: 1 if (x["同所属場_1"] * x["好走_1"] + x["同所属場_2"] * x["好走_2"] + x["同所属場_3"] * x["好走_3"] + x["同所属場_4"] * x["好走_4"] + x["同所属場_5"] * x["好走_5"]) >= 3 else 0, axis=1)
        self.base_df.loc[:, "同所属場◎"] = self.base_df.apply(lambda x: 1 if (x["同所属場_1"] * x["激走_1"] + x["同所属場_2"] * x["激走_2"] + x["同所属場_3"] * x["激走_3"] + x["同所属場_4"] * x["激走_4"] + x["同所属場_5"] * x["激走_5"]) >= 3 else 0, axis=1)
        self.base_df.loc[:, "同所属騎手○"] = self.base_df.apply(lambda x: 1 if (x["同所属騎手_1"] * x["好走_1"] + x["同所属騎手_2"] * x["好走_2"] + x["同所属騎手_3"] * x["好走_3"] + x["同所属騎手_4"] * x["好走_4"] + x["同所属騎手_5"] * x["好走_5"]) >= 3 else 0, axis=1)
        self.base_df.loc[:, "同所属騎手◎"] = self.base_df.apply(lambda x: 1 if (x["同所属騎手_1"] * x["激走_1"] + x["同所属騎手_2"] * x["激走_2"] + x["同所属騎手_3"] * x["激走_3"] + x["同所属騎手_4"] * x["激走_4"] + x["同所属騎手_5"] * x["激走_5"]) >= 3 else 0, axis=1)



    def _drop_columns_base_df(self):
        self.base_df.drop(
            ["主催者コード_1", "主催者コード_2", "主催者コード_3", "主催者コード_4", "主催者コード_5",
             "発走時刻_1", "発走時刻_2", "発走時刻_3", "発走時刻_4", "発走時刻_5",
             "場コード_1", "場コード_2", "場コード_3", "場コード_4", "場コード_5",
             "トラックコード_1", "トラックコード_2", "トラックコード_3", "トラックコード_4", "トラックコード_5",
             "ナイター_1", "ナイター_2", "ナイター_3", "ナイター_4", "ナイター_5",
             "距離グループ_1", "距離グループ_2", "距離グループ_3", "距離グループ_4", "距離グループ_5",
             "同季節_1", "同季節_2", "同季節_3", "同季節_4", "同季節_5",
             "連闘_1", "連闘_2", "連闘_3", "連闘_4", "連闘_5",
             "頭数グループ_1", "頭数グループ_2", "頭数グループ_3", "頭数グループ_4", "頭数グループ_5",
             "同騎手_1", "同騎手_2", "同騎手_3", "同騎手_4", "同騎手_5",
             "３角先頭_1", "３角先頭_2", "３角先頭_3", "３角先頭_4", "３角先頭_5",
             "４角先頭_1", "４角先頭_2", "４角先頭_3", "４角先頭_4", "４角先頭_5",
             "上がり最速_1", "上がり最速_2", "上がり最速_3", "上がり最速_4", "上がり最速_5",
             "凡走_1", "凡走_2", "凡走_3", "凡走_4", "凡走_5",
             "好走_1", "好走_2", "好走_3", "好走_4", "好走_5",
             "激走_1", "激走_2", "激走_3", "激走_4", "激走_5",
             "逃げそびれ_1", "逃げそびれ_2", "逃げそびれ_3", "逃げそびれ_4", "逃げそびれ_5",
             "季節_1", "季節_2", "季節_3", "季節_4", "季節_5",
             "テン乗り_1", "テン乗り_2", "テン乗り_3", "テン乗り_4", "テン乗り_5",
             "騎手名_1", "騎手名_2", "騎手名_3", "騎手名_4", "騎手名_5",
             "勝ち_1", "勝ち_2", "勝ち_3", "勝ち_4", "勝ち_5",
             "大差負け_1", "大差負け_2", "大差負け_3", "大差負け_4", "大差負け_5",
             "同場騎手_1", "同場騎手_2", "同場騎手_3", "同場騎手_4", "同場騎手_5",
             "同所属場_1", "同所属場_2", "同所属場_3", "同所属場_4", "同所属場_5",
             "同所属騎手_1", "同所属騎手_2", "同所属騎手_3", "同所属騎手_4", "同所属騎手_5",
             "同距離グループ_1", "同距離グループ_2", "同距離グループ_3", "同距離グループ_4", "同距離グループ_5",
             "同場_1", "同場_2", "同場_3", "同場_4", "同場_5",
             "発走時刻", "年月日", "近走競走コード1", "近走馬番1", "近走競走コード2", "近走馬番2", "近走競走コード3", "近走馬番3", "近走競走コード4", "近走馬番4",
             "近走競走コード5", "近走馬番5"
             ],
        axis=1, inplace=True)

    def drop_base_columns(self):
        self.base_df.drop(["発走時刻", "年月日", "近走競走コード1", "近走馬番1", "近走競走コード2", "近走馬番2", "近走競走コード3", "近走馬番3", "近走競走コード4", "近走馬番4", "近走競走コード5", "近走馬番5"], axis=1, inplace=True)


    def _scale_df(self):
        print("scale_df")
        mmsc_columns = ["距離", "頭数", "枠番", "休養後出走回数", "予想人気", "先行指数順位", "馬齢", "距離増減", "前走頭数"]
        mmsc_dict_name = "sc_base_mmsc"
        stdsc_columns = ["予想勝ち指数", "休養週数", "キャリア", "斤量比", "負担重量", "デフォルト得点", "得点V1", "得点V2", "得点V3"
            , "fa_1_1", "fa_2_1", "fa_3_1", "fa_4_1", "fa_5_1", "fa_1_2", "fa_2_2", "fa_3_2", "fa_4_2", "fa_5_2"
            , "fa_1_3", "fa_2_3", "fa_3_3", "fa_4_3", "fa_5_3", "fa_1_4", "fa_2_4", "fa_3_4", "fa_4_4", "fa_5_4"
            , "fa_1_5", "fa_2_5", "fa_3_5", "fa_4_5", "fa_5_5"]
        stdsc_dict_name = "sc_base_stdsc"

        self.base_df = mu.scale_df_for_fa(self.base_df, mmsc_columns, mmsc_dict_name, stdsc_columns, stdsc_dict_name, self.dict_folder)
        self.base_df.loc[:, "競走種別コード_h"] = self.base_df["競走種別コード"]
        self.base_df.loc[:, "場コード_h"] = self.base_df["場コード"]
        hash_track_columns = ["主催者コード", "競走種別コード_h", "場コード_h", "競走条件コード", "トラックコード"]
        hash_track_dict_name = "sc_base_hash_track"
        self.base_df = mu.hash_eoncoding(self.base_df, hash_track_columns, 10, hash_track_dict_name, self.dict_folder)

class SkModel(LBSkModel):
    class_list = ['競走種別コード', '場コード', 'コース']

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
    if test_flag:
        dict_path = 'C:\HRsystem\HRsystem/for_test_'
    else:
        dict_path = 'C:\HRsystem\HRsystem/'
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
            sk_model.set_table_name(table_name)

        luigi.build([End_baoz_predict(start_date=start_date, end_date=end_date, skmodel=sk_model,
                                      intermediate_folder=INTERMEDIATE_FOLDER)], local_scheduler=True)

    else:
        print("error")