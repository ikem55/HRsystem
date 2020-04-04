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

basedir = os.path.dirname(__file__)[:-8]
print(basedir)
sys.path.append(basedir)

# 呼び出し方
# python lb_v3.py learning True True
# ====================================== パラメータ　要変更 =====================================================
## モデルV3、騎手・厩舎・馬主・血統を重視。過去走は最小限に


MODEL_VERSION = 'lb_v3'
MODEL_NAME = 'raceuma_ens'
TABLE_NAME = '地方競馬レース馬V3'

class Ext(LBExtract):
    pass

class Tf(LBTransform):
    def create_feature_race_df(self, race_df):
        """ 特徴となる値を作成する。ナイター、季節、非根幹、距離グループ、頭数グループ、コースを作成して列として付与する。

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
        temp_race_df.loc[:, "コース"] = race_df["場コード"].astype(str) + race_df["トラックコード"].astype(str)
        return temp_race_df

    def choose_race_result_column(self, race_df):
        """ レースデータから必要な列に絞り込む。列は'主催者コード', '発走時刻', '月日', '競走コード', '距離', '場コード', '頭数', 'ペース', 'トラックコード'

        :param dataframe race_df:
        :return: dataframe
        """
        temp_race_df = race_df[
            ['主催者コード', '発走時刻', '月日', '競走コード', '距離', '場コード', '頭数', 'ペース', 'トラックコード']]
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
        temp_merge_df.loc[:, "逃げ勝ち"] = temp_merge_df["コーナー順位4"].apply(lambda x: 1 if x == 1 else 0)
        temp_merge_df.loc[:, "内勝ち"] = temp_merge_df["枠番"].apply(lambda x: 1 if x in (1, 2, 3) else 0)
        temp_merge_df.loc[:, "外勝ち"] = temp_merge_df["枠番"].apply(lambda x: 1 if x in (6, 7, 8) else 0)
        temp_merge_df.loc[:, "短縮勝ち"] = temp_merge_df["距離増減"].apply(lambda x: 1 if x < 0 else 0)
        temp_merge_df.loc[:, "延長勝ち"] = temp_merge_df["距離増減"].apply(lambda x: 1 if x > 0 else 0)
        temp_merge_df.loc[:, "人気勝ち"] = temp_merge_df["単勝人気"].apply(lambda x: 1 if x == 1 else 0)
        merge_df = pd.merge(temp_race_df,
                            temp_merge_df[["競走コード", "逃げ勝ち", "内勝ち", "外勝ち", "短縮勝ち", "延長勝ち", "人気勝ち"]],
                            on="競走コード")
        return merge_df

    def encode_raceuma_before_df(self, raceuma_df, dict_folder):
        """  列をエンコードする処理。騎手名、所属、転厩をラベルエンコーディングして値を置き換える。

        :param dataframe raceuma_df:
        :return: dataframe
        """
        temp_raceuma_df = raceuma_df.copy()
        temp_raceuma_df = self.choose_upper_n_count(temp_raceuma_df, "騎手名", 150, dict_folder)
        temp_raceuma_df.loc[:, '騎手名'] = mu.label_encoding(raceuma_df['騎手名'], '騎手名', dict_folder).astype(str)
        temp_raceuma_df = self.choose_upper_n_count(temp_raceuma_df, "調教師名", 150, dict_folder)
        temp_raceuma_df.loc[:, '調教師名'] = mu.label_encoding(raceuma_df['調教師名'], '調教師名', dict_folder).astype(str)
        temp_raceuma_df.loc[:, '所属'] = mu.label_encoding(raceuma_df['所属'], '所属', dict_folder).astype(str)
        temp_raceuma_df.loc[:, '転厩'] = mu.label_encoding(raceuma_df['転厩'], '転厩', dict_folder).astype(str)
        temp_raceuma_df.loc[:, '予想展開'] = temp_raceuma_df["予想展開"].astype(str)
        return temp_raceuma_df.copy()

    def create_feature_raceuma_df(self, raceuma_df):
        """  raceuma_dfの特徴量を作成する。馬番グループ、休養週数、休養後出走回数、先行指数順位、キャリア、馬齢、距離増減を数値変換して列を追加する。

        :param dataframe raceuma_df:
        :return: dataframe
        """
        temp_raceuma_df = raceuma_df.copy()
        temp_raceuma_df.loc[:, "馬番グループ"] = raceuma_df["馬番"] // 4
        temp_raceuma_df.loc[:, "休養週数"] = raceuma_df["休養週数"].apply(lambda x: 1 if x == 0 else 1 / x)
        temp_raceuma_df.loc[:, "休養後出走回数"] = raceuma_df["休養後出走回数"].apply(lambda x: 5 if x >= 5 else x)
        temp_raceuma_df.loc[:, "先行指数順位"] = raceuma_df["先行指数順位"].apply(lambda x: 1 if x == 0 else 1 / x)
        temp_raceuma_df.loc[:, "キャリア"] = raceuma_df["キャリア"].apply(lambda x: 10 if x >= 10 else x)
        temp_raceuma_df.loc[:, "馬齢"] = raceuma_df["馬齢"].apply(lambda x: 7 if x >= 7 else x)
        temp_raceuma_df.loc[:, "距離増減"] = raceuma_df["距離増減"] // 200
        return temp_raceuma_df

    def standardize_raceuma_result_df(self, raceuma_df):
        return raceuma_df

    def drop_columns_raceuma_df(self, raceuma_df):
        return raceuma_df.drop(["データ作成年月日", "予想タイム指数", "予想タイム指数順位", "デフォルト得点",
                                "近走競走コード2", "近走馬番2", "近走競走コード3", "近走馬番3", "近走競走コード4", "近走馬番4", "近走競走コード5", "近走馬番5",
                                "予想オッズ", "予想人気", "血統距離評価", "血統トラック評価", "血統成長力評価", "血統総合評価", "血統距離評価B", "血統トラック評価B", "血統成長力評価B", "血統総合評価B",
                                "先行指数", "クラス変動", "騎手コード", "騎手評価", "調教師評価", "枠順評価", "脚質評価", "調教師コード", "前走着順", "前走人気", "前走着差", "前走トラック種別コード",
                                "前走馬体重", "前走頭数","タイム指数上昇係数", "タイム指数回帰推定値", "タイム指数回帰標準偏差", "前走休養週数", "騎手ランキング", "調教師ランキング","得点V1", "得点V2",
                                "得点V3", "得点V1順位", "得点V2順位", "デフォルト得点順位", "得点V3順位"], axis=1)


    def choose_raceuma_result_column(self, raceuma_df):
        """  レース馬データから必要な列に絞り込む。

        :param dataframe raceuma_df:
        :return: dataframe
        """
        temp_raceuma_df = raceuma_df[
            ['競走コード', '馬番', '枠番', '年月日', '血統登録番号',  '単勝人気', '休養週数',  '確定着順', '着差', '展開コード', '騎手名', 'テン乗り', '負担重量', '馬体重', 'コーナー順位3', 'コーナー順位4', '距離増減']].copy()
        return temp_raceuma_df

    def encode_raceuma_result_df(self, raceuma_df, dict_folder):
        """  列をエンコードする処理。騎手名をラベルエンコーディングして値を置き換える。

        :param dataframe raceuma_df:
        :return: dataframe
        """
        temp_raceuma_df = raceuma_df.copy()
        temp_raceuma_df = self.choose_upper_n_count(temp_raceuma_df, "騎手名", 150, dict_folder)
        temp_raceuma_df.loc[:, '騎手名'] = mu.label_encoding(raceuma_df['騎手名'], '騎手名', dict_folder).astype(str)
        return temp_raceuma_df.copy()

    def normalize_raceuma_result_df(self, raceuma_df):
        """ レース馬の成績データのノーマライズ。馬体重を偏差化と順位計算 """
        grouped_df = raceuma_df[['競走コード', '馬体重']].groupby('競走コード').agg(
            ['mean', 'std']).reset_index()
        grouped_df.columns = ['競走コード', '馬体重_mean', '馬体重_std']
        merged_df = pd.merge(raceuma_df, grouped_df, on='競走コード')
        merged_df['馬体重偏差'] = merged_df.apply(lambda x: 50 if x['馬体重_std'] == 0 else (x['馬体重_mean'] - x['馬体重'])/ x['馬体重_std'] * 10 + 50 , axis=1)
        merged_df['馬体重順位'] = raceuma_df.groupby("競走コード")["馬体重"].rank(ascending=False, method='max')
        merged_df['馬体重順位'] = merged_df.apply(lambda x: np.nan if x["馬体重"] == 0 else x["馬体重順位"], axis=1)
        merged_df = merged_df[["競走コード", "馬番", "馬体重偏差", "馬体重順位"]]
        return_df = pd.merge(raceuma_df, merged_df, on=["競走コード", "馬番"]).drop(["馬体重"], axis=1).rename(columns={"馬体重偏差": "馬体重"})
        return return_df


    def create_feature_raceuma_result_df(self, race_df, raceuma_df):
        """  raceuma_dfの特徴量を作成する。馬番→馬番グループを作成して列を追加する。

        :param dataframe raceuma_df:
        :return: dataframe
        """
        temp_raceuma_df = raceuma_df.copy()
        temp_merge_df = pd.merge(race_df, raceuma_df, on="競走コード")
        print(temp_merge_df.shape)
        temp_raceuma_df.loc[:, '展開脚質'] = raceuma_df['展開コード'].astype(str).str[:1]
        temp_raceuma_df.loc[:, '展開脚色'] = raceuma_df['展開コード'].astype(str).str[-1:]
        temp_raceuma_df.loc[:, "追込率"] = (temp_merge_df["コーナー順位4"] - temp_merge_df["確定着順"]) / temp_merge_df["頭数"]
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

    def choose_horse_column(self, horse_df):
        """ 馬データから必要な列に絞り込む。対象は血統登録番号、繁殖登録番号１、繁殖登録番号５、東西所属コード、生産者コード、馬主コード

        :param dataframe raceuma_df:
        :return: dataframe
        """
        temp_horse_df = horse_df[['血統登録番号', '繁殖登録番号1', '繁殖登録番号3', '繁殖登録番号5', '東西所属コード', '生産者コード', '馬主コード']]
        return temp_horse_df

    def encode_horse_df(self, horse_df, dict_folder):
        """  列をエンコードする処理。騎手名、所属、転厩をラベルエンコーディングして値を置き換える。learning_modeがTrueの場合は辞書生成がされる。

        :param dataframe horse_df:
        :return: dataframe
        """
        temp_horse_df = horse_df.copy()
        temp_horse_df = self.choose_upper_n_count(temp_horse_df, "繁殖登録番号1", 70, dict_folder)
        temp_horse_df.loc[:, '繁殖登録番号1'] = mu.label_encoding(horse_df['繁殖登録番号1'], '騎手名', dict_folder).astype(str)
        temp_horse_df = self.choose_upper_n_count(temp_horse_df, "繁殖登録番号3", 70, dict_folder)
        temp_horse_df.loc[:, '繁殖登録番号3'] = mu.label_encoding(horse_df['繁殖登録番号3'], '調教師名', dict_folder).astype(str)
        temp_horse_df = self.choose_upper_n_count(temp_horse_df, "繁殖登録番号5", 70, dict_folder)
        temp_horse_df.loc[:, '繁殖登録番号5'] = mu.label_encoding(horse_df['繁殖登録番号5'], '調教師名', dict_folder).astype(str)
        temp_horse_df = self.choose_upper_n_count(temp_horse_df, "生産者コード", 30, dict_folder)
        temp_horse_df.loc[:, '生産者コード'] = mu.label_encoding(horse_df['生産者コード'], '調教師名', dict_folder).astype(str)
        temp_horse_df = self.choose_upper_n_count(temp_horse_df, "馬主コード", 150, dict_folder)
        temp_horse_df.loc[:, '馬主コード'] = mu.label_encoding(horse_df['馬主コード'], '調教師名', dict_folder).astype(str)
        return temp_horse_df.copy()


    def factory_analyze_raceuma_result_df(self, race_df, input_raceuma_df, dict_folder):
        """ RaceUmaの因子分析を行うためのデータを取得。このモデルでは因子分解を行わずにそのままパスする """
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
        return race_df.drop(["トラック種別コード", "競走番号", "場名", 'グレードコード', '競走条件コード', '予想勝ち指数'
                                        , "初出走頭数", "混合", "予想決着指数", "登録頭数", "回次", "日次"], axis=1)


    def _proc_horse_df(self, horse_base_df):
        horse_df = self.tf.choose_horse_column(horse_base_df)
        horse_df = self.tf.encode_horse_df(horse_df, self.dict_folder)
        return horse_df.copy()


    def set_prev_df(self):
        """  prev_dfを作成するための処理。prev1_raceuma_dfに処理がされたデータをセットする。前走のデータをセットする  """
        print("set_prev_df")
        race_result_df, raceuma_result_df = self._get_prev_base_df(1)
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
        merged_df = merged_df.drop(["年月日", "月日"], axis=1)
        return merged_df

    def _proc_scale_df_for_fa(self, raceuma_df):
        print("-- check! this is LBLoad class: " + sys._getframe().f_code.co_name)
        mmsc_columns = ["距離増減", "負担重量"]
        mmsc_dict_name = "sc_fa_mmsc"
        stdsc_columns = ["馬体重"]
        stdsc_dict_name = "sc_fa_stdsc"
        raceuma_df = mu.scale_df_for_fa(raceuma_df, mmsc_columns, mmsc_dict_name, stdsc_columns, stdsc_dict_name,self.dict_folder)
        return raceuma_df

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

    def _create_feature(self,):
        """ 過去走と今回を比較した特徴量等、最終的な特徴量を生成する """
        #        self.base_df.loc[:, "継続騎乗"] = self.base_df.apply(lambda x: 1 if x["騎手名"] == x["騎手名_1"] else 0 )
        self.base_df.loc[:, "継続騎乗"] = (self.base_df["騎手名"] == self.base_df["騎手名_1"]).astype(int)
        self.base_df.loc[:, "同場騎手"] = (self.base_df["騎手所属場コード"] == self.base_df["場コード"]).astype(int)
        self.base_df.loc[:, "同所属場"] = (self.base_df["調教師所属場コード"] == self.base_df["場コード"]).astype(int)
        self.base_df.loc[:, "同所属騎手"] = (self.base_df["騎手所属場コード"] == self.base_df["調教師所属場コード"]).astype(int)
        self.base_df.loc[:, "同主催者"] = (self.base_df["主催者コード"] == self.base_df["主催者コード_1"]).astype(int)
        self.base_df.loc[:, "同場コード"] = (self.base_df["場コード"] == self.base_df["場コード_1"]).astype(int)
        self.base_df.loc[:, "同場_1"] = (self.base_df["場コード"] == self.base_df["場コード_1"]).astype(int)
        self.base_df.loc[:, "同距離グループ_1"] = (self.base_df["距離グループ"] == self.base_df["距離グループ_1"]).astype(int)
        self.base_df.loc[:, "同季節_1"] = (self.base_df["季節"] == self.base_df["季節_1"]).astype(int)
        self.base_df.loc[:, "負担重量_1"] = self.base_df["負担重量_1"] - self.base_df["負担重量"]
        self.base_df.loc[:, "頭数差"] = self.base_df["頭数グループ"] - self.base_df["頭数グループ_1"]
        self.base_df.loc[:, "休み明け"] = self.base_df["休養週数"].apply(lambda x: True if x >= 20 else False)
        self.base_df.loc[:, "継続騎乗好走"] = self.base_df.apply(lambda x: 1 if x["継続騎乗"] * x["激走_1"] == 1 else 0, axis=1)

    def _drop_columns_base_df(self):
        self.base_df.drop(
            ["発走時刻_1", "発走時刻", "年月日", "近走競走コード1", "近走馬番1"],
        axis=1, inplace=True)

    def drop_base_columns(self):
        print("つかってない？")
        self.base_df.drop(["発走時刻", "年月日", "近走競走コード1", "近走馬番1"], axis=1, inplace=True)


    def _scale_df(self):
        print("scale_df")
        mmsc_columns = ["距離", "頭数", "枠番", "休養後出走回数", "先行指数順位", "馬齢", "距離増減"]
        mmsc_dict_name = "sc_base_mmsc"
        stdsc_columns = ["休養週数", "負担重量"]
        stdsc_dict_name = "sc_base_stdsc"

        self.base_df = mu.scale_df_for_fa(self.base_df, mmsc_columns, mmsc_dict_name, stdsc_columns, stdsc_dict_name, self.dict_folder)
        self.base_df.loc[:, "競走種別コード_h"] = self.base_df["競走種別コード"]
        self.base_df.loc[:, "場コード_h"] = self.base_df["場コード"]
        hash_track_columns = ["主催者コード", "競走種別コード_h", "場コード_h", "トラックコード"]
        hash_track_dict_name = "sc_base_hash_track"
        self.base_df = mu.hash_eoncoding(self.base_df, hash_track_columns, 10, hash_track_dict_name, self.dict_folder)

class SkModel(LBSkModel):
    class_list = ['競走種別コード', 'コース', '距離グループ', 'ナイター', '季節']
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
                                      intermediate_folder=INTERMEDIATE_FOLDER)], local_scheduler=True, export_mode=False)

    else:
        print("error")