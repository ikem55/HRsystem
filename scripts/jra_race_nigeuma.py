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
from sklearn.model_selection import train_test_split
import optuna.integration.lightgbm as lgb
import featuretools as ft



# 呼び出し方
# python jra_race_nigeuma.py learning True True
# ====================================== パラメータ　要変更 =====================================================
# 逃げ馬、上がり最速馬を予測する

MODEL_VERSION = 'jra_rc_nigeuma'
MODEL_NAME = 'race_lgm'

# ====================================== クラス　要変更 =========================================================

class Ext(JRAExtract):
    pass

class Tf(JRATransform):
    def drop_columns_race_df(self, race_df):
        race_df = race_df.drop(["KAISAI_KEY", "発走時間", "レース名", "回数", "レース名短縮", "レース名９文字", "データ区分", "１着賞金", "２着賞金",
                                "３着賞金", "４着賞金", "５着賞金", "１着算入賞金", "２着算入賞金", "馬券発売フラグ", "WIN5フラグ", "曜日",
                                "芝馬場状態内", "芝馬場状態中", "芝馬場状態外", "芝馬場差", "直線馬場差最内", "直線馬場差内", "直線馬場差中",
                                "直線馬場差外", "直線馬場差大外", "ダ馬場状態コード", "ダ馬場状態内", "ダ馬場状態中", "ダ馬場状態外", "ダ馬場差", "中間降水量"], axis=1)
        return race_df


    def encode_race_before_df(self, race_df, dict_folder):
        """  列をエンコードする処理。調教師所属、所属、転厩をラベルエンコーディングして値を置き換える。辞書がない場合は作成される
        騎手名とかはコードがあるからそちらを使う（作成しない）

        :param dataframe raceuma_df:
        :return: dataframe
        """
        temp_race_df = race_df.copy()
        temp_race_df.loc[:, '場名'] = temp_race_df['RACE_KEY'].str[:2]
        return temp_race_df.copy()

    def drop_columns_raceuma_df(self, raceuma_df):
        raceuma_df = raceuma_df.drop(["騎手名", "調教師名", "馬名", "枠確定馬体重", "枠確定馬体重増減", "取消フラグ", "調教年月日", "調教コメント", "コメント年月日",
                                      "基準オッズ", "基準複勝オッズ", "基準複勝人気順位", "特定情報◎", "特定情報○", "特定情報▲", "特定情報△", "特定情報×", "総合情報◎",
                                      "総合情報○", "総合情報○", "総合情報▲", "総合情報△", "総合情報×", "騎手期待連対率", "クラスコード", "調教師所属", "総合印", "ＩＤＭ印",
                                      "情報印", "騎手印", "厩舎印", "調教印", "激走印", "獲得賞金", "収得賞金", "条件クラス", "馬主会コード", "参考前走", "参考前走騎手コード",
                                      "万券印", "降級フラグ", "入厩年月日", "入厩何日前", "厩舎ＢＢ印", "厩舎ＢＢ◎単勝回収率", "厩舎ＢＢ◎連対率", "騎手ＢＢ印", "騎手ＢＢ◎単勝回収率",
                                      "騎手ＢＢ◎連対率", "調教曜日", "併せ年齢", "併せクラス", "父馬産駒芝連対率", "父馬産駒ダ連対率", "父馬産駒連対平均距離", "母父馬産駒芝連対率",
                                      "母父馬産駒ダ連対率", "母父馬産駒連対平均距離", "ＪＲＡ成績", "交流成績", "他成績", "芝ダ障害別成績", "芝ダ障害別距離成績", "トラック距離成績",
                                      "ローテ成績", "回り成績", "騎手成績", "良成績", "稍成績", "重成績", "Ｓペース成績", "Ｍペース成績", "Ｈペース成績", "季節成績", "枠成績",
                                      "騎手距離成績", "騎手トラック距離成績", "騎手調教師別成績", "騎手馬主別成績", "騎手ブリンカ成績", "調教師馬主別成績",
                                      "ローテーション", "騎手期待単勝率", "騎手期待３着内率", "CID調教素点", "CID厩舎素点", "CID素点", "LS指数", "LS評価", "EM", "調教回数",
                                      "総合指数", "人気指数", "厩舎指数", "基準人気グループ", "芝適性コード", "ダ適性コード", "ペース予想", "道中差", "後３Ｆ差" ,"ゴール差",
                                      "馬記号コード", "乗り役", "調教Ｆ", "テンＦ", "中間Ｆ", "終いＦ", "調教評価", "調教コース坂", "調教コースW",
                                      "調教コースダ", "調教コース芝", "調教コースプール", "調教コース障害", "調教コースポリ"], axis=1)
        return raceuma_df


    def drop_columns_horse_df(self, horse_df):
        horse_df = horse_df.drop(["馬名", "生年月日", "父馬生年", "母馬生年", "母父馬生年", "馬主名", "馬主会コード", "母馬名", "性別コード", "馬記号コード", "毛色コード", "生産者名", "産地名", "登録抹消フラグ", "NENGAPPI"], axis=1)
        return horse_df


    def encode_horse_df(self, horse_df, dict_folder):
        """  列をエンコードする処理。調教師所属、所属、転厩をラベルエンコーディングして値を置き換える。辞書がない場合は作成される
        騎手名とかはコードがあるからそちらを使う（作成しない）

        :param dataframe raceuma_df:
        :return: dataframe
        """
        temp_horse_df = horse_df.copy()
        temp_horse_df.loc[:, '父馬名'] = mu.label_encoding(horse_df['父馬名'], '父馬名', dict_folder).astype(str)
        temp_horse_df.loc[:, '母父馬名'] = mu.label_encoding(horse_df['母父馬名'], '母父馬名', dict_folder).astype(str)
        return temp_horse_df.copy()

    def drop_columns_raceuma_result_df(self, raceuma_df):
        raceuma_df = raceuma_df.drop(["NENGAPPI", "馬名", "レース名", "レース名略称", "タイム", "斤量", "騎手名", "調教師名", "確定単勝オッズ", "確定単勝人気順位",
                                      "ＩＤＭ結果", "素点", "馬場差", "ペース", "出遅", "位置取", "不利", "前不利", "中不利", "中不利", "後不利", "レース", "コース取り",
                                      "上昇度コード", "クラスコード", "馬体コード", "気配コード", "レースペース", "1(2)着馬名", "前３Ｆタイム", "後３Ｆタイム", "確定複勝オッズ下",
                                      "パドックコメント", "脚元コメント", "馬具(その他)コメント", "天候コード", "コース", "本賞金", "収得賞金", "レースペース流れ",
                                      "10時単勝オッズ", "10時複勝オッズ", "馬体重", "馬体重増減", "レースコメント", "異常区分", "血統登録番号", "単勝", "複勝", "KYOSO_RESULT_KEY",
                                      "種別", "条件", "記号", "重量", "グレード", "異常区分"], axis=1)
        return raceuma_df

    def encode_raceuma_result_df(self, raceuma_df, dict_folder):
        """  列をエンコードする処理。騎手名、所属、転厩をラベルエンコーディングして値を置き換える。learning_modeがTrueの場合は辞書生成がされる。

        :param dataframe raceuma_df:
        :return: dataframe
        """
        temp_raceuma_df = raceuma_df.copy()
        temp_raceuma_df.loc[:, '馬ペース'] = temp_raceuma_df['馬ペース'].apply(lambda x: self._convert_pace(x))
        hash_tokki_column = ["特記コード１", "特記コード２", "特記コード３", "特記コード４", "特記コード５", "特記コード６"]
        hash_tokki_dict_name = "raceuma_result_tokki"
        temp_raceuma_df = mu.hash_eoncoding(temp_raceuma_df, hash_tokki_column, 3, hash_tokki_dict_name, dict_folder)
        hash_bagu_column = ["馬具コード１", "馬具コード２", "馬具コード３", "馬具コード４", "馬具コード５", "馬具コード６", "馬具コード７", "馬具コード８", "ハミ", "バンテージ", "蹄鉄"]
        hash_bagu_dict_name = "raceuma_result_bagu"
        temp_raceuma_df = mu.hash_eoncoding(temp_raceuma_df, hash_bagu_column, 3, hash_bagu_dict_name, dict_folder)
        hash_taikei_column = ["総合１", "総合２", "総合３", "左前１", "左前２", "左前３", "右前１", "右前２", "右前３", "左後１", "左後２", "左後３", "右後１", "右後２", "右後３", "蹄状態", "ソエ", "骨瘤"]
        hash_taikei_dict_name = "raceuma_result_taikei"
        temp_raceuma_df = mu.hash_eoncoding(temp_raceuma_df, hash_taikei_column, 3, hash_taikei_dict_name, dict_folder)
        return temp_raceuma_df.copy()


    def normalize_raceuma_result_df(self, raceuma_df):
        """ 数値系データを平準化する処理。偏差値に変換して置き換える。対象列は負担重量、予想タイム指数、デフォルト得点、得点V1、得点V2、得点V3。偏差がない場合は５０に設定

        :param dataframe raceuma_df:
        :return: dataframe
        """
        norm_list = [ 'テン指数結果', '上がり指数結果', 'ペース指数結果', 'レースＰ指数結果']
        temp_raceuma_df = raceuma_df[norm_list].astype(float)
        temp_raceuma_df.loc[:, "RACE_KEY"] = raceuma_df["RACE_KEY"]
        temp_raceuma_df.loc[:, "UMABAN"] = raceuma_df["UMABAN"]
        grouped_df = temp_raceuma_df[['RACE_KEY'] + norm_list].groupby('RACE_KEY').agg(['mean', 'std']).reset_index()
        grouped_df.columns = ['RACE_KEY', 'テン指数結果_mean', 'テン指数結果_std', '上がり指数結果_mean', '上がり指数結果_std',
                              'ペース指数結果_mean', 'ペース指数結果_std', 'レースＰ指数結果_mean', 'レースＰ指数結果_std']
        temp_raceuma_df = pd.merge(temp_raceuma_df, grouped_df, on='RACE_KEY')
        for norm in norm_list:
            temp_raceuma_df[f'{norm}偏差'] = temp_raceuma_df.apply(lambda x: (x[norm] - x[f'{norm}_mean']) / x[f'{norm}_std'] * 10 + 50 if x[f'{norm}_std'] != 0 else 50, axis=1)
            temp_raceuma_df = temp_raceuma_df.drop([norm, f'{norm}_mean', f'{norm}_std'], axis=1)
            raceuma_df = raceuma_df.drop(norm, axis=1)
            temp_raceuma_df = temp_raceuma_df.rename(columns={f'{norm}偏差': norm})
        raceuma_df = pd.merge(raceuma_df, temp_raceuma_df, on=["RACE_KEY", "UMABAN"])
        return raceuma_df.copy()

    def create_feature_raceuma_result_df(self, raceuma_df):
        """  raceuma_dfの特徴量を作成する。馬番→馬番グループを作成して列を追加する。

        :param dataframe raceuma_df:
        :return: dataframe
        """
        temp_raceuma_df = raceuma_df.copy()
        temp_raceuma_df.loc[:, "非根幹"] = temp_raceuma_df["距離"].apply(lambda x: 0 if x % 400 == 0 else 1)
        temp_raceuma_df.loc[:, "距離グループ"] = temp_raceuma_df["距離"] // 400
        temp_raceuma_df.loc[:, "追込率"] = (temp_raceuma_df["コーナー順位４"] - temp_raceuma_df["着順"]) / temp_raceuma_df["頭数"]
        temp_raceuma_df.loc[:, "コーナー順位１"] = (temp_raceuma_df["コーナー順位１"] / temp_raceuma_df["頭数"])
        temp_raceuma_df.loc[:, "コーナー順位２"] = (temp_raceuma_df["コーナー順位２"] / temp_raceuma_df["頭数"])
        temp_raceuma_df.loc[:, "コーナー順位３"] = (temp_raceuma_df["コーナー順位３"] / temp_raceuma_df["頭数"])
        temp_raceuma_df.loc[:, "コーナー順位４"] = (temp_raceuma_df["コーナー順位４"] / temp_raceuma_df["頭数"])
        return temp_raceuma_df

    def factory_analyze_raceuma_result_df(self, input_raceuma_df, dict_folder):
        return input_raceuma_df


class Ld(JRALoad):
    def _get_extract_object(self, start_date, end_date, mock_flag):
        """ 利用するExtクラスを指定する """
        ext = Ext(start_date, end_date, mock_flag)
        return ext

    def _get_transform_object(self, start_date, end_date):
        """ 利用するTransformクラスを指定する """
        tf = Tf(start_date, end_date)
        return tf

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

    def set_result_df(self):
        """ 目的変数作成用のresult_dfを作成するための処理。result_dfに処理がされたデータをセットする """
        self.result_raceuma_df = self.ext.get_raceuma_table_base()

class SkProc(JRASkProc):
    """
    地方競馬の機械学習処理プロセスを取りまとめたクラス。
    """
    index_list = ["RACE_KEY", "target_date"]
    # LightGBM のハイパーパラメータ
    obj_column_list = ['R_NIGEUMA', 'R_AGARI_SAISOKU', 'R_TEN_SAISOKU']
    lgbm_params = {
        'R_NIGEUMA':{'objective': 'multiclass', 'metric': 'multi_logloss', 'num_class': 19},#0も含める
        'R_AGARI_SAISOKU':{'objective': 'multiclass', 'metric': 'multi_logloss', 'num_class': 19},
        'R_TEN_SAISOKU':{'objective': 'multiclass', 'metric': 'multi_logloss', 'num_class': 19},
                   }
    class_dict = [{"name": "芝", "code": "1", "except_list": ["芝ダ障害コード", "転圧", "凍結防止剤"]},
                  {"name": "ダ", "code": "2", "except_list": ["芝ダ障害コード", "芝馬場状態コード", "芝種類", "草丈"]}]

    def _get_load_object(self, version_str, start_date, end_date, mock_flag, test_flag):
        ld = Ld(version_str, start_date, end_date, mock_flag, test_flag)
        return ld

    def _create_feature(self,):
        """ 過去走と今回を比較した特徴量等、最終的な特徴良を生成する """
        print(self.base_df.shape)
        self.base_df.loc[:, "継続騎乗"] = (self.base_df["騎手コード"] == self.base_df["騎手コード_1"]).astype(int)
        self.base_df.loc[:, "距離増減"] = self.base_df["距離"] - self.base_df["距離_1"]
        self.base_df.loc[:, "頭数増減"] = self.base_df["頭数"] - self.base_df["頭数_1"]


    def proc_create_learning_data(self):
        self._proc_create_base_df()
        self._drop_unnecessary_columns()
        self._set_target_variables()
        learning_df = pd.merge(self.base_df, self.result_df, on ="RACE_KEY")
        return learning_df

    def _proc_create_base_df(self):
        self._set_ld_data()
        self._merge_df()
        self._create_feature()
        feature_summary_df = self._feature_summary_data()
        self._drop_columns_base_df()
        self._flat_base_df(feature_summary_df)
        self.base_df = self._rename_key(self.base_df)

    def _merge_df(self):
        self.base_df = pd.merge(self.ld.race_df, self.ld.raceuma_df, on=["RACE_KEY", "target_date", "NENGAPPI"])
        self.base_df = pd.merge(self.base_df, self.ld.horse_df, on=["血統登録番号", "target_date"])
        self.base_df = pd.merge(self.base_df, self.ld.prev1_raceuma_df, on=["RACE_KEY", "UMABAN", "target_date"], how='left')
        self.base_df = pd.merge(self.base_df, self.ld.prev2_raceuma_df, on=["RACE_KEY", "UMABAN", "target_date"], how='left')
        self.base_df = pd.merge(self.base_df, self.ld.prev3_raceuma_df, on=["RACE_KEY", "UMABAN", "target_date"], how='left')
        self.base_df = pd.merge(self.base_df, self.ld.prev4_raceuma_df, on=["RACE_KEY", "UMABAN", "target_date"], how='left')
        self.base_df = pd.merge(self.base_df, self.ld.prev5_raceuma_df, on=["RACE_KEY", "UMABAN", "target_date"], how='left')


    def _feature_summary_data(self):
        raceuma_df = self.base_df[["RACE_KEY", "UMABAN", "激走指数", "馬スタート指数", "馬出遅率", "IDM", "騎手指数", "テン指数",
                                   "ペース指数", "上がり指数", "位置指数", "テンＦ指数", "中間Ｆ指数", "終いＦ指数",
                                   "コーナー順位３_1", "コーナー順位４_1", "前３Ｆ先頭差_1", "後３Ｆ先頭差_1",
                                   "レース脚質_1", "テン指数結果_1", "上がり指数結果_1", "ペース指数結果_1", "レースＰ指数結果_1", "追込率_1",
                                    "コーナー順位３_2", "コーナー順位４_2", "前３Ｆ先頭差_2", "後３Ｆ先頭差_2",
                                    "レース脚質_2", "テン指数結果_2", "上がり指数結果_2", "ペース指数結果_2", "レースＰ指数結果_2", "追込率_2",
                                    "コーナー順位３_3", "コーナー順位４_3", "前３Ｆ先頭差_3", "後３Ｆ先頭差_3",
                                    "レース脚質_3", "テン指数結果_3", "上がり指数結果_3", "ペース指数結果_3", "レースＰ指数結果_3", "追込率_3",
                                    "コーナー順位３_4", "コーナー順位４_4", "前３Ｆ先頭差_4", "後３Ｆ先頭差_4",
                                    "レース脚質_4", "テン指数結果_4", "上がり指数結果_4", "ペース指数結果_4", "レースＰ指数結果_4", "追込率_4",
                                    "コーナー順位３_5", "コーナー順位４_5", "前３Ｆ先頭差_5", "後３Ｆ先頭差_5",
                                    "レース脚質_5", "テン指数結果_5", "上がり指数結果_5", "ペース指数結果_5", "レースＰ指数結果_5", "追込率_5"]]
        raceuma_df.loc[:, "RACE_UMA_KEY"] = raceuma_df["RACE_KEY"].astype(str).str.cat(raceuma_df["UMABAN"].astype(str))
        raceuma_df.drop("UMABAN", axis=1, inplace=True)
        es = ft.EntitySet(id="race")

        es.entity_from_dataframe(entity_id='race', dataframe=self.ld.race_df[["RACE_KEY", "target_date"]], index="RACE_KEY")
        es.entity_from_dataframe(entity_id='raceuma', dataframe=raceuma_df, index="RACE_UMA_KEY")
        relationship = ft.Relationship(es['race']["RACE_KEY"], es['raceuma']["RACE_KEY"])
        es = es.add_relationship(relationship)
        print(es)
        # 集約関数
        aggregation_list = ['mean', 'skew']
        transform_list = []
        # run dfs
        print("un dfs")
        feature_matrix, features_dfs = ft.dfs(entityset=es, target_entity='race', agg_primitives=aggregation_list,
                                              trans_primitives=transform_list, max_depth=2)
        feature_summary_df = pd.merge(feature_matrix, self.ld.race_df, on=["RACE_KEY", "target_date"])
        print("_create_feature: feature_summary_df", feature_summary_df.shape)
        return feature_summary_df


    def _drop_columns_base_df(self):
        self.base_df = self.base_df[["RACE_KEY", "UMABAN", "脚質", "上昇度", "調教矢印コード", "厩舎評価コード", "激走指数", "蹄コード",
                                     "見習い区分", "騎手コード", "道中順位", "道中内外", "後３Ｆ順位", "後３Ｆ内外", "展開記号",
                                     "LS指数順位", "テン指数順位", "ペース指数順位", "上がり指数順位", "位置指数順位", "輸送区分", "馬スタート指数", "馬出遅率",
                                     "raceuma_before_tokki_0", "raceuma_before_tokki_1", "IDM", "騎手指数", "テン指数", "ペース指数", "上がり指数",
                                     "テンＦ指数", "終いＦ指数", "父系統コード", "母父系統コード",

                                     "頭数_1", "着順_1", "馬ペース_1", "コーナー順位３_1", "コーナー順位４_1", "前３Ｆ先頭差_1", "後３Ｆ先頭差_1",
                                     "レース脚質_1", "４角コース取り_1", "raceuma_result_tokki_0_1", "raceuma_result_tokki_1_1", "raceuma_result_tokki_2_1",
                                     "テン指数結果_1", "上がり指数結果_1", "ペース指数結果_1", "レースＰ指数結果_1", "追込率_1",
                                     "TRACK_BIAS_ZENGO_1", "TRACK_BIAS_UCHISOTO_1", "テン指数結果順位_1", "上がり指数結果順位_1",
                                     "ru_cluster_1", "fa_1_1", "fa_2_1", "fa_3_1", "fa_4_1", "fa_5_1",

                                     "頭数_2", "着順_2", "馬ペース_2", "コーナー順位３_2", "コーナー順位４_2", "レース脚質_2",
                                     "テン指数結果_2", "上がり指数結果_2", "ペース指数結果_2", "レースＰ指数結果_2", "追込率_2",
                                     "TRACK_BIAS_ZENGO_2", "TRACK_BIAS_UCHISOTO_2", "テン指数結果順位_2", "上がり指数結果順位_2",
                                     "ru_cluster_2", "fa_1_2", "fa_2_2", "fa_3_2", "fa_4_2", "fa_5_2",

                                     "コーナー順位３_3", "コーナー順位４_3", "レース脚質_3",
                                     "テン指数結果_3", "上がり指数結果_3", "ペース指数結果_3", "レースＰ指数結果_3", "追込率_3",
                                     "TRACK_BIAS_ZENGO_3", "TRACK_BIAS_UCHISOTO_3", "テン指数結果順位_3", "上がり指数結果順位_3",
                                     "ru_cluster_3", "fa_1_3", "fa_2_3", "fa_3_3", "fa_4_3", "fa_5_3",

                                     "コーナー順位３_4", "コーナー順位４_4", "レース脚質_4",
                                     "テン指数結果_4", "上がり指数結果_4", "ペース指数結果_4", "レースＰ指数結果_4", "追込率_4",
                                     "TRACK_BIAS_ZENGO_4", "TRACK_BIAS_UCHISOTO_4", "テン指数結果順位_4", "上がり指数結果順位_4",
                                     "ru_cluster_4", "fa_1_4", "fa_2_4", "fa_3_4", "fa_4_4", "fa_5_4",

                                     "コーナー順位３_5", "コーナー順位４_5", "レース脚質_5",
                                     "テン指数結果_5", "上がり指数結果_5", "ペース指数結果_5", "レースＰ指数結果_5", "追込率_5",
                                     "TRACK_BIAS_ZENGO_5", "TRACK_BIAS_UCHISOTO_5", "テン指数結果順位_5", "上がり指数結果順位_5",
                                     "ru_cluster_5", "fa_1_5", "fa_2_5", "fa_3_5", "fa_4_5", "fa_5_5",
                                     ]]


    def _flat_base_df(self, feature_summary_df):
        """ レース馬情報を１行に並べてレース情報をつなげたものをbase_dfとして再作成する """
        temp_df = self.base_df.set_index(["RACE_KEY", "UMABAN"])
        print(temp_df.shape)
        #import collections
        #c = collections.Counter(temp_df.columns.tolist())
        #print(c)

        temp_unstack_df = temp_df.unstack()
        unstack_columns = ["__".join(pair) for pair in temp_unstack_df.columns]
        temp_unstack_df.columns = unstack_columns
        columns_base = temp_df.columns.values.tolist()
        columns_list = []
        for i in range(1, 19):
            columns_list += [s + "__" + str(i).zfill(2) for s in columns_base]
        dif_list = columns_list.copy()
        for col in unstack_columns:
            try:
                dif_list.remove(col)
            except ValueError:
                continue
        for col in dif_list:
            temp_unstack_df[col] = np.NaN
        self.base_df = pd.merge(temp_unstack_df, feature_summary_df, on="RACE_KEY")
        print(self.base_df.shape)

    def _flat_function(self, a):
        a.index = [0 for i in range(len(a))]
        del a['ID']
        out = a[0:1]
        for i in range(1, len(a)):
            out = out.join(a[i:i + 1], rsuffix='{0}'.format(i))
        return out

    def _drop_unnecessary_columns(self):
        """ predictに不要な列を削除してpredict_dfを作成する。削除する列は血統登録番号、確定着順、タイム指数、単勝オッズ、単勝人気  """
        self.base_df.drop(["NENGAPPI", "COURSE_KEY"], axis=1, inplace=True)

    def _set_target_variables(self):
        self.ld.set_result_df()
        nigeuma_df = self.ld.result_raceuma_df.query("レース脚質 == '1'")[["RACE_KEY", "UMABAN"]].rename(columns={"UMABAN": "R_NIGEUMA"})
        agari_saisoku_df = self.ld.result_raceuma_df.query("上がり指数結果順位 == 1")[["RACE_KEY", "UMABAN"]].rename(columns={"UMABAN": "R_AGARI_SAISOKU"})
        ten_saisoku_df = self.ld.result_raceuma_df.query("テン指数結果順位 == 1")[["RACE_KEY", "UMABAN"]].rename(columns={"UMABAN": "R_TEN_SAISOKU"})
        self.result_df = pd.merge(nigeuma_df, agari_saisoku_df, on ="RACE_KEY")
        self.result_df = pd.merge(self.result_df, ten_saisoku_df, on ="RACE_KEY")

    def load_learning_target_encoding(self):
        pass

    def _set_predict_target_encoding(self, df):
        return df

    def learning_race_lgb(self, this_model_name, target):
        # テスト用のデータを評価用と検証用に分ける
        X_eval, X_valid, y_eval, y_valid = train_test_split(self.X_test, self.y_test, random_state=42)

        # データセットを生成する
        lgb_train = lgb.Dataset(self.X_train, self.y_train)
        lgb_eval = lgb.Dataset(X_eval, y_eval, reference=lgb_train)

        # 上記のパラメータでモデルを学習する
        best_params, history = {}, []
        this_param = self.lgbm_params[target]
        model = lgb.train(this_param, lgb_train,valid_sets=lgb_eval,
                          verbose_eval=False,
                          num_boost_round=100,
                          early_stopping_rounds=5,
                          best_params=best_params,
                          tuning_history=history)
        print("Bset Paramss:", best_params)
        print('Tuning history:', history)

        self._save_learning_model(model, this_model_name)

    def _sub_create_pred_df(self, temp_df, y_pred):
        pred_df = pd.DataFrame(y_pred, columns=range(y_pred.shape[1])).reset_index(drop=True)
        base_df = pd.DataFrame({"RACE_KEY": temp_df["RACE_KEY"], "target_date": temp_df["target_date"]}).reset_index(drop=True)
        pred_df = pd.concat([base_df, pred_df], axis=1)
        pred_df = pred_df.set_index(["RACE_KEY", "target_date"])
        pred_df = pred_df.stack().reset_index().rename(columns={"level_2": "UMABAN", 0: "prob"})
        pred_df.loc[:, "UMABAN"] = pred_df["UMABAN"].astype(str).str.zfill(2)
        pred_df = pred_df[pred_df["UMABAN"] != "00"]
        pred_df = self._calc_grouped_data(pred_df)
        return pred_df


class SkModel(JRASkModel):
    obj_column_list = ['R_NIGEUMA', 'R_AGARI_SAISOKU', 'R_TEN_SAISOKU']

    def _get_skproc_object(self, version_str, start_date, end_date, model_name, mock_flag, test_flag):
        proc = SkProc(version_str, start_date, end_date, model_name, mock_flag, test_flag, self.obj_column_list)
        return proc

    def create_featrue_select_data(self, learning_df):
        pass

    def _eval_check_df(self, result_df, target_df, target):
        target_df = target_df.query("predict_rank == 1")
        temp_df = result_df[["RACE_KEY", target]].rename(columns={target: "result"})
        check_df = pd.merge(target_df, temp_df, on="RACE_KEY")
        check_df.loc[:, "的中"] = check_df.apply(lambda x: 1 if x["UMABAN"] == x["result"] else 0, axis=1)
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
    dict_path = mc.return_jra_path(test_flag)
    INTERMEDIATE_FOLDER = dict_path + 'intermediate/' + MODEL_VERSION + '_' + args[1] + '/' + MODEL_NAME + '/'
    print("intermediate_folder:" + INTERMEDIATE_FOLDER)

    pd.set_option('display.max_rows', 300)

    if mode == "learning":
        if test_flag:
            print("Test mode")
            start_date = '2018/01/01'
            end_date = '2018/01/31'
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
            start_date = '2018/02/01'
            end_date = '2018/02/28'
        else:
            base_start_date = '2019/01/01'
            pred_folder = dict_path + 'pred/' + MODEL_VERSION
            start_date = SkModel.get_recent_day(base_start_date, pred_folder)
            end_date = (dt.now() + timedelta(days=1)).strftime('%Y/%m/%d')
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