from modules.base_transform import BaseTransform
import modules.util as mu

import pandas as pd
import numpy as np
import os
import math
import sys
from sklearn.decomposition import PCA

class JRATransform(BaseTransform):
    """
    地方競馬に関するデータ変換処理をまとめたクラス
    """

    def __init__(self, start_date, end_date):
        """ LocalBaozExtractオブジェクトを生成してデータを呼び出す準備をする

        :param str start_date:
        :param str end_date:
        """
        self.start_date = start_date
        self.end_date = end_date


    def drop_columns_race_df(self, race_df):
        """ 処理前に不要なカラムを削除する """
        race_df = race_df.drop(["KAISAI_KEY", "レース名", "回数", "レース名短縮", "レース名９文字", "データ区分", "１着賞金", "２着賞金", "３着賞金", "４着賞金", "５着賞金", "１着算入賞金", "２着算入賞金", "馬券発売フラグ", "WIN5フラグ"], axis=1)
        return race_df

    def encode_race_before_df(self, race_df, dict_folder):
        """  列をエンコードする処理。調教師所属、所属、転厩をラベルエンコーディングして値を置き換える。辞書がない場合は作成される
        騎手名とかはコードがあるからそちらを使う（作成しない）

        :param dataframe raceuma_df:
        :return: dataframe
        """
        temp_race_df = race_df.copy()
        temp_race_df.loc[:, '場名'] = temp_race_df['RACE_KEY'].str[:2]
        temp_race_df.loc[:, '曜日'] = temp_race_df['曜日'].apply(lambda x: self._convert_weekday(x))
        return temp_race_df.copy()

    def create_feature_race_df(self, race_df):
        """ 特徴となる値を作成する。月日→月、距離→根幹、距離グループを作成して列として付与する。

        :param dataframe race_df:
        :return: dataframe
        """
        temp_race_df = race_df.copy()
        temp_race_df.loc[:, '条件'] = temp_race_df['条件'].apply(lambda x: self._convert_joken(x))
        temp_race_df.loc[:, '月'] = race_df['NENGAPPI'].str[4:6].astype(int)
        temp_race_df.loc[:, "非根幹"] = race_df["距離"].apply(lambda x: 0 if x % 400 == 0 else 1)
        temp_race_df.loc[:, "距離グループ"] = race_df["距離"] // 400
        return temp_race_df

    def encode_raceuma_before_df(self, raceuma_df, dict_folder):
        """  列をエンコードする処理。調教師所属、所属、転厩をラベルエンコーディングして値を置き換える。辞書がない場合は作成される
        騎手名とかはコードがあるからそちらを使う（作成しない）

        :param dataframe raceuma_df:
        :return: dataframe
        """
        temp_raceuma_df = raceuma_df.copy()
        temp_raceuma_df.loc[:, 'ペース予想'] = temp_raceuma_df['ペース予想'].apply(lambda x: self._convert_pace(x))
        temp_raceuma_df.loc[:, '調教曜日'] = temp_raceuma_df['調教曜日'].apply(lambda x: self._convert_weekday(x))
        temp_raceuma_df.loc[:, '放牧先ランク'] = temp_raceuma_df['放牧先ランク'].apply(lambda x: self._convert_rank(x))
        temp_raceuma_df.loc[:, '調教量評価'] = temp_raceuma_df['調教量評価'].apply(lambda x: self._convert_rank(x))
        temp_raceuma_df.loc[:, '調教師所属'] = mu.label_encoding(raceuma_df['調教師所属'], '調教師所属', dict_folder).astype(str)
        temp_raceuma_df.loc[:, '馬主名'] = mu.label_encoding(raceuma_df['馬主名'], '馬主名', dict_folder).astype(str)
        temp_raceuma_df.loc[:, '放牧先'] = mu.label_encoding(raceuma_df['放牧先'], '放牧先', dict_folder).astype(str)
        temp_raceuma_df.loc[:, '激走タイプ'] = mu.label_encoding(raceuma_df['激走タイプ'], '激走タイプ', dict_folder).astype(str)
        temp_raceuma_df.loc[:, '調教コースコード'] = mu.label_encoding(raceuma_df['調教コースコード'], '調教コースコード', dict_folder).astype(str)
        hash_taikei_column = ["走法", "体型０１", "体型０２", "体型０３", "体型０４", "体型０５", "体型０６", "体型０７", "体型０８", "体型０９", "体型１０", "体型１１", "体型１２", "体型１３", "体型１４", "体型１５", "体型１６", "体型１７", "体型１８", "体型総合１", "体型総合２", "体型総合３"]
        hash_taikei_dict_name = "raceuma_before_taikei"
        temp_raceuma_df = mu.hash_eoncoding(temp_raceuma_df, hash_taikei_column, 3, hash_taikei_dict_name, dict_folder)
        hash_tokki_column = ["馬特記１", "馬特記２", "馬特記３"]
        hash_tokki_dict_name = "raceuma_before_tokki"
        temp_raceuma_df = mu.hash_eoncoding(temp_raceuma_df, hash_tokki_column, 2, hash_tokki_dict_name, dict_folder)


        return temp_raceuma_df.copy()


    def normalize_raceuma_df(self, raceuma_df):
        """ 数値系データを平準化する処理。偏差値に変換して置き換える。対象列は負担重量、予想タイム指数、デフォルト得点、得点V1、得点V2、得点V3。偏差がない場合は５０に設定

        :param dataframe raceuma_df:
        :return: dataframe
        """
        norm_list = ['IDM', '騎手指数', '情報指数', '総合指数', '人気指数', '調教指数', '厩舎指数', 'テン指数', 'ペース指数', '上がり指数', '位置指数', '万券指数',
                                 'テンＦ指数', '中間Ｆ指数', '終いＦ指数', '追切指数', '仕上指数']
        temp_raceuma_df = raceuma_df[norm_list].astype(float)
        temp_raceuma_df.loc[:, "RACE_KEY"] = raceuma_df["RACE_KEY"]
        temp_raceuma_df.loc[:, "UMABAN"] = raceuma_df["UMABAN"]
        grouped_df = temp_raceuma_df[['RACE_KEY'] + norm_list].groupby('RACE_KEY').agg(['mean', 'std']).reset_index()
        grouped_df.columns = ['RACE_KEY', 'IDM_mean', 'IDM_std', '騎手指数_mean', '騎手指数_std', '情報指数_mean', '情報指数_std', '総合指数_mean', '総合指数_std',
                              '人気指数_mean', '人気指数_std', '調教指数_mean', '調教指数_std', '厩舎指数_mean', '厩舎指数_std', 'テン指数_mean', 'テン指数_std',
                              'ペース指数_mean', 'ペース指数_std', '上がり指数_mean', '上がり指数_std', '位置指数_mean', '位置指数_std', '万券指数_mean', '万券指数_std',
                              'テンＦ指数_mean', 'テンＦ指数_std', '中間Ｆ指数_mean', '中間Ｆ指数_std', '終いＦ指数_mean', '終いＦ指数_std', '追切指数_mean', '追切指数_std', '仕上指数_mean', '仕上指数_std']
        temp_raceuma_df = pd.merge(temp_raceuma_df, grouped_df, on='RACE_KEY')
        for norm in norm_list:
            temp_raceuma_df[f'{norm}偏差'] = temp_raceuma_df.apply(lambda x: (x[norm] - x[f'{norm}_mean']) / x[f'{norm}_std'] * 10 + 50 if x[f'{norm}_std'] != 0 else 50, axis=1)
            temp_raceuma_df = temp_raceuma_df.drop([norm, f'{norm}_mean', f'{norm}_std'], axis=1)
            raceuma_df = raceuma_df.drop(norm, axis=1)
            temp_raceuma_df = temp_raceuma_df.rename(columns={f'{norm}偏差': norm})
        raceuma_df = pd.merge(raceuma_df, temp_raceuma_df, on=["RACE_KEY", "UMABAN"])
        return raceuma_df.copy()

    def standardize_raceuma_df(self, raceuma_df):
        """ 数値データを整備する。無限大(inf）をnanに置き換える

        :param dataframe raceuma_df:
        :return: dataframe
        """
        return raceuma_df

    def create_feature_raceuma_df(self, raceuma_df):
        """  raceuma_dfの特徴量を作成する。馬番→馬番グループを作成して列を追加する。

        :param dataframe raceuma_df:
        :return: dataframe
        """
        temp_raceuma_df = raceuma_df.copy()
        temp_raceuma_df.loc[:, "馬番グループ"] = raceuma_df["UMABAN"].apply(lambda x: int(x) // 4)
        temp_raceuma_df.loc[:, "基準人気グループ"] = raceuma_df["基準人気順位"].apply(lambda x : self._convert_ninki_group(x))
        result_type_list = ["ＪＲＡ成績", "交流成績", "他成績", "芝ダ障害別成績", "芝ダ障害別距離成績", "トラック距離成績", "ローテ成績", "回り成績", "騎手成績", "良成績", "稍成績",
                       "重成績", "Ｓペース成績", "Ｍペース成績", "Ｈペース成績", "季節成績", "枠成績", "騎手距離成績", "騎手トラック距離成績", "騎手調教師別成績", "騎手馬主別成績",
                       "騎手ブリンカ成績", "調教師馬主別成績"]
        for type in result_type_list:
            temp_raceuma_df[f'{type}１着'] = temp_raceuma_df[f'{type}１着'].apply(lambda x: x if x else 0)
            temp_raceuma_df[f'{type}２着'] = temp_raceuma_df[f'{type}２着'].apply(lambda x: x if x else 0)
            temp_raceuma_df[f'{type}３着'] = temp_raceuma_df[f'{type}３着'].apply(lambda x: x if x else 0)
            temp_raceuma_df[f'{type}着外'] = temp_raceuma_df[f'{type}着外'].apply(lambda x: x if x else 0)
            temp_raceuma_df.loc[:, type] = temp_raceuma_df.apply(lambda x: np.nan if (x[f'{type}１着'] + x[f'{type}２着'] + x[f'{type}３着'] + x[f'{type}着外'] == 0)
            else (x[f'{type}１着'] + x[f'{type}２着'] + x[f'{type}３着'])/ (x[f'{type}１着'] + x[f'{type}２着'] + x[f'{type}３着'] + x[f'{type}着外']), axis=1)
            temp_raceuma_df.drop([f'{type}１着', f'{type}２着', f'{type}３着', f'{type}着外'], axis=1, inplace=True)
        return temp_raceuma_df



    def drop_columns_raceuma_df(self, raceuma_df):
        raceuma_df = raceuma_df.drop(["騎手名", "調教師名", "馬名", "枠確定馬体重", "枠確定馬体重増減", "取消フラグ", "調教年月日", "調教コメント", "コメント年月日"], axis=1)
        return raceuma_df

    def drop_columns_horse_df(self, horse_df):
        temp_horse_df = horse_df.drop(["馬名", "生年月日", "父馬生年", "母馬生年", "母父馬生年", "馬主名", "馬主会コード", "母馬名", "性別コード", "馬記号コード", "NENGAPPI"], axis=1)
        return temp_horse_df


    def encode_horse_df(self, horse_df, dict_folder):
        """  列をエンコードする処理。調教師所属、所属、転厩をラベルエンコーディングして値を置き換える。辞書がない場合は作成される
        騎手名とかはコードがあるからそちらを使う（作成しない）

        :param dataframe raceuma_df:
        :return: dataframe
        """
        temp_horse_df = horse_df.copy()
        temp_horse_df.loc[:, '父馬名'] = mu.label_encoding(horse_df['父馬名'], '父馬名', dict_folder).astype(str)
        temp_horse_df.loc[:, '母父馬名'] = mu.label_encoding(horse_df['母父馬名'], '母父馬名', dict_folder).astype(str)
        temp_horse_df.loc[:, '生産者名'] = mu.label_encoding(horse_df['生産者名'], '生産者名', dict_folder).astype(str)
        temp_horse_df.loc[:, '産地名'] = mu.label_encoding(horse_df['産地名'], '産地名', dict_folder).astype(str)
        return temp_horse_df.copy()

    def drop_columns_raceuma_result_df(self, raceuma_df):
        raceuma_df = raceuma_df.drop(["NENGAPPI", "馬名", "レース名", "レース名略称", "騎手名", "調教師名", "1(2)着馬名", "パドックコメント", "脚元コメント", "馬具(その他)コメント", "レースコメント", "異常区分", "血統登録番号", "単勝", "複勝", "馬体重増減", "KYOSO_RESULT_KEY"], axis=1)
        return raceuma_df

    def encode_raceuma_result_df(self, raceuma_df, dict_folder):
        """  列をエンコードする処理。騎手名、所属、転厩をラベルエンコーディングして値を置き換える。learning_modeがTrueの場合は辞書生成がされる。

        :param dataframe raceuma_df:
        :return: dataframe
        """
        temp_raceuma_df = raceuma_df.copy()
        temp_raceuma_df.loc[:, 'RAP_TYPE'] = temp_raceuma_df['RAP_TYPE'].apply(lambda x: mu.encode_rap_type(x))
        temp_raceuma_df.loc[:, '条件'] = temp_raceuma_df['条件'].apply(lambda x: self._convert_joken(x))
        temp_raceuma_df.loc[:, 'レースペース'] = temp_raceuma_df['レースペース'].apply(lambda x: self._convert_pace(x))
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
        norm_list = ['ＩＤＭ結果', '斤量', 'テン指数結果', '上がり指数結果', 'ペース指数結果', 'レースＰ指数結果', '馬体重']
        temp_raceuma_df = raceuma_df[norm_list].astype(float)
        temp_raceuma_df.loc[:, "RACE_KEY"] = raceuma_df["RACE_KEY"]
        temp_raceuma_df.loc[:, "UMABAN"] = raceuma_df["UMABAN"]
        grouped_df = temp_raceuma_df[['RACE_KEY'] + norm_list].groupby('RACE_KEY').agg(['mean', 'std']).reset_index()
        grouped_df.columns = ['RACE_KEY', 'ＩＤＭ結果_mean', 'ＩＤＭ結果_std', '斤量_mean', '斤量_std', 'テン指数結果_mean',
                              'テン指数結果_std', '上がり指数結果_mean', '上がり指数結果_std', 'ペース指数結果_mean', 'ペース指数結果_std', 'レースＰ指数結果_mean',
                              'レースＰ指数結果_std', '馬体重_mean', '馬体重_std']
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
        temp_raceuma_df.loc[:, "先行率"] = (temp_raceuma_df["コーナー順位４"] / temp_raceuma_df["頭数"])
        temp_raceuma_df.loc[:, "人気率"] = (temp_raceuma_df["確定単勝人気順位"] / temp_raceuma_df["頭数"])
        temp_raceuma_df.loc[:, "着順率"] = (temp_raceuma_df["着順"] / temp_raceuma_df["頭数"])
        temp_raceuma_df.loc[:, "追込率"] = (temp_raceuma_df["コーナー順位４"] - temp_raceuma_df["着順"]) / temp_raceuma_df["頭数"]
        temp_raceuma_df.loc[:, "平均タイム"] = temp_raceuma_df["タイム"] / temp_raceuma_df["距離"] * 200
        return temp_raceuma_df

    def factory_analyze_race_result_df(self, raceuma_result_df, dict_path):
        """ レースの因子を計算。それぞれ以下の意味
        # fa1:　数値大：底力指数
        # fa2:　数値大：末脚指数
        # fa3:　数値大：スタミナ指数
        # fa4: 両方向：レースタイプ
        # fa5:　数値大：高速スピード指数
        """
        dict_folder = dict_path + 'dict/jra_common/'
        fa_dict_name = "fa_raceuma_result_df"
        fa = mu.load_dict(fa_dict_name, dict_folder)
        fa_list = ['１着算入賞金', 'ラップ差４ハロン', 'ラップ差３ハロン', 'ラップ差２ハロン', 'ラップ差１ハロン',
               'TRACK_BIAS_ZENGO', 'TRACK_BIAS_UCHISOTO', 'ハロン数', '芝', '外', '重', '軽',
               '１ハロン平均_mean', 'ＩＤＭ結果_mean', 'テン指数結果_mean', '上がり指数結果_mean',
               'ペース指数結果_mean', '追走力_mean', '追上力_mean', '後傾指数_mean', '１ハロン平均_std',
               '上がり指数結果_std', 'ペース指数結果_std']
        df_data_org = raceuma_result_df[fa_list]
        sc_dict_name = "fa_sc_raceuma_result_df"
        scaler = mu.load_dict(sc_dict_name, dict_folder)
        df_data = pd.DataFrame(scaler.transform(df_data_org), columns=df_data_org.columns)
        fa_df = pd.DataFrame(fa.transform(df_data.fillna(0)), columns=["fa_1", "fa_2", "fa_3", "fa_4", "fa_5"])
        raceuma_result_df = pd.concat([raceuma_result_df, fa_df], axis=1)
        return raceuma_result_df

    def cluster_raceuma_result_df(self, raceuma_result_df, dict_path):
        """ 出走結果をクラスタリング。それぞれ以下の意味
        # 激走: 4:前目の位置につけて能力以上の激走
        # 好走：1:後方から上がり上位で能力通り好走 　7:前目の位置につけて能力通り好走
        # ふつう：0:なだれ込み能力通りの凡走    5:前目の位置につけて上りの足が上がり能力通りの凡走 6:後方から足を使うも能力通り凡走
        # 凡走（下位）：2:前目の位置から上りの足が上がって能力以下の凡走　
        # 大凡走　3:後方追走いいとこなしで能力以下の大凡走
        # 障害、出走取消等→8

        """
        dict_folder = dict_path + 'dict/jra_common/'
        fa_dict_name = "cluster_raceuma_result"
        cluster = mu.load_dict(fa_dict_name, dict_folder)
        fa_list = ["RACE_KEY", "UMABAN", "IDM", "RAP_TYPE", "着順", "確定単勝人気順位", "ＩＤＭ結果", "コーナー順位２",
                   "コーナー順位３", "コーナー順位４", "タイム", "距離", "芝ダ障害コード", "後３Ｆタイム", "テン指数結果順位",
                   "上がり指数結果順位", "頭数", "前３Ｆ先頭差", "後３Ｆ先頭差", "異常区分"]
        temp_df = raceuma_result_df.query("異常区分 not in ('1','2') and 芝ダ障害コード != '3' and 頭数 != 0")
        df = temp_df[fa_list].copy()
        df.loc[:, "追走力"] = df.apply(
            lambda x: x["コーナー順位２"] - x["コーナー順位４"] if x["コーナー順位２"] != 0 else x["コーナー順位３"] - x["コーナー順位４"], axis=1)
        df.loc[:, "追上力"] = df.apply(lambda x: x["コーナー順位４"] - x["着順"], axis=1)
        df.loc[:, "１ハロン平均"] = df.apply(lambda x: x["タイム"] / x["距離"] * 200, axis=1)
        df.loc[:, "後傾指数"] = df.apply(lambda x: x["１ハロン平均"] * 3 / x["後３Ｆタイム"] if x["後３Ｆタイム"] != 0 else 1, axis=1)
        df.loc[:, "馬番"] = df["UMABAN"].astype(int) / df["頭数"]
        df.loc[:, "IDM差"] = df["ＩＤＭ結果"] - df["IDM"]
        df.loc[:, "コーナー順位４"] = df["コーナー順位４"] / df["頭数"]
        df.loc[:, "CHAKU_RATE"] = df["着順"] / df["頭数"]
        df.loc[:, "確定単勝人気順位"] = df["確定単勝人気順位"] / df["頭数"]
        df.loc[:, "テン指数結果順位"] = df["テン指数結果順位"] / df["頭数"]
        df.loc[:, "上がり指数結果順位"] = df["上がり指数結果順位"] / df["頭数"]
        df.loc[:, "上り最速"] = df["上がり指数結果順位"].apply(lambda x: 1 if x == 1 else 0)
        df.loc[:, "逃げ"] = df["テン指数結果順位"].apply(lambda x: 1 if x == 1 else 0)
        df.loc[:, "勝ち"] = df["着順"].apply(lambda x: 1 if x == 1 else 0)
        df.loc[:, "連対"] = df["着順"].apply(lambda x: 1 if x in (1, 2) else 0)
        df.loc[:, "３着内"] = df["着順"].apply(lambda x: 1 if x in (1, 2, 3) else 0)
        df.loc[:, "掲示板前後"] = df["着順"].apply(lambda x: 1 if x in (4, 5, 6) else 0)
        df.loc[:, "着外"] = df["CHAKU_RATE"].apply(lambda x: 1 if x >= 0.4 else 0)
        df.loc[:, "凡走"] = df.apply(lambda x: 1 if x["CHAKU_RATE"] >= 0.6 and x["確定単勝人気順位"] <= 0.3 else 0, axis=1)
        df.loc[:, "激走"] = df.apply(lambda x: 1 if x["CHAKU_RATE"] <= 0.3 and x["確定単勝人気順位"] >= 0.7 else 0, axis=1)
        df.loc[:, "異常"] = df["異常区分"].apply(lambda x: 1 if x != '0' else 0)
        numerical_feats = df.dtypes[df.dtypes != "object"].index
        km_df = df[numerical_feats].drop(["タイム", "後３Ｆタイム", "頭数", 'ＩＤＭ結果', "IDM", "１ハロン平均", 'コーナー順位２', 'コーナー順位３', '距離'],axis=1)
        # print(km_df.columns)
        # Index(['着順', '確定単勝人気順位', 'コーナー順位４', 'テン指数結果順位', '上がり指数結果順位', '前３Ｆ先頭差',
        #        '後３Ｆ先頭差', '追走力', '追上力', '後傾指数', '馬番', 'IDM差', 'CHAKU_RATE', '上り最速',
        #        '逃げ', '勝ち', '連対', '３着内', '掲示板前後', '着外', '凡走', '激走', '異常'],
        pred = cluster.predict(km_df)
        temp_df.loc[:, "ru_cluster"] = pred
        other_df = raceuma_result_df.query("異常区分 in ('1','2') or 芝ダ障害コード == '3'")
        other_df.loc[:, "ru_cluster"] = 8
        return_df = pd.concat([temp_df, other_df])
        return return_df


    def cluster_course_df(self, raceuma_result_df, dict_path):
        """ コースを10(0-9)にクラスタリング。当てはまらないものは10にする

        """
        dict_folder = dict_path + 'dict/jra_common/'
        fa_dict_name = "course_cluster_df.pkl"
        cluster_df = pd.read_pickle(dict_folder + fa_dict_name)
        raceuma_result_df.loc[:, "COURSE_KEY"] = raceuma_result_df["RACE_KEY"].str[:2] + raceuma_result_df["距離"].astype(str) + raceuma_result_df["芝ダ障害コード"] + raceuma_result_df["内外"]
        raceuma_result_df.loc[:, "場コード"] = raceuma_result_df["RACE_KEY"].str[:2]
        df = pd.merge(raceuma_result_df, cluster_df, on="COURSE_KEY", how="left").rename(columns={"cluster": "course_cluster"})
        df.loc[:, "course_cluster"] = df["course_cluster"].fillna(10)
        return df

    def bk_factory_analyze_raceuma_result_df(self, input_raceuma_df, dict_folder):
        """ RaceUmaの因子分析を行うためのデータを取得 """
        fa_list = ['芝ダ障害コード', '右左', '内外', '種別', '条件', '記号', '重量',
             "頭数", "着順", '馬場状態', "タイム", "確定単勝オッズ", "コース取り", "上昇度コード", "馬体コード", "気配コード", "レースペース", "馬ペース",
             "1(2)着タイム差", "前３Ｆタイム", "後３Ｆタイム", "確定複勝オッズ下", "10時単勝オッズ", "10時複勝オッズ",
             "コーナー順位１", "コーナー順位２", "コーナー順位３", "コーナー順位４", "前３Ｆ先頭差", "後３Ｆ先頭差", "天候コード",
             "コース", "レース脚質", "４角コース取り", "確定単勝人気順位", "素点", "素点", "馬場差", "ペース", "出遅", "位置取",
             "前不利", "中不利", "後不利", "レース", "クラスコード", "レースペース流れ", "馬ペース流れ", "グレード"]
        X = input_raceuma_df[fa_list].fillna(0)
        str_list = X.select_dtypes(include=object).columns.tolist()
        X[str_list] = X[str_list].astype(int)
        X.iloc[0] = X.iloc[0] + 0.000001

        dict_name = "fa_raceuma_result_df"
        filename = dict_folder + dict_name + '.pkl'
        if os.path.exists(filename):
            fa = mu.load_dict(dict_name, dict_folder)
        else:
            fa = PCA(n_components=5)
            #fa = FactorAnalyzer(n_factors=5, rotation='promax', impute='drop')
            fa.fit(X)
            mu.save_dict(fa, dict_name, dict_folder)

        fa_np = fa.transform(X)
        fa_df = pd.DataFrame(fa_np, columns=["fa_1", "fa_2", "fa_3", "fa_4", "fa_5"])
        X_fact = pd.concat([input_raceuma_df.drop(fa_list, axis=1), fa_df], axis=1)
        return X_fact


    def _convert_pace(self, pace):
        if pace == "S":
            return 1
        elif pace == "M":
            return 2
        elif pace == "H":
            return 3
        else:
            return 0

    def _convert_joken(self, joken):
        if joken == 'A1':
            return 1
        elif joken == 'A2':
            return 2
        elif joken == 'A3':
            return 3
        elif joken == 'OP':
            return 99
        elif joken is None:
            return 0
        else:
            return int(joken)


    def _convert_ninki_group(self, ninki):
        if ninki == 1:
            return 1
        elif ninki in (2,3):
            return 2
        elif ninki in (4,5,6,7):
            return 3
        elif ninki in (8, 9, 10, 11, 12):
            return 4
        else:
            return 5

    def _convert_weekday(self, yobi):
        if yobi == "日":
            return '0'
        elif yobi == "月":
            return '1'
        elif yobi == "火":
            return '2'
        elif yobi == "水":
            return '3'
        elif yobi == "木":
            return '4'
        elif yobi == "金":
            return '5'
        elif yobi == "土":
            return '6'
        else:
            return yobi

    def _convert_rank(self, rank):
        if rank == "A":
            return 1
        elif rank == "B":
            return 2
        elif rank == "C":
            return 3
        elif rank == "D":
            return 4
        elif rank == "E":
            return 5
        elif rank == "F":
            return 6
        elif rank == "G":
            return 7
        elif rank == "H":
            return 8
        elif rank == "I":
            return 9
        else:
            return rank
