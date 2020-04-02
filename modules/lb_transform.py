from modules.base_transform import BaseTransform
import modules.util as mu

import pandas as pd
import numpy as np
import os
import math
import sys
from factor_analyzer import FactorAnalyzer

class LBTransform(BaseTransform):
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


    def factory_analyze_raceuma_result_df(self, race_df, input_raceuma_df, dict_folder):
        """ RaceUmaの因子分析を行うためのデータを取得 """
        print("-- check! this is BaseTransform class: " + sys._getframe().f_code.co_name)
        temp_df = pd.merge(input_raceuma_df, race_df, on="競走コード")
        X = temp_df[
            ['競走コード', '馬番',  'タイム指数', '単勝オッズ', '先行率', 'ペース偏差値', '距離増減', '斤量比', '追込率', '平均タイム',
             "距離", "頭数", "非根幹", "上り係数", "逃げ勝ち", "内勝ち", "外勝ち", "短縮勝ち", "延長勝ち", "人気勝ち"]]

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

    
    def create_feature_race_base_df(self, race_df):
        """ 特徴となる値を作成する。月日→月、距離→根幹、距離グループを作成して列として付与する。

        :param dataframe race_df:
        :return: dataframe
        """
        temp_race_df = race_df.copy()
        temp_race_df.loc[:, '月'] = race_df['月日'].apply(lambda x: x.month)
        temp_race_df.loc[:, "非根幹"] = race_df["距離"].apply(lambda x: 0 if x % 400 == 0 else 1)
        temp_race_df.loc[:, "距離グループ"] = race_df["距離"] // 400
        return temp_race_df

    
    def create_feature_race_df(self, race_df):
        """ 特徴となる値を作成する。月日→月、距離→根幹、距離グループを作成して列として付与する。

        :param dataframe race_df:
        :return: dataframe
        """
        print("-- check! this is BaseTransform class: " + sys._getframe().f_code.co_name)
        temp_race_df = race_df.copy()
        temp_race_df.loc[:, '月'] = race_df['月日'].apply(lambda x: x.month)
        temp_race_df.loc[:, "非根幹"] = race_df["距離"].apply(lambda x: 0 if x % 400 == 0 else 1)
        temp_race_df.loc[:, "距離グループ"] = race_df["距離"] // 400
        return temp_race_df

    
    def create_feature_raceuma_base_df(self, raceuma_df):
        """  raceuma_dfの特徴量を作成する。馬番→馬番グループを作成して列を追加する。

        :param dataframe raceuma_df:
        :return: dataframe
        """
        temp_raceuma_df = raceuma_df.copy()
        temp_raceuma_df.loc[:, "馬番グループ"] = raceuma_df["馬番"] // 4
        return temp_raceuma_df

    
    def group_prev_raceuma_df(self, raceuma_prev_df, raceuma_base_df):
        """
        | set_all_prev_raceuma_dfで生成された過去レコードに対して、条件毎にタイム指数の最大値と平均値を取得してdataframeとして返す。normalize_group_prev_raceuma_dfで最終的に偏差値化する。
        | 条件は、同場、同距離、同根幹、同距離グループ、同馬番グループ

        :param dataframe raceuma_prev_df: dataframe（過去走のdataframe)
        :param dataframe raceuma_base_df: dataframe（軸となるdataframe)
        :return: dataframe
        """
        # ベースとなるデータフレームを作成する
        raceuma_base_df = raceuma_base_df[["競走コード", "馬番"]]
        # print(raceuma_prev_df.head())
        # 集計対象の条件のみに絞り込んだデータを作成し、平均値と最大値を計算する
        same_ba_df = raceuma_prev_df.query("場コード_x == 場コード_y").groupby(["競走コード", "馬番"])["タイム指数"].agg(['max', 'mean']).rename(columns=lambda x: '同場_' + x).reset_index()
        same_kyori_df = raceuma_prev_df.query("距離_x == 距離_y").groupby(["競走コード", "馬番"])["タイム指数"].agg(['max', 'mean']).rename(columns=lambda x: '同距離_' + x).reset_index()
        same_konkan_df = raceuma_prev_df.query("非根幹_x == 非根幹_y").groupby(["競走コード", "馬番"])["タイム指数"].agg(['max', 'mean']).rename(columns=lambda x: '同根幹_' + x).reset_index()
        same_kyori_group_df = raceuma_prev_df.query("距離グループ_x == 距離グループ_y").groupby(["競走コード", "馬番"])["タイム指数"].agg(['max', 'mean']).rename(columns=lambda x: '同距離グループ_' + x).reset_index()
        same_umaban_group_df = raceuma_prev_df.query("馬番グループ_x == 馬番グループ_y").groupby(["競走コード", "馬番"])["タイム指数"].agg(['max', 'mean']).rename(columns=lambda x: '同馬番グループ_' + x).reset_index()
        # 集計したデータをベースのデータフレームに結合追加する
        raceuma_base_df = pd.merge(raceuma_base_df, same_ba_df, on=["競走コード", "馬番"], how='left')
        raceuma_base_df = pd.merge(raceuma_base_df, same_kyori_df, on=["競走コード", "馬番"], how='left')
        raceuma_base_df = pd.merge(raceuma_base_df, same_konkan_df, on=["競走コード", "馬番"], how='left')
        raceuma_base_df = pd.merge(raceuma_base_df, same_kyori_group_df, on=["競走コード", "馬番"], how='left')
        raceuma_base_df = pd.merge(raceuma_base_df, same_umaban_group_df, on=["競走コード", "馬番"], how='left')

        # 集計値を標準化する
        # 偏差値計算するための平均値と偏差を計算する
        grouped_df = raceuma_base_df[['競走コード', '同場_max', '同場_mean', '同距離_max', '同距離_mean', '同根幹_max', '同根幹_mean', '同距離グループ_max',
                                 '同距離グループ_mean', '同馬番グループ_max', '同馬番グループ_mean']].groupby('競走コード').agg(['mean', 'std']).reset_index()
        # 計算するために列名を変更する
        grouped_df.columns = ['競走コード', '同場_max_mean', '同場_max_std', '同場_mean_mean', '同場_mean_std', '同距離_max_mean', '同距離_max_std', '同距離_mean_mean', '同距離_mean_std', '同根幹_max_mean', '同根幹_max_std',
                              '同根幹_mean_mean', '同根幹_mean_std', '同距離グループ_max_mean', '同距離グループ_max_std', '同距離グループ_mean_mean', '同距離グループ_mean_std', '同馬番グループ_max_mean', '同馬番グループ_max_std', '同馬番グループ_mean_mean', '同馬番グループ_mean_std']
        # 偏差値計算するためのデータフレームを作成する
        merged_df = pd.merge(raceuma_base_df, grouped_df, on='競走コード')
        # ベースとなるデータフレームを作成する
        df = merged_df[["競走コード", "馬番"]].copy()
        # 各偏差値をベースとなるデータフレームに追加していく
        df.loc[:, '同場_max'] = (merged_df['同場_max'] - merged_df['同場_max_mean']) / merged_df['同場_max_std'] * 10 + 50
        df.loc[:, '同場_mean'] = (merged_df['同場_mean'] - merged_df['同場_mean_mean']) / merged_df['同場_mean_std'] * 10 + 50
        df.loc[:, '同距離_max'] = (merged_df['同距離_max'] - merged_df['同距離_max_mean']) / merged_df['同距離_max_std'] * 10 + 50
        df.loc[:, '同距離_mean'] = (merged_df['同距離_mean'] - merged_df['同距離_mean_mean']) / merged_df['同距離_mean_std'] * 10 + 50
        df.loc[:, '同根幹_max'] = (merged_df['同根幹_max'] - merged_df['同根幹_max_mean']) / merged_df['同根幹_max_std'] * 10 + 50
        df.loc[:, '同根幹_mean'] = (merged_df['同根幹_mean'] - merged_df['同根幹_mean_mean']) / merged_df['同根幹_mean_std'] * 10 + 50
        df.loc[:, '同距離グループ_max'] = (merged_df['同距離グループ_max'] - merged_df['同距離グループ_max_mean']) / merged_df['同距離グループ_max_std'] * 10 + 50
        df.loc[:, '同距離グループ_mean'] = (merged_df['同距離グループ_mean'] - merged_df['同距離グループ_mean_mean']) / merged_df['同距離グループ_mean_std'] * 10 + 50
        df.loc[:, '同馬番グループ_max'] = (merged_df['同馬番グループ_max'] - merged_df['同馬番グループ_max_mean']) / merged_df['同馬番グループ_max_std'] * 10 + 50
        df.loc[:, '同馬番グループ_mean'] = (merged_df['同馬番グループ_mean'] - merged_df['同馬番グループ_mean_mean']) / merged_df['同馬番グループ_mean_std'] * 10 + 50
        return df

    
    def choose_race_result_column(self, race_df):
        """ レースデータから必要な列に絞り込む。列はデータ区分、主催者コード、競走コード、月日、距離、場コード、頭数、予想勝ち指数、予想決着指数, 競走種別コード

        :param dataframe race_df:
        :return: dataframe
        """
        print("-- check! this is BaseTransform class: " + sys._getframe().f_code.co_name)
        temp_race_df = race_df[
            ['データ区分', '主催者コード', '競走コード', '月日', '距離', '場コード', '頭数', '天候コード', '馬場状態コード', '競走種別コード', 'ペース', '後３ハロン']]
        return temp_race_df

    
    def create_feature_race_result_df(self, race_df, race_winner_df):
        """  race_ddfのデータから特徴量を作成して列を追加する。月日→月、距離→非根幹、距離グループを作成

        :param dataframe race_df:
        :return: dataframe
        """
        print("-- check! this is BaseTransform class: " + sys._getframe().f_code.co_name)
        temp_race_df = race_df.copy()
        temp_merge_df = pd.merge(race_df, race_winner_df, on="競走コード")
        temp_race_df.loc[:, 'cos_day'] = race_df['月日'].dt.dayofyear.apply(
            lambda x: np.cos(math.radians(90 - (x / 365) * 360)))
        temp_race_df.loc[:, 'sin_day'] = race_df['月日'].dt.dayofyear.apply(
            lambda x: np.sin(math.radians(90 - (x / 365) * 360)))
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

    
    def encode_race_df(self, race_df):
        """  列をエンコードする処理（ラベルエンコーディング、onehotエンコーディング等）。デフォルトでは特にないのでそのまま返す。

        :param dataframe race_df:
        :return: dataframe
        """
        temp_race_df = race_df.copy()
        return temp_race_df


    
    def choose_raceuma_result_column(self, raceuma_df):
        """  レース馬データから必要な列に絞り込む。

        :param dataframe raceuma_df:
        :return: dataframe
        """
        print("-- check! this is BaseTransform class: " + sys._getframe().f_code.co_name)
        temp_raceuma_df = raceuma_df[
            ['データ区分', '競走コード', '馬番', '年月日', '血統登録番号', 'タイム指数', '単勝オッズ', '単勝人気', '確定着順', '着差', '休養週数', '先行率', 'タイム',
             'ペース偏差値', '展開コード', 'クラス変動', '騎手所属場コード', '騎手名', 'テン乗り', '負担重量', '馬体重', '馬体重増減', 'コーナー順位4', '距離増減', '所属', '調教師所属場コード',
             '転厩', '斤量比', '騎手ランキング', '調教師ランキング', 'デフォルト得点', '得点V1', '得点V2', '得点V3']].copy()
        return temp_raceuma_df

    
    def encode_raceuma_result_df(self, raceuma_df, dict_folder):
        """  列をエンコードする処理。騎手名、所属、転厩をラベルエンコーディングして値を置き換える。learning_modeがTrueの場合は辞書生成がされる。

        :param dataframe raceuma_df:
        :return: dataframe
        """
        print("-- check! this is BaseTransform class: " + sys._getframe().f_code.co_name)
        temp_raceuma_df = raceuma_df.copy()
        temp_raceuma_df.loc[:, '騎手名'] = mu.label_encoding(raceuma_df['騎手名'], '騎手名', dict_folder).astype(str)
        temp_raceuma_df.loc[:, '所属'] = mu.label_encoding(raceuma_df['所属'], '所属', dict_folder).astype(str)
        temp_raceuma_df.loc[:, '転厩'] = mu.label_encoding(raceuma_df['転厩'], '転厩', dict_folder).astype(str)
        return temp_raceuma_df.copy()


    
    def normalize_raceuma_result_df(self, raceuma_df):
        """ 数値系データを平準化する処理。偏差値に変換して置き換える。対象列は負担重量、予想タイム指数、デフォルト得点、得点V1、得点V2、得点V3。偏差がない場合は５０に設定

        :param dataframe raceuma_df:
        :return: dataframe
        """
        print("-- check! this is BaseTransform class: " + sys._getframe().f_code.co_name)
        grouped_df = raceuma_df[['競走コード', '負担重量', 'タイム指数', 'デフォルト得点', '得点V1', '得点V2', '得点V3']].groupby('競走コード').agg(
            ['mean', 'std']).reset_index()
        grouped_df.columns = ['競走コード', '負担重量_mean', '負担重量_std', 'タイム指数_mean', 'タイム指数_std', 'デフォルト得点_mean',
                              'デフォルト得点_std', '得点V1_mean', '得点V1_std', '得点V2_mean', '得点V2_std', '得点V3_mean',
                              '得点V3_std']
        merged_df = pd.merge(raceuma_df, grouped_df, on='競走コード')
        merged_df['負担重量偏差'] = (merged_df['負担重量'] - merged_df['負担重量_mean']) / merged_df['負担重量_std'] * 10 + 50
        merged_df['タイム指数偏差'] = (merged_df['タイム指数'] - merged_df['タイム指数_mean']) / merged_df['タイム指数_std'] * 10 + 50
        merged_df['デフォルト得点偏差'] = (merged_df['デフォルト得点'] - merged_df['デフォルト得点_mean']) / merged_df[
            'デフォルト得点_std'] * 10 + 50
        merged_df['得点V1偏差'] = (merged_df['得点V1'] - merged_df['得点V1_mean']) / merged_df['得点V1_std'] * 10 + 50
        merged_df['得点V2偏差'] = (merged_df['得点V2'] - merged_df['得点V2_mean']) / merged_df['得点V2_std'] * 10 + 50
        merged_df['得点V3偏差'] = (merged_df['得点V3'] - merged_df['得点V3_mean']) / merged_df['得点V3_std'] * 10 + 50
        merged_df.drop(
            ['負担重量_mean', '負担重量_std', 'タイム指数_mean', 'タイム指数_std', 'デフォルト得点_mean', 'デフォルト得点_std', '得点V1_mean',
             '得点V1_std',
             '得点V2_mean', '得点V2_std', '得点V3_mean', '得点V3_std', '負担重量', 'タイム指数', 'デフォルト得点', '得点V1', '得点V2', '得点V3'],
            axis=1, inplace=True)
        raceuma_df = merged_df.rename(columns={'負担重量偏差': '負担重量', 'タイム指数偏差': 'タイム指数',
                                               'デフォルト得点偏差': 'デフォルト得点', '得点V1偏差': '得点V1', '得点V2偏差': '得点V2',
                                               '得点V3偏差': '得点V3'}).copy()
        raceuma_df.replace([np.inf, -np.inf], np.nan)
        raceuma_df.fillna({'負担重量': 50, 'タイム指数': 50, 'デフォルト得点': 50, '得点V1': 50, '得点V2': 50, '得点V3': 50},
                          inplace=True)
        return raceuma_df.copy()

    
    def create_feature_raceuma_result_df(self,  race_df, raceuma_df):
        """  raceuma_dfの特徴量を作成する。馬番→馬番グループを作成して列を追加する。

        :param dataframe raceuma_df:
        :return: dataframe
        """
        print("-- check! this is BaseTransform class: " + sys._getframe().f_code.co_name)
        temp_raceuma_df = raceuma_df.copy()
        temp_merge_df = pd.merge(raceuma_df, race_df, on="競走コード")
        temp_raceuma_df.loc[:, "同場騎手"] = (temp_merge_df["騎手所属場コード"] == temp_merge_df["場コード"]).astype(int)
        temp_raceuma_df.loc[:, "同所属場"] = (temp_merge_df["調教師所属場コード"] == temp_merge_df["場コード"]).astype(int)
        temp_raceuma_df.loc[:, "同所属騎手"] = (temp_merge_df["騎手所属場コード"] == temp_merge_df["調教師所属場コード"]).astype(int)
        temp_raceuma_df.loc[:, "追込率"] = (temp_merge_df["コーナー順位4"] - temp_merge_df["確定着順"]) / temp_merge_df["頭数"]
        temp_raceuma_df.loc[:, "平均タイム"] = temp_merge_df["タイム"] / temp_merge_df["距離"] * 200
        return temp_raceuma_df

    
    def encode_raceuma_before_df(self, raceuma_df, dict_folder):
        """  列をエンコードする処理。騎手名、所属、転厩をラベルエンコーディングして値を置き換える。learning_modeがTrueの場合は辞書生成がされる。

        :param dataframe raceuma_df:
        :return: dataframe
        """
        print("-- check! this is BaseTransform class: " + sys._getframe().f_code.co_name)
        temp_raceuma_df = raceuma_df.copy()
        temp_raceuma_df.loc[:, '騎手名'] = mu.label_encoding(raceuma_df['騎手名'], '騎手名', dict_folder).astype(str)
        temp_raceuma_df.loc[:, '調教師名'] = mu.label_encoding(raceuma_df['調教師名'], '調教師名', dict_folder).astype(str)
        temp_raceuma_df.loc[:, '所属'] = mu.label_encoding(raceuma_df['所属'], '所属', dict_folder).astype(str)
        temp_raceuma_df.loc[:, '転厩'] = mu.label_encoding(raceuma_df['転厩'], '転厩', dict_folder).astype(str)
        return temp_raceuma_df.copy()

    
    def normalize_raceuma_df(self, raceuma_df):
        """ 数値系データを平準化する処理。偏差値に変換して置き換える。対象列は負担重量、予想タイム指数、デフォルト得点、得点V1、得点V2、得点V3。偏差がない場合は５０に設定

        :param dataframe raceuma_df:
        :return: dataframe
        """
        grouped_df = raceuma_df[['競走コード', '負担重量', '予想タイム指数', 'デフォルト得点', '得点V1', '得点V2', '得点V3']].groupby('競走コード').agg(['mean', 'std']).reset_index()
        grouped_df.columns = ['競走コード', '負担重量_mean', '負担重量_std', '予想タイム指数_mean', '予想タイム指数_std', 'デフォルト得点_mean',
                              'デフォルト得点_std', '得点V1_mean', '得点V1_std', '得点V2_mean', '得点V2_std', '得点V3_mean', '得点V3_std']
        merged_df = pd.merge(raceuma_df, grouped_df, on='競走コード')
        merged_df['負担重量偏差'] = (merged_df['負担重量'] - merged_df['負担重量_mean']) / merged_df['負担重量_std'] * 10 + 50
        merged_df['予想タイム指数偏差'] = (merged_df['予想タイム指数'] - merged_df['予想タイム指数_mean']) / merged_df['予想タイム指数_std'] * 10 + 50
        merged_df['デフォルト得点偏差'] = (merged_df['デフォルト得点'] - merged_df['デフォルト得点_mean']) / merged_df['デフォルト得点_std'] * 10 + 50
        merged_df['得点V1偏差'] = (merged_df['得点V1'] - merged_df['得点V1_mean']) / merged_df['得点V1_std'] * 10 + 50
        merged_df['得点V2偏差'] = (merged_df['得点V2'] - merged_df['得点V2_mean']) / merged_df['得点V2_std'] * 10 + 50
        merged_df['得点V3偏差'] = (merged_df['得点V3'] - merged_df['得点V3_mean']) / merged_df['得点V3_std'] * 10 + 50
        merged_df.drop(['負担重量_mean', '負担重量_std', '予想タイム指数_mean', '予想タイム指数_std', 'デフォルト得点_mean', 'デフォルト得点_std', '得点V1_mean', '得点V1_std',
                        '得点V2_mean', '得点V2_std', '得点V3_mean', '得点V3_std', '負担重量', '予想タイム指数', 'デフォルト得点', '得点V1', '得点V2', '得点V3'], axis=1, inplace=True)
        raceuma_df = merged_df.rename(columns={'負担重量偏差': '負担重量', '予想タイム指数偏差': '予想タイム指数',
                                               'デフォルト得点偏差': 'デフォルト得点', '得点V1偏差': '得点V1', '得点V2偏差': '得点V2', '得点V3偏差': '得点V3'}).copy()
        raceuma_df.fillna({'負担重量': 50, '予想タイム指数': 50, 'デフォルト得点': 50, '得点V1': 50, '得点V2': 50, '得点V3': 50}, inplace=True)
        return raceuma_df.copy()

    
    def drop_columns_raceuma_df(self, raceuma_df):
        print("-- check! this is BaseTransform class: " + sys._getframe().f_code.co_name)
        return raceuma_df

    
    def standardize_raceuma_df(self, raceuma_df):
        """ 数値データを整備する。無限大(inf）をnanに置き換える

        :param dataframe raceuma_df:
        :return: dataframe
        """
        temp_raceuma_df = raceuma_df.copy()
        temp_raceuma_df.loc[:, '予想タイム指数'] = raceuma_df['予想タイム指数'].replace([np.inf, -np.inf], np.nan)
        return temp_raceuma_df.copy()

    
    def create_feature_raceuma_df(self, raceuma_df):
        """  raceuma_dfの特徴量を作成する。馬番→馬番グループを作成して列を追加する。

        :param dataframe raceuma_df:
        :return: dataframe
        """
        print("-- check! this is BaseTransform class: " + sys._getframe().f_code.co_name)
        temp_raceuma_df = raceuma_df.copy()
        temp_raceuma_df.loc[:, "馬番グループ"] = raceuma_df["馬番"] // 4
        return temp_raceuma_df

    
    def standardize_raceuma_result_df(self, raceuma_df):
        """ 数値データを整備する。無限大(inf）をnanに置き換える

        :param dataframe raceuma_df:
        :return: dataframe
        """
        temp_raceuma_df = raceuma_df.copy()
        temp_raceuma_df.loc[:, 'タイム指数'] = raceuma_df['タイム指数'].replace([np.inf, -np.inf], np.nan)
        return temp_raceuma_df.copy()


    
    def choose_horse_column(self, horse_df):
        """ 馬データから必要な列に絞り込む。対象は血統登録番号、繁殖登録番号１、繁殖登録番号５、東西所属コード、生産者コード、馬主コード

        :param dataframe raceuma_df:
        :return: dataframe
        """
        print("-- check! this is BaseTransform class: " + sys._getframe().f_code.co_name)
        temp_horse_df = horse_df[['血統登録番号', '繁殖登録番号1', '繁殖登録番号5', '東西所属コード', '生産者コード', '馬主コード']]
        return temp_horse_df

    
    def normalize_prev_merged_df(self, merged_df):
        """ 前走レース馬データ用に値を平準化した値に置き換える。頭数で割る or 逆数化。対象は単勝人気、確定着順、コーナー順位４、騎手ランキング、調教師ランキング

        :param dataframe merged_df:
        :return: dataframe
        """
        temp_merged_df = merged_df.copy()
        temp_merged_df.loc[:, '単勝人気'] = merged_df.apply(lambda x: 0 if x['頭数'] == 0 else x['単勝人気'] / x['頭数'], axis=1)
        temp_merged_df.loc[:, '確定着順'] = merged_df.apply(lambda x: 0 if x['頭数'] == 0 else x['確定着順'] / x['頭数'], axis=1)
        temp_merged_df.loc[:, 'コーナー順位4'] = merged_df.apply(lambda x: 0 if x['頭数'] == 0 else x['コーナー順位4'] / x['頭数'], axis=1)
        temp_merged_df.loc[:, '騎手ランキング'] = merged_df.apply(lambda x: np.nan if x['騎手ランキング'] == 0 else 1 / x['騎手ランキング'], axis=1)
        temp_merged_df.loc[:, '調教師ランキング'] = merged_df.apply(lambda x: np.nan if x['調教師ランキング'] == 0 else 1 / x['調教師ランキング'], axis=1)
        return temp_merged_df