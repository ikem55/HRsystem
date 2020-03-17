from modules.base_load import BaseLoad
from modules.lb_extract import LBExtract
from modules.lb_transform import LBTransform as Tf
import modules.util as mu

import sys
import pandas as pd

class LBLoad(BaseLoad):
    """
    地方競馬のデータロードクラス
    """
    dict_folder = './for_test_dict/base/'
    """ 辞書フォルダのパス """

    def _get_extract_object(self, start_date, end_date, mock_flag):
        """ 利用するExtクラスを指定する """
        print("-- check! this is LBLoad class: " + sys._getframe().f_code.co_name)
        ext = LBExtract(start_date, end_date, mock_flag)
        return ext

    def _get_transform_object(self, start_date, end_date):
        """ 利用するTransformクラスを指定する """
        print("-- check! this is LBLoad class: " + sys._getframe().f_code.co_name)
        tf = Tf(start_date, end_date)
        return tf

    def set_prev_df(self):
        """  prev_dfを作成するための処理。prev1_raceuma_df,prev2_raceuma_dfに処理がされたデータをセットする。過去２走のデータと過去走を集計したデータをセットする  """
        print("-- check! this is LBLoad class: " + sys._getframe().f_code.co_name)
        race_result_df, raceuma_result_df = self._get_prev_base_df(2)
        self.prev2_raceuma_df = self._get_prev_df(2, race_result_df, raceuma_result_df)
        self.prev2_raceuma_df.rename(columns=lambda x: x + "_2", inplace=True)
        self.prev2_raceuma_df.rename(columns={"競走コード_2": "競走コード", "馬番_2": "馬番"}, inplace=True)
        self.prev1_raceuma_df = self._get_prev_df(1, race_result_df, raceuma_result_df)
        self.prev1_raceuma_df.rename(columns=lambda x: x + "_1", inplace=True)
        self.prev1_raceuma_df.rename(columns={"競走コード_1": "競走コード", "馬番_1": "馬番"}, inplace=True)
        self._set_grouped_raceuma_prev_df(race_result_df, raceuma_result_df)

    def _get_race_winner_df(self, raceuma_base_df):
        race_winner_df = raceuma_base_df[raceuma_base_df["確定着順"] == 1].drop_duplicates(subset='競走コード')
        return race_winner_df

    def _proc_scale_df_for_fa(self, raceuma_df):
        print("-- check! this is LBLoad class: " + sys._getframe().f_code.co_name)
        mmsc_columns = ["距離増減", "斤量比", "負担重量"]
        mmsc_dict_name = "sc_fa_mmsc"
        stdsc_columns = ["タイム指数", "単勝オッズ", "ペース偏差値", "平均タイム", "馬体重"]
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
        merged_df = pd.merge(this_race_df, this_raceuma_df, on=["競走コード", "馬番"])
        merged_df = merged_df.drop(['タイム指数', '単勝オッズ', '先行率', 'ペース偏差値', '距離増減', '斤量比', '追込率', '平均タイム',
             "距離", "頭数", "上り係数", "逃げ勝ち", "内勝ち", "外勝ち", "短縮勝ち", "延長勝ち", "人気勝ち",
             "年月日", "月日", "距離", "血統登録番号"], axis=1)

        return merged_df

    def _set_grouped_raceuma_prev_df(self, race_result_df, raceuma_result_df):
        """  過去走の集計データを作成する。開始日の1年前のデータを取得して条件毎の集計データを作成する

        :return: dataframe
        """
        # レースデータを取得
        race_base_df = self.race_df[["競走コード", "場コード", "距離", "月日"]]
        race_base_df = self.tf.create_feature_race_base_df(race_base_df)

        # レース馬データを取得
        raceuma_base_df = self.raceuma_df[["競走コード", "馬番", "血統登録番号", "年月日"]]
        raceuma_base_df = self.tf.create_feature_raceuma_base_df(raceuma_base_df)

        # レースとレース馬データを結合したデータフレーム（母数となるテーブル）を作成
        raceuma_df = pd.merge(raceuma_base_df, race_base_df, on="競走コード").rename(columns={"年月日": "年月日_x"})

        # 過去のレース全データを取得
        raceuma_prev_all_base_df = raceuma_result_df[["競走コード", "血統登録番号", "年月日", "馬番", "タイム指数"]]
        raceuma_prev_all_base_df = self.tf.create_feature_raceuma_base_df(raceuma_prev_all_base_df).drop("馬番", axis=1)
        race_prev_all_base_df = self.tf.create_feature_race_df(race_result_df)
        raceuma_prev_all_df = pd.merge(raceuma_prev_all_base_df, race_prev_all_base_df, on="競走コード").drop("競走コード", axis=1).rename(columns={"年月日": "年月日_y"})

        # 母数テーブルと過去走テーブルを血統登録番号で結合
        raceuma_prev_df = pd.merge(raceuma_df, raceuma_prev_all_df, on="血統登録番号")
        # 対象レースより前のレースのみに絞り込む
        raceuma_prev_df = raceuma_prev_df.query("年月日_x > 年月日_y")
        # 過去レースの結果を集計する
        self.grouped_raceuma_prev_df = self.tf.group_prev_raceuma_df(raceuma_prev_df, raceuma_base_df)

    def _proc_result_df(self, result_race_df, result_raceuma_df):
        result_df = pd.merge(result_race_df, result_raceuma_df, on="競走コード")
        return result_df[["競走コード", "馬番", "確定着順", "単勝オッズ"]].copy()