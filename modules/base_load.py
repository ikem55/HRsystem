from modules.base_extract import BaseExtract
from modules.base_transform import BaseTransform as Tf
import modules.util as mu
import my_config as mc

from datetime import datetime as dt
from datetime import timedelta
import sys

class BaseLoad(object):
    """
    データロードに関する共通処理を定義する。
    race,raceuma,prev_raceといった塊ごとのデータを作成する。learning_df等の最終データの作成はsk_proc側に任せる

    """
    dict_folder = ""
    """ 辞書フォルダのパス """
    mock_flag = False
    """ mockデータを利用する場合はTrueにする """
    race_df = ""
    raceuma_df = ""
    horse_df = ""
    prev2_raceuma_df = ""
    prev1_raceuma_df = ""
    grouped_raceuma_prev_df = ""

    def __init__(self, version_str, start_date, end_date, mock_flag, test_flag):
        self.start_date = start_date
        self.end_date = end_date
        self.mock_flag = mock_flag
        self.dict_path = mc.return_base_path(test_flag)
        self._set_folder_path(version_str)
        self.ext = self._get_extract_object(start_date, end_date, mock_flag)
        self.tf = self._get_transform_object(start_date, end_date)

    def _set_folder_path(self, version_str):
        self.dict_folder = self.dict_path + 'dict/' + version_str + '/'
        print("self.dict_folder:", self.dict_folder)

    def _get_extract_object(self, start_date, end_date, mock_flag):
        """ 利用するExtクラスを指定する """
        print("-- check! this is BaseLoad class: " + sys._getframe().f_code.co_name)
        ext = BaseExtract(start_date, end_date, mock_flag)
        return ext

    def _get_transform_object(self, start_date, end_date):
        """ 利用するTransformクラスを指定する """
        print("-- check! this is BaseLoad class: " + sys._getframe().f_code.co_name)
        tf = Tf(start_date, end_date)
        return tf

    def set_race_df(self):
        print("-- check! this is BaseLoad class: " + sys._getframe().f_code.co_name)

    def _proc_race_df(self, race_base_df):
        print("-- check! this is BaseLoad class: " + sys._getframe().f_code.co_name)
        race_df = self.tf.create_feature_race_df(race_base_df)
        return race_df

    def set_raceuma_df(self):
        print("-- check! this is BaseLoad class: " + sys._getframe().f_code.co_name)

    def _proc_raceuma_df(self, raceuma_base_df):
        raceuma_df = self.tf.encode_raceuma_before_df(raceuma_base_df, self.dict_folder)
        raceuma_df = self.tf.normalize_raceuma_df(raceuma_df)
        raceuma_df = self.tf.standardize_raceuma_df(raceuma_df)
        raceuma_df = self.tf.create_feature_raceuma_df(raceuma_df)
        raceuma_df = self.tf.drop_columns_raceuma_df(raceuma_df)
        return raceuma_df.copy()

    def set_horse_df(self):
        """  horse_dfを作成するための処理。horse_dfに処理がされたデータをセットする """
        horse_base_df = self.ext.get_horse_table_base()
        self.horse_df = self._proc_horse_df(horse_base_df)
        print("set_horse_df: horse_df", self.horse_df.shape)

    def _proc_horse_df(self, horse_base_df):
        print("-- check! this is BaseLoad class: " + sys._getframe().f_code.co_name)
        horse_df = self.tf.choose_horse_column(horse_base_df)
        return horse_df.copy()

    def set_prev_df(self):
        print("-- check! this is BaseLoad class: " + sys._getframe().f_code.co_name)

    def _get_prev_base_df(self, num):
        """ 過去データを計算するためのベースとなるDataFrameを作成する

        :param int num: int(計算前走数）
        :return: dataframe
        """
        print("_get_prev_base_df" + str(num))
        dt_start_date = dt.strptime(self.start_date, '%Y/%m/%d')
        prev_start_date = (dt_start_date - timedelta(days=(180 + 60 * int(num)))).strftime('%Y/%m/%d')
        ext_prev = self._get_extract_object(prev_start_date, self.end_date, self.mock_flag)
        race_base_df = ext_prev.get_race_table_base()
        raceuma_base_df = ext_prev.get_raceuma_table_base()
        race_winner_df = self._get_race_winner_df(raceuma_base_df)
        race_result_df = self._proc_race_result_df(race_base_df, race_winner_df)
        raceuma_result_df = self._proc_raceuma_result_df(race_result_df, raceuma_base_df)
        return race_result_df, raceuma_result_df

    def _get_race_winner_df(self, raceuma_base_df):
        print("-- check! this is BaseLoad class: " + sys._getframe().f_code.co_name)

    def _proc_race_result_df(self, race_base_df, race_winner_df):
        """ レーステーブルに必要な処理をまとめたもの、最終的にrace_dfに格納する。処理の流れはカラム選択→特徴良作成→エンコーディング """
        race_df = self.tf.choose_race_result_column(race_base_df)
        race_df = self.tf.create_feature_race_df(race_df)
        race_df = self.tf.create_feature_race_result_df(race_df, race_winner_df)
        race_df = self.tf.encode_race_df(race_df)
        print("_proc_race_result_df: race_df", race_df.shape)
        return race_df.copy()

    def _proc_raceuma_result_df(self, race_result_df, raceuma_base_df):
        """  レース馬テーブルに必要な処理をまとめたもの、最終的にraceuma_dfに格納する。処理の流れはカラム選択→エンコーディング→平準化＆標準化 """
        raceuma_df = self.tf.choose_raceuma_result_column(raceuma_base_df)
        raceuma_df = self.tf.encode_raceuma_result_df(raceuma_df, self.dict_folder)
        raceuma_df = self.tf.normalize_raceuma_result_df(raceuma_df)
        raceuma_df = self.tf.standardize_raceuma_result_df(raceuma_df)
        raceuma_df = self.tf.create_feature_raceuma_result_df(race_result_df, raceuma_df)
        raceuma_df = self._proc_scale_df_for_fa(raceuma_df)
        raceuma_df = self.tf.factory_analyze_raceuma_result_df(race_result_df, raceuma_df, self.dict_folder)
        print("_proc_raceuma_result_df: raceuma_df", raceuma_df.shape)
        return raceuma_df.copy()

    def _proc_scale_df_for_fa(self, raceuma_df):
        print("-- check! this is BaseLoad class: " + sys._getframe().f_code.co_name)

    def _get_prev_df(self, num, race_result_df, raceuma_result_df):
        print("-- check! this is BaseLoad class: " + sys._getframe().f_code.co_name)

    def _set_grouped_raceuma_prev_df(self, race_result_df, raceuma_result_df):
        print("-- check! this is BaseLoad class: " + sys._getframe().f_code.co_name)

    def set_result_df(self):
        """ result_dfを作成するための処理。result_dfに処理がされたデータをセットする """
        result_race_df = self.ext.get_race_table_base()
        result_raceuma_df = self.ext.get_raceuma_table_base()
        self.result_df = self._proc_result_df(result_race_df, result_raceuma_df)

    def _proc_result_df(self, result_race_df, result_raceuma_df):
        print("-- check! this is BaseLoad class: " + sys._getframe().f_code.co_name)
