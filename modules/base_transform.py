import os
import modules.util as mu

import sys

class BaseTransform(object):
    """
    データ変換に関する共通処理を定義する
    辞書データの格納等の処理があるため、基本的にはインスタンス化して実行するが、不要な場合はクラスメソッドでの実行を可とする。
    辞書データの作成は、ない場合は作成、ある場合は読み込み、を基本方針とする(learning_modeによる判別はしない）
    """

    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date

    
    def factory_analyze_raceuma_result_df(self, race_df, input_raceuma_df, dict_folder):
        print("-- check! this is BaseTransform class: " + sys._getframe().f_code.co_name)

    
    def create_feature_race_base_df(self, race_df):
        print("-- check! this is BaseTransform class: " + sys._getframe().f_code.co_name)

    
    def create_feature_race_df(self, race_df):
        print("-- check! this is BaseTransform class: " + sys._getframe().f_code.co_name)

    
    def create_feature_raceuma_base_df(self, raceuma_df):
        print("-- check! this is BaseTransform class: " + sys._getframe().f_code.co_name)

    
    def group_prev_raceuma_df(self, raceuma_prev_df, raceuma_base_df):
        print("-- check! this is BaseTransform class: " + sys._getframe().f_code.co_name)

    
    def choose_race_result_column(self, race_df):
        print("-- check! this is BaseTransform class: " + sys._getframe().f_code.co_name)

    def create_feature_race_result_df(self, race_df, race_winner_df):
        print("-- check! this is BaseTransform class: " + sys._getframe().f_code.co_name)

    
    def encode_race_df(self, race_df):
        print("-- check! this is BaseTransform class: " + sys._getframe().f_code.co_name)

    
    def choose_raceuma_result_column(self, raceuma_df):
        print("-- check! this is BaseTransform class: " + sys._getframe().f_code.co_name)

    
    def encode_raceuma_result_df(self, raceuma_df, dict_folder):
        print("-- check! this is BaseTransform class: " + sys._getframe().f_code.co_name)

    
    def normalize_raceuma_result_df(self, raceuma_df):
        print("-- check! this is BaseTransform class: " + sys._getframe().f_code.co_name)

    
    def standardize_raceuma_result_df(self, raceuma_df):
        print("-- check! this is BaseTransform class: " + sys._getframe().f_code.co_name)

    
    def create_feature_raceuma_result_df(self,  race_df, raceuma_df):
        print("-- check! this is BaseTransform class: " + sys._getframe().f_code.co_name)

    
    def encode_raceuma_before_df(self, raceuma_df, dict_folder):
        print("-- check! this is BaseTransform class: " + sys._getframe().f_code.co_name)

    
    def normalize_raceuma_df(self, raceuma_df):
        print("-- check! this is BaseTransform class: " + sys._getframe().f_code.co_name)

    
    def standardize_raceuma_df(self, raceuma_df):
        print("-- check! this is BaseTransform class: " + sys._getframe().f_code.co_name)

    
    def create_feature_raceuma_df(self, raceuma_df):
        print("-- check! this is BaseTransform class: " + sys._getframe().f_code.co_name)

    
    def drop_columns_raceuma_df(self, raceuma_df):
        print("-- check! this is BaseTransform class: " + sys._getframe().f_code.co_name)

    
    def choose_horse_column(self, horse_df):
        print("-- check! this is BaseTransform class: " + sys._getframe().f_code.co_name)

    
    def normalize_prev_merged_df(self, horse_df):
        print("-- check! this is BaseTransform class: " + sys._getframe().f_code.co_name)

    
    def choose_upper_n_count(self, df, column_name, n, dict_folder):
        """ 指定したカラム名の上位N出現以外をその他にまとめる

        :param df:
        :param column_name:
        :param n:
        :return: df
        """
        dict_name = "choose_upper_" + str(n) + "_" + column_name
        file_name = dict_folder + dict_name + ".pkl"
        if os.path.exists(file_name):
            temp_df = mu.load_dict(dict_name, dict_folder)
        else:
            temp_df = df[column_name].value_counts().iloc[:n].index
            mu.save_dict(temp_df, dict_name, dict_folder)
        df.loc[:, column_name] = df[column_name].apply(lambda x: x if x in temp_df else 'その他')
        return df