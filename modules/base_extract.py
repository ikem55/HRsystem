import pandas as pd
import pyodbc
import sys

class BaseExtract(object):
    """
    データ抽出に関する共通モデル
    """
    mock_flag = False
    """ mockデータを使うかの判断に使用するフラグ。Trueの場合はMockデータを使う """
    mock_path = '../mock_data/base/'
    """ mockファイルが格納されているフォルダのパス """

    def __init__(self, start_date, end_date, mock_flag):
        self.start_date = start_date
        self.end_date = end_date
        if mock_flag:
            self.set_mock_path()
            self.mock_flag = mock_flag

    def set_mock_path(self):
        """ mock_flagをTrueにしてmockのパスを設定する。  """
        self.mock_path_race = self.mock_path + 'race.pkl'
        self.mock_path_raceuma = self.mock_path + 'raceuma.pkl'
        self.mock_path_bet = self.mock_path + 'bet.pkl'
        self.mock_path_haraimodoshi = self.mock_path + 'haraimodoshi.pkl'
        self.mock_path_zandaka = self.mock_path + 'zandaka.pkl'
        self.mock_path_horse = self.mock_path + 'horse.pkl'
        self.mock_path_mydb = self.mock_path + 'mydb.pkl'
        self.mock_path_tansho = self.mock_path + 'tansho.pkl'
        self.mock_path_fukusho = self.mock_path + 'fukusho.pkl'
        self.mock_path_umaren = self.mock_path + 'umaren.pkl'
        self.mock_path_umatan = self.mock_path + 'umatan.pkl'
        self.mock_path_wide = self.mock_path + 'wide.pkl'
        self.mock_path_sanrenpuku = self.mock_path + 'sanrenpuku.pkl'

    def create_mock_data(self):
        """ mock dataを作成する  """
        self.mock_flag = False
        race_df = self.get_race_table_base()
        raceuma_df = self.get_raceuma_table_base()
        bet_df = self.get_bet_table_base()
        haraimodoshi_df = self.get_haraimodoshi_table_base()
        zandaka_df = self.get_zandaka_table_base()
        horse_df = self.get_horse_table_base()
        mydb_df = self.get_mydb_table_base()
        tansho_df = self.get_tansho_table_base()
        fukusho_df = self.get_fukusho_table_base()
        umaren_df = self.get_umaren_table_base()
        umatan_df = self.get_umatan_table_base()
        wide_df = self.get_wide_table_base()
        sanrenpuku_df = self.get_sanrenpuku_table_base()

        self.set_mock_path()
        race_df.to_pickle(self.mock_path_race)
        raceuma_df.to_pickle(self.mock_path_raceuma)
        bet_df.to_pickle(self.mock_path_bet)
        haraimodoshi_df.to_pickle(self.mock_path_haraimodoshi)
        zandaka_df.to_pickle(self.mock_path_zandaka)
        horse_df.to_pickle(self.mock_path_horse)
        mydb_df.to_pickle(self.mock_path_mydb)
        tansho_df.to_pickle(self.mock_path_tansho)
        fukusho_df.to_pickle(self.mock_path_fukusho)
        umaren_df.to_pickle(self.mock_path_umaren)
        umatan_df.to_pickle(self.mock_path_umatan)
        wide_df.to_pickle(self.mock_path_wide)
        sanrenpuku_df.to_pickle(self.mock_path_sanrenpuku)

    def get_bet_table_base(self):
        print("-- check! this is BaseExtract class: " + sys._getframe().f_code.co_name)

    def get_haraimodoshi_table_base(self):
        print("-- check! this is BaseExtract class: " + sys._getframe().f_code.co_name)

    def get_zandaka_table_base(self):
        print("-- check! this is BaseExtract class: " + sys._getframe().f_code.co_name)

    def get_mydb_table_base(self):
        print("-- check! this is BaseExtract class: " + sys._getframe().f_code.co_name)

    def get_race_table_base(self):
        print("-- check! this is BaseExtract class: " + sys._getframe().f_code.co_name)

    def get_race_before_table_base(self):
        print("-- check! this is BaseExtract class: " + sys._getframe().f_code.co_name)

    def get_raceuma_table_base(self):
        print("-- check! this is BaseExtract class: " + sys._getframe().f_code.co_name)

    def get_raceuma_before_table_base(self):
        print("-- check! this is BaseExtract class: " + sys._getframe().f_code.co_name)

    def get_horse_table_base(self):
        print("-- check! this is BaseExtract class: " + sys._getframe().f_code.co_name)

    def get_tansho_table_base(self):
        print("-- check! this is BaseExtract class: " + sys._getframe().f_code.co_name)

    def get_fukusho_table_base(self):
        print("-- check! this is BaseExtract class: " + sys._getframe().f_code.co_name)

    def get_umaren_table_base(self):
        print("-- check! this is BaseExtract class: " + sys._getframe().f_code.co_name)

    def get_umatan_table_base(self):
        print("-- check! this is BaseExtract class: " + sys._getframe().f_code.co_name)

    def get_wide_table_base(self):
        print("-- check! this is BaseExtract class: " + sys._getframe().f_code.co_name)

    def get_sanrenpuku_table_base(self):
        print("-- check! this is BaseExtract class: " + sys._getframe().f_code.co_name)

