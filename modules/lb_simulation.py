from modules.lb_extract import LBExtract
from modules.base_simulation import BaseSimulation

import modules.util as mu

class LBSimulation(BaseSimulation):
    """
    地方競馬用馬券シミュレーションに関する処理をまとめたクラス
    """

    def _get_extract_object(self, start_date, end_date, mock_flag):
        """ 利用するExtクラスを指定する """
        ext = LBExtract(start_date, end_date, mock_flag)
        return ext

    def _set_haraimodoshi_dict(self):
        haraimodoshi_df = self.ext.get_haraimodoshi_table_base()
        self.dict_haraimodoshi = mu.get_haraimodoshi_dict(haraimodoshi_df)

    def _set_tansho_df(self):
        """ tansho_df[umaban] """
        base_df = self.ext.get_tansho_table_base()
        odds_df = self._add_odds_list(base_df)
        self.tansho_df = odds_df[["データ作成年月日", "競走コード", "票数合計", "odds"]].copy()

    def _set_fukusho_df(self):
        """ fukusho_df[umaban] """
        base_df = self.ext.get_fukusho_table_base()
        odds_df = self._add_odds_list(base_df, fuku=True)
        self.fukusho_df = odds_df[["データ作成年月日", "競走コード", "票数合計", "odds"]].copy()

    def _set_umaren_df(self):
        """ umaren_df[umaban1][umaban2] """
        base_df = self.ext.get_umaren_table_base()
        odds_df = self._add_odds_array(base_df, 6)
        self.umaren_df = odds_df[["データ作成年月日", "競走コード", "票数合計", "odds"]].copy()

    def _set_wide_df(self):
        """ wide_df[umaban1][umaban2] """
        base_df = self.ext.get_wide_table_base()
        odds_df = self._add_odds_array(base_df, 5, fuku=True)
        self.wide_df = odds_df[["データ作成年月日", "競走コード", "票数合計", "odds"]].copy()

    def _set_umatan_df(self):
        """ umatan_df[umaban1][umaban2] """
        base_df = self.ext.get_umatan_table_base()
        odds_df = self._add_odds_array(base_df, 6, combi=False)
        self.umatan_df = odds_df[["データ作成年月日", "競走コード", "票数合計", "odds"]].copy()

    def _set_sanrenpuku_df(self):
        """ sanrenpuku_df[umaban1][umaban2][umaban3] """
        base_df = self.ext.get_sanrenpuku_table_base()
        odds_df = self._add_odds_panel(base_df, 6)
        self.sanrenpuku_df = odds_df[["データ作成年月日", "競走コード", "票数合計", "odds"]].copy()

