from modules.base_extract import BaseExtract
from modules.lb_extract import LBExtract
import modules.util as mu

import pandas as pd
from datetime import datetime as dt


class BaseReport(object):
    def __init__(self, start_date, end_date, mock_flag):
        self.start_date = start_date
        self.end_date = end_date
        self.ext = BaseExtract(start_date, end_date, mock_flag)

    def get_bet_df(self):
        base_bet_df = self.ext.get_bet_table_base()
        bet_df = base_bet_df[["式別", "日付", "結果", "金額"]].copy()
        bet_df.loc[:, "結果"] = bet_df["結果"] * bet_df["金額"] / 100
        return bet_df

    def get_filterd_bet_df(self, bet_df, n_value = 200):
        tansho_df = bet_df[bet_df["式別"] == 1].sort_values("日付").tail(n_value)
        umaren_df = bet_df[bet_df["式別"] == 5].sort_values("日付").tail(n_value)
        umatan_df = bet_df[bet_df["式別"] == 6].sort_values("日付").tail(n_value)
        wide_df = bet_df[bet_df["式別"] == 7].sort_values("日付").tail(n_value)
        sanrenpuku_df = bet_df[bet_df["式別"] == 8].sort_values("日付").tail(n_value)
        filter_bet_df = pd.concat([tansho_df, umaren_df, umatan_df, wide_df, sanrenpuku_df])
        return filter_bet_df

    def get_todays_bet_df(self, bet_df):
        today_bet_df = bet_df[bet_df["日付"] == self.end_date].groupby("式別").sum()
        today_all_bet_df = today_bet_df.sum()
        today_all_bet_df.name = 0
        today_bet_df = today_bet_df.append(today_all_bet_df)
        today_bet_df.loc[:, "回収率"] = today_bet_df["結果"] / today_bet_df["金額"] * 100
        today_bet_df.reset_index(inplace=True)
        return today_bet_df

    def get_todays_text(self, today_bet_df):
        bet_text = dt.now().strftime('%Y/%m/%d %H:%M:%S') + '\r\n'
        for index, row in today_bet_df.iterrows():
            bet_text += mu.trans_baken_type(row['式別']) + ' ' + round(row['回収率']).astype('str') + '% (' + round(
                row['結果']).astype('str') + ' / ' + round(row['金額']).astype('str') + ')\r\n'
        return bet_text


class LBReport(BaseReport):
    def __init__(self, start_date, end_date, mock_flag):
        self.start_date = start_date
        self.end_date = end_date
        self.ext = LBExtract(start_date, end_date, mock_flag)

