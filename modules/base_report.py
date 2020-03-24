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
        self._set_predata()

    def _set_predata(self):
        self._set__bet_df()
        self._set_haraimodoshi_dict()
        self._set_raceuma_df()
        self._check_result_data()

    def _check_result_data(self):
        if len(self.bet_df[self.bet_df["日付"] == self.end_date]) != 0:
            self.check_flag = True
        else:
            self.check_flag = False

    def _set__bet_df(self):
        base_bet_df = self.ext.get_bet_table_base()
        bet_df = base_bet_df[["式別", "日付", "結果", "金額"]].copy()
        bet_df.loc[:, "結果"] = bet_df["結果"] * bet_df["金額"] / 100
        self.bet_df = bet_df

    def _set_haraimodoshi_dict(self):
        base_haraimodoshi_df = self.ext.get_haraimodoshi_table_base()
        end_date = self.end_date
        self.todays_haraimodoshi_dict = mu.get_haraimodoshi_dict(base_haraimodoshi_df.query("データ作成年月日 == @end_date"))
        self.haraimodoshi_dict = mu.get_haraimodoshi_dict(base_haraimodoshi_df)

    def _set_raceuma_df(self):
        base_race_df = self.ext.get_race_table_base()
        base_raceuma_df = self.ext.get_raceuma_table_base()
        self.race_df = base_race_df[["競走コード", "データ区分", "月日", "距離", "競走番号", "場名", "発走時刻", "投票フラグ"]]
        raceuma_df = base_raceuma_df[["競走コード", "馬番", "年月日", "馬券評価順位", "得点", "単勝オッズ", "単勝人気", "確定着順", "単勝配当", "複勝配当", "デフォルト得点順位"]].copy()
        raceuma_df.loc[:, "ck1"] = raceuma_df["確定着順"].apply(lambda x: 1 if x == 1 else 0)
        raceuma_df.loc[:, "ck2"] = raceuma_df["確定着順"].apply(lambda x: 1 if x == 2 else 0)
        raceuma_df.loc[:, "ck3"] = raceuma_df["確定着順"].apply(lambda x: 1 if x == 3 else 0)
        raceuma_df.loc[:, "ckg"] = raceuma_df["確定着順"].apply(lambda x: 1 if x > 3 else 0)
        self.raceuma_df = pd.merge(raceuma_df, self.race_df[["競走コード", "場名", "距離", "データ区分"]], on="競走コード")

    def get_filterd_bet_df(self, n_value = 200):
        tansho_df = self.bet_df[self.bet_df["式別"] == 1].sort_values("日付").tail(n_value)
        umaren_df = self.bet_df[self.bet_df["式別"] == 5].sort_values("日付").tail(n_value)
        umatan_df = self.bet_df[self.bet_df["式別"] == 6].sort_values("日付").tail(n_value)
        wide_df = self.bet_df[self.bet_df["式別"] == 7].sort_values("日付").tail(n_value)
        sanrenpuku_df = self.bet_df[self.bet_df["式別"] == 8].sort_values("日付").tail(n_value)
        filter_bet_df = pd.concat([tansho_df, umaren_df, umatan_df, wide_df, sanrenpuku_df])
        return filter_bet_df

    def _get_bet_summary_df(self, bet_df):
        today_bet_df = bet_df.groupby("式別").sum()
        today_all_bet_df = today_bet_df.sum()
        today_all_bet_df.name = 0
        today_bet_df = today_bet_df.append(today_all_bet_df)
        today_bet_df.loc[:, "回収率"] = today_bet_df["結果"] / today_bet_df["金額"] * 100
        today_bet_df.reset_index(inplace=True)
        return today_bet_df

    def get_todays_bet_text(self):
        bet_text = '== 本日結果 == \r\n'
        today_bet_df = self._get_bet_summary_df(self.bet_df[self.bet_df["日付"] == self.end_date])
        bet_text += self._get_bet_text(today_bet_df)
        return bet_text

    def get_recent_bet_text(self):
        bet_text = '== 直近結果 == \r\n'
        recent_bet_df = self._get_bet_summary_df(self.bet_df)
        bet_text += self._get_bet_text(recent_bet_df)
        return bet_text

    def _get_bet_text(self, bet_df):
        bet_text =''
        for index, row in bet_df.iterrows():
            bet_text += mu.trans_baken_type(row['式別']) + ' ' + round(row['回収率']).astype('str') + '% (' + round(
                row['結果']).astype('str') + ' / ' + round(row['金額']).astype('str') + ')\r\n'
        return bet_text

    def _get_todays_df(self):
        race_df = self.race_df[self.race_df["月日"] == self.end_date][["データ区分", "場名", "発走時刻", "投票フラグ"]]
        end_date = self.end_date
        raceuma_df = self.raceuma_df.query("年月日 == @end_date & データ区分 == '7'")[["馬券評価順位", "単勝オッズ", "単勝人気", "確定着順", "単勝配当"
            , "複勝配当", "デフォルト得点順位", "ck1", "ck2", "ck3", "ckg"]]
        return race_df, raceuma_df

    def get_current_text(self):
        current_text = ''
        race_df, raceuma_df = self._get_todays_df()
        kaisai_text = str(set(race_df["場名"]))
        race_status = race_df["データ区分"].value_counts()
        all_race_count = ''
        for key, val in race_status.iteritems():
            all_race_count += 'St' + str(key) + ':' + str(val) + 'R | '
        final_race_time = ' 最終レース:' + race_df["発走時刻"].max().strftime('%H:%M')
        current_text += '== 開催情報(' + dt.now().strftime('%Y/%m/%d %H') + '時点情報) ==\r\n'
        current_text += ' 開催場所: ' + kaisai_text + '\r\n'
        current_text += ' レース進捗：' + all_race_count + '\r\n'
        current_text += final_race_time + '\r\n'
        return current_text

    def get_trend_text(self):
        trend_text = '== レース配当トレンド == \r\n'
        tansho_df = self.haraimodoshi_dict["tansho_df"]
        kpi_tansho_df = tansho_df.query("払戻 >= 1000")
        kpi_tansho = round(len(kpi_tansho_df)/ len(tansho_df) * 100 , 1)
        kpi_tansho_cnt = len(kpi_tansho_df)
        avg_tansho = round(tansho_df["払戻"].mean())

        td_tansho_df = self.todays_haraimodoshi_dict["tansho_df"]
        td_kpi_tansho_df = td_tansho_df.query("払戻 >= 1000")
        td_kpi_tansho = round(len(td_kpi_tansho_df)/ len(td_tansho_df) * 100 , 1)
        td_kpi_tansho_cnt = len(td_kpi_tansho_df)
        td_avg_tansho = round(td_tansho_df["払戻"].mean())
        trend_text += ' 単勝平均: ' + str(td_avg_tansho) + '(' + str(td_kpi_tansho) + '%, ' + str(td_kpi_tansho_cnt) + '件)\r\n ーー平均: '\
                      + str(avg_tansho) + '(' + str(kpi_tansho) + '%, ' + str(kpi_tansho_cnt) +'件)\r\n'

        umaren_df = self.haraimodoshi_dict["umaren_df"]
        kpi_umaren_df = umaren_df.query("払戻 >= 5000")
        kpi_umaren = round(len(kpi_umaren_df)/ len(umaren_df) * 100 , 1)
        kpi_umaren_cnt = len(kpi_umaren_df)
        avg_umaren = round(umaren_df["払戻"].mean())
        td_umaren_df = self.todays_haraimodoshi_dict["umaren_df"]
        td_kpi_umaren_df = td_umaren_df.query("払戻 >= 5000")
        td_kpi_umaren = round(len(td_kpi_umaren_df)/ len(td_umaren_df) * 100 , 1)
        td_kpi_umaren_cnt = len(td_kpi_umaren_df)
        td_avg_umaren = round(td_umaren_df["払戻"].mean())
        trend_text += ' 馬連平均: ' + str(td_avg_umaren) + '(' + str(td_kpi_umaren) + '%, ' + str(td_kpi_umaren_cnt) + '件)\r\n ーー平均: '\
                      + str(avg_umaren) + '(' + str(kpi_umaren) + '%, ' + str(kpi_umaren_cnt) +'件)\r\n'

        umatan_df = self.haraimodoshi_dict["umatan_df"]
        kpi_umatan_df = umatan_df.query("払戻 >= 5000")
        kpi_umatan = round(len(kpi_umatan_df) / len(umatan_df) * 100, 1)
        kpi_umatan_cnt = len(kpi_umatan_df)
        avg_umatan = round(umatan_df["払戻"].mean())
        td_umatan_df = self.todays_haraimodoshi_dict["umatan_df"]
        td_kpi_umatan_df = td_umatan_df.query("払戻 >= 5000")
        td_kpi_umatan = round(len(td_kpi_umatan_df) / len(td_umatan_df) * 100, 1)
        td_kpi_umatan_cnt = len(td_kpi_umatan_df)
        td_avg_umatan = round(td_umatan_df["払戻"].mean())
        trend_text += ' 馬単平均: ' + str(td_avg_umatan) + '(' + str(td_kpi_umatan) + '%, ' + str(td_kpi_umatan_cnt) + '件)\r\n ーー平均: '\
                      + str(avg_umatan) + '(' + str(kpi_umatan) + '%, ' + str(kpi_umatan_cnt) +'件)\r\n'


        wide_df = self.haraimodoshi_dict["wide_df"]
        kpi_wide_df = wide_df.query("払戻 >= 3500")
        kpi_wide = round(len(kpi_wide_df) / len(wide_df) * 100, 1)
        kpi_wide_cnt = len(kpi_wide_df)
        avg_wide = round(wide_df["払戻"].mean())
        td_wide_df = self.todays_haraimodoshi_dict["wide_df"]
        td_kpi_wide_df = td_wide_df.query("払戻 >= 3500")
        td_kpi_wide = round(len(td_kpi_wide_df) / len(td_wide_df) * 100, 1)
        td_kpi_wide_cnt = len(td_kpi_wide_df)
        td_avg_wide = round(td_wide_df["払戻"].mean())
        trend_text += ' ワイド平均: ' + str(td_avg_wide) + '(' + str(td_kpi_wide) + '%, ' + str(td_kpi_wide_cnt) + '件)\r\n ーーー平均: '\
                      + str(avg_wide) + '(' + str(kpi_wide) + '%, ' + str(kpi_wide_cnt) +'件)\r\n'

        sanrenpuku_df = self.haraimodoshi_dict["sanrenpuku_df"]
        kpi_sanrenpuku_df = sanrenpuku_df.query("払戻 >= 7500")
        kpi_sanrenpuku = round(len(kpi_sanrenpuku_df) / len(sanrenpuku_df) * 100, 1)
        kpi_sanrenpuku_cnt = len(kpi_sanrenpuku_df)
        avg_sanrenpuku = round(sanrenpuku_df["払戻"].mean())
        td_sanrenpuku_df = self.todays_haraimodoshi_dict["sanrenpuku_df"]
        td_kpi_sanrenpuku_df = td_sanrenpuku_df.query("払戻 >= 7500")
        td_kpi_sanrenpuku = round(len(td_kpi_sanrenpuku_df) / len(td_sanrenpuku_df) * 100, 1)
        td_kpi_sanrenpuku_cnt = len(td_kpi_sanrenpuku_df)
        td_avg_sanrenpuku = round(td_sanrenpuku_df["払戻"].mean())
        trend_text += ' 三連複平均: ' + str(td_avg_sanrenpuku) + '(' + str(td_kpi_sanrenpuku) + '%, ' + str(td_kpi_sanrenpuku_cnt) + '件)\r\n ーーー平均: '\
                      + str(avg_sanrenpuku) + '(' + str(kpi_sanrenpuku) + '%, ' + str(kpi_sanrenpuku_cnt) +'件)\r\n'

        return trend_text

    def get_summary_text(self):
        summary_text = '== KPI集計結果 == \r\n'
        race_df, raceuma_df = self._get_todays_df()
        score_raceuma_df = raceuma_df.query("馬券評価順位 == 1")
        default_raceuma_df = raceuma_df.query("デフォルト得点順位 == 1")
        ninki_raceuma_df = raceuma_df.query("単勝人気 == 1")
        score_result_txt = self._calc_raceuma_result(score_raceuma_df)
        default_result_txt = self._calc_raceuma_result(default_raceuma_df)
        ninki_result_txt = self._calc_raceuma_result(ninki_raceuma_df)

        total_score_raceuma_df = self.raceuma_df.query("馬券評価順位 == 1")
        total_default_raceuma_df = self.raceuma_df.query("デフォルト得点順位 == 1")
        total_ninki_raceuma_df = self.raceuma_df.query("単勝人気 == 1")
        total_score_result_txt = self._calc_raceuma_result(total_score_raceuma_df)
        total_default_result_txt = self._calc_raceuma_result(total_default_raceuma_df)
        total_ninki_result_txt = self._calc_raceuma_result(total_ninki_raceuma_df)

        summary_text += '馬券評価順位１位' + "\r\n" + score_result_txt + "\r\n" + total_score_result_txt + "\r\n"
        summary_text += 'デフォルト得点１位' + "\r\n" + default_result_txt + "\r\n" + total_default_result_txt + "\r\n"
        summary_text += '一番人気' + "\r\n" + ninki_result_txt + "\r\n" + total_ninki_result_txt + "\r\n"
        return summary_text

    def _calc_raceuma_result(self, df):
        summary_df = df.describe()
        sum_df = df[["ck1", "ck2", "ck3", "ckg"]].sum()
        av_ninki = round(summary_df["単勝人気"]["mean"], 1)
        av_chakujn = round(summary_df["確定着順"]["mean"], 1)
        tansho_return = round(summary_df["単勝配当"]["mean"], 1)
        fukusho_return = round(summary_df["複勝配当"]["mean"], 1)
        chaku_text = str(sum_df["ck1"])
        for key, val in sum_df.iteritems():
            if key != 'ck1':
                chaku_text += '-' + str(val)
        res_text = ' Av着:' + str(av_chakujn) + '(' + chaku_text + ') 単:' + str(tansho_return) + ' 複:' + str(fukusho_return) + ' Av人:' + str(av_ninki)
        return res_text


class LBReport(BaseReport):
    def __init__(self, start_date, end_date, mock_flag):
        self.start_date = start_date
        self.end_date = end_date
        self.ext = LBExtract(start_date, end_date, mock_flag)
        self._set_predata()

"""
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 200)
print("test")
start_date = '2020/03/22'
end_date = '2020/03/22'
mock_flag = False
test = LBReport(start_date,end_date,mock_flag)
#print(test.race_df)
#print(test.raceuma_df)
#print(test.haraimodoshi_dict)
#print(test.bet_df)
current_text = test.get_current_text()
print(current_text)
bet_text = test.get_bet_text()
print(bet_text)
summary_text = test.get_summary_text()
print(summary_text)
trend_text = test.get_trend_text()
print(trend_text)
"""
