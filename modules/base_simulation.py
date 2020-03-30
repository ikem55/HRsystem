from modules.base_extract import BaseExtract

import sys
import pandas as pd
import numpy as np
import re
import itertools

class BaseSimulation(object):
    """
    馬券シミュレーションに関する処理をまとめたクラス
    """

    def __init__(self, start_date, end_date, mock_flag):
        self.start_date = start_date
        self.end_date = end_date
        self.mock_flag = mock_flag
        self.ext = self._get_extract_object(start_date, end_date, mock_flag)

    def set_raceuma_df(self, raceuma_df):
        self.raceuma_df = raceuma_df

    def _get_extract_object(self, start_date, end_date, mock_flag):
        """ 利用するExtクラスを指定する """
        ext = BaseExtract(start_date, end_date, mock_flag)
        return ext

    def _add_odds_list(self, df, fuku=False):
        df = df.assign(odds = "")
        for index, row in df.iterrows():
            odds_list = []
            for i in range(4):
                odds_list.append(row["全オッズ"][i::4])
            odds_list = ["".join(i) for i in zip(*odds_list)]
            odds_list = [int(s)/ 10 for s in odds_list]
            if fuku:
                odds_list = odds_list[0::2]
            odds_list.insert(0,0)
            df["odds"][index] = odds_list
        return df

    def _add_odds_array(self, df, n, combi=True, fuku=False):
        df = df.assign(odds="")
        empty_check = ' ' * n
        for index, row in df.iterrows():
            odds_list = []
            tosu = row["登録頭数"] + 1
            for i in range(n):
                odds_list.append(row["全オッズ"][i::n])
            odds_list = ["".join(i) for i in zip(*odds_list)]
            odds_list = [i for i in odds_list if i != empty_check]
            odds_list = [int(s)/ 10 for s in odds_list]
            if fuku:
                odds_list = odds_list[0::2]
            odds_array = np.zeros((tosu, tosu))
            idx = 0
            if combi:
                for element in itertools.combinations(range(1, tosu), 2):
                    odds_array[element[0], element[1]] = odds_list[idx]
                    odds_array[element[1], element[0]] = odds_list[idx]
                    idx = idx + 1
            else:
                for element in itertools.permutations(range(1, tosu), 2):
                    odds_array[element[0], element[1]] = odds_list[idx]
                    idx = idx + 1
            df["odds"][index] = odds_array
        return df

    def _add_odds_panel(self, df, n, combi=True):
        df = df.assign(odds="")
        empty_check = ' ' * n
        for index, row in df.iterrows():
            odds_list = []
            tosu = row["登録頭数"] + 1
            for i in range(n):
                odds_list.append(row["全オッズ"][i::n])
            odds_list = ["".join(i) for i in zip(*odds_list)]
            odds_list = [i for i in odds_list if i != empty_check]
            odds_list = [int(s)/ 10 for s in odds_list]
            odds_array = np.zeros((tosu, tosu, tosu))
            idx = 0
            if combi:
                for element in itertools.combinations(range(1, tosu), 3):
                    odds_array[element[0], element[1], element[2]] = odds_list[idx]
                    odds_array[element[0], element[2], element[1]] = odds_list[idx]
                    odds_array[element[1], element[0], element[2]] = odds_list[idx]
                    odds_array[element[1], element[2], element[0]] = odds_list[idx]
                    odds_array[element[2], element[0], element[1]] = odds_list[idx]
                    odds_array[element[2], element[1], element[0]] = odds_list[idx]
                    idx = idx + 1
            else:
                for element in itertools.permutations(range(1, tosu), 3):
                    odds_array[element[0], element[1], element[2]] = odds_list[idx]
                    idx = idx + 1
            df["odds"][index] = odds_array
        return df

    def get_umaren_kaime(self, df1, df2):
        """ df1とdf2の組み合わせの馬連の買い目リストを作成, dfは競走コード,馬番のセット """
        # df1の馬番を横持に変換
        df1_gp = df1.groupby("競走コード")["馬番"].apply(list)
        df2_gp = df2.groupby("競走コード")["馬番"].apply(list)
        merge_df = pd.merge(df1_gp, df2_gp, on="競走コード")
        merge_df = pd.merge(merge_df, self.umaren_df, on="競走コード")
        race_key_list = []
        umaban_list = []
        odds_list = []
        for index, row in merge_df.iterrows():
            uma1 = row["馬番_x"]
            uma2 = [i for i in row["馬番_y"] if i not in uma1]
            for element in itertools.product(uma1, uma2):
                odds = row["odds"][element[0]][element[1]]
                race_key_list += [row["競走コード"]]
                umaban_list += [sorted(element)] # 連系なのでソート
                odds_list += [odds]
        kaime_df = pd.DataFrame(
            data={"競走コード": race_key_list, "馬番": umaban_list, "オッズ": odds_list},
            columns=["競走コード","馬番","オッズ"]
        )
        kaime_df = kaime_df[kaime_df["オッズ"] != 0]
        return kaime_df

    def check_result_kaime(self, kaime_df, result_df):
        """ 買い目DFと的中結果を返す """
        kaime_df["馬番"] = kaime_df["馬番"].apply(lambda x: ', '.join(map(str, x)))
        result_df["馬番"] = result_df["馬番"].apply(lambda x: ', '.join(map(str, x)))
        merge_df = pd.merge(kaime_df, result_df, on=["競走コード", "馬番"], how="left").fillna(0)
        return merge_df

    def calc_summary(self, df, cond_text):
        all_count = len(df)
        race_count = len(df["競走コード"].drop_duplicates())
        hit_df = df[df["払戻"] != 0]
        hit_count = len(hit_df)
        avg_return = round(hit_df["払戻"].mean(), 0)
        std_return = round(hit_df["払戻"].std(), 0)
        max_return = hit_df["払戻"].max()
        sum_return = hit_df["払戻"].sum()
        avg = round(df["払戻"].mean() , 1)
        hit_rate = round(hit_count / all_count * 100 , 1)
        race_hit_rate = round(hit_count / race_count * 100 , 1)
        sr = pd.Series(data=[cond_text, all_count, hit_count, race_count, avg, hit_rate, race_hit_rate, avg_return, std_return, max_return, all_count * 100 , sum_return]
                       , index=["条件", "件数", "的中数", "レース数", "回収率", "的中率", "R的中率", "払戻平均", "払戻偏差", "最大払戻", "購入総額", "払戻総額"])
        return sr

    def simulation_umaren(self, cond1, cond2, filter_odds_low, filter_odds_high):
        self.sim_umaren()
        df1 = self.raceuma_df.query(cond1)[["競走コード", "馬番"]]
        df2 = self.raceuma_df.query(cond2)[["競走コード", "馬番"]]
        umaren_kaime_df = self.get_umaren_kaime(df1, df2)
        umaren_kaime_df = umaren_kaime_df[(umaren_kaime_df["オッズ"] >= filter_odds_low) & (umaren_kaime_df["オッズ"] <= filter_odds_high)]
        check_umaren_df = self.check_result_kaime(umaren_kaime_df, self.result_umaren_df)
        cond_text = "馬1: " + cond1 + " 馬2:" + cond2
        sr = self.calc_summary(check_umaren_df, cond_text)
        return sr


    def sim_umaren(self):
        self._set_haraimodoshi_dict()
        self.result_umaren_df = self.dict_haraimodoshi["umaren_df"]
        self._set_umaren_df()

    def _set_haraimodoshi_dict(self):
        print("-- check! this is BaseExtract class: " + sys._getframe().f_code.co_name)

    def _set_tansho_df(self):
        print("-- check! this is BaseExtract class: " + sys._getframe().f_code.co_name)

    def _set_fukusho_df(self):
        print("-- check! this is BaseExtract class: " + sys._getframe().f_code.co_name)

    def _set_umaren_df(self):
        print("-- check! this is BaseExtract class: " + sys._getframe().f_code.co_name)

    def _set_wide_df(self):
        print("-- check! this is BaseExtract class: " + sys._getframe().f_code.co_name)

    def _set_umatan_df(self):
        print("-- check! this is BaseExtract class: " + sys._getframe().f_code.co_name)

    def _set_sanrenpuku_df(self):
        print("-- check! this is BaseExtract class: " + sys._getframe().f_code.co_name)



"""

        sr_odds_text = df["全オッズ"]
        print(sr_odds_text)
        print(reg_text)
        print(type(reg_text))
        print('([0-9]{4})([0-9]{4})([0-9]{4})')
        print(type('([0-9]{4})([0-9]{4})([0-9]{4})'))
        odds_list_df = sr_odds_text.str.extract(reg_text, expand=True)
        #        odds_list_df = sr_odds_text.str.extract('([0-9]{4})([0-9]{4})', expand=True)
        odds_df = pd.concat([df[["競走コード", "全オッズ", "登録頭数"]], odds_list_df], axis=1)
"""
