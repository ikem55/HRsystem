from modules.base_extract import BaseExtract
import modules.util as mu

import pandas as pd
import numpy as np
from scipy import stats
from statistics import mean
import collections

class BaseAnalyze(object):
    """
    データ分析に関する共通処理を定義する。
    騎手のコースごとの成績とか
    """
    class_list = ["場コード", "距離", "条件コード"]
    analyze_target = ["騎手コード", "調教師コード", "父名", "母父名"]
    dict_folder = './for_test_dict/base/'

    def __init__(self, start_date, end_date, mock_flag):
        self.start_date = start_date
        self.end_date = end_date
        self.mock_flag = mock_flag
        self.ext = self._get_extract_object(start_date, end_date, mock_flag)

    def _get_extract_object(self, start_date, end_date, mock_flag):
        """ 利用するExtクラスを指定する """
        ext = BaseExtract(start_date, end_date, mock_flag)
        return ext

    def get_base_df(self):
        race_df = self.ext.get_race_table_base()
        raceuma_df = self.ext.get_raceuma_table_base()
        horse_df = self.ext.get_horse_table_base()
        merge_df = pd.merge(race_df, raceuma_df, on="競走コード")
        merge_df = pd.merge(merge_df, horse_df, on="血統登録番号")
        columns_list = self.class_list + self.analyze_target + ["競走コード", "馬番", "複勝配当"]
        base_df = merge_df[columns_list]
        return base_df

    def calc_analyzed_df(self, base_df):
        df = pd.DataFrame()
        for target in self.analyze_target:
            target_list = base_df[target]
            # https://note.nkmk.me/python-collections-counter/
            c = collections.Counter(target_list)
            target_tuple = c.most_common(100)
            for tuple in target_tuple:
                target_val = tuple[0]
                target_df = base_df[base_df[target] == target_val]
                target_base_return = target_df["複勝配当"]
                target_base_mean = mean(target_base_return)
                for cls_val_type in self.class_list:
                    cls_list = base_df[cls_val_type].drop_duplicates()
                    for cls_val in cls_list:
                        target_class_return = target_df[target_df[cls_val_type] == cls_val]["複勝配当"]
                        res = stats.ttest_ind(target_base_return, target_class_return)
                        if res[1] < 0.05:
                            print("target:" + target + " - " + target_val + "  class:" + cls_val_type + " - " + cls_val)
                            avg_diff = mean(target_class_return) - target_base_mean
                            sr = pd.Series([target_val, cls_val, avg_diff], index=[target, cls_val_type, "平均差"])
                            df = df.append(sr)
        mu.save_dict(df, "analyzed_df", self.dict_folder)

    @classmethod
    def get_analyzed_result_df(cls, base_df):
        analyzed_df = mu.load_dict("analyzed_df", cls.dict_folder)
        for target in cls.analyze_target:
            for cls_val_type in cls.class_list:
                filterd_analyze_df = analyzed_df[[target, cls_val_type, "平均差"]].dropna()
                base_df = pd.merge(base_df, filterd_analyze_df, on=[target, cls_val_type], how='left').fillna({"平均差":0}).rename(columns={"平均差": "平均差_" + target + cls_val_type})
        return base_df


