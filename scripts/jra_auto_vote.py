from scripts.jra_target_script import Ext, Ld as TargetLd, CreateFile
from modules.jra_jrdb_download import JrdbDownload
import my_config as mc
import modules.util as mu
import pandas as pd

from datetime import datetime as dt
from datetime import timedelta
import sys
import os

class Ld(TargetLd):
    target_path = mc.TARGET_PATH

    def set_odds_df(self, type):
        self.odds_df = self.ext.get_odds_df(type)

    def set_target_mark_df(self):
        raceuma_df = self.ext.get_raceuma_before_table_base()[["RACE_KEY", "UMABAN"]].copy()
        main_mark_df = self._get_target_mark_df("")
        win_mark_df = self._get_target_mark_df("UmaMark2/")
        jiku_mark_df = self._get_target_mark_df("UmaMark3/")
        tb_us_mark_df = self._get_target_mark_df("UmaMark4/")
        tb_zg_mark_df = self._get_target_mark_df("UmaMark5/")
        nige_mark_df = self._get_target_mark_df("UmaMark6/")
        agari_mark_df = self._get_target_mark_df("UmaMark7/")
        sim_mark_df = self._get_target_mark_df("UmaMark8/")
        raceuma_df = pd.merge(raceuma_df, main_mark_df, on=["RACE_KEY", "UMABAN"], how="left")
        raceuma_df = pd.merge(raceuma_df, win_mark_df.rename(columns={"印": "勝"}), on=["RACE_KEY", "UMABAN"], how="left")
        raceuma_df = pd.merge(raceuma_df, jiku_mark_df.rename(columns={"印": "軸"}), on=["RACE_KEY", "UMABAN"], how="left")
        raceuma_df = pd.merge(raceuma_df, tb_us_mark_df.rename(columns={"印": "US"}), on=["RACE_KEY", "UMABAN"], how="left")
        raceuma_df = pd.merge(raceuma_df, tb_zg_mark_df.rename(columns={"印": "ZG"}), on=["RACE_KEY", "UMABAN"], how="left")
        raceuma_df = pd.merge(raceuma_df, nige_mark_df.rename(columns={"印": "逃"}), on=["RACE_KEY", "UMABAN"], how="left")
        raceuma_df = pd.merge(raceuma_df, agari_mark_df.rename(columns={"印": "上"}), on=["RACE_KEY", "UMABAN"], how="left")
        raceuma_df = pd.merge(raceuma_df, sim_mark_df.rename(columns={"印": "類"}), on=["RACE_KEY", "UMABAN"], how="left")
        self.target_mark_df = raceuma_df.copy()

    def set_target_race_mark(self):
        self.race_mark_df = self._get_main_race_mark_df()

    def _get_main_race_mark_df(self):
        target_file_list = self.race_file_df["file_id"].drop_duplicates().tolist()
        mark_df = pd.DataFrame()
        for file in target_file_list:
            file_df = self.race_file_df.query(f"file_id == '{file}'").copy()
            with open(self.target_path + "UM" + file + ".DAT", 'r', encoding="ms932") as f:
                file_dat = f.readlines()
            file_dat = file_dat[:len(file_df.index)]
            file_df.loc[:, "mark_text"] = file_dat
            mark_df = pd.concat([mark_df, file_df])
        mark_df.loc[:, "mark_text"] = mark_df["mark_text"].apply(lambda x: self.replace_line(x))
        mark_df.loc[:, "レース印"] = mark_df["mark_text"].str[0:6]
        race_mark_df = mark_df[["RACE_KEY", "レース印"]].copy()
        return race_mark_df

    def _get_target_mark_df(self, type):
        target_file_list = self.race_file_df["file_id"].drop_duplicates().tolist()
        mark_df = pd.DataFrame()
        for file in target_file_list:
            print(file)
            file_df = self.race_file_df.query(f"file_id == '{file}'").copy()
            with open(self.target_path + type + "UM" + file + ".DAT", 'r', encoding="ms932") as f:
                file_dat = f.readlines()
            file_dat = file_dat[:len(file_df.index)]
            file_df.loc[:, "mark_text"] = file_dat
            mark_df = pd.concat([mark_df, file_df])
        mark_df.loc[:, "mark_text"] = mark_df["mark_text"].apply(lambda x: self.replace_line(x))
        mark_df.loc[:, "レース印１"] = mark_df["mark_text"].str[0:6]
        mark_df.loc[:, "01"] = mark_df["mark_text"].str[6:8]
        mark_df.loc[:, "02"] = mark_df["mark_text"].str[8:10]
        mark_df.loc[:, "03"] = mark_df["mark_text"].str[10:12]
        mark_df.loc[:, "04"] = mark_df["mark_text"].str[12:14]
        mark_df.loc[:, "05"] = mark_df["mark_text"].str[14:16]
        mark_df.loc[:, "06"] = mark_df["mark_text"].str[16:18]
        mark_df.loc[:, "07"] = mark_df["mark_text"].str[18:20]
        mark_df.loc[:, "08"] = mark_df["mark_text"].str[20:22]
        mark_df.loc[:, "09"] = mark_df["mark_text"].str[22:24]
        mark_df.loc[:, "10"] = mark_df["mark_text"].str[24:26]
        mark_df.loc[:, "11"] = mark_df["mark_text"].str[26:28]
        mark_df.loc[:, "12"] = mark_df["mark_text"].str[28:30]
        mark_df.loc[:, "13"] = mark_df["mark_text"].str[30:32]
        mark_df.loc[:, "14"] = mark_df["mark_text"].str[32:34]
        mark_df.loc[:, "15"] = mark_df["mark_text"].str[34:36]
        mark_df.loc[:, "16"] = mark_df["mark_text"].str[36:38]
        mark_df.loc[:, "17"] = mark_df["mark_text"].str[38:40]
        mark_df.loc[:, "18"] = mark_df["mark_text"].str[40:42]
        uma_mark_df = mark_df[["RACE_KEY", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18"]].copy().set_index("RACE_KEY")
        uma_mark_df = uma_mark_df.stack()
        uma_mark_df = uma_mark_df.reset_index()
        uma_mark_df.columns = ["RACE_KEY", "UMABAN", "印"]
        return uma_mark_df.copy()

    def replace_line(self, line):
        import unicodedata
        count = 0
        new_line = ''
        for c in line:
            if unicodedata.east_asian_width(c) in 'FWA':
                new_line += c + ' '
                count += 2
            else:
                new_line += c
                count += 1
        return new_line

class Simlation(CreateFile):
    def __init__(self, start_date, end_date, term_start_date, term_end_date, test_flag):
        self.start_date = start_date
        self.end_date = end_date
        self.test_flag = test_flag
        self.dict_path = mc.return_jra_path(test_flag)
        self.target_path = mc.TARGET_PATH
        self.ext_score_path = self.target_path + 'ORIGINAL_DATA/'
        self.ld = self._get_load_object("dummy", start_date, end_date, False, test_flag)
        self._set_base_df(term_start_date, term_end_date)

    def _get_load_object(self, version_str, start_date, end_date, mock_flag, test_flag):
        ld = Ld(version_str, start_date, end_date, mock_flag, test_flag)
        return ld

    def _set_base_df(self, term_start_date, term_end_date):
        self.ld.set_race_df()
        self.ld.set_race_file_df()
        self.ld.set_target_mark_df()
        self.ld.set_target_race_mark()
        self.ld.set_haraimodoshi_df()
        base_term_df = self.ld.race_df.query(f"NENGAPPI >= '{term_start_date}' and NENGAPPI <= '{term_end_date}'")[["RACE_KEY"]].copy()
        self.res_raceuma_df = self.ld.ext.get_raceuma_table_base()[["RACE_KEY", "UMABAN", "着順", "確定単勝オッズ", "確定単勝人気順位", "レース脚質", "単勝", "複勝", "テン指数結果順位", "上がり指数結果順位"]].copy()
        self.race_df = self.ld.race_df[["RACE_KEY", "場コード", "距離", "芝ダ障害コード", "種別", "条件", "天候コード", "芝馬場状態コード", "ダ馬場状態コード", "COURSE_KEY", "target_date", "距離グループ", "非根幹"]].copy()
        self.race_df.loc[:, "年月"] = self.race_df["target_date"].str[0:6]
        self.race_df = pd.merge(self.race_df, self.ld.race_mark_df, on ="RACE_KEY")
        self.race_df = pd.merge(self.race_df, base_term_df, on="RACE_KEY")
        self.target_mark_df = self.ld.target_mark_df.copy()
        self.target_mark_df = pd.merge(self.target_mark_df, base_term_df, on="RACE_KEY")
        self.haraimodoshi_dict = self.ld.dict_haraimodoshi

    def get_sim_tanpuku_df(self, uma1_df):
        add_uma1_df = pd.merge(uma1_df, self.res_raceuma_df, on=["RACE_KEY", "UMABAN"])
        self.ld.set_odds_df("馬連")
        odds_df = self.ld.odds_df[["RACE_KEY", "UMABAN", "単勝オッズ", "複勝オッズ"]]
        target_df = pd.merge(add_uma1_df, odds_df, on=["RACE_KEY", "UMABAN"])
        target_df = pd.merge(target_df, self.race_df, on="RACE_KEY")
        return target_df

    def calc_tanpuku_result(self, target_df):
        total_count = len(target_df.index)
        if total_count == 0:
            return pd.Series()
        race_count = len(target_df["RACE_KEY"].drop_duplicates())
        uma1_count = len(target_df[["RACE_KEY", "UMABAN"]].drop_duplicates().index)
        tansho_hit = len(target_df.query("単勝 != 0").index)
        fukusho_hit = len(target_df.query("複勝 != 0").index)
        tansho_hit_rate = round(tansho_hit / uma1_count * 100, 1)
        fukusho_hit_rate = round(fukusho_hit / uma1_count * 100, 1)
        race_tansho_hit_rate = round(tansho_hit / race_count * 100, 1)
        race_fukusho_hit_rate = round(fukusho_hit / race_count * 100, 1)
        tan_return_qua = target_df.query("単勝 != 0")["単勝"].quantile(q=[0, 0.25, 0.5, 0.75, 1])
        tan_return_min = round(tan_return_qua[0])
        tan_return_25 = round(tan_return_qua[0.25])
        tan_return_med = round(tan_return_qua[0.50])
        tan_return_75 = round(tan_return_qua[0.75])
        tan_return_max = round(tan_return_qua[1])
        tan_return_all = round(target_df["単勝"].sum())
        tan_return_avg = round(target_df["単勝"].mean(),1)
        fuku_return_qua = target_df.query("複勝 != 0")["複勝"].quantile(q=[0, 0.25, 0.5, 0.75, 1])
        fuku_return_min = round(fuku_return_qua[0])
        fuku_return_25 = round(fuku_return_qua[0.25])
        fuku_return_med = round(fuku_return_qua[0.50])
        fuku_return_75 = round(fuku_return_qua[0.75])
        fuku_return_max = round(fuku_return_qua[1])
        fuku_return_all = round(target_df["複勝"].sum())
        fuku_return_avg = round(target_df["複勝"].mean(),1)
        res_sr = pd.Series({"総数": uma1_count, "レース数": race_count,
                            "単勝的中数": tansho_hit, "単勝的中率": tansho_hit_rate, "単勝的中R率": race_tansho_hit_rate, "単勝払戻総額": tan_return_all,  "単勝回収率": tan_return_avg,
                            "複勝的中数": fukusho_hit, "複勝的中率": fukusho_hit_rate, "複勝的中R率": race_fukusho_hit_rate, "複勝払戻総額": fuku_return_all,  "複勝回収率": fuku_return_avg,
                            "単勝最低配当": tan_return_min, "単勝配当25%": tan_return_25, "単勝配当中央値": tan_return_med, "単勝配当75%": tan_return_75, "単勝最高配当": tan_return_max,
                            "複勝最低配当": fuku_return_min, "複勝配当25%": fuku_return_25, "複勝配当中央値": fuku_return_med, "複勝配当75%": fuku_return_75, "複勝最高配当": fuku_return_max})
        return res_sr

    def get_sim_umaren_df(self, uma1_df, uma2_df):
        target_df = self.get_umaren_target_df(uma1_df, uma2_df)
        result_df = self.haraimodoshi_dict["umaren_df"]
        target_df = pd.merge(target_df, result_df, on="RACE_KEY")
        target_df.loc[:, "馬1結果"] = target_df.apply(lambda x: True if x["UMABAN_1"] in x["UMABAN"] else False, axis=1)
        target_df.loc[:, "馬2結果"] = target_df.apply(lambda x: True if x["UMABAN_2"] in x["UMABAN"] else False, axis=1)
        target_df.loc[:, "結果"] = target_df.apply(lambda x: x["払戻"] if x["馬1結果"] and x["馬2結果"] else 0, axis=1)
        return target_df

    def get_umaren_target_df(self, uma1_df, uma2_df):
        add_uma1_df = pd.merge(uma1_df, self.res_raceuma_df, on=["RACE_KEY", "UMABAN"]).add_suffix("_1").rename(columns={"RACE_KEY_1":"RACE_KEY"})
        add_uma2_df = pd.merge(uma2_df, self.res_raceuma_df, on=["RACE_KEY", "UMABAN"]).add_suffix("_2").rename(columns={"RACE_KEY_2":"RACE_KEY"})
        self.ld.set_odds_df("馬連")
        odds_df = self.ld.odds_df
        base_uma1_df = pd.merge(uma1_df[["RACE_KEY", "UMABAN"]], odds_df, on=["RACE_KEY", "UMABAN"]).set_index(["RACE_KEY", "UMABAN"])
        umaren_uma1_df = base_uma1_df[['馬連オッズ０１', '馬連オッズ０２', '馬連オッズ０３', '馬連オッズ０４', '馬連オッズ０５', '馬連オッズ０６', '馬連オッズ０７', '馬連オッズ０８', '馬連オッズ０９',
                                       '馬連オッズ１０', '馬連オッズ１１', '馬連オッズ１２', '馬連オッズ１３', '馬連オッズ１４', '馬連オッズ１５', '馬連オッズ１６', '馬連オッズ１７', '馬連オッズ１８']].copy()
        umaren_uma1_df.columns = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18"]
        umaren_uma1_df = umaren_uma1_df.stack().reset_index()
        umaren_uma1_df.columns = ["RACE_KEY", "UMABAN_1", "UMABAN_2", "オッズ"]
        #umaren_uma1_df = umaren_uma1_df.astype({"RACE_KEY": 'str', "UMABAN_1": 'str', "UMABAN_2": 'str'})
        target_df = pd.merge(umaren_uma1_df, self.race_df, on="RACE_KEY")
        target_df = pd.merge(target_df, add_uma1_df, on=["RACE_KEY", "UMABAN_1"])
        target_df = pd.merge(target_df, add_uma2_df, on=["RACE_KEY", "UMABAN_2"])
        target_df = target_df.drop_duplicates(subset=["RACE_KEY", "UMABAN_1", "UMABAN_2"])
        return target_df

    def calc_umaren_result(self, target_df):
        target_df = target_df.query("UMABAN_1 != UMABAN_2").copy()
        total_count = len(target_df.index)
        if total_count == 0:
            return pd.Series()
        race_count = len(target_df["RACE_KEY"].drop_duplicates())
        uma1_count = len(target_df[["RACE_KEY", "UMABAN_1"]].drop_duplicates().index)
        uma1_hit = len(target_df.query("馬1結果 == True")[["RACE_KEY", "UMABAN_1"]].drop_duplicates().index)
        uma2_count = len(target_df[["RACE_KEY", "UMABAN_2"]].drop_duplicates().index)
        uma2_hit = len(target_df.query("馬2結果 == True")[["RACE_KEY", "UMABAN_2"]].drop_duplicates().index)
        all_hit = len(target_df.query("馬1結果 == True and 馬2結果 == True")[["RACE_KEY", "UMABAN_2"]].drop_duplicates().index)
        uma1_hit_rate = round(uma1_hit / uma1_count * 100, 1)
        uma2_hit_rate = round(uma2_hit / uma2_count * 100, 1)
        all_hit_rate = round(all_hit / total_count * 100, 1)
        race_hit_rate = round(all_hit / race_count * 100, 1)
        return_qua = target_df.query("結果 != 0")["結果"].quantile(q=[0, 0.25, 0.5, 0.75, 1])
        return_min = round(return_qua[0])
        return_25 = round(return_qua[0.25])
        return_med = round(return_qua[0.50])
        return_75 = round(return_qua[0.75])
        return_max = round(return_qua[1])
        return_all = round(target_df["結果"].sum())
        return_avg = round(target_df["結果"].mean(),1)
        res_sr = pd.Series({"総数": total_count, "レース数": race_count, "馬１総数": uma1_count, "馬１的中数": uma1_hit, "馬１的中率": uma1_hit_rate,
                            "馬２総数": uma2_count, "馬２的中数": uma2_hit, "馬２的中率": uma2_hit_rate, "的中数": all_hit, "的中率": all_hit_rate,
                            "レース的中率": race_hit_rate, "回収率": return_avg, "払戻総額": return_all,  "最低配当": return_min,
                            "配当25%": return_25, "配当中央値": return_med, "配当75%": return_75, "最高配当": return_max})
        return res_sr

    def get_sim_wide_df(self, uma1_df, uma2_df):
        target_df = self.get_wide_target_df(uma1_df, uma2_df)
        result_df = self.haraimodoshi_dict["wide_df"]
        result_df.loc[:, "UMABAN_1"] = result_df["UMABAN"].apply(lambda x: x[0])
        result_df.loc[:, "UMABAN_2"] = result_df["UMABAN"].apply(lambda x: x[1])
        target_df = pd.merge(target_df, result_df, on=["RACE_KEY", "UMABAN_1", "UMABAN_2"], how='left')
        target_df.loc[:, "馬1結果"] = target_df.apply(lambda x: True if x["UMABAN"] == x["UMABAN"] else False, axis=1)
        target_df.loc[:, "馬2結果"] = target_df.apply(lambda x: True if x["UMABAN"] == x["UMABAN"] else False, axis=1)
        target_df.loc[:, "結果"] = target_df.apply(lambda x: x["払戻"] if x["馬1結果"] and x["馬2結果"] else 0, axis=1)
        return target_df.fillna(0)

    def get_wide_target_df(self, uma1_df, uma2_df):
        add_uma1_df = pd.merge(uma1_df, self.res_raceuma_df, on=["RACE_KEY", "UMABAN"]).add_suffix("1").rename(
            columns={"RACE_KEY1": "RACE_KEY"})
        add_uma2_df = pd.merge(uma2_df, self.res_raceuma_df, on=["RACE_KEY", "UMABAN"]).add_suffix("2").rename(
            columns={"RACE_KEY2": "RACE_KEY"})
        base_df = pd.merge(add_uma1_df, add_uma2_df, on="RACE_KEY")
        base_df = base_df.query("UMABAN1 != UMABAN2")
        base_df.loc[:, "UMABAN_bet"] = base_df.apply(lambda x: sorted([x["UMABAN1"], x["UMABAN2"]]), axis=1)
        base_df.loc[:, "UMABAN_1"] = base_df["UMABAN_bet"].apply(lambda x: x[0])
        base_df.loc[:, "UMABAN_2"] = base_df["UMABAN_bet"].apply(lambda x: x[1])
        self.ld.set_odds_df("ワイド")
        odds_df = self.ld.odds_df
        target_df = pd.merge(base_df, odds_df, on=["RACE_KEY", "UMABAN_1", "UMABAN_2"])
        target_df = pd.merge(target_df, self.race_df, on=["RACE_KEY", "target_date"])
        target_df = target_df.drop_duplicates(subset=["RACE_KEY", "UMABAN_1", "UMABAN_2"])
        target_df = target_df.rename(columns={"ワイドオッズ": "オッズ"})
        return target_df

    def calc_wide_result(self, target_df):
        total_count = len(target_df.index)
        if total_count == 0:
            return pd.Series()
        race_count = len(target_df["RACE_KEY"].drop_duplicates())
        uma1_count = len(target_df[["RACE_KEY", "UMABAN1"]].drop_duplicates().index)
        uma1_hit = len(target_df.query("馬1結果 == True")[["RACE_KEY", "UMABAN1"]].drop_duplicates().index)
        uma2_count = len(target_df[["RACE_KEY", "UMABAN2"]].drop_duplicates().index)
        uma2_hit = len(target_df.query("馬2結果 == True")[["RACE_KEY", "UMABAN2"]].drop_duplicates().index)
        all_hit = len(target_df.query("馬1結果 == True and 馬2結果 == True")[
                          ["RACE_KEY", "UMABAN_2"]].drop_duplicates().index)
        uma1_hit_rate = round(uma1_hit / uma1_count * 100, 1)
        uma2_hit_rate = round(uma2_hit / uma2_count * 100, 1)
        all_hit_rate = round(all_hit / total_count * 100, 1)
        race_hit_rate = round(all_hit / race_count * 100, 1)
        return_qua = target_df.query("結果 != 0")["結果"].quantile(q=[0, 0.25, 0.5, 0.75, 1])
        return_min = round(return_qua[0])
        return_25 = round(return_qua[0.25])
        return_med = round(return_qua[0.50])
        return_75 = round(return_qua[0.75])
        return_max = round(return_qua[1])
        return_all = round(target_df["結果"].sum())
        return_avg = round(target_df["結果"].mean(), 1)
        res_sr = pd.Series(
            {"総数": total_count, "レース数": race_count, "馬１総数": uma1_count, "馬１的中数": uma1_hit, "馬１的中率": uma1_hit_rate,
             "馬２総数": uma2_count, "馬２的中数": uma2_hit, "馬２的中率": uma2_hit_rate, "的中数": all_hit, "的中率": all_hit_rate,
             "レース的中率": race_hit_rate, "回収率": return_avg, "払戻総額": return_all, "最低配当": return_min,
             "配当25%": return_25, "配当中央値": return_med, "配当75%": return_75, "最高配当": return_max})
        return res_sr

    def get_sim_umatan_df(self, uma1_df, uma2_df):
        target_df = self.get_umatan_target_df(uma1_df, uma2_df)
        result_df = self.haraimodoshi_dict["umatan_df"]
        target_df = pd.merge(target_df, result_df, on="RACE_KEY")
        target_df.loc[:, "馬1結果"] = target_df.apply(lambda x: True if x["UMABAN_1"] == x["UMABAN"][0] else False, axis=1)
        target_df.loc[:, "馬2結果"] = target_df.apply(lambda x: True if x["UMABAN_2"] == x["UMABAN"][1] else False, axis=1)
        target_df.loc[:, "結果"] = target_df.apply(lambda x: x["払戻"] if x["馬1結果"] and x["馬2結果"] else 0, axis=1)
        return target_df

    def get_umatan_target_df(self, uma1_df, uma2_df):
        add_uma1_df = pd.merge(uma1_df, self.res_raceuma_df, on=["RACE_KEY", "UMABAN"]).add_suffix("_1").rename(
            columns={"RACE_KEY_1": "RACE_KEY"})
        add_uma2_df = pd.merge(uma2_df, self.res_raceuma_df, on=["RACE_KEY", "UMABAN"]).add_suffix("_2").rename(
            columns={"RACE_KEY_2": "RACE_KEY"})
        base_df = pd.merge(add_uma1_df, add_uma2_df, on="RACE_KEY")
        base_df = base_df.query("UMABAN_1 != UMABAN_2")
        self.ld.set_odds_df("馬単")
        odds_df = self.ld.odds_df
        target_df = pd.merge(base_df, odds_df, on=["RACE_KEY", "UMABAN_1", "UMABAN_2"])
        target_df = pd.merge(target_df, self.race_df, on=["RACE_KEY", "target_date"])
        target_df = target_df.rename(columns={"馬単オッズ": "オッズ"})
        return target_df

    def calc_umatan_result(self, target_df):
        total_count = len(target_df.index)
        if total_count == 0:
            return pd.Series()
        race_count = len(target_df["RACE_KEY"].drop_duplicates())
        uma1_count = len(target_df[["RACE_KEY", "UMABAN_1"]].drop_duplicates().index)
        uma1_hit = len(target_df.query("馬1結果 == True")[["RACE_KEY", "UMABAN_1"]].drop_duplicates().index)
        uma2_count = len(target_df[["RACE_KEY", "UMABAN_2"]].drop_duplicates().index)
        uma2_hit = len(target_df.query("馬2結果 == True")[["RACE_KEY", "UMABAN_2"]].drop_duplicates().index)
        all_hit = len(target_df.query("馬1結果 == True and 馬2結果 == True")[
                          ["RACE_KEY", "UMABAN_2"]].drop_duplicates().index)
        uma1_hit_rate = round(uma1_hit / uma1_count * 100, 1)
        uma2_hit_rate = round(uma2_hit / uma2_count * 100, 1)
        all_hit_rate = round(all_hit / total_count * 100, 1)
        race_hit_rate = round(all_hit / race_count * 100, 1)
        return_qua = target_df.query("結果 != 0")["結果"].quantile(q=[0, 0.25, 0.5, 0.75, 1])
        return_min = round(return_qua[0])
        return_25 = round(return_qua[0.25])
        return_med = round(return_qua[0.50])
        return_75 = round(return_qua[0.75])
        return_max = round(return_qua[1])
        return_all = round(target_df["結果"].sum())
        return_avg = round(target_df["結果"].mean(), 1)
        res_sr = pd.Series(
            {"総数": total_count, "レース数": race_count, "馬１総数": uma1_count, "馬１的中数": uma1_hit, "馬１的中率": uma1_hit_rate,
             "馬２総数": uma2_count, "馬２的中数": uma2_hit, "馬２的中率": uma2_hit_rate, "的中数": all_hit, "的中率": all_hit_rate,
             "レース的中率": race_hit_rate, "回収率": return_avg, "払戻総額": return_all, "最低配当": return_min,
             "配当25%": return_25, "配当中央値": return_med, "配当75%": return_75, "最高配当": return_max})
        return res_sr

    def get_sim_sanrenpuku_df(self, uma1_df, uma2_df, uma3_df):
        target_df = self.get_sanrenpuku_target_df(uma1_df, uma2_df, uma3_df)
        result_df = self.haraimodoshi_dict["sanrenpuku_df"]
        target_df = pd.merge(target_df, result_df, on="RACE_KEY")
        target_df.loc[:, "馬1結果"] = target_df.apply(lambda x: True if x["UMABAN1"] in x["UMABAN"] else False, axis=1)
        target_df.loc[:, "馬2結果"] = target_df.apply(lambda x: True if x["UMABAN2"] in x["UMABAN"] else False, axis=1)
        target_df.loc[:, "馬3結果"] = target_df.apply(lambda x: True if x["UMABAN3"] in x["UMABAN"] else False, axis=1)
        target_df.loc[:, "結果"] = target_df.apply(lambda x: x["払戻"] if x["馬1結果"] and x["馬2結果"] and x["馬3結果"] else 0, axis=1)
        return target_df

    def get_sanrenpuku_target_df(self, uma1_df, uma2_df, uma3_df):
        add_uma1_df = pd.merge(uma1_df, self.res_raceuma_df, on=["RACE_KEY", "UMABAN"]).add_suffix("1").rename(columns={"RACE_KEY1":"RACE_KEY"})
        add_uma2_df = pd.merge(uma2_df, self.res_raceuma_df, on=["RACE_KEY", "UMABAN"]).add_suffix("2").rename(columns={"RACE_KEY2":"RACE_KEY"})
        add_uma3_df = pd.merge(uma3_df, self.res_raceuma_df, on=["RACE_KEY", "UMABAN"]).add_suffix("3").rename(columns={"RACE_KEY3":"RACE_KEY"})
        base_df = pd.merge(add_uma1_df, add_uma2_df, on="RACE_KEY")
        base_df = pd.merge(base_df, add_uma3_df, on="RACE_KEY")
        base_df = base_df.query("(UMABAN1 != UMABAN2) and (UMABAN2 != UMABAN3) and (UMABAN3) != (UMABAN1)")
        base_df.loc[:, "UMABAN_bet"] = base_df.apply(lambda x: sorted([x["UMABAN1"], x["UMABAN2"], x["UMABAN3"]]), axis=1)
        base_df.loc[:, "UMABAN_1"] = base_df["UMABAN_bet"].apply(lambda x: x[0])
        base_df.loc[:, "UMABAN_2"] = base_df["UMABAN_bet"].apply(lambda x: x[1])
        base_df.loc[:, "UMABAN_3"] = base_df["UMABAN_bet"].apply(lambda x: x[2])
        self.ld.set_odds_df("三連複")
        odds_df = self.ld.odds_df
        target_df = pd.merge(base_df, odds_df, on=["RACE_KEY", "UMABAN_1", "UMABAN_2", "UMABAN_3"])
        target_df = pd.merge(target_df, self.race_df, on=["RACE_KEY", "target_date"])
        target_df = target_df.drop_duplicates(subset=["RACE_KEY", "UMABAN_1", "UMABAN_2", "UMABAN_3"])
        target_df = target_df.rename(columns={"３連複オッズ": "オッズ"})
        return target_df

    def calc_sanrenpuku_result(self, target_df):
        total_count = len(target_df.index)
        if total_count == 0:
            return pd.Series()
        race_count = len(target_df["RACE_KEY"].drop_duplicates())
        uma1_count = len(target_df[["RACE_KEY", "UMABAN1"]].drop_duplicates().index)
        uma1_hit = len(target_df.query("馬1結果 == True")[["RACE_KEY", "UMABAN1"]].drop_duplicates().index)
        uma2_count = len(target_df[["RACE_KEY", "UMABAN2"]].drop_duplicates().index)
        uma2_hit = len(target_df.query("馬2結果 == True")[["RACE_KEY", "UMABAN2"]].drop_duplicates().index)
        uma3_count = len(target_df[["RACE_KEY", "UMABAN3"]].drop_duplicates().index)
        uma3_hit = len(target_df.query("馬3結果 == True")[["RACE_KEY", "UMABAN3"]].drop_duplicates().index)
        all_hit = len(target_df.query("馬1結果 == True and 馬2結果 == True and 馬3結果 == True")[["RACE_KEY", "UMABAN_2"]].drop_duplicates().index)
        uma1_hit_rate = round(uma1_hit / uma1_count * 100, 1)
        uma2_hit_rate = round(uma2_hit / uma2_count * 100, 1)
        uma3_hit_rate = round(uma3_hit / uma3_count * 100, 1)
        all_hit_rate = round(all_hit / total_count * 100, 1)
        race_hit_rate = round(all_hit / race_count * 100, 1)
        return_qua = target_df.query("結果 != 0")["結果"].quantile(q=[0, 0.25, 0.5, 0.75, 1])
        return_min = round(return_qua[0])
        return_25 = round(return_qua[0.25])
        return_med = round(return_qua[0.50])
        return_75 = round(return_qua[0.75])
        return_max = round(return_qua[1])
        return_all = round(target_df["結果"].sum())
        return_avg = round(target_df["結果"].mean(),1)
        res_sr = pd.Series({"総数": total_count, "レース数": race_count, "馬１総数": uma1_count, "馬１的中数": uma1_hit, "馬１的中率": uma1_hit_rate,
                            "馬２総数": uma2_count, "馬２的中数": uma2_hit, "馬２的中率": uma2_hit_rate, "馬３総数": uma3_count, "馬３的中数": uma3_hit,
                            "馬３的中率": uma3_hit_rate, "的中数": all_hit, "的中率": all_hit_rate,
                            "レース的中率": race_hit_rate, "回収率": return_avg, "払戻総額": return_all,  "最低配当": return_min,
                            "配当25%": return_25, "配当中央値": return_med, "配当75%": return_75, "最高配当": return_max})
        return res_sr



class AutoVote(Simlation):
    def _set_base_df(self, term_start_date, term_end_date):
        self.auto_bet_path = self.target_path + 'AUTO_BET/'
        self.ld.set_race_df()
        self.ld.set_race_file_df()
        self.ld.set_target_mark_df()
        self.ld.set_target_race_mark()
        base_term_df = self.ld.race_df.query(f"NENGAPPI >= '{term_start_date}' and NENGAPPI <= '{term_end_date}'")[["RACE_KEY"]].copy()
        self.res_raceuma_df = self.ld.ext.get_raceuma_before_table_base()[["RACE_KEY", "UMABAN"]].copy()
        self.race_df = self.ld.race_df[["RACE_KEY", "場コード", "距離", "芝ダ障害コード", "種別", "条件", "天候コード", "芝馬場状態コード", "ダ馬場状態コード", "COURSE_KEY", "target_date", "距離グループ", "非根幹"]].copy()
        self.race_df = pd.merge(self.race_df, self.ld.race_mark_df, on ="RACE_KEY")
        self.race_df = pd.merge(self.race_df, base_term_df, on="RACE_KEY")
        self.target_mark_df = self.ld.target_mark_df.copy()
        self.target_mark_df = pd.merge(self.target_mark_df, base_term_df, on="RACE_KEY")

    def export_bet_csv(self):
        target_df = self.target_mark_df.copy()
        tansho_ipat_bet_df, tansho_target_bet_df = self.get_tansho_bet_df(target_df)
        fukusho_ipat_bet_df, fukusho_target_bet_df = self.get_fukusho_bet_df(target_df)
        umaren_ipat_bet_df, umaren_target_bet_df = self.get_umaren_bet_df(target_df, target_df)
        umatan_ipat_bet_df, umatan_target_bet_df = self.get_umatan_bet_df(target_df, target_df)
        wide_ipat_bet_df, wide_target_bet_df = self.get_wide_bet_df(target_df, target_df)
        sanrenpuku_uma1_df = target_df.query("印 == '◎ '").copy()
        sanrenpuku_uma2_df = target_df.query("印 in ['× ', '△ ', '▲ ', '○ ']").copy()
        sanrenpuku_uma3_df = target_df.query("印 != '◎ '").copy()
        sanrenpuku_ipat_bet_df, sanrenpuku_target_bet_df = self.get_sanrenpuku_bet_df(sanrenpuku_uma1_df, sanrenpuku_uma2_df, sanrenpuku_uma3_df)
        ipat_bet_df = pd.concat([tansho_ipat_bet_df, fukusho_ipat_bet_df, umaren_ipat_bet_df, umatan_ipat_bet_df, wide_ipat_bet_df, sanrenpuku_ipat_bet_df])
        ipat_bet_df.to_csv(self.auto_bet_path + "ipat_bet.csv", index=False, header=False)
        target_bet_df = pd.concat([tansho_target_bet_df, fukusho_target_bet_df, umaren_target_bet_df, umatan_target_bet_df, wide_target_bet_df, sanrenpuku_target_bet_df])
        target_bet_df = target_bet_df.sort_values(["RACE_ID", "エリア", "券種", "購入金額", "目１", "目２", "目３"])
        target_bet_df.to_csv(self.auto_bet_path + "target_bet.csv", index=False, header=False)

    def get_tansho_bet_df(self, target_df):
        tansho_base_df = self.get_sim_tanpuku_df(target_df)
        bet_df = tansho_base_df.query("印 in ['△ ', '▲ ', '○ ', '◎ '] and 勝 in ['☆ ','▲ ', '○ ', '◎ '] and 単勝オッズ >= 10").copy()
        bet_df.loc[:, "コード"] =bet_df["UMABAN"]
        bet_df.loc[:, "目１"] =bet_df["UMABAN"]
        bet_df.loc[:, "目２"] =""
        bet_df.loc[:, "目３"] =""
        bet_df.loc[:, "馬券式"] = "TANSYO"
        bet_df.loc[:, "券種"] = '0'
        bet_df.loc[:, "オッズ"] = bet_df["単勝オッズ"]
        ipat_bet_df = self._get_ipatgo_bet_df(bet_df)
        target_bet_df = self._get_target_bet_df(bet_df)
        return ipat_bet_df, target_bet_df

    def get_fukusho_bet_df(self, target_df):
        tansho_base_df = self.get_sim_tanpuku_df(target_df)
        bet_df = tansho_base_df.query("印 in ['△ ', '▲ ', '○ ', '◎ '] and 軸 in ['☆ ', '○ ', '◎ '] and 複勝オッズ >= 5 and レース印 != '000000'").copy()
        bet_df.loc[:, "コード"] =bet_df["UMABAN"]
        bet_df.loc[:, "目１"] =bet_df["UMABAN"]
        bet_df.loc[:, "目２"] =""
        bet_df.loc[:, "目３"] =""
        bet_df.loc[:, "馬券式"] = "FUKUSHO"
        bet_df.loc[:, "券種"] = '1'
        bet_df.loc[:, "オッズ"] = bet_df["複勝オッズ"]
        ipat_bet_df = self._get_ipatgo_bet_df(bet_df)
        target_bet_df = self._get_target_bet_df(bet_df)
        return ipat_bet_df, target_bet_df

    def get_umaren_bet_df(self, uma1_df, uma2_df):
        umaren_base_df = self.get_umaren_target_df(uma1_df, uma2_df)
        bet_df = umaren_base_df.query(
            "印_1 == '◎ ' and 印_2 in ['△ ', '▲ ', '○ '] and 軸_1 != '◎ ' and 軸_2 not in ['◎ ', '▲ '] and オッズ >= 50").copy()
        bet_df.loc[:, "コード"] =bet_df.apply(lambda x: x["UMABAN_1"] + "-" + x["UMABAN_2"], axis=1)
        bet_df.loc[:, "目１"] =bet_df["UMABAN_1"]
        bet_df.loc[:, "目２"] =bet_df["UMABAN_2"]
        bet_df.loc[:, "目３"] =""
        bet_df.loc[:, "馬券式"] = "UMAREN"
        bet_df.loc[:, "券種"] = '3'
        ipat_bet_df = self._get_ipatgo_bet_df(bet_df)
        target_bet_df = self._get_target_bet_df(bet_df)
        return ipat_bet_df, target_bet_df

    def get_umatan_bet_df(self, uma1_df, uma2_df):
        umatan_base_df = self.get_umatan_target_df(uma1_df, uma2_df)
        bet_df = umatan_base_df.query(
            "勝_1 == '☆ ' and 印_1 in ['▲ ', '○ ', '◎ '] and 軸_2 in ['☆ ','▲ ', '○ ', '◎ '] and オッズ >= 50").copy()
        bet_df.loc[:, "コード"] =bet_df.apply(lambda x: x["UMABAN_1"] + "-" + x["UMABAN_2"], axis=1)
        bet_df.loc[:, "目１"] =bet_df["UMABAN_1"]
        bet_df.loc[:, "目２"] =bet_df["UMABAN_2"]
        bet_df.loc[:, "目３"] =""
        bet_df.loc[:, "馬券式"] = "UMATAN"
        bet_df.loc[:, "券種"] = '5'
        ipat_bet_df = self._get_ipatgo_bet_df(bet_df)
        target_bet_df = self._get_target_bet_df(bet_df)
        return ipat_bet_df, target_bet_df

    def get_wide_bet_df(self, uma1_df, uma2_df):
        wide_base_df = self.get_wide_target_df(uma1_df, uma2_df)
        bet_df = wide_base_df.query(
            "印1 in ['▲ ', '○ ', '◎ '] and (軸1 != '  ' or 勝1 != '  ') and 軸2 in ['☆ ', '◎ '] and オッズ >= 20 and オッズ <= 100").copy()
        bet_df.loc[:, "コード"] =bet_df.apply(lambda x: x["UMABAN_1"] + "-" + x["UMABAN_2"], axis=1)
        bet_df.loc[:, "目１"] =bet_df["UMABAN_1"]
        bet_df.loc[:, "目２"] =bet_df["UMABAN_2"]
        bet_df.loc[:, "目３"] =""
        bet_df.loc[:, "馬券式"] = "WIDE"
        bet_df.loc[:, "券種"] = '4'
        ipat_bet_df = self._get_ipatgo_bet_df(bet_df)
        target_bet_df = self._get_target_bet_df(bet_df)
        return ipat_bet_df, target_bet_df

    def get_sanrenpuku_bet_df(self, uma1_df, uma2_df, uma3_df):
        sanrenpuku_base_df = self.get_sanrenpuku_target_df(uma1_df, uma2_df, uma3_df)
        bet_df = sanrenpuku_base_df.query(
            "軸1 in ['☆ ', '▲ ', '○ ', '◎ '] and 印2 in ['△ ', '▲ ', '○ '] and 軸3 in ['▲ ', '○ ', '◎ '] and オッズ >= 50 and オッズ <= 150").copy()
        bet_df.loc[:, "コード"] =bet_df.apply(lambda x: x["UMABAN_1"] + "-" + x["UMABAN_2"] + "-" + x["UMABAN_3"], axis=1)
        bet_df.loc[:, "目１"] =bet_df["UMABAN_1"]
        bet_df.loc[:, "目２"] =bet_df["UMABAN_2"]
        bet_df.loc[:, "目３"] =bet_df["UMABAN_3"]
        bet_df.loc[:, "馬券式"] = "SANRENPUKU"
        bet_df.loc[:, "券種"] = '6'
        ipat_bet_df = self._get_ipatgo_bet_df(bet_df)
        target_bet_df = self._get_target_bet_df(bet_df)
        return ipat_bet_df, target_bet_df

    def _get_ipatgo_bet_df(self, bet_df):
        bet_df["年月日"] = bet_df["target_date"]
        bet_df.loc[:, "競馬場"] = bet_df["RACE_KEY"].apply(lambda x: self._convert_keibajo(x[0:2]))
        bet_df.loc[:, "レース番号"] = bet_df["RACE_KEY"].str[6:8].astype(int)
        bet_df.loc[:, "投票方式"] = "NORMAL"
        bet_df.loc[:, "マルチ"] = "MULTI"
        bet_df.loc[:, "金額"] = 100
        bet_df = bet_df[["年月日", "競馬場", "レース番号", "馬券式", "投票方式", "マルチ", "コード", "金額"]].copy()
        return bet_df

    def _get_target_bet_df(self, bet_df):
        bet_df = pd.merge(bet_df, self.ld.race_file_df, on="RACE_KEY")
        bet_df.loc[:, "変換フラグ"] = 0
        bet_df.loc[:, "購入金額"] = 100
        bet_df.loc[:, "的中時の配当"] = 0
        bet_df.loc[:, "エリア"] = "H"
        bet_df.loc[:, "マーク"] = ""
        bet_df = bet_df[["RACE_ID", "変換フラグ", "券種", "目１", "目２", "目３", "購入金額", "オッズ", "的中時の配当", "エリア", "マーク"]].copy()
        return bet_df

    def _convert_keibajo(self, str):
        if str == '01': return "SAPPORO"
        if str == '02': return "HAKODATE"
        if str == '03': return "FUKUSHIMA"
        if str == '04': return "NIIGATA"
        if str == '05': return "TOKYO"
        if str == '06': return "NAKAYAMA"
        if str == '07': return "CHUKYO"
        if str == '08': return "KYOTO"
        if str == '09': return "HANSHIN"
        if str == '10': return "KOKURA"

    def export_pbi_data(self):
        race_df = self.ld.race_df.copy()
        race_df.loc[:, "場名"] = race_df["場名"].apply(lambda x: mu.convert_basho(x))
        race_df.loc[:, "レースNo"] = race_df["RACE_KEY"].str[6:8]
        race_df.loc[:, "種別"] = race_df["種別"].apply(lambda x: mu.convert_shubetsu(x))
        race_df.loc[:, "条件"] = race_df["条件"].apply(lambda x: self._convert_joken(x))
        race_df.loc[:, "芝ダ"] = race_df["芝ダ障害コード"].apply(lambda x: mu.convert_shida(x))
        race_df.loc[:, "コース名"] = race_df.apply(lambda x: self._get_course_name(x), axis=1)
        race_df = pd.merge(race_df, self.ld.race_file_df[["RACE_KEY", "RACE_ID"]], on="RACE_KEY")
        race_df = race_df[["RACE_ID", "RACE_KEY", "場名", "レースNo", "距離", "芝ダ", "種別", "条件", "target_date", "コース名"]].copy()
        race_df.to_csv(self.auto_bet_path + "race.csv", index=False, header=True)
        raceuma_df = self.ld.ext.get_raceuma_before_table_base()[["RACE_KEY", "UMABAN", "基準オッズ", "騎手名", "調教師名", "馬名"]].copy()
        raceuma_df = pd.merge(raceuma_df, self.target_mark_df, on=["RACE_KEY", "UMABAN"])
        raceuma_df.to_csv(self.auto_bet_path + "raceuma.csv", index=False, header=True)

    def _convert_joken(self, joken):
        if joken == 1: return "A1"
        if joken == 2: return "A2"
        if joken == 3: return "A3"
        if joken == 99: return "OP"
        if joken == 5: return "500万下"
        if joken == 10: return "1000万下"
        if joken == 16: return "1600万下"
        else: return ""

    def _get_course_name(self, sr):
        soto = "外" if sr["内外"] == "2" else ""
        return sr["場名"] + sr["芝ダ"] + str(sr["距離"]) +"m" + soto

class Sokuho(object):
    def __init__(self, target_date, test_flag):
        self.ext = Ext(target_date, target_date, test_flag)
        self.dict_path = mc.return_jra_path(test_flag)
        self.target_path = mc.TARGET_PATH
        self.auto_bet_path = self.target_path + 'AUTO_BET/'

    def export_pbi_real_data(self):
        jrdb = JrdbDownload()
        jrdb.procedure_download_sokuho()
        filelist = os.listdir(self.dict_path + "jrdb_data/sokuho/")
        for file in filelist:
            if file[0:3] == "SED":
                temp_df = self.ext.get_sed_sokuho_df(file)
                temp_df.to_csv(self.auto_bet_path + "result.csv", index=False, header=True)
            #elif file[0:3] == "TYB":
            #    temp_df = self.ext.get_tyb_sokuho_df(file)
            # elif file[0:3] == "SRB":
            #     temp_df = self.ext.get_srb_sokuho_df(file)
            elif file[0:3] == "HJC":
                temp_df = self.ext.get_hjc_sokuho_df(file)
                dict_haraimodoshi = self.ext.get_haraimodoshi_dict(temp_df)
                tansho_df = dict_haraimodoshi["tansho_df"]
                tansho_df.loc[:, "NENGAPPI"] = "20" + file[3:9]
                tansho_df.loc[:, "RACE_ID"] = tansho_df.apply(lambda x: mu.convert_jrdb_id(x["RACE_KEY"], x["NENGAPPI"]), axis=1)
                tansho_df.to_csv(self.auto_bet_path + "tansho.csv", index=False, header=True)
                fukusho_df = dict_haraimodoshi["fukusho_df"]
                fukusho_df.loc[:, "NENGAPPI"] = "20" + file[3:9]
                fukusho_df.loc[:, "RACE_ID"] = fukusho_df.apply(lambda x: mu.convert_jrdb_id(x["RACE_KEY"], x["NENGAPPI"]), axis=1)
                fukusho_df.to_csv(self.auto_bet_path + "fukusho.csv", index=False, header=True)
                umaren_df = dict_haraimodoshi["umaren_df"]
                umaren_df.loc[:, "NENGAPPI"] = "20" + file[3:9]
                umaren_df.loc[:, "RACE_ID"] = umaren_df.apply(lambda x: mu.convert_jrdb_id(x["RACE_KEY"], x["NENGAPPI"]), axis=1)
                umaren_df.loc[:, "馬1"] = umaren_df["UMABAN"].apply(lambda x: int(x[0]))
                umaren_df.loc[:, "馬2"] = umaren_df["UMABAN"].apply(lambda x: int(x[1]))
                umaren_df.to_csv(self.auto_bet_path + "umaren.csv", index=False, header=True)
                wide_df = dict_haraimodoshi["wide_df"]
                wide_df.loc[:, "NENGAPPI"] = "20" + file[3:9]
                wide_df.loc[:, "RACE_ID"] = wide_df.apply(lambda x: mu.convert_jrdb_id(x["RACE_KEY"], x["NENGAPPI"]), axis=1)
                wide_df.loc[:, "馬1"] = wide_df["UMABAN"].apply(lambda x: int(x[0]))
                wide_df.loc[:, "馬2"] = wide_df["UMABAN"].apply(lambda x: int(x[1]))
                wide_df.to_csv(self.auto_bet_path + "wide.csv", index=False, header=True)
                umatan_df = dict_haraimodoshi["umatan_df"]
                umatan_df.loc[:, "NENGAPPI"] = "20" + file[3:9]
                umatan_df.loc[:, "RACE_ID"] = umatan_df.apply(lambda x: mu.convert_jrdb_id(x["RACE_KEY"], x["NENGAPPI"]), axis=1)
                umatan_df.loc[:, "馬1"] = umatan_df["UMABAN"].apply(lambda x: int(x[0]))
                umatan_df.loc[:, "馬2"] = umatan_df["UMABAN"].apply(lambda x: int(x[1]))
                umatan_df.to_csv(self.auto_bet_path + "umatan.csv", index=False, header=True)
                sanrenpuku_df = dict_haraimodoshi["sanrenpuku_df"]
                sanrenpuku_df.loc[:, "NENGAPPI"] = "20" + file[3:9]
                sanrenpuku_df.loc[:, "RACE_ID"] = sanrenpuku_df.apply(lambda x: mu.convert_jrdb_id(x["RACE_KEY"], x["NENGAPPI"]), axis=1)
                sanrenpuku_df.loc[:, "馬1"] = sanrenpuku_df["UMABAN"].apply(lambda x: int(x[0]))
                sanrenpuku_df.loc[:, "馬2"] = sanrenpuku_df["UMABAN"].apply(lambda x: int(x[1]))
                sanrenpuku_df.loc[:, "馬3"] = sanrenpuku_df["UMABAN"].apply(lambda x: int(x[2]))
                sanrenpuku_df.to_csv(self.auto_bet_path + "sanrenpuku.csv", index=False, header=True)
            else:
                continue


if __name__ == "__main__":
    args = sys.argv
    print(args)
    print("mode：" + args[1])  # test or init or prod
    mock_flag = False
    test_flag = False
    mode = args[1]
    dict_path = mc.return_jra_path(test_flag)
    version_str = "dummy" #dict_folderを取得するのに使用
    pd.set_option('display.max_columns', 3000)
    pd.set_option('display.max_rows', 3000)
    if mode == "test":
        print("Test mode")
        start_date = '2020/01/01'
        # end_date = '2020/05/31'
        end_date = (dt.now() + timedelta(days=1)).strftime('%Y/%m/%d')
        term_start_date = '20200501'
        term_end_date = '20200531'
    elif mode == "init":
        start_date = '2019/01/01'
        end_date = (dt.now() + timedelta(days=1)).strftime('%Y/%m/%d')
        term_start_date = '20190101'
        term_end_date = (dt.now() + timedelta(days=1)).strftime('%Y%m%d')
    elif mode == "prod":
        start_date = (dt.now() + timedelta(days=-90)).strftime('%Y/%m/%d')
        end_date = (dt.now() + timedelta(days=1)).strftime('%Y/%m/%d')
        term_start_date = (dt.now() + timedelta(days=-9)).strftime('%Y%m%d')
        term_end_date = (dt.now() + timedelta(days=1)).strftime('%Y%m%d')
    elif mode == "sokuho":
        start_date = (dt.now() + timedelta(days=-90)).strftime('%Y/%m/%d')
        end_date = (dt.now() + timedelta(days=1)).strftime('%Y/%m/%d')
        term_start_date = (dt.now()).strftime('%Y%m%d')
        term_end_date = (dt.now()).strftime('%Y%m%d')
        sokuho = Sokuho(end_date, test_flag)
        sokuho.export_pbi_real_data()

    print("MODE:" + str(args[1]) + "  update_start_date:" + term_start_date + " update_end_date:" + term_end_date)

    av = AutoVote(start_date, end_date, term_start_date, term_end_date, test_flag)
    av.export_bet_csv()
    av.export_pbi_data()
