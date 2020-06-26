from scripts.jra_auto_vote import Ld, Simlation, AutoVote
import my_config as mc
import modules.util as mu
import pandas as pd

from datetime import datetime as dt
from datetime import timedelta
import sys
import os

class PbiExport():
    def __init__(self, start_date, end_date, test_flag):
        self.start_date = start_date[0:4] + "/" + start_date[4:6] + "/" + start_date[6:8]
        self.end_date = end_date[0:4] + "/" + end_date[4:6] + "/" + end_date[6:8]
        ld_start_date = start_date[0:4] + "/" + start_date[4:6] + "/01"
        ld_end_date = end_date[0:4] + "/" + end_date[4:6] + "/" + end_date[6:8]
        term_start_date = start_date[0:6] + "01"
        term_end_date = end_date
        base_start_date = start_date[0:4] + "/01/01"
        base_end_date = end_date[0:4] + "/" + end_date[4:6] + "/" + end_date[6:8]
        self.ld = Ld("dummy", ld_start_date, ld_end_date, False, test_flag)
        self.ld.set_race_file_df()
        self.ld.set_result_df()
        self.ld.set_pred_df()
        self.av = AutoVote(base_start_date, base_end_date, term_start_date, term_end_date, test_flag)
        self.pbi_path = mc.return_jrdb_path() + "pbi/"


    def export_race_pred_data(self):
        pred_ld = Ld("dummy", self.start_date, self.end_date, False, False)
        pred_ld.set_race_file_df()
        pred_ld.set_pred_df()
        race_df = pred_ld.ext.get_race_before_table_base()
        race_course_df = self.ld.tf.cluster_course_df(race_df, self.ld.dict_path)[["RACE_KEY", "course_cluster"]].drop_duplicates()
        race_df = pd.merge(race_df, race_course_df, on="RACE_KEY")
        race_df = race_df[["RACE_KEY", "NENGAPPI", "距離", "芝ダ障害コード", "内外", "条件", "種別", "グレード", "レース名９文字", "WIN5フラグ", "場名",
                           "天候コード", "芝馬場状態コード", "芝馬場差", "ダ馬場状態コード", "ダ馬場差", "連続何日目", "芝種類", "草丈", "転圧",
                           "凍結防止剤", "course_cluster"]].copy()
        race_df = race_df.groupby("RACE_KEY").first().reset_index()
        race_df.loc[:, "年月"] = race_df["NENGAPPI"].str[0:6]
        race_df.loc[:, "レースNo"] = race_df["RACE_KEY"].str[6:8]
        race_df.loc[:, "種別"] = race_df["種別"].apply(lambda x: mu.convert_shubetsu(x))
        race_df.loc[:, "天候"] = race_df["天候コード"].apply(lambda x: mu.convert_tenko(x))
        race_df.loc[:, "芝ダ"] = race_df["芝ダ障害コード"].apply(lambda x: mu.convert_shida(x))
        race_df.loc[:, "条件"] = race_df["条件"].apply(lambda x: mu.convert_joken(x))
        race_df.loc[:, "芝種類"] = race_df["芝種類"].apply(lambda x: mu.convert_shibatype(x))
        race_df.loc[:, "コース名"] = race_df.apply(lambda x: self._get_course_name(x), axis=1)
        race_df.drop(["芝ダ障害コード", "芝馬場状態コード", "ダ馬場状態コード", "天候コード", "内外"], axis=1, inplace=True)
        race_df.to_csv(self.pbi_path + "race.csv", index=False, header=True)

        raceuma_df = pred_ld.ext.get_raceuma_before_table_base()
        horse_df = pred_ld.ext.get_horse_table_base()
        uma_mark_df = pred_ld.uma_mark_df[["RACE_KEY", "UMABAN", "win_std", "jiku_std", "ana_std", "nige_std", "agari_std", "ten_std"]].copy()

        raceuma_df = raceuma_df[["RACE_KEY", "UMABAN", "NENGAPPI", "血統登録番号", "騎手名", "負担重量", "調教師名", "調教師所属", "枠番", "放牧先", "馬名",
                                 "調教タイプ", "調教コース種別", "調教重点"]].copy()
        horse_df = horse_df[["血統登録番号", "NENGAPPI", "性別コード", "父馬名", "母父馬名", "生産者名", "産地名", "父系統コード", "母父系統コード"]].copy()
        raceuma_df = pd.merge(raceuma_df, horse_df, on=["血統登録番号", "NENGAPPI"])
        raceuma_df = pd.merge(raceuma_df, self.av.target_mark_df, on=["RACE_KEY", "UMABAN"])
        raceuma_df = pd.merge(raceuma_df, uma_mark_df, on=["RACE_KEY", "UMABAN"])
        raceuma_df.loc[:, "年月"] = raceuma_df["NENGAPPI"].str[0:6]
        raceuma_df.loc[:, "調教タイプ"] = raceuma_df["調教タイプ"].apply(lambda x: mu.convert_chokyo_type(x))
        raceuma_df.loc[:, "調教コース種別"] = raceuma_df["調教コース種別"].apply(lambda x: mu.convert_chokyo_course_shubetsu(x))
        raceuma_df.loc[:, "調教重点"] = raceuma_df["調教重点"].apply(lambda x: mu.convert_chokyo_juten(x))
        raceuma_df.loc[:, "性別"] = raceuma_df["性別コード"].apply(lambda x: mu.convert_sex(x))
        raceuma_df.loc[:, "父系統"] = raceuma_df["父系統コード"].apply(lambda x: mu.convert_keito(x))
        raceuma_df.loc[:, "母父系統"] = raceuma_df["母父系統コード"].apply(lambda x: mu.convert_keito(x))
        raceuma_df.loc[:, "UMA_KEY"] = raceuma_df["血統登録番号"]
        raceuma_df.drop(["血統登録番号", "NENGAPPI", "性別コード", "父系統コード", "母父系統コード"], axis=1, inplace=True)
        raceuma_df.to_csv(self.pbi_path + "raceuma.csv", index=False, header=True)


    def export_race_result_data(self):
        race_df = self.ld.ext.get_race_table_base()
        race_result_df = self.ld.race_result_df[["RACE_KEY", "RACE_ID", "fa_1", "fa_2", "fa_3", "fa_4", "fa_5", "TB_ZENGO", "TB_UCHISOTO"]].drop_duplicates()
        race_df = pd.merge(race_df, race_result_df, on="RACE_KEY")
        race_course_df = self.ld.tf.cluster_course_df(race_df, self.ld.dict_path)[["RACE_KEY", "course_cluster"]].drop_duplicates()
        race_df = pd.merge(race_df, race_course_df, on="RACE_KEY")
        race_df = race_df[["RACE_ID", "RACE_KEY", "NENGAPPI", "距離", "芝ダ障害コード", "内外", "条件", "種別", "グレード", "レース名９文字", "WIN5フラグ", "場名",
                           "天候コード", "芝馬場状態コード", "芝馬場差", "ダ馬場状態コード", "ダ馬場差", "連続何日目", "芝種類", "草丈", "転圧",
                           "凍結防止剤", "ハロンタイム０１", "ハロンタイム０２", "ハロンタイム０３", "ハロンタイム０４", "ハロンタイム０５", "ハロンタイム０６",
                           "ハロンタイム０７", "ハロンタイム０８", "ハロンタイム０９", "ハロンタイム１０", "ハロンタイム１１", "ハロンタイム１２", "ハロンタイム１３",
                           "ハロンタイム１４", "ハロンタイム１５", "ハロンタイム１６", "ハロンタイム１７", "ハロンタイム１８", "レースコメント",
                           "ラスト５ハロン", "ラスト４ハロン", "ラスト３ハロン", "ラスト２ハロン", "ラスト１ハロン", "RAP_TYPE",
                           "fa_1", "fa_2", "fa_3", "fa_4", "fa_5", "TRACK_BIAS_ZENGO", "TRACK_BIAS_UCHISOTO", "TB_ZENGO", "TB_UCHISOTO", "course_cluster"]].copy()
        race_df = race_df.groupby("RACE_KEY").first().reset_index()
        race_df.loc[:, "年月"] = race_df["NENGAPPI"].str[0:6]
        race_df.loc[:, "レースNo"] = race_df["RACE_KEY"].str[6:8]
        race_df.loc[:, "種別"] = race_df["種別"].apply(lambda x: mu.convert_shubetsu(x))
        race_df.loc[:, "天候"] = race_df["天候コード"].apply(lambda x: mu.convert_tenko(x))
        race_df.loc[:, "芝ダ"] = race_df["芝ダ障害コード"].apply(lambda x: mu.convert_shida(x))
        race_df.loc[:, "馬場状態"] = race_df.apply(lambda x: self._convert_babajotai(x["芝馬場状態コード"]) if x["芝ダ障害コード"] == "1" else self._convert_babajotai(x["ダ馬場状態コード"]), axis=1 )
        race_df.loc[:, "条件"] = race_df["条件"].apply(lambda x: mu.convert_joken(x))
        race_df.loc[:, "芝種類"] = race_df["芝種類"].apply(lambda x: mu.convert_shibatype(x))
        race_df.loc[:, "コース名"] = race_df.apply(lambda x: self._get_course_name(x), axis=1)
        race_df.drop(["芝ダ障害コード", "芝馬場状態コード", "ダ馬場状態コード", "天候コード", "内外"], axis=1, inplace=True)
        print(race_df.iloc[0])
        ym_list = race_df["年月"].drop_duplicates()
        for ym in ym_list:
            temp_df = race_df.query(f"年月 == '{ym}'").copy()
            temp_df.to_csv(self.pbi_path + "race_result/"+ ym + ".csv", index=False, header=True)

    def _convert_babajotai(self, joken):
        if joken == "1": return "良"
        if joken == "2": return "稍重"
        if joken == "3": return "重"
        if joken == "4": return "不良"
        else: return ""

    def _get_course_name(self, sr):
        soto = "外" if sr["内外"] == "2" else ""
        return sr["場名"] + sr["芝ダ"] + str(sr["距離"]) +"m" + soto

    def export_raceuma_result_data(self):
        raceuma_df = self.ld.ext.get_raceuma_table_base()
        horse_df = self.ld.ext.get_horse_table_base()
        uma_mark_df = self.ld.uma_mark_df[["RACE_KEY", "UMABAN", "win_std", "jiku_std", "ana_std", "nige_std", "agari_std", "ten_std"]].copy()

        raceuma_df = raceuma_df[["RACE_KEY", "UMABAN", "NENGAPPI", "血統登録番号", "騎手名", "負担重量", "調教師名", "調教師所属", "枠番", "放牧先", "馬名",
                                 "調教タイプ", "調教コース種別", "調教重点", "着順", "確定単勝オッズ", "確定単勝人気順位", "ＩＤＭ結果", "テン指数結果", "上がり指数結果",
                                 "ペース指数結果", "前３Ｆタイム", "後３Ｆタイム", "コーナー順位１", "コーナー順位２", "コーナー順位３", "コーナー順位４", "馬体重",
                                 "レース脚質", "単勝", "複勝", "レース馬コメント"]].copy()
        horse_df = horse_df[["血統登録番号", "NENGAPPI", "性別コード", "父馬名", "母父馬名", "生産者名", "産地名", "父系統コード", "母父系統コード"]].copy()
        raceuma_df = pd.merge(raceuma_df, horse_df, on=["血統登録番号", "NENGAPPI"])
        raceuma_df = pd.merge(raceuma_df, self.av.target_mark_df, on=["RACE_KEY", "UMABAN"])
        raceuma_df = pd.merge(raceuma_df, uma_mark_df, on=["RACE_KEY", "UMABAN"])
        raceuma_df.loc[:, "年月"] = raceuma_df["NENGAPPI"].str[0:6]
        raceuma_df.loc[:, "調教タイプ"] = raceuma_df["調教タイプ"].apply(lambda x: mu.convert_chokyo_type(x))
        raceuma_df.loc[:, "調教コース種別"] = raceuma_df["調教コース種別"].apply(lambda x: mu.convert_chokyo_course_shubetsu(x))
        raceuma_df.loc[:, "調教重点"] = raceuma_df["調教重点"].apply(lambda x: mu.convert_chokyo_juten(x))
        raceuma_df.loc[:, "性別"] = raceuma_df["性別コード"].apply(lambda x: mu.convert_sex(x))
        raceuma_df.loc[:, "脚質"] = raceuma_df["レース脚質"].apply(lambda x: mu.convert_kyakushitsu(x))
        raceuma_df.loc[:, "父系統"] = raceuma_df["父系統コード"].apply(lambda x: mu.convert_keito(x))
        raceuma_df.loc[:, "母父系統"] = raceuma_df["母父系統コード"].apply(lambda x: mu.convert_keito(x))
        raceuma_df.loc[:, "UMA_KEY"] = raceuma_df["血統登録番号"]
        raceuma_df.drop(["血統登録番号", "NENGAPPI", "性別コード", "レース脚質", "父系統コード", "母父系統コード"], axis=1, inplace=True)
        print(raceuma_df.iloc[0])
        ym_list = raceuma_df["年月"].drop_duplicates()
        for ym in ym_list:
            temp_df = raceuma_df.query(f"年月 == '{ym}'").copy()
            temp_df.to_csv(self.pbi_path + "raceuma_result/"+ ym + ".csv", index=False, header=True)

    def export_bet_data(self):
        target_df = self.av.target_mark_df.copy()
        self.ld.set_haraimodoshi_df()
        haraimodoshi_dict = self.ld.dict_haraimodoshi
        tansho_df = haraimodoshi_dict["tansho_df"]
        fukusho_df = haraimodoshi_dict["fukusho_df"]
        umaren_df = haraimodoshi_dict["umaren_df"]
        wide_df = haraimodoshi_dict["wide_df"]
        umatan_df = haraimodoshi_dict["umatan_df"]
        sanrenpuku_df = haraimodoshi_dict["sanrenpuku_df"]
        tansho_ipat_bet_df, tansho_target_bet_df = self.av.get_tansho_bet_df(target_df)
        fukusho_ipat_bet_df, fukusho_target_bet_df = self.av.get_fukusho_bet_df(target_df)
        umaren_ipat_bet_df, umaren_target_bet_df = self.av.get_umaren_bet_df(target_df, target_df)
        umatan_ipat_bet_df, umatan_target_bet_df = self.av.get_umatan_bet_df(target_df, target_df)
        wide_ipat_bet_df, wide_target_bet_df = self.av.get_wide_bet_df(target_df, target_df)
        sanrenpuku_uma1_df = target_df.query("印 == '◎ '").copy()
        sanrenpuku_uma2_df = target_df.query("印 in ['× ', '△ ', '▲ ', '○ ']").copy()
        sanrenpuku_uma3_df = target_df.query("印 != '◎ '").copy()
        sanrenpuku_ipat_bet_df, sanrenpuku_target_bet_df = self.av.get_sanrenpuku_bet_df(sanrenpuku_uma1_df, sanrenpuku_uma2_df, sanrenpuku_uma3_df)
        target_bet_df = pd.concat([tansho_target_bet_df, fukusho_target_bet_df, umaren_target_bet_df, umatan_target_bet_df, wide_target_bet_df, sanrenpuku_target_bet_df])
        target_bet_df = target_bet_df.sort_values(["RACE_ID", "エリア", "券種", "購入金額", "目１", "目２", "目３"])
        target_bet_df.loc[:, "年月"] = target_bet_df["RACE_ID"].str[0:6]
        target_bet_df = pd.merge(target_bet_df, self.av.ld.race_file_df[["RACE_KEY", "RACE_ID"]], on="RACE_ID")
        bet_tansho_df = pd.merge(target_bet_df.query("券種 == '0'"), tansho_df, on="RACE_KEY").reset_index(drop=True)
        bet_tansho_df.loc[:, "買い目"] = bet_tansho_df.apply(lambda x: x["目１"], axis=1)
        bet_tansho_df.loc[:, "結果"] = bet_tansho_df.apply(lambda x: x["払戻"] if x["目１"] == x["UMABAN"] else 0, axis=1)
        bet_fukusho_df = pd.merge(target_bet_df.query("券種 == '1'"), fukusho_df, on="RACE_KEY").reset_index(drop=True)
        bet_fukusho_df.loc[:, "買い目"] = bet_fukusho_df.apply(lambda x: x["目１"], axis=1)
        bet_fukusho_df.loc[:, "結果"] = bet_fukusho_df.apply(lambda x: x["払戻"] if x["目１"] == x["UMABAN"] else 0, axis=1)
        bet_umaren_df = pd.merge(target_bet_df.query("券種 == '3'"), umaren_df, on="RACE_KEY").reset_index(drop=True)
        bet_umaren_df.loc[:, "買い目"] = bet_umaren_df.apply(lambda x: sorted([x["目１"], x["目２"]]), axis=1)
        bet_umaren_df.loc[:, "結果"] = bet_umaren_df.apply(lambda x: x["払戻"] if x["買い目"][0] == x["UMABAN"][0] and x["買い目"][1] == x["UMABAN"][1] else 0, axis=1)
        bet_wide_df = pd.merge(target_bet_df.query("券種 == '4'"), wide_df, on="RACE_KEY").reset_index(drop=True)
        bet_wide_df.loc[:, "買い目"] = bet_wide_df.apply(lambda x: sorted([x["目１"], x["目２"]]), axis=1)
        bet_wide_df.loc[:, "結果"] = bet_wide_df.apply(lambda x: x["払戻"] if x["買い目"][0] == x["UMABAN"][0] and x["買い目"][1] == x["UMABAN"][1] else 0, axis=1)
        bet_umatan_df = pd.merge(target_bet_df.query("券種 == '5'"), umatan_df, on="RACE_KEY").reset_index(drop=True)
        bet_umatan_df.loc[:, "買い目"] = bet_umatan_df.apply(lambda x: [x["目１"], x["目２"]], axis=1)
        bet_umatan_df.loc[:, "結果"] = bet_umatan_df.apply(lambda x: x["払戻"] if x["目１"] == x["UMABAN"][0] and x["目２"] == x["UMABAN"][1] else 0, axis=1)
        bet_sanrenpuku_df = pd.merge(target_bet_df.query("券種 == '6'"), sanrenpuku_df, on="RACE_KEY").reset_index(drop=True)
        bet_sanrenpuku_df.loc[:, "買い目"] = bet_sanrenpuku_df.apply(lambda x: sorted([x["目１"], x["目２"], x["目３"]]), axis=1)
        bet_sanrenpuku_df.loc[:, "結果"] = bet_sanrenpuku_df.apply(lambda x: x["払戻"] if x["買い目"][0] == x["UMABAN"][0] and x["買い目"][1] == x["UMABAN"][1] and x["買い目"][2] == x["UMABAN"][2] else 0, axis=1)
        bet_df = pd.concat([bet_tansho_df[["RACE_KEY", "券種", "購入金額", "買い目", "結果", "年月"]],
                            bet_fukusho_df[["RACE_KEY", "券種", "購入金額", "買い目", "結果", "年月"]],
                            bet_umaren_df[["RACE_KEY", "券種", "購入金額", "買い目", "結果", "年月"]],
                            bet_wide_df[["RACE_KEY", "券種", "購入金額", "買い目", "結果", "年月"]],
                            bet_umatan_df[["RACE_KEY", "券種", "購入金額", "買い目", "結果", "年月"]],
                            bet_sanrenpuku_df[["RACE_KEY", "券種", "購入金額", "買い目", "結果", "年月"]]])
        ym_list = bet_df["年月"].drop_duplicates()
        for ym in ym_list:
            temp_df = bet_df.query(f"年月 == '{ym}'").copy()
            temp_df.to_csv(self.pbi_path + "target_bet/"+ ym + ".csv", index=False, header=True)


if __name__ == "__main__":
    args = sys.argv
    print(args)
    print("mode：" + args[1])  # test or init or prod
    test_flag = False
    mode = args[1]
    dict_path = mc.return_jra_path(test_flag)
    version_str = "dummy" #dict_folderを取得するのに使用
    pd.set_option('display.max_columns', 3000)
    pd.set_option('display.max_rows', 3000)
    if mode == "test":
        print("Test mode")
        start_date = '20200618'
        # end_date = '20200531'
        end_date = (dt.now() + timedelta(days=1)).strftime('%Y%m%d')
    elif mode == "init":
        start_date = '20190101'
        end_date = (dt.now() + timedelta(days=1)).strftime('%Y%m%d')
    elif mode == "prod":
        start_date = (dt.now() + timedelta(days=-3)).strftime('%Y%m%d')
        end_date = (dt.now() + timedelta(days=1)).strftime('%Y%m%d')
    print("MODE:" + str(args[1]) + "  start_date:" + start_date + " end_date:" + end_date)

    pe = PbiExport(start_date, end_date, test_flag)
    pe.export_race_pred_data()
    pe.export_race_result_data()
    pe.export_raceuma_result_data()
    pe.export_bet_data()
