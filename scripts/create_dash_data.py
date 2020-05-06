from modules.lb_extract import LBExtract
from datetime import datetime as dt
from datetime import timedelta
import os
import sys

from ftplib import FTP_TLS
import my_config as mc
import shutil

test_flag = True # Trueの場合はＦＴＰアップを行わない

args = sys.argv
print("------------- upload file to Azure ----------------")
print(args)
print("mode：" + args[1])  # init or daily or real
run_mode = args[1]


stored_folder_path = "./scripts/data"
## clean folder
if not os.path.exists(stored_folder_path + "/race/"): os.makedirs(stored_folder_path + "/race/")
if not os.path.exists(stored_folder_path + "/raceuma/"): os.makedirs(stored_folder_path + "/raceuma/")
if not os.path.exists(stored_folder_path + "/bet/"): os.makedirs(stored_folder_path + "/bet/")
if not os.path.exists(stored_folder_path + "/haraimodoshi/"): os.makedirs(stored_folder_path + "/haraimodoshi/")
if not os.path.exists(stored_folder_path + "/current/"): os.makedirs(stored_folder_path + "/current/")

## create data

if run_mode == "init":
    if test_flag:
        start_date = '2020/4/01'
    else:
        start_date = '2019/12/01'
elif run_mode == "daily":
    start_date = (dt.now() + timedelta(days=-1)).strftime('%Y/%m/%d')
else:
    start_date = (dt.now() + timedelta(days=0)).strftime('%Y/%m/%d')

end_date = (dt.now() + timedelta(days=0)).strftime('%Y/%m/%d')
mock_flag = False


ext = LBExtract(start_date, end_date, mock_flag)

base_race_df = ext.get_race_table_for_view()[["データ区分","競走コード", "月日", "距離", "トラック種別コード", "競走番号", "場名", "競走種別コード", "競走条件名称", "トラックコード",
                                              "発走時刻", "頭数", "天候コード", "前３ハロン", "前４ハロン", "後３ハロン", "後４ハロン", "馬場状態コード", "前半タイム", "予想勝ち指数",
                                              "ペース", "予想決着指数", "投票フラグ", "波乱度", "ラップタイム", "競走名略称", "メインレース", "競走名", "UMAREN_ARE", "UMATAN_ARE", "SANRENPUKU_ARE", "UMAREN_ARE_RATE", "UMATAN_ARE_RATE", "SANRENPUKU_ARE_RATE"]]
base_race_df.loc[:, "年月"] = base_race_df["月日"].apply(lambda x: x.strftime("%Y%m"))
base_race_df.loc[:, "競走条件名称"] = base_race_df["競走条件名称"].replace(' ', '').replace('　', '')
base_raceuma_df = ext.get_raceuma_table_base()[["データ区分", "競走コード", "馬番", "枠番", "馬名", "予想人気", "血統登録番号", "性別コード", "年月日", "タイム", "タイム指数", "予想タイム指数", "予想タイム指数順位", "デフォルト得点", "得点",
                                                "馬券評価順位", "単勝配当", "複勝配当", "単勝オッズ", "単勝人気", "単勝支持率", "複勝オッズ1", "複勝オッズ2", "先行指数", "先行率", "予想展開", "展開コード", "騎手名",
                                                "馬齢", "調教師名", "負担重量", "馬体重", "異常区分コード", "確定着順", "コーナー順位1", "コーナー順位2", "コーナー順位3", "コーナー順位4", "上がりタイム",
                                                "所属", "得点V3", "WIN_RATE", "JIKU_RATE", "ANA_RATE", "WIN_RANK", "JIKU_RANK", "ANA_RANK", "SCORE", "SCORE_RANK",
                                                "CK1_RATE", "CK2_RATE", "CK3_RATE", "CK1_RANK", "CK2_RANK", "CK3_RANK"]]
base_raceuma_df.loc[:, "年月"] = base_raceuma_df["年月日"].apply(lambda x: x.strftime("%Y%m"))
base_bet_df = ext.get_bet_table_base()[["競走コード", "式別", "結果", "日付", "金額", "番号"]]
base_bet_df.loc[:, "年月"] = base_bet_df["日付"].apply(lambda x: x.strftime("%Y%m"))
base_haraimodoshi_df = ext.get_haraimodoshi_table_base()
base_haraimodoshi_df.loc[:, "年月"] = base_haraimodoshi_df["データ作成年月日"].apply(lambda x: x.strftime("%Y%m"))

if test_flag:
    if not base_race_df.empty: print(base_race_df.iloc[0])
    if not base_raceuma_df.empty: print(base_raceuma_df.iloc[0])
    if not base_bet_df.empty: print(base_bet_df.iloc[0])
    if not base_haraimodoshi_df.empty: print(base_haraimodoshi_df.iloc[0])

if run_mode == "real":
    base_race_df.to_pickle(stored_folder_path + "/current/race_df.pickle")
    base_raceuma_df.to_pickle(stored_folder_path + "/current/raceuma_df.pickle")
    base_bet_df.to_pickle(stored_folder_path + "/current/bet_df.pickle")
    base_haraimodoshi_df.to_pickle(stored_folder_path + "/current/haraimodoshi_df.pickle")
else:
    start_dt = dt.strptime(start_date, "%Y/%m/%d")
    end_dt = dt.strptime(end_date, "%Y/%m/%d")
    days_num = (end_dt - start_dt).days + 1

    yearmonth_list = []
    for i in range(days_num):
        yearmonth_list.append((start_dt + timedelta(days=i)).strftime("%Y%m"))

    yearmonth_list = sorted(list(set(yearmonth_list)))
    for ym in yearmonth_list:
        print("Yearmonth:", ym)
        ym_race_df = base_race_df[base_race_df["年月"] == ym]
        ym_race_df.to_pickle(stored_folder_path + "/race/" + ym + ".pickle")
        ym_raceuma_df = base_raceuma_df[base_raceuma_df["年月"] == ym]
        ym_raceuma_df.to_pickle(stored_folder_path + "/raceuma/" + ym + ".pickle")
        ym_bet_df = base_bet_df[base_bet_df["年月"] == ym]
        ym_bet_df.to_pickle(stored_folder_path + "/bet/" + ym + ".pickle")
        ym_haraimodoshi_df = base_haraimodoshi_df[base_haraimodoshi_df["年月"] == ym]
        ym_haraimodoshi_df.to_pickle(stored_folder_path + "/haraimodoshi/" + ym + ".pickle")

    ## currentファイル(exp_data取得)
    intermediate_folder = "E:\python/intermediate"
    if not os.path.exists(stored_folder_path + "/current/lb_v4_predict/raceuma_lgm"): os.makedirs(stored_folder_path + "/current/lb_v4_predict/raceuma_lgm")
    if not os.path.exists(stored_folder_path + "/current/lb_v5_predict/raceuma_lgm"): os.makedirs(stored_folder_path + "/current/lb_v5_predict/raceuma_lgm")
    if not os.path.exists(stored_folder_path + "/current/lbr_v1_predict/race_lgm"): os.makedirs(stored_folder_path + "/current/lbr_v1_predict/race_lgm")

    int_end_date = (dt.now() + timedelta(days=0)).strftime('%Y%m%d')

    if os.path.exists(intermediate_folder + "/lb_v4_predict/raceuma_lgm/" + int_end_date + "_exp_data.pkl"):
        shutil.copy(intermediate_folder + "/lb_v4_predict/raceuma_lgm/" + int_end_date + "_exp_data.pkl", stored_folder_path + "/current/lb_v4_predict/raceuma_lgm/")
    if os.path.exists(intermediate_folder + "/lb_v5_predict/raceuma_lgm/" + int_end_date + "_exp_data.pkl"):
        shutil.copy(intermediate_folder + "/lb_v5_predict/raceuma_lgm/" + int_end_date + "_exp_data.pkl", stored_folder_path + "/current/lb_v5_predict/raceuma_lgm/")
    if os.path.exists(intermediate_folder + "/lbr_v1_predict/race_lgm/" + int_end_date + "_exp_data.pkl"):
        shutil.copy(intermediate_folder + "/lbr_v1_predict/race_lgm/" + int_end_date + "_exp_data.pkl", stored_folder_path + "/current/lbr_v1_predict/race_lgm/")


## ftp upload
def connect():
    ftp = FTP_TLS()
    ftp.debugging = 2
    ftp.connect(mc.FTP_HOST)
    ftp.login(mc.FTP_USER, mc.FTP_PASSWORD)
    return ftp

if not test_flag:
    ftp = connect()
    ftp.cwd('site/wwwroot/static/data/')
    if run_mode == "real":
        with open(stored_folder_path + "/current/race_df.pickle", 'rb') as f:
            ftp.storbinary('STOR {}'.format("race_df.pickle"), f)
        with open(stored_folder_path + "/current/raceuma_df.pickle", 'rb') as f:
            ftp.storbinary('STOR {}'.format("raceuma_df.pickle"), f)
        with open(stored_folder_path + "/current/bet_df.pickle", 'rb') as f:
            ftp.storbinary('STOR {}'.format("bet_df.pickle"), f)
        with open(stored_folder_path + "/current/haraimodoshi_df.pickle", 'rb') as f:
            ftp.storbinary('STOR {}'.format("haraimodoshi_df.pickle"), f)
    else:
        # race
        race_list = os.listdir(stored_folder_path + "/race/")
        ftp.cwd('./race/')
        for file in race_list:
            with open(stored_folder_path + "/race/" + file, 'rb') as f:
                ftp.storbinary('STOR {}'.format(file), f)

        # raceuma
        raceuma_list = os.listdir(stored_folder_path + "/raceuma/")
        ftp.cwd('../raceuma/')
        for file in raceuma_list:
            with open(stored_folder_path + "/raceuma/" + file, 'rb') as f:
                ftp.storbinary('STOR {}'.format(file), f)

        # bet
        bet_list = os.listdir(stored_folder_path + "/bet/")
        ftp.cwd('../bet/')
        for file in bet_list:
            with open(stored_folder_path + "/bet/" + file, 'rb') as f:
                ftp.storbinary('STOR {}'.format(file), f)

        # haraimodoshi
        haraimodoshi_list = os.listdir(stored_folder_path + "/haraimodoshi/")
        ftp.cwd('../haraimodoshi/')
        for file in haraimodoshi_list:
            with open(stored_folder_path + "/haraimodoshi/" + file, 'rb') as f:
                ftp.storbinary('STOR {}'.format(file), f)

        # current
        if os.path.exists(stored_folder_path + "/current/lb_v4_predict/raceuma_lgm/" + int_end_date + "_exp_data.pkl"):
            ftp.cwd('../current/lb_v4/')
            with open(stored_folder_path + "/current/lb_v4_predict/raceuma_lgm/" + int_end_date + "_exp_data.pkl", 'rb') as f:
                ftp.storbinary('STOR {}'.format(int_end_date + "_exp_data.pkl"), f)
        if os.path.exists(stored_folder_path + "/current/lb_v5_predict/raceuma_lgm/" + int_end_date + "_exp_data.pkl"):
            ftp.cwd('../lb_v5/')
            with open(stored_folder_path + "/current/lb_v5_predict/raceuma_lgm/" + int_end_date + "_exp_data.pkl", 'rb') as f:
                ftp.storbinary('STOR {}'.format(int_end_date + "_exp_data.pkl"), f)
        if os.path.exists(stored_folder_path + "/current/lbr_v1_predict/race_lgm/" + int_end_date + "_exp_data.pkl"):
            ftp.cwd('../lbr_v1/')
            with open(stored_folder_path + "/current/lbr_v1_predict/race_lgm/" + int_end_date + "_exp_data.pkl", 'rb') as f:
                ftp.storbinary('STOR {}'.format(int_end_date + "_exp_data.pkl"), f)

    ftp.quit()

    ## clean folder

    shutil.rmtree(stored_folder_path + "/race/")
    shutil.rmtree(stored_folder_path + "/raceuma/")
    shutil.rmtree(stored_folder_path + "/bet/")
    shutil.rmtree(stored_folder_path + "/haraimodoshi/")
    shutil.rmtree(stored_folder_path + "/current/")
