from modules.lb_extract import LBExtract
from modules.lb_simulation import LBSimulation
import pyodbc
import pandas as pd
import numpy as np
import my_config as mc
import pickle

## 各指数の適切な配分を計算する
## 勝ち指数：単勝回収率・勝率を重視
## 軸指数：複勝回収率・複勝率を重視
## 穴指数：１番人気との馬連の回収率・的中率を重視


# データ取得

start_date = '2019/01/01'
end_date = '2019/12/31'
mock_flag = False

ext = LBExtract(start_date, end_date, mock_flag)
sim = LBSimulation(start_date, end_date, mock_flag)

"""
def connect_baoz_my_mdb():
    conn_str = (
        r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
        r'DBQ=C:\BaoZ\DB\MasterDB\MyDB.MDB;'
    )
    cnxn = pyodbc.connect(conn_str)
    return cnxn


cnxn = connect_baoz_my_mdb()
select_sql_v1 = 'SELECT * FROM 地方競馬レース馬V1 WHERE target_date >= #' + \
             start_date + '# AND target_date <= #' + end_date + '#'
lb_v1_df = pd.read_sql(select_sql_v1, cnxn)

select_sql = 'SELECT * FROM 地方競馬レース馬V2 WHERE target_date >= #' + \
             start_date + '# AND target_date <= #' + end_date + '#'
lb_v2_df = pd.read_sql(select_sql, cnxn)
"""

dict_path = mc.return_base_path(False)
intermediate_folder = dict_path + 'intermediate/'

with open(intermediate_folder + 'lb_v1_lb_v1/raceuma_ens/export_data.pkl', 'rb') as f:
    lb_v1_df = pickle.load(f)
with open(intermediate_folder + 'lb_v2_lb_v2/raceuma_ens/export_data.pkl', 'rb') as f:
    lb_v2_df = pickle.load(f)

lb_v1_win_df = lb_v1_df[lb_v1_df["target"] == "WIN_FLAG"][["RACE_KEY", "UMABAN", "predict_std"]].rename(columns={"predict_std": "偏差v1"})
lb_v2_win_df = lb_v2_df[lb_v2_df["target"] == "WIN_FLAG"][["RACE_KEY", "UMABAN", "predict_std"]].rename(columns={"predict_std": "偏差v2"})
lb_v1_jiku_df = lb_v1_df[lb_v1_df["target"] == "JIKU_FLAG"][["RACE_KEY", "UMABAN", "predict_std"]].rename(columns={"predict_std": "偏差v1"})
lb_v2_jiku_df = lb_v2_df[lb_v2_df["target"] == "JIKU_FLAG"][["RACE_KEY", "UMABAN", "predict_std"]].rename(columns={"predict_std": "偏差v2"})
lb_v1_ana_df = lb_v1_df[lb_v1_df["target"] == "ANA_FLAG"][["RACE_KEY", "UMABAN", "predict_std"]].rename(columns={"predict_std": "偏差v1"})
lb_v2_ana_df = lb_v2_df[lb_v2_df["target"] == "ANA_FLAG"][["RACE_KEY", "UMABAN", "predict_std"]].rename(columns={"predict_std": "偏差v2"})

raceuma_df = ext.get_raceuma_table_base()
result_df = raceuma_df[["RACE_KEY", "UMABAN", "確定着順", "単勝配当", "複勝配当"]].copy()
result_df.loc[:, "勝"] = result_df["確定着順"].apply(lambda x: 1 if x == 1 else 0)
result_df.loc[:, "連"] = result_df["確定着順"].apply(lambda x: 1 if x in (1, 2) else 0)
result_df.loc[:, "複"] = result_df["確定着順"].apply(lambda x: 1 if x in (1, 2, 3) else 0)

def sim_rate(df1, df2, result_df):
    merge_df = pd.merge(df1, df2, on =["RACE_KEY", "UMABAN"])
    merge_df = pd.merge(merge_df, result_df, on =["RACE_KEY", "UMABAN"])
    iter_range = 5
    lb_v1_rate = range(20, 81, iter_range)
    lb_v2_rate = range(20, 81, iter_range)

    lb_v1_list = []
    lb_v2_list = []

    cnt_list = []
    av_win_list = []
    av_ren_list = []
    av_fuku_list = []
    tan_ret_list = []
    fuku_ret_list = []

    for v1 in lb_v1_rate:
        for v2 in lb_v2_rate:
            if v1 + v2 == 100:
                # print(str(v1) + "  " + str(v2))
                merge_df["配分値"] = merge_df["偏差v1"] * v1 / 100 + merge_df["偏差v2"] * v2 / 100
                grouped = merge_df.groupby("RACE_KEY")
                merge_df["順位"] = grouped['配分値'].rank("dense", ascending=False)
                target_df = merge_df[merge_df["順位"] <= 2].copy()
                cnt_list.append(len(target_df))
                lb_v1_list.append(v1)
                lb_v2_list.append(v2)
                av_win_list.append(round(target_df["勝"].mean() * 100, 2))
                av_ren_list.append(round(target_df["連"].mean() * 100, 2))
                av_fuku_list.append(round(target_df["複"].mean() * 100, 2))
                tan_ret_list.append(round(target_df["単勝配当"].mean(), 2))
                fuku_ret_list.append(round(target_df["複勝配当"].mean(), 2))

    score_df = pd.DataFrame(
        data={'count': cnt_list, 'v1_rate': lb_v1_list, 'v2_rate': lb_v2_list,
              'av_win': av_win_list, 'av_ren': av_ren_list, 'av_fuku': av_fuku_list, 'tan_return': tan_ret_list,
              'fuku_return': fuku_ret_list},
        columns=['count', 'v1_rate', 'v2_rate', 'av_win', 'av_ren', 'av_fuku', 'tan_return', 'fuku_return']
    )
    score_df.loc[:, 'tan_return_rank'] = score_df['tan_return'].rank(ascending=False)
    score_df.loc[:, 'fuku_return_rank'] = score_df['tan_return'].rank(ascending=False)
    score_df.loc[:, 'av_win_rank'] = score_df['av_win'].rank(ascending=False)
    score_df.loc[:, 'av_ren_rank'] = score_df['av_ren'].rank(ascending=False)
    score_df.loc[:, 'av_fuku_rank'] = score_df['av_fuku'].rank(ascending=False)
    score_df.loc[:, 'win_rank'] = score_df['tan_return_rank'] + score_df['av_win_rank']
    score_df.loc[:, 'jiku_rank'] = score_df['fuku_return_rank'] + score_df['av_ren_rank']
    score_df.loc[:, 'ana_rank'] = score_df['tan_return_rank'] + score_df['fuku_return_rank']
    return score_df


def sim_rate_umaren(df1, df2, pair_df, umaren_df):
    merge_df = pd.merge(df1, df2, on =["RACE_KEY", "UMABAN"])
    merge_df = pd.merge(merge_df, result_df, on =["RACE_KEY", "UMABAN"])
    iter_range = 5
    lb_v1_rate = range(20, 81, iter_range)
    lb_v2_rate = range(20, 81, iter_range)

    lb_v1_list = []
    lb_v2_list = []

    cnt_list = []
    umaren_return_list = []
    umaren_hit_list = []
    tan_ret_list = []
    fuku_ret_list = []

    for v1 in lb_v1_rate:
        for v2 in lb_v2_rate:
            if v1 + v2 == 100:
                # print(str(v1) + "  " + str(v2))
                merge_df["配分値"] = merge_df["偏差v1"] * v1 / 100 + merge_df["偏差v2"] * v2 / 100
                grouped = merge_df.groupby("RACE_KEY")
                merge_df["順位"] = grouped['配分値'].rank("dense", ascending=False)
                candidate_df = merge_df[merge_df["順位"] <= 2].copy()
                target_df = pd.merge(candidate_df, pair_df, on ="RACE_KEY").fillna(0)
                target_df = pd.merge(target_df, umaren_df , on ="RACE_KEY", how="left").fillna(0)
                target_df["削除フラグ"] = target_df.apply(lambda x: 1 if x["UMABAN_x"] == x["UMABAN_y"] else 0, axis=1)
                target_df["削除フラグ"] = target_df.apply(lambda x: 1 if type(x["UMABAN"]) is int  else x["削除フラグ"], axis=1)
                target_df = target_df[target_df["削除フラグ"] == 0]
                target_df["結果"] = target_df.apply(lambda x: x["払戻"] if x["UMABAN_x"] in x["UMABAN"] and x["UMABAN_y"] in x["UMABAN"] else 0 , axis=1 )
                target_df["的中"] = target_df.apply(lambda x: 0 if x["結果"] == 0 else 1, axis=1)
                # print(target_df.head())
                cnt_list.append(len(target_df))
                lb_v1_list.append(v1)
                lb_v2_list.append(v2)
                umaren_return_list.append(round(target_df["結果"].mean() , 2))
                umaren_hit_list.append(round(target_df["的中"].sum()))
                tan_ret_list.append(round(target_df["単勝配当"].mean(), 2))
                fuku_ret_list.append(round(target_df["複勝配当"].mean(), 2))

    score_df = pd.DataFrame(
        data={'count': cnt_list, 'v1_rate': lb_v1_list, 'v2_rate': lb_v2_list,
              'umaren_return': umaren_return_list, 'umaren_hit': umaren_hit_list, 'tan_return': tan_ret_list,
              'fuku_return': fuku_ret_list},
        columns=['count', 'v1_rate', 'v2_rate', 'umaren_return', 'umaren_hit', 'tan_return', 'fuku_return']
    )
    score_df.loc[:, 'tan_return_rank'] = score_df['tan_return'].rank(ascending=False)
    score_df.loc[:, 'fuku_return_rank'] = score_df['tan_return'].rank(ascending=False)
    score_df.loc[:, 'umaren_return_rank'] = score_df['umaren_return'].rank(ascending=False)
    score_df.loc[:, 'umaren_hit_rank'] = score_df['umaren_hit'].rank(ascending=False)
    score_df.loc[:, 'ana_rank'] = score_df['umaren_return_rank'] + score_df['umaren_hit_rank']
    return score_df


pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 200)

print("----------- WIN_FLAG -----------------")
score_win_df = sim_rate(lb_v1_win_df, lb_v2_win_df, result_df)
print(score_win_df.sort_values('win_rank').head())

print("----------- JIKU_FLAG -----------------")
score_jiku_df = sim_rate(lb_v1_jiku_df, lb_v2_jiku_df, result_df)
print(score_jiku_df.sort_values('jiku_rank').head())

print("----------- ANA_FLAG -----------------")
sim.sim_umaren()
umaren_df = sim.result_umaren_df
ninki_df =raceuma_df[raceuma_df["単勝人気"] == 1][["RACE_KEY", "UMABAN"]]

score_ana_df = sim_rate_umaren(lb_v1_ana_df, lb_v2_ana_df, ninki_df, umaren_df)
print(score_ana_df.sort_values('ana_rank').head())

score_ana_df = sim_rate_umaren(lb_v1_win_df, lb_v2_win_df, ninki_df, umaren_df)
print(score_ana_df.sort_values('ana_rank').head())
score_ana_df = sim_rate_umaren(lb_v1_jiku_df, lb_v2_jiku_df, ninki_df, umaren_df)
print(score_ana_df.sort_values('ana_rank').head())