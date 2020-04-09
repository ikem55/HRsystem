from pulp import *
from modules.lb_extract import LBExtract
from modules.lb_transform import LBTransform

import numpy as np
import pandas as pd
import pickle
import my_config as mc

start_date = '2019/01/01'
end_date = '2019/12/31'

ext = LBExtract(start_date, end_date, False)
tr = LBTransform(start_date, end_date)
#ext.mock_flag = True
#ext.set_mock_path()

pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 200)

raceuma_base_df = ext.get_raceuma_table_base()
temp_df = tr.normalize_raceuma_df(raceuma_base_df)

df = temp_df[["競走コード", "馬番", "デフォルト得点", "確定着順", "単勝配当", "複勝配当",  "得点V3"]]

dict_path = mc.return_base_path(False)
intermediate_folder = dict_path + 'intermediate/'
with open(intermediate_folder + 'lb_v1_lb_v1/raceuma_ens/export_data.pkl', 'rb') as f:
    lb_v1_df = pickle.load(f)
with open(intermediate_folder + 'lb_v2_lb_v2/raceuma_ens/export_data.pkl', 'rb') as f:
    lb_v2_df = pickle.load(f)
with open(intermediate_folder + 'lb_v3_lb_v3/raceuma_ens/export_data.pkl', 'rb') as f:
    lb_v3_df = pickle.load(f)

my_df = pd.merge(lb_v1_df, lb_v2_df , on=["RACE_KEY", "UMABAN", "target"]).rename(columns={"predict_std_x": "偏差v1", "predict_std_y":"偏差v2"})
my_df = pd.merge(my_df, lb_v3_df , on=["RACE_KEY", "UMABAN", "target"]).rename(columns={"predict_std": "偏差v3", "RACE_KEY": "競走コード", "UMABAN": "馬番"})
win_df = my_df[my_df["target"] == "WIN_FLAG"]
jiku_df = my_df[my_df["target"] == "JIKU_FLAG"]
ana_df = my_df[my_df["target"] == "ANA_FLAG"]
win_df.loc[:, "勝ち偏差"] = win_df["偏差v1"] * 0.50 + win_df["偏差v2"] * 0.30 + win_df["偏差v3"] * 0.20
jiku_df.loc[:, "軸偏差"] = jiku_df["偏差v1"] * 0.50 + jiku_df["偏差v2"] * 0.25 + jiku_df["偏差v3"] * 0.25
ana_df.loc[:, "穴偏差"] = ana_df["偏差v1"] * 0.45 + ana_df["偏差v2"] * 0.10 + ana_df["偏差v3"] * 0.45

my_score_df = pd.merge(win_df[["競走コード", "馬番", "勝ち偏差"]], jiku_df[["競走コード", "馬番", "軸偏差"]], on=["競走コード", "馬番"])
my_score_df = pd.merge(my_score_df, ana_df[["競走コード", "馬番", "穴偏差"]], on=["競走コード", "馬番"])

df = pd.merge(df, my_score_df, on=["競走コード", "馬番"])
df.loc[:, "勝"] = df["確定着順"].apply(lambda x: 1 if x == 1 else 0)
df.loc[:, "連"] = df["確定着順"].apply(lambda x: 1 if x in (1, 2) else 0)
df.loc[:, "複"] = df["確定着順"].apply(lambda x: 1 if x in (1, 2, 3) else 0)
print(df.head())
print("------ check ------")
print(df.shape)

iter_range = 5
score_rate = range(0, 51, iter_range)
v3_rate = range(0, 41, iter_range)
win_rate = range(0, 101, iter_range)
jiku_rate = range(0, 101, iter_range)
ana_rate = range(0, 101, iter_range)

s1_list = []
v3_list = []
win_list = []
jiku_list = []
ana_list = []

cnt_list = []
av_win_list = []
av_ren_list = []
av_fuku_list = []
tan_ret_list = []
fuku_ret_list = []

#df = df.head(200)
total_count = len(df)

for s1 in score_rate:
    print(s1)
    for v3 in v3_rate:
        for win in win_rate:
            for jiku in jiku_rate:
                for ana in ana_rate:
                    if s1 + v3 + win + jiku + ana == 100:
                        #print("s1:" + str(s1) + " win:" + str(win) + " jiku:" + str(jiku) + " ana:" + str(ana))
                        temp_df = df
                        temp_df.loc[:, "最適得点"] = df["デフォルト得点"] * s1/100 + df["得点V3"] * v3/100 + df["勝ち偏差"] * win/100 + df["軸偏差"] * jiku/100 + df["穴偏差"] * ana/100
                        target_df = temp_df[temp_df["最適得点"] >= 55]
                        if len(target_df) > total_count / 5:
                            cnt_list.append(len(target_df))
                            s1_list.append(s1)
                            v3_list.append(v3)
                            win_list.append(win)
                            jiku_list.append(jiku)
                            ana_list.append(ana)
                            av_win_list.append(round(target_df["勝"].mean() * 100, 2))
                            av_ren_list.append(round(target_df["連"].mean() * 100, 2))
                            av_fuku_list.append(round(target_df["複"].mean() * 100, 2))
                            tan_ret_list.append(round(target_df["単勝配当"].mean(), 2))
                            fuku_ret_list.append(round(target_df["複勝配当"].mean(), 2))


score_df = pd.DataFrame(
    data={'score_rate': s1_list, 'win_rate': win_list, 'jiku_rate': jiku_list, 'ana_rate': ana_list,
          'count': cnt_list, 'v3_rate': v3_list,
        'av_win': av_win_list, 'av_ren': av_ren_list, 'av_fuku': av_fuku_list, 'tan_return': tan_ret_list , 'fuku_return': fuku_ret_list},
    columns=['score_rate', 'v3_rate', 'win_rate', 'jiku_rate', 'ana_rate', 'count', 'av_win', 'av_ren', 'av_fuku', 'tan_return', 'fuku_return']
)
score_df.loc[:,'tan_return_rank'] = score_df['tan_return'].rank(ascending=False)
score_df.loc[:,'fuku_return_rank'] = score_df['tan_return'].rank(ascending=False)
score_df.loc[:,'av_win_rank'] = score_df['av_win'].rank(ascending=False)
score_df.loc[:,'av_ren_rank'] = score_df['av_ren'].rank(ascending=False)
score_df.loc[:,'av_fuku_rank'] = score_df['av_fuku'].rank(ascending=False)
score_df.loc[:,'total_rank'] = score_df['tan_return_rank'] + score_df['fuku_return_rank'] \
                               + score_df['av_win_rank'] + score_df['av_ren_rank'] + score_df['av_fuku_rank']

print("----------- tan_return -----------------")
print(score_df.sort_values('tan_return', ascending=False).head())
print("----------- fuku_return -----------------")
print(score_df.sort_values('fuku_return', ascending=False).head())
print("----------- av_win -----------------")
print(score_df.sort_values('av_win', ascending=False).head())
print("----------- av_ren -----------------")
print(score_df.sort_values('av_ren', ascending=False).head())
print("----------- av_fuku -----------------")
print(score_df.sort_values('av_fuku', ascending=False).head())

print("----------- total_rank -----------------")
dump_df = score_df.sort_values('total_rank').head(10)
print(dump_df)
print(score_df.describe())

with open(dict_path + 'temp_analysis/output/find_parameter.pkl', 'wb') as f:
    pickle.dump(dump_df, f)
