from pulp import *
from modules.lb_extract import LBExtract
from modules.lb_transform import LBTransform

import numpy as np
import pandas as pd

start_date = '2019/01/01'
end_date = '2019/12/31'

ext = LBExtract(start_date, end_date, False)
tr = LBTransform(start_date, end_date)
#ext.mock_flag = True
#ext.set_mock_path()

pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 200)

df = ext.get_raceuma_table_base()
df = tr.normalize_raceuma_df(df)
#print(df.iloc[0])

df = df[["競走コード", "馬番", "デフォルト得点", "確定着順", "単勝配当", "複勝配当", "勝ち偏差", "軸偏差", "穴偏差", "勝ち偏差２", "軸偏差２", "穴偏差２", "得点V3"]]

df.loc[:, "勝"] = df["確定着順"].apply(lambda x: 1 if x == 1 else 0)
df.loc[:, "連"] = df["確定着順"].apply(lambda x: 1 if x in (1, 2) else 0)
df.loc[:, "複"] = df["確定着順"].apply(lambda x: 1 if x in (1, 2, 3) else 0)
print(df.iloc[0])
print("------ check ------")
print(df.shape)
total_count = len(df)

iter_range = 0.05
score_rate = np.arange(0,0.6, iter_range)
v3_rate = np.arange(0,0.6, iter_range)
win_rate = np.arange(0,0.6, iter_range)
jiku_rate = np.arange(0,0.6, iter_range)
ana_rate = np.arange(0,0.6, iter_range)
win2_rate = np.arange(0,0.6, iter_range)
jiku2_rate = np.arange(0,0.6, iter_range)
ana2_rate = np.arange(0,0.6, iter_range)
#score_range = np.arange(50, 65, 5)
score_range = [55]

sr_list = []
s1_list = []
v3_list = []
win_list = []
jiku_list = []
ana_list = []
win2_list = []
jiku2_list = []
ana2_list = []

cnt_list = []
av_win_list = []
av_ren_list = []
av_fuku_list = []
tan_ret_list = []
fuku_ret_list = []

for s1 in score_rate:
    print(s1)
    for v3 in v3_rate:
        print(v3)
        for win in win_rate:
            print(win)
            for jiku in jiku_rate:
                for ana in ana_rate:
                    for win2 in win2_rate:
                        for jiku2 in jiku2_rate:
                            for ana2 in ana2_rate:
                                if s1 + v3 + win + jiku + ana + win2 + jiku2 + ana2 == 1:
                                    #print("s1:" + str(s1) + " win:" + str(win) + " jiku:" + str(jiku) + " ana:" + str(ana))
                                    temp_df = df
                                    temp_df.loc[:, "最適得点"] = df["デフォルト得点"] * s1 + df["得点V3"] * v3 + df["勝ち偏差"] * win + df["軸偏差"] * jiku + df["穴偏差"] * ana + df["勝ち偏差２"] * win2 + df["軸偏差２"] * jiku2 + df["穴偏差２"] * ana2
                                    for sr in score_range:
                                        target_df = temp_df[temp_df["最適得点"] >= sr]
                                        if len(target_df) > total_count / 5:
                                            cnt_list.append(len(target_df))
                                            sr_list.append(sr)
                                            s1_list.append(s1)
                                            v3_list.append(v3)
                                            win_list.append(win)
                                            jiku_list.append(jiku)
                                            ana_list.append(ana)
                                            win2_list.append(win2)
                                            jiku2_list.append(jiku2)
                                            ana2_list.append(ana2)
                                            av_win_list.append(round(target_df["勝"].mean() * 100, 2))
                                            av_ren_list.append(round(target_df["連"].mean() * 100, 2))
                                            av_fuku_list.append(round(target_df["複"].mean() * 100, 2))
                                            tan_ret_list.append(round(target_df["単勝配当"].mean(), 2))
                                            fuku_ret_list.append(round(target_df["複勝配当"].mean(), 2))


score_df = pd.DataFrame(
    data={'score_rate': s1_list, 'win_rate': win_list, 'jiku_rate': jiku_list, 'ana_rate': ana_list,
          'win2_rate': win2_list, 'jiku2_rate': jiku2_list, 'ana2_rate': ana2_list,
          'count': cnt_list, 'score_range': sr_list, 'v3_rate': v3_list,
        'av_win': av_win_list, 'av_ren': av_ren_list, 'av_fuku': av_fuku_list, 'tan_return': tan_ret_list , 'fuku_return': fuku_ret_list},
    columns=['score_rate', 'v3_rate', 'win_rate', 'jiku_rate', 'ana_rate', 'win2_rate', 'jiku2_rate', 'ana2_rate', 'score_range', 'count', 'av_win', 'av_ren', 'av_fuku', 'tan_return', 'fuku_return']
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
print(score_df.sort_values('total_rank').head(10))
score_df.to_pickle('score_df.pkl')
print(score_df.describe())
