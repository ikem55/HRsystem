from modules.lb_extract import LBExtract
from modules.lb_simulation import LBSimulation
import pandas as pd
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

dict_path = mc.return_base_path(False)
intermediate_folder = dict_path + 'intermediate/'

def get_type_df_list(df, type):
    df.rename(columns={"RACE_KEY": "競走コード", "UMABAN": "馬番", "predict_std": "予測値偏差"}, inplace=True)

    win_df = df[df["target"] == "WIN_FLAG"][["競走コード", "馬番", "予測値偏差"]].rename(
        columns={"予測値偏差": "偏差" + type})
    jiku_df = df[df["target"] == "JIKU_FLAG"][["競走コード", "馬番", "予測値偏差"]].rename(
        columns={"予測値偏差": "偏差" + type})
    ana_df = df[df["target"] == "ANA_FLAG"][["競走コード", "馬番", "予測値偏差"]].rename(
        columns={"予測値偏差": "偏差" + type})
    return [win_df, jiku_df, ana_df]

"""
print(lb_v1_df.iloc[0])
lb_v1_df.rename(columns={"RACE_KEY":"競走コード", "UMABAN": "馬番", "predict_std": "予測値偏差"}, inplace=True)
lb_v2_df.rename(columns={"RACE_KEY":"競走コード", "UMABAN": "馬番", "predict_std": "予測値偏差"}, inplace=True)

lb_v1_win_df = lb_v1_df[lb_v1_df["target"] == "WIN_FLAG"][["競走コード", "馬番", "予測値偏差"]].rename(columns={"予測値偏差": "偏差v1"})
lb_v2_win_df = lb_v2_df[lb_v2_df["target"] == "WIN_FLAG"][["競走コード", "馬番", "予測値偏差"]].rename(columns={"予測値偏差": "偏差v2"})
lb_v1_jiku_df = lb_v1_df[lb_v1_df["target"] == "JIKU_FLAG"][["競走コード", "馬番", "予測値偏差"]].rename(columns={"予測値偏差": "偏差v1"})
lb_v2_jiku_df = lb_v2_df[lb_v2_df["target"] == "JIKU_FLAG"][["競走コード", "馬番", "予測値偏差"]].rename(columns={"予測値偏差": "偏差v2"})
lb_v1_ana_df = lb_v1_df[lb_v1_df["target"] == "ANA_FLAG"][["競走コード", "馬番", "予測値偏差"]].rename(columns={"予測値偏差": "偏差v1"})
lb_v2_ana_df = lb_v2_df[lb_v2_df["target"] == "ANA_FLAG"][["競走コード", "馬番", "予測値偏差"]].rename(columns={"予測値偏差": "偏差v2"})
"""

def sim_rate(df1, df2, df3, result_df):
    merge_df = pd.merge(df1, df2, on =["競走コード", "馬番"])
    merge_df = pd.merge(merge_df, df3, on =["競走コード", "馬番"])
    merge_df = pd.merge(merge_df, result_df, on =["競走コード", "馬番"])
    iter_range = 5
    lb_v1_rate = range(0, 101, iter_range)
    lb_v2_rate = range(0, 101, iter_range)
    lb_v3_rate = range(0, 101, iter_range)

    lb_v1_list = []
    lb_v2_list = []
    lb_v3_list = []

    cnt_list = []
    av_win_list = []
    av_ren_list = []
    av_fuku_list = []
    tan_ret_list = []
    fuku_ret_list = []

    for v1 in lb_v1_rate:
        for v2 in lb_v2_rate:
            for v3 in lb_v3_rate:
                if v1 + v2 + v3 == 100:
                    # print(str(v1) + "  " + str(v2))
                    merge_df["配分値"] = merge_df["偏差v1"] * v1 / 100 + merge_df["偏差v2"] * v2 / 100 + merge_df["偏差v3"] * v3 / 100
                    grouped = merge_df.groupby("競走コード")
                    merge_df["順位"] = grouped['配分値'].rank("dense", ascending=False)
                    target_df = merge_df[merge_df["順位"] <= 2].copy()
                    cnt_list.append(len(target_df))
                    lb_v1_list.append(v1)
                    lb_v2_list.append(v2)
                    lb_v3_list.append(v3)
                    av_win_list.append(round(target_df["勝"].mean() * 100, 2))
                    av_ren_list.append(round(target_df["連"].mean() * 100, 2))
                    av_fuku_list.append(round(target_df["複"].mean() * 100, 2))
                    tan_ret_list.append(round(target_df["単勝配当"].mean(), 2))
                    fuku_ret_list.append(round(target_df["複勝配当"].mean(), 2))

    score_df = pd.DataFrame(
        data={'count': cnt_list, 'v1_rate': lb_v1_list, 'v2_rate': lb_v2_list, 'v3_rate': lb_v3_list,
              'av_win': av_win_list, 'av_ren': av_ren_list, 'av_fuku': av_fuku_list, 'tan_return': tan_ret_list,
              'fuku_return': fuku_ret_list},
        columns=['count', 'v1_rate', 'v2_rate', 'v3_rate', 'av_win', 'av_ren', 'av_fuku', 'tan_return', 'fuku_return']
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


def sim_rate_umaren(df1, df2, df3, result_df, pair_df, umaren_df):
    merge_df = pd.merge(df1, df2, on =["競走コード", "馬番"])
    merge_df = pd.merge(merge_df, df3, on =["競走コード", "馬番"])
    merge_df = pd.merge(merge_df, result_df, on =["競走コード", "馬番"])
    iter_range = 5
    lb_v1_rate = range(0, 101, iter_range)
    lb_v2_rate = range(0, 101, iter_range)
    lb_v3_rate = range(0, 101, iter_range)

    lb_v1_list = []
    lb_v2_list = []
    lb_v3_list = []

    cnt_list = []
    umaren_return_list = []
    umaren_hit_list = []
    tan_ret_list = []
    fuku_ret_list = []

    for v1 in lb_v1_rate:
        for v2 in lb_v2_rate:
            for v3 in lb_v3_rate:
                if v1 + v2 + v3 == 100:
                    # print(str(v1) + "  " + str(v2))
                    merge_df["配分値"] = merge_df["偏差v1"] * v1 / 100 + merge_df["偏差v2"] * v2 / 100 + merge_df["偏差v3"] * v3 / 100
                    grouped = merge_df.groupby("競走コード")
                    merge_df["順位"] = grouped['配分値'].rank("dense", ascending=False)
                    candidate_df = merge_df[merge_df["順位"] <= 2].copy()
                    target_df = pd.merge(candidate_df, pair_df, on ="競走コード").fillna(0)
                    target_df = pd.merge(target_df, umaren_df , on ="競走コード", how="left").fillna(0)
                    target_df["削除フラグ"] = target_df.apply(lambda x: 1 if x["馬番_x"] == x["馬番_y"] else 0, axis=1)
                    target_df["削除フラグ"] = target_df.apply(lambda x: 1 if type(x["馬番"]) is int  else x["削除フラグ"], axis=1)
                    target_df = target_df[target_df["削除フラグ"] == 0]
                    target_df["結果"] = target_df.apply(lambda x: x["払戻"] if x["馬番_x"] in x["馬番"] and x["馬番_y"] in x["馬番"] else 0 , axis=1 )
                    target_df["的中"] = target_df.apply(lambda x: 0 if x["結果"] == 0 else 1, axis=1)
                    # print(target_df.head())
                    cnt_list.append(len(target_df))
                    lb_v1_list.append(v1)
                    lb_v2_list.append(v2)
                    lb_v3_list.append(v3)
                    umaren_return_list.append(round(target_df["結果"].mean() , 2))
                    umaren_hit_list.append(round(target_df["的中"].sum()))
                    tan_ret_list.append(round(target_df["単勝配当"].mean(), 2))
                    fuku_ret_list.append(round(target_df["複勝配当"].mean(), 2))

    score_df = pd.DataFrame(
        data={'count': cnt_list, 'v1_rate': lb_v1_list, 'v2_rate': lb_v2_list, 'v3_rate': lb_v3_list,
              'umaren_return': umaren_return_list, 'umaren_hit': umaren_hit_list, 'tan_return': tan_ret_list,
              'fuku_return': fuku_ret_list},
        columns=['count', 'v1_rate', 'v2_rate', 'v3_rate', 'umaren_return', 'umaren_hit', 'tan_return', 'fuku_return']
    )
    score_df.loc[:, 'tan_return_rank'] = score_df['tan_return'].rank(ascending=False)
    score_df.loc[:, 'fuku_return_rank'] = score_df['tan_return'].rank(ascending=False)
    score_df.loc[:, 'umaren_return_rank'] = score_df['umaren_return'].rank(ascending=False)
    score_df.loc[:, 'umaren_hit_rank'] = score_df['umaren_hit'].rank(ascending=False)
    score_df.loc[:, 'ana_rank'] = score_df['umaren_return_rank'] + score_df['umaren_hit_rank']
    return score_df


pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 200)


with open(intermediate_folder + 'lb_v1_lb_v1/raceuma_ens/export_data.pkl', 'rb') as f:
    lb_v1_df = pickle.load(f)
with open(intermediate_folder + 'lb_v2_lb_v2/raceuma_ens/export_data.pkl', 'rb') as f:
    lb_v2_df = pickle.load(f)
with open(intermediate_folder + 'lb_v3_lb_v3/raceuma_ens/export_data.pkl', 'rb') as f:
    lb_v3_df = pickle.load(f)

lb_v1_list = get_type_df_list(lb_v1_df, "v1")
lb_v2_list = get_type_df_list(lb_v2_df, "v2")
lb_v3_list = get_type_df_list(lb_v3_df, "v3")


raceuma_df = ext.get_raceuma_table_base()
result_df = raceuma_df[["競走コード", "馬番", "確定着順", "単勝配当", "複勝配当"]].copy()
result_df.loc[:, "勝"] = result_df["確定着順"].apply(lambda x: 1 if x == 1 else 0)
result_df.loc[:, "連"] = result_df["確定着順"].apply(lambda x: 1 if x in (1, 2) else 0)
result_df.loc[:, "複"] = result_df["確定着順"].apply(lambda x: 1 if x in (1, 2, 3) else 0)

print("----------- WIN_FLAG -----------------")
score_win_df = sim_rate(lb_v1_list[0], lb_v2_list[0], lb_v3_list[0], result_df)
score_win_df = score_win_df.sort_values('win_rank')
print(score_win_df.head())
win_rate = score_win_df.iloc[0]

print("----------- JIKU_FLAG -----------------")
score_jiku_df = sim_rate(lb_v1_list[1], lb_v2_list[1], lb_v3_list[1], result_df)
score_jiku_df = score_jiku_df.sort_values('jiku_rank')
print(score_jiku_df.head())
jiku_rate = score_jiku_df.iloc[0]

print("----------- ANA_FLAG -----------------")
sim.sim_umaren()
umaren_df = sim.result_umaren_df
ninki_df =raceuma_df[raceuma_df["単勝人気"] == 1][["競走コード", "馬番"]]

score_ana_df = sim_rate_umaren(lb_v1_list[2], lb_v2_list[2], lb_v3_list[2], result_df, ninki_df, umaren_df)
score_ana_df = score_ana_df.sort_values('ana_rank')
print(score_ana_df.head())
ana_rate = score_ana_df.iloc[0]

dict_rate = {"win_rate": win_rate, "jiku_rate": jiku_rate, "ana_rate": ana_rate}
print(dict_rate)
with open('temp_analysis/output/score_rete.pkl', 'w') as f:
    pickle.dump(dict_rate, f)
