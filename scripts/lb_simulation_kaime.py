from modules.lb_extract import LBExtract
from modules.lb_simulation import LBSimulation

import pandas as pd
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 200)

start_date = '2019/01/01'
end_date = '2019/01/01'
mock_flag = False
ext = LBExtract(start_date, end_date, mock_flag)
raceuma_df = ext.get_raceuma_table_base()

sim = LBSimulation(start_date, end_date, mock_flag)
sim.set_raceuma_df(raceuma_df)

flag_tansho = False
flag_fukusho = False

# オッズとかのフィルター方法を検討する
cond_1tou_list = [
    {"cond1": "馬券評価順位 == 1"},
    {"cond1": "馬券評価順位 <= 2"},
]

if flag_tansho:
    print("------- 単勝 ----------")
    sim_tansho_df = pd.DataFrame([]
                                 , columns=["条件", "件数", "的中数", "レース数", "回収率", "的中率", "R的中率", "払戻平均", "払戻偏差", "最大払戻",
                                            "購入総額", "払戻総額"]
                                 , index=range(len(cond_1tou_list)))
    i = 0
    for cond in cond_1tou_list:
        sr = sim.simulation_tansho(cond["cond1"])
        print(str(sr["回収率"]) + '%:  ' + sr["条件"])
        sim_tansho_df.iloc[i, :] = sr
        i += 1

    print(sim_tansho_df)

if flag_fukusho:
    print("------- 複勝 ----------")
    sim_fukusho_df = pd.DataFrame([]
                                 , columns=["条件", "件数", "的中数", "レース数", "回収率", "的中率", "R的中率", "払戻平均", "払戻偏差", "最大払戻",
                                            "購入総額", "払戻総額"]
                                 , index=range(len(cond_1tou_list)))
    i = 0
    for cond in cond_1tou_list:
        sr = sim.simulation_fukusho(cond["cond1"])
        print(str(sr["回収率"]) + '%:  ' + sr["条件"])
        sim_fukusho_df.iloc[i, :] = sr
        i += 1

    print(sim_fukusho_df)

flag_umaren = False
flag_wide = False
flag_umatan = False

cond_2tou_list = [
{"cond1": "馬券評価順位 == 1", "cond2": "軸偏差 >= 45"},
{"cond1": "馬券評価順位 == 1", "cond2": "馬券評価順位 <= 5"},
]

if flag_umaren:
    print("------- 馬連 ----------")
    sim_umaren_df = pd.DataFrame([]
                                 , columns=["条件", "件数", "的中数", "レース数", "回収率", "的中率", "R的中率", "払戻平均", "払戻偏差", "最大払戻", "購入総額", "払戻総額"]
                                 , index=range(len(cond_2tou_list)))
    i = 0
    for cond in cond_2tou_list:
        sr = sim.simulation_umaren(cond["cond1"], cond["cond2"])
        print(str(sr["回収率"]) + '%:  ' + sr["条件"])
        sim_umaren_df.iloc[i,:] = sr
        i += 1

    print(sim_umaren_df)

if flag_wide:
    print("------- ワイド ----------")
    sim_wide_df = pd.DataFrame([]
                               , columns=["条件", "件数", "的中数", "レース数", "回収率", "的中率", "R的中率", "払戻平均", "払戻偏差", "最大払戻", "購入総額", "払戻総額"]
                               , index=range(len(cond_2tou_list)))
    i = 0
    for cond in cond_2tou_list:
        sr = sim.simulation_wide(cond["cond1"], cond["cond2"])
        print(str(sr["回収率"]) + '%:  ' + sr["条件"])
        sim_wide_df.iloc[i,:] = sr
        i += 1

    print(sim_wide_df)

if flag_umatan:
    print("------- 馬単 ----------")
    sim_umatan_df = pd.DataFrame([]
                                 , columns=["条件", "件数", "的中数", "レース数", "回収率", "的中率", "R的中率", "払戻平均", "払戻偏差", "最大払戻", "購入総額", "払戻総額"]
                                 , index=range(len(cond_2tou_list)))
    i = 0
    for cond in cond_2tou_list:
        sr = sim.simulation_umatan(cond["cond1"], cond["cond2"])
        print(str(sr["回収率"]) + '%:  ' + sr["条件"])
        sim_umatan_df.iloc[i,:] = sr
        i += 1

    print(sim_umatan_df)

flag_sanrenpuku = True

cond_3tou_list = [
{"cond1": "馬券評価順位 == 1", "cond2": "軸偏差 >= 50", "cond3": "軸偏差 >= 45"},
{"cond1": "馬券評価順位 == 1", "cond2": "馬券評価順位 <= 3", "cond3": "軸偏差 >= 45"},
]

if flag_sanrenpuku:
    print("------- 三連複 ----------")
    sim_sanrenpuku_df = pd.DataFrame([]
                                 , columns=["条件", "件数", "的中数", "レース数", "回収率", "的中率", "R的中率", "払戻平均", "払戻偏差", "最大払戻", "購入総額", "払戻総額"]
                                 , index=range(len(cond_3tou_list)))
    i = 0
    for cond in cond_3tou_list:
        sr = sim.simulation_sanrenpuku(cond["cond1"], cond["cond2"], cond["cond3"])
        print(str(sr["回収率"]) + '%:  ' + sr["条件"])
        sim_sanrenpuku_df.iloc[i,:] = sr
        i += 1

    print(sim_sanrenpuku_df)