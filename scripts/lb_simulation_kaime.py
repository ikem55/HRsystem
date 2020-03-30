from modules.lb_extract import LBExtract
from modules.lb_simulation import LBSimulation

import pandas as pd
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 200)

start_date = '2019/01/01'
end_date = '2019/01/31'
mock_flag = False
ext = LBExtract(start_date, end_date, mock_flag)
raceuma_df = ext.get_raceuma_table_base()

sim = LBSimulation(start_date, end_date, mock_flag)
sim.set_raceuma_df(raceuma_df)

cond_list = [
{"cond1": "馬券評価順位 == 1", "cond2": "軸偏差 >= 45", "filter_odds_low": 30, "filter_odds_high": 150},
{"cond1": "馬券評価順位 == 1", "cond2": "軸偏差 >= 50", "filter_odds_low": 30, "filter_odds_high": 150},
{"cond1": "馬券評価順位 <= 2", "cond2": "軸偏差 >= 50", "filter_odds_low": 50, "filter_odds_high": 150},
{"cond1": "馬券評価順位 == 1", "cond2": "馬券評価順位 <= 5", "filter_odds_low": 50, "filter_odds_high": 150},
]

sim_df = pd.DataFrame([]
                      , columns=["条件", "件数", "的中数", "レース数", "回収率", "的中率", "R的中率", "払戻平均", "払戻偏差", "最大払戻", "購入総額", "払戻総額"]
                      , index=range(len(cond_list)))
i = 0
for cond in cond_list:
    sr = sim.simulation_umaren(cond["cond1"], cond["cond2"], cond["filter_odds_low"], cond["filter_odds_high"])
    print(str(sr["回収率"]) + '%:  ' + sr["条件"])
    sim_df.iloc[i,:] = sr
    i += 1

print(sim_df)
