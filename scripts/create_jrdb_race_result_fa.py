import pickle
import pandas as pd
from pylab import rcParams
from sklearn.preprocessing import StandardScaler
from modules.jra_extract import JRAExtract
from factor_analyzer import FactorAnalyzer

start_date = '2010/01/01'
end_date = '2018/12/31'
mock_flag = False
ext = JRAExtract(start_date, end_date, mock_flag)
dict_path = "E:\python/for_test_"

dict_folder = dict_path + 'dict/jra_common/'
fa_dict_name = "fa_raceuma_result_df.pkl"
sc_dict_name = "fa_sc_raceuma_result_df.pkl"

race_result_df = ext.get_race_table_base()
raceuma_result_df = ext.get_raceuma_table_base()

raceuma_result_df.loc[:, "追走力"] = raceuma_result_df.apply(lambda x: x["コーナー順位２"] - x["コーナー順位４"] if x["コーナー順位２"] != 0 else x["コーナー順位３"] - x["コーナー順位４"], axis=1 )
raceuma_result_df.loc[:, "追上力"] = raceuma_result_df.apply(lambda x: x["コーナー順位４"] - x["着順"], axis=1)
raceuma_result_df.loc[:, "ハロン数"] = raceuma_result_df.apply(lambda x: x["距離"] / 200, axis=1)
raceuma_result_df.loc[:, "１ハロン平均"] = raceuma_result_df.apply(lambda x: x["タイム"] / x["距離"] * 200, axis=1)
raceuma_result_df.loc[:, "後傾指数"] = raceuma_result_df.apply(lambda x: x["１ハロン平均"] *3 / x["後３Ｆタイム"] if x["後３Ｆタイム"] != 0 else 1 , axis=1)
gp_mean_raceuma_result_df = raceuma_result_df.query("着順 in (1,2,3,4,5)")[["RACE_KEY", '１ハロン平均', 'ＩＤＭ結果', 'テン指数結果', '上がり指数結果', 'ペース指数結果', '前３Ｆタイム', '後３Ｆタイム',
                                                                     'コーナー順位１', 'コーナー順位２', 'コーナー順位３', 'コーナー順位４', '前３Ｆ先頭差', '後３Ｆ先頭差', '追走力', '追上力',
                                                                     '後傾指数']].groupby("RACE_KEY").mean()\
    .reset_index().add_suffix('_mean').rename(columns={"RACE_KEY_mean": "RACE_KEY"})
gp_std_raceuma_result_df = raceuma_result_df.query("着順 in (1,2,3,4,5)")[["RACE_KEY", '１ハロン平均', '上がり指数結果', 'ペース指数結果']].groupby("RACE_KEY").std()\
    .reset_index().add_suffix('_std').rename(columns={"RACE_KEY_std": "RACE_KEY"})
gp_raceuma_df = pd.merge(gp_mean_raceuma_result_df, gp_std_raceuma_result_df, on="RACE_KEY")
print(gp_raceuma_df.head())

race_base_df = race_result_df[race_result_df["芝ダ障害コード"] != "3"][["RACE_KEY","距離","芝ダ障害コード" ,"内外", "条件", "グレード", "レース名", "１着算入賞金", "芝馬場状態コード",
                               "ダ馬場状態コード",  "芝種類", "草丈", "転圧", "凍結防止剤", "中間降水量", "ラスト５ハロン", "ラスト４ハロン",
                               "ラスト３ハロン", "ラスト２ハロン", "ラスト１ハロン", "ラップ差４ハロン", "ラップ差３ハロン", "ラップ差２ハロン",
                               "ラップ差１ハロン", 'レース名９文字', '場名', "RAP_TYPE", "TRACK_BIAS_ZENGO", "TRACK_BIAS_UCHISOTO"]]
race_base_df.loc[:, "ハロン数"] = race_base_df.apply(lambda x: x["距離"] / 200, axis=1)
race_base_df.loc[:, "芝"] = race_base_df["芝ダ障害コード"].apply(lambda x: 1 if x == "1" else 0)
race_base_df.loc[:, "外"] = race_base_df["内外"].apply(lambda x: 1 if x == "1" else 0)
race_base_df.loc[:, "重"] = race_base_df.apply(lambda x: 1 if (x["芝ダ障害コード"] == "1" and x["芝種類"] == "2") or (x["芝ダ障害コード"] == "2" and x["凍結防止剤"] == "1") else 0, axis=1)
race_base_df.loc[:, "軽"] = race_base_df.apply(lambda x: 1 if (x["芝ダ障害コード"] == "1" and x["芝種類"] == "1") or (x["芝ダ障害コード"] == "2" and x["転圧"] == "1") else 0, axis=1)
race_base_df.loc[:, "馬場状態"] = race_base_df.apply(lambda x: x["ダ馬場状態コード"] if x["芝ダ障害コード"] == "2" else x["芝馬場状態コード"], axis=1)

df = pd.merge(race_base_df, gp_raceuma_df, on="RACE_KEY")
print(df.shape)

numerical_feats = df.dtypes[df.dtypes != "object"].index
categorical_feats = df.dtypes[df.dtypes == "object"].index
print("----numerical_feats-----")
print(numerical_feats.tolist())
print("----categorical_feats-----")
print(categorical_feats.tolist())

df_data_org = df[numerical_feats].drop(['ラスト５ハロン', 'ラスト４ハロン', 'ラスト３ハロン', 'ラスト２ハロン', 'ラスト１ハロン','前３Ｆ先頭差_mean', '後３Ｆ先頭差_mean','草丈', '中間降水量', 'コーナー順位１_mean',
                                        'コーナー順位２_mean', 'コーナー順位３_mean', 'コーナー順位４_mean', '前３Ｆタイム_mean', '後３Ｆタイム_mean' ], axis=1)
print(df_data_org.columns)
scaler = StandardScaler()
df_data = pd.DataFrame(scaler.fit_transform(df_data_org), columns=df_data_org.columns)

n=5

fa = FactorAnalyzer(n_factors=n, rotation='promax', impute='drop')
fa.fit(df_data)

with open(dict_folder + fa_dict_name, 'wb') as f:
    pickle.dump(fa, f)
with open(dict_folder + sc_dict_name, 'wb') as f:
    pickle.dump(scaler, f)

# fa1:　数値大：底力指数
# 高：ペース指数が高く上りがかかる前傾レース。ダート的な要素で失速ラップの要素　→　底力評価
# 低：スローの瞬発力レース。３ハロンの加速が大きく後傾ラップで、テンや道中のスピードが遅いほど指数が高くなる　→スロー適性、瞬発力評価（？）

# fa2:　数値大：末脚指数
# IDMや賞金が高い高グレードレース。１ハロン平均が速くばらつきが少ないほど指数が高くなる　→　高レベル・末脚評価
# 賞金が低く時計がかかるレースだと指数が高くなる。ダート中距離の新馬戦　→　基本的には低レベルレース

# fa3:　数値大：スタミナ指数
# 時計が遅い　→　長距離スタミナ評価
# 時計が速く内決着　→　新潟１０００ｍレース →　純粋スピード・内前決着　短距離評価

# fa4: 両方向：レースタイプ
# 後方差し決着、時計はややかかり目のダートに多い　→　後方決着レース
# 前前決着　時計はスロー　→　前方決着レース

# fa5:　数値大：高速スピード指数
# 軽い芝決着。時計速くラスト１ハロンの失速率高い　→　高速馬場　スピード評価
# 重いダート、時計遅いがラスト１ハロンの失速は高くない　→　基本的には低レベルレース