import pickle
import pandas as pd
from sklearn.cluster import KMeans

from modules.jra_transform import JRATransform
from modules.jra_extract import JRAExtract


start_date = '2018/12/01'
end_date = '2018/12/31'
mock_flag = False
ext = JRAExtract(start_date, end_date, mock_flag)
dict_path = "E:\python/for_test_"
dict_folder = dict_path + 'dict/jra_common/'

race_result_df = ext.get_race_table_base().drop(["NENGAPPI", "距離", "芝ダ障害コード", "右左", "内外", "種別", "条件", "記号", "重量", "グレード", "レース名",
                                                                                  "頭数", "コース", "馬場状態"], axis=1)
raceuma_result_df = ext.get_raceuma_table_base().query("異常区分 not in ('1','2') and 芝ダ障害コード != '3'")

raceuma_df = pd.merge(race_result_df, raceuma_result_df, on="RACE_KEY")

df = raceuma_df.copy()[["RACE_KEY", "UMABAN", "IDM", "RAP_TYPE", "着順", "確定単勝人気順位",
                                  "ＩＤＭ結果", "コーナー順位２", "コーナー順位３", "コーナー順位４", "タイム", "距離", "芝ダ障害コード",
                                  "後３Ｆタイム", "テン指数結果順位", "上がり指数結果順位", "頭数", "前３Ｆ先頭差", "後３Ｆ先頭差", "異常区分"]]
df.loc[:, "追走力"] = df.apply(lambda x: x["コーナー順位２"] - x["コーナー順位４"] if x["コーナー順位２"] != 0 else x["コーナー順位３"] - x["コーナー順位４"], axis=1 )
df.loc[:, "追上力"] = df.apply(lambda x: x["コーナー順位４"] - x["着順"], axis=1)
df.loc[:, "１ハロン平均"] = df.apply(lambda x: x["タイム"] / x["距離"] * 200, axis=1)
df.loc[:, "後傾指数"] = df.apply(lambda x: x["１ハロン平均"] *3 / x["後３Ｆタイム"] if x["後３Ｆタイム"] != 0 else 1 , axis=1)
df.loc[:, "馬番"] = df["UMABAN"].astype(int) / df["頭数"]
df.loc[:, "IDM差"] = df["ＩＤＭ結果"] - df["IDM"]
df.loc[:, "コーナー順位４"] = df["コーナー順位４"] / df["頭数"]
df.loc[:, "CHAKU_RATE"] = df["着順"] / df["頭数"]
df.loc[:, "確定単勝人気順位"] = df["確定単勝人気順位"] / df["頭数"]
df.loc[:, "テン指数結果順位"] = df["テン指数結果順位"] / df["頭数"]
df.loc[:, "上がり指数結果順位"] = df["上がり指数結果順位"] / df["頭数"]
df.loc[:, "上り最速"] = df["上がり指数結果順位"].apply(lambda x: 1 if x == 1 else 0)
df.loc[:, "逃げ"] = df["テン指数結果順位"].apply(lambda x: 1 if x == 1 else 0)
df.loc[:, "勝ち"] = df["着順"].apply(lambda x: 1 if x == 1 else 0)
df.loc[:, "連対"] = df["着順"].apply(lambda x: 1 if x in (1,2) else 0)
df.loc[:, "３着内"] = df["着順"].apply(lambda x: 1 if x in (1,2,3) else 0)
df.loc[:, "掲示板前後"] = df["着順"].apply(lambda x: 1 if x in (4,5,6) else 0)
df.loc[:, "着外"] = df["CHAKU_RATE"].apply(lambda x: 1 if x >= 0.4 else 0)
df.loc[:, "凡走"] = df.apply(lambda x: 1 if x["CHAKU_RATE"] >= 0.6 and x["確定単勝人気順位"] <= 0.3 else 0, axis=1)
df.loc[:, "激走"] = df.apply(lambda x: 1 if x["CHAKU_RATE"] <= 0.3 and x["確定単勝人気順位"] >= 0.7 else 0, axis=1)
df.loc[:, "異常"] = df["異常区分"].apply(lambda x: 1 if x != '0' else 0)

numerical_feats = df.dtypes[df.dtypes != "object"].index
categorical_feats = df.dtypes[df.dtypes == "object"].index
print("----numerical_feats-----")
print(numerical_feats.tolist())
print("----categorical_feats-----")
print(categorical_feats.tolist())

k=8
km_df = df[numerical_feats].drop(["タイム", "後３Ｆタイム", "頭数", 'ＩＤＭ結果', "IDM", "１ハロン平均",'コーナー順位２', 'コーナー順位３'], axis=1)
print(km_df.columns)
cluster = KMeans(n_clusters=k).fit(km_df)

cluster_dict_name = "cluster_raceuma_result.pkl"

#with open(dict_folder + cluster_dict_name, 'wb') as f:
#    pickle.dump(cluster, f)

# 激走: 4:前目の位置につけて能力以上の激走
# 好走：1:後方から上がり上位で能力通り好走 　7:前目の位置につけて能力通り好走
# ふつう：0:なだれ込み能力通りの凡走    5:前目の位置につけて上りの足が上がり能力通りの凡走 6:後方から足を使うも能力通り凡走
# 凡走（下位）：2:前目の位置から上りの足が上がって能力以下の凡走　
# 大凡走　3:後方追走いいとこなしで能力以下の大凡走
