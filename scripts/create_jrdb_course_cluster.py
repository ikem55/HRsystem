import pandas as pd
from sklearn.cluster import KMeans

from modules.jra_extract import JRAExtract
start_date = '2010/01/01'
end_date = '2018/12/31'
mock_flag = False
ext = JRAExtract(start_date, end_date, mock_flag)
dict_path = "E:\python/for_test_"

dict_folder = dict_path + 'dict/jra_common/'
cluster_df_name = "course_cluster_df.pkl"

race_result_df = ext.get_race_table_base()
raceuma_result_df = ext.get_raceuma_table_base()
horse_df = ext.get_horse_table_base()

race_result_df.loc[:, "COURSE_KEY"] = race_result_df["RACE_KEY"].str[:2] + race_result_df["距離"].astype(str) + race_result_df["芝ダ障害コード"] + race_result_df["内外"]
race_columns = ['RACE_KEY', 'COURSE_KEY', 'RAP_TYPE', 'ラスト５ハロン', 'ラスト４ハロン', 'ラスト３ハロン', 'ラスト２ハロン', 'ラスト１ハロン', 'ラップ差４ハロン', 'ラップ差３ハロン', 'ラップ差２ハロン', 'ラップ差１ハロン',
                'TRACK_BIAS_ZENGO', 'TRACK_BIAS_UCHISOTO']
race_df = race_result_df[race_columns]
race_df.loc[:, "RAP_TYPE_一貫"] = race_df["RAP_TYPE"].apply(lambda x: 1 if x =="一貫" else 0)
race_df.loc[:, "RAP_TYPE_L4加速"] = race_df["RAP_TYPE"].apply(lambda x: 1 if x =="L4加速" else 0)
race_df.loc[:, "RAP_TYPE_L3加速"] = race_df["RAP_TYPE"].apply(lambda x: 1 if x =="L3加速" else 0)
race_df.loc[:, "RAP_TYPE_L2加速"] = race_df["RAP_TYPE"].apply(lambda x: 1 if x =="L2加速" else 0)
race_df.loc[:, "RAP_TYPE_L1加速"] = race_df["RAP_TYPE"].apply(lambda x: 1 if x =="L1加速" else 0)
race_df.loc[:, "RAP_TYPE_L4失速"] = race_df["RAP_TYPE"].apply(lambda x: 1 if x =="L4失速" else 0)
race_df.loc[:, "RAP_TYPE_L3失速"] = race_df["RAP_TYPE"].apply(lambda x: 1 if x =="L3失速" else 0)
race_df.loc[:, "RAP_TYPE_L2失速"] = race_df["RAP_TYPE"].apply(lambda x: 1 if x =="L2失速" else 0)
race_df.loc[:, "RAP_TYPE_L1失速"] = race_df["RAP_TYPE"].apply(lambda x: 1 if x =="L1失速" else 0)
raceuma_columns = ['RACE_KEY', 'UMABAN', 'NENGAPPI', '血統登録番号', '着順', 'テン指数順位', 'ペース指数順位', '上がり指数順位', '位置指数順位', '確定単勝人気順位','コーナー順位２','コーナー順位３', 'コーナー順位４',
                   '調教コース坂', '調教コースW', '調教コースダ', '調教コース芝', '調教コースプール', '調教コース障害', '調教コースポリ', 'レース脚質', '距離', 'タイム', '後３Ｆタイム']
horse_columns = ['NENGAPPI', '血統登録番号',  '父系統コード', '母父系統コード', '父馬名', '母父馬名']
raceuma_df = pd.merge(raceuma_result_df[raceuma_columns], horse_df[horse_columns],on=["NENGAPPI", "血統登録番号"]).query("着順 in (1,2,3)").reset_index(drop=True)
raceuma_df.loc[:, '１ハロン平均'] = raceuma_df.apply(lambda x: x["タイム"] / x["距離"] * 200, axis=1)
raceuma_df.loc[:, '上り平均'] = raceuma_df.apply(lambda x: x["後３Ｆタイム"] / 3, axis=1)
raceuma_df.loc[:, "追走力"] = raceuma_df.apply(lambda x: x["コーナー順位２"] - x["コーナー順位４"] if x["コーナー順位２"] != 0 else x["コーナー順位３"] - x["コーナー順位４"], axis=1 )
raceuma_df.loc[:, "追上力"] = raceuma_df.apply(lambda x: x["コーナー順位４"] - x["着順"], axis=1)
raceuma_df.loc[:, "後傾指数"] = raceuma_df.apply(lambda x: x["１ハロン平均"] *3 / x["後３Ｆタイム"] if x["後３Ｆタイム"] != 0 else 1 , axis=1)
raceuma_df.loc[:, '父系統コード_1206'] = raceuma_df['父系統コード'].apply(lambda x : 1 if x == '1206' else 0)
raceuma_df.loc[:, '父系統コード_1503'] = raceuma_df['父系統コード'].apply(lambda x : 1 if x == '1503' else 0)
raceuma_df.loc[:, '父系統コード_1207'] = raceuma_df['父系統コード'].apply(lambda x : 1 if x == '1206' else 0)
raceuma_df.loc[:, '父系統コード_1106'] = raceuma_df['父系統コード'].apply(lambda x : 1 if x == '1106' else 0)
raceuma_df.loc[:, '母父系統コード_1206'] = raceuma_df['母父系統コード'].apply(lambda x : 1 if x == '1206' else 0)
raceuma_df.loc[:, '母父系統コード_1503'] = raceuma_df['母父系統コード'].apply(lambda x : 1 if x == '1503' else 0)
raceuma_df.loc[:, '母父系統コード_1103'] = raceuma_df['母父系統コード'].apply(lambda x : 1 if x == '1103' else 0)
raceuma_df.loc[:, '母父系統コード_1207'] = raceuma_df['母父系統コード'].apply(lambda x : 1 if x == '1207' else 0)
raceuma_df.loc[:, '父馬名_ディープインパクト'] = raceuma_df['父馬名'].apply(lambda x : 1 if x == 'ディープインパクト' else 0)
raceuma_df.loc[:, '父馬名_ハーツクライ'] = raceuma_df['父馬名'].apply(lambda x : 1 if x == 'ハーツクライ' else 0)
raceuma_df.loc[:, '父馬名_ステイゴールド'] = raceuma_df['父馬名'].apply(lambda x : 1 if x == 'ステイゴールド' else 0)
raceuma_df.loc[:, '父馬名_ダイワメジャー'] = raceuma_df['父馬名'].apply(lambda x : 1 if x == 'ダイワメジャー' else 0)
raceuma_df.loc[:, '父馬名_キングカメハメハ'] = raceuma_df['父馬名'].apply(lambda x : 1 if x == 'キングカメハメハ' else 0)
raceuma_df.loc[:, '父馬名_ロードカナロア'] = raceuma_df['父馬名'].apply(lambda x : 1 if x == 'ロードカナロア' else 0)
raceuma_df.loc[:, '父馬名_ルーラーシップ'] = raceuma_df['父馬名'].apply(lambda x : 1 if x == 'ルーラーシップ' else 0)
raceuma_df.loc[:, '父馬名_ハービンジャー'] = raceuma_df['父馬名'].apply(lambda x : 1 if x == 'ハービンジャー' else 0)
raceuma_df.loc[:, '父馬名_ゴールドアリュール'] = raceuma_df['父馬名'].apply(lambda x : 1 if x == 'ゴールドアリュール' else 0)
raceuma_df.loc[:, '父馬名_キンシャサノキセキ'] = raceuma_df['父馬名'].apply(lambda x : 1 if x == 'キンシャサノキセキ' else 0)
raceuma_df.loc[:, '父馬名_クロフネ'] = raceuma_df['父馬名'].apply(lambda x : 1 if x == '' else 0)
raceuma_df.loc[:, '父馬名_オルフェーヴル'] = raceuma_df['父馬名'].apply(lambda x : 1 if x == 'オルフェーヴル' else 0)
raceuma_df.loc[:, '父馬名_マンハッタンカフェ'] = raceuma_df['父馬名'].apply(lambda x : 1 if x == 'マンハッタンカフェ' else 0)
raceuma_df.loc[:, '父馬名_エンパイアメーカー'] = raceuma_df['父馬名'].apply(lambda x : 1 if x == 'エンパイアメーカー' else 0)
raceuma_df.loc[:, '父馬名_ヴィクトワールピサ'] = raceuma_df['父馬名'].apply(lambda x : 1 if x == 'ヴィクトワールピサ' else 0)
raceuma_df.loc[:, '母父馬名_サンデーサイレンス'] = raceuma_df['母父馬名'].apply(lambda x : 1 if x == 'サンデーサイレンス' else 0)
raceuma_df.loc[:, '母父馬名_アグネスタキオン'] = raceuma_df['母父馬名'].apply(lambda x : 1 if x == 'アグネスタキオン' else 0)
raceuma_df.loc[:, '母父馬名_フジキセキ'] = raceuma_df['母父馬名'].apply(lambda x : 1 if x == 'フジキセキ' else 0)
raceuma_df.loc[:, '母父馬名_クロフネ'] = raceuma_df['母父馬名'].apply(lambda x : 1 if x == 'クロフネ' else 0)
raceuma_df.loc[:, '母父馬名_フレンチデピュティ'] = raceuma_df['母父馬名'].apply(lambda x : 1 if x == 'フレンチデピュティ' else 0)
raceuma_df.loc[:, '母父馬名_キングカメハメハ'] = raceuma_df['母父馬名'].apply(lambda x : 1 if x == 'キングカメハメハ' else 0)
raceuma_df.loc[:, '母父馬名_ダンスインザダーク'] = raceuma_df['母父馬名'].apply(lambda x : 1 if x == 'ダンスインザダーク' else 0)
raceuma_df.loc[:, '母父馬名_ブライアンズタイム'] = raceuma_df['母父馬名'].apply(lambda x : 1 if x == 'ブライアンズタイム' else 0)
raceuma_df.loc[:, '母父馬名_スペシャルウィーク'] = raceuma_df['母父馬名'].apply(lambda x : 1 if x == 'スペシャルウィーク' else 0)
raceuma_df.loc[:, '母父馬名_シンボリクリスエス'] = raceuma_df['母父馬名'].apply(lambda x : 1 if x == 'シンボリクリスエス' else 0)
raceuma_df.loc[:, '母父馬名_サクラバクシンオー'] = raceuma_df['母父馬名'].apply(lambda x : 1 if x == 'サクラバクシンオー' else 0)
raceuma_df.loc[:, '母父馬名_タイキシャトル'] = raceuma_df['母父馬名'].apply(lambda x : 1 if x == 'タイキシャトル' else 0)
raceuma_df.loc[:, '母父馬名_ディープインパクト'] = raceuma_df['母父馬名'].apply(lambda x : 1 if x == 'ディープインパクト' else 0)
raceuma_df.loc[:, '母父馬名_ネオユニヴァース'] = raceuma_df['母父馬名'].apply(lambda x : 1 if x == 'ネオユニヴァース' else 0)
raceuma_df.loc[:, '母父馬名_トニービン'] = raceuma_df['母父馬名'].apply(lambda x : 1 if x == 'トニービン' else 0)
race_df = pd.merge(race_df, raceuma_df.drop(["タイム", "着順"], axis=1), on="RACE_KEY")

numerical_feats = race_df.dtypes[race_df.dtypes != "object"].index
categorical_feats = race_df.dtypes[race_df.dtypes == "object"].index
print("----numerical_feats-----")
print(numerical_feats.tolist())
print("----categorical_feats-----")
print(categorical_feats.tolist())

data_df = race_df.groupby('COURSE_KEY').mean().reset_index()
print(data_df.columns)
print(len(data_df.columns))
master_df = race_result_df[["COURSE_KEY", "場名", "距離", "芝ダ障害コード", "内外"]].drop_duplicates()
master_df.loc[:, "芝ダ"] = master_df["芝ダ障害コード"].apply(lambda x: "芝" if x == "1" else ("ダ" if x =="2" else "障害"))
master_df.loc[:, "コース名"] = master_df["芝ダ"] + master_df["場名"] + master_df["距離"].astype(str) + master_df["内外"]
master_df.loc[:, "距離"] = master_df["距離"].astype(int)
master_df = master_df.drop(["場名", "芝ダ障害コード", "内外", "芝ダ"], axis=1)

# k-means
k=10
km_df = data_df.drop(["COURSE_KEY", "確定単勝人気順位", "後３Ｆタイム"], axis=1)
pred = KMeans(n_clusters=k).fit_predict(km_df)
course_cluster_df = pd.DataFrame({"COURSE_KEY": data_df["COURSE_KEY"], "cluster": pred})
course_cluster_df.to_pickle(dict_folder + cluster_df_name)
