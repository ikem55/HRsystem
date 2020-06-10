from scripts.jra_target_script import Ld, CreateFile
import pandas as pd

start_date = '2020/01/01'
end_date = '2020/01/31'
mock_flag = False
test_flag = False
version_str = "dummy"
pd.set_option('display.max_columns', 3000)
pd.set_option('display.max_rows', 3000)

ld = Ld(version_str, start_date, end_date, mock_flag, test_flag)
ld.set_race_file_df()
ld.set_pred_df()
ld.set_contents_based_filtering_df()

df = ld.raceuma_cbf_df.copy()
print(df.iloc[0])
numerical_feats = df.dtypes[df.dtypes != "object"].index.tolist()
categorical_feats = df.dtypes[df.dtypes == "object"].index.tolist()
print("----numerical_feats-----")
print(numerical_feats)
print(df[numerical_feats].iloc[0])
print("----categorical_feats-----")
print(categorical_feats)
print(df[categorical_feats].iloc[0])

cf = CreateFile(start_date, end_date, test_flag)

num_columns_list = numerical_feats + ["RACE_KEY", "UMABAN"]

race_key = df["RACE_KEY"].iloc[0]
print(race_key)
target_df = df.query(f"RACE_KEY == '{race_key}'").reset_index(drop=True)
sim_df = cf.create_sim_score(target_df)
print(sim_df)
"""
#target_df = df.query(f"RACE_KEY == '{race_key}'")[num_columns_list].reset_index(drop=True).fillna(0)
#sim_score_df = cf.create_cos_sim_score(target_df)
#print(sim_score_df)

target_df = df.query(f"RACE_KEY == '{race_key}'")[categorical_feats].reset_index(drop=True).fillna("")
target_df.loc[:, "テキスト"] = ""
for column_name, item in target_df.iteritems():
    target_df.loc[:, "テキスト"] = target_df["テキスト"].str.cat(item, sep=',')

print(target_df["テキスト"])
target_df["テキスト"] = target_df["テキスト"].apply(lambda x: x.split(','))
print(target_df["テキスト"])
"""
