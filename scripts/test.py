import pandas as pd
pd.set_option('display.max_columns', 3000)
pd.set_option('display.max_rows', 3000)

from scripts.jra_raceuma_mark import Ld

start_date = '2012/01/01'
end_date = '2018/12/31'
model_version = 'jra_rc_raptype'
test_flag = True
mock_flag = False

ld = Ld(model_version, start_date, end_date, mock_flag, test_flag)
ld.set_race_df()
race_df = ld.race_df
print(race_df.shape)

dup_race_df = race_df[["RACE_KEY"]]
dup_race_df = dup_race_df[dup_race_df.duplicated()]
print(dup_race_df.head())

print(race_df.query("RACE_KEY == '06134501'"))
