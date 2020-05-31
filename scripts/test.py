import pandas as pd
pd.set_option('display.max_columns', 3000)
pd.set_option('display.max_rows', 3000)

from scripts.jra_raceuma_mark import Ld

start_date = '2012/01/01'
end_date = '2018/12/31'
model_version = 'jra_ru_mark'
test_flag = True
mock_flag = False

ld = Ld(model_version, start_date, end_date, mock_flag, test_flag)
ld.set_race_df()
ld.set_raceuma_df()
ld.set_horse_df()
ld.set_prev_df()


