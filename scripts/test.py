import pandas as pd
from modules.jra_extract import JRAExtract
start_date = '2019/01/01'
end_date = '2020/06/20'
mock_flag = False
ext = JRAExtract(start_date, end_date, mock_flag)
pd.set_option('display.max_columns', 3000)
pd.set_option('display.max_rows', 3000)

df = pd.read_pickle("C:/Users/ikem5/Dropbox/jrdb_data/OZ/OZ200613.txt.pkl")
check_df = df[["RACE_KEY", "UMABAN", "単勝オッズ", "複勝オッズ"]].copy()

print(check_df.head(100))
print(check_df.describe())