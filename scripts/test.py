import pandas as pd
from modules.jra_extract import JRAExtract
start_date = '2019/01/01'
end_date = '2020/06/20'
mock_flag = False
ext = JRAExtract(start_date, end_date, mock_flag)
pd.set_option('display.max_columns', 3000)
pd.set_option('display.max_rows', 3000)

