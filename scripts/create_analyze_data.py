from modules.base_report import LBReport
from datetime import datetime as dt
from datetime import timedelta
import codecs
import json

n = -30
start_date = (dt.now() + timedelta(days=n)).strftime('%Y/%m/%d')
end_date = dt.now().strftime('%Y/%m/%d')
mock_flag = False
rep = LBReport(start_date, end_date, mock_flag)
export_path = "C:\python/localapp/static/data/"

bet_df = rep.get_filterd_bet_df(100)

print(bet_df)

summary_bet_df = rep._get_bet_summary_df(bet_df)
print(summary_bet_df)
summary_bet_df.to_pickle(export_path + "summary_bet_df.pickle")

race_list_df = rep.ext.get_race_before_table_base()
print(race_list_df)
race_list_df.to_pickle(export_path + "race_list_df.pickle")