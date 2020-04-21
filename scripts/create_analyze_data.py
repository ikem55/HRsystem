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

bet_df = rep.get_filterd_bet_df(100)

print(bet_df)

summary_bet_df = rep._get_bet_summary_df(bet_df)
print(summary_bet_df)
export_path = "C:\Dashboard/mydashboard/assets/data/summary_bet.json"
summary_bet_df.to_json(export_path)
