from modules.base_report import LBReport
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta
pd.set_option('display.max_columns', 3000)
pd.set_option('display.max_rows', 3000)

n = 0
start_date = (dt.now() + timedelta(days=n)).strftime('%Y/%m/%d')
end_date = dt.now().strftime('%Y/%m') + '/01'

print(start_date)
print(end_date)
#br = LBReport(start_date, end_date, False)

#print(br.bet_df.head())
#print(br.haraimodoshi_dict)
#print(br.raceuma_df.iloc[0])

#br.export_raceuma_df()