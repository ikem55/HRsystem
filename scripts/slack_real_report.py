from modules.base_slack import RealtimeSlack
from modules.base_report import LBReport
from datetime import datetime as dt
from datetime import timedelta

start_date = (dt.now() + timedelta(days=0)).strftime('%Y/%m/%d')
end_date = (dt.now() + timedelta(days=0)).strftime('%Y/%m/%d')
mock_flag = False
slack = RealtimeSlack()
rep = LBReport(start_date, end_date, mock_flag)

bet_df = rep.get_bet_df()
today_bet_df = rep.get_todays_bet_df(bet_df)
bet_text = rep.get_todays_text(today_bet_df)

slack.post_slack_text(bet_text)