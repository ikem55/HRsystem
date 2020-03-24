from modules.base_slack import SummarySlack
from modules.base_report import LBReport
from datetime import datetime as dt
from datetime import timedelta

n = -60
start_date = (dt.now() + timedelta(days=n)).strftime('%Y/%m/%d')
end_date = (dt.now() + timedelta(days=0)).strftime('%Y/%m/%d')
mock_flag = False
slack = SummarySlack()
rep = LBReport(start_date, end_date, mock_flag)


post_text = ''

if rep.check_flag:
    todays_bet_text = rep.get_todays_bet_text()
    recent_bet_text = rep.get_recent_bet_text()
    trend_text = rep.get_trend_text()
    summary_text = rep.get_summary_text()

    post_text += todays_bet_text + "\r\n"
    post_text += recent_bet_text + "\r\n"
    post_text += trend_text + "\r\n"
    post_text += summary_text + "\r\n"

else:
    post_text = "no data"

test = False

if test:
    print(post_text)
else:
    slack.post_slack_text(post_text)