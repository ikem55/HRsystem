from modules.base_slack import RealtimeSlack
from modules.base_report import LBReport
from datetime import datetime as dt
from datetime import timedelta

n = 0
start_date = (dt.now() + timedelta(days=n)).strftime('%Y/%m/%d')
end_date = (dt.now() + timedelta(days=n)).strftime('%Y/%m/%d')
mock_flag = False
slack = RealtimeSlack()
rep = LBReport(start_date, end_date, mock_flag)

post_text = ''

if rep.check_flag:
    current_text = rep.get_current_text()
    bet_text = rep.get_todays_bet_text()

    post_text += current_text + "\r\n"
    post_text += bet_text + "\r\n"

else:
    post_text = "no data"

test = False

if test:
    print(post_text)
else:
    slack.post_slack_text(post_text)