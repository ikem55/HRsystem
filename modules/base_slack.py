import slackweb
import os
import my_config as mc
from oauth2client.service_account import ServiceAccountCredentials
import gspread
from gspread_dataframe import set_with_dataframe

class BaseSlack(object):
    def __init__(self):
        self.slack_url = mc.SLACK_URL

    def post_slack_text(self, post_text):
        slack = slackweb.Slack(url=self.slack_url)
        slack.notify(text=post_text)

class OperationSlack(BaseSlack):
    def __init__(self):
        self.slack_url = mc.SLACK_operation_webhook_url

class SummarySlack(BaseSlack):
    key_name = '../localBaoz-510313930ad5.json'
    def __init__(self):
        self.slack_url = mc.SLACK_summary_webhook_url

    def upload_gsheet(self, bet_df):
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        credentials = ServiceAccountCredentials.from_json_keyfile_name(self.key_name, scope)
        gc = gspread.authorize(credentials)
        bet_sheet_name = 'localBaozResult'
        bet_update_sheet = gc.open(bet_sheet_name).worksheet("Sheet1")
        set_with_dataframe(bet_update_sheet, bet_df, resize=True, include_index=True)


class RealtimeSlack(BaseSlack):
    def __init__(self):
        self.slack_url = mc.SLACK_realtime_webhook_url