import slackweb
import os
import my_config as mc

class BaseSlack(object):
    def __init__(self):
        self.slack_url = mc.SLACK_URL

    def post_slack_text(self, post_text):
        slack = slackweb.Slack(url=self.slack_url)
        slack.notify(text=post_text)

class OperationSlack(BaseSlack):
    def __init__(self):
        print(os.environ)
        self.slack_url = mc.SLACK_operation_webhook_url

