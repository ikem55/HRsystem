from scripts.lb_v1 import SkModel
import modules.util as mu

import pickle

import os
import sys

basedir = os.path.dirname(__file__)[:-8]
print(basedir)
sys.path.append(basedir)

MODEL_NAME = 'raceuma_ens'
start_date = '2018/01/01'
end_date = '2018/01/11'
mock_flag = False
MODEL_VERSION = 'lb_v1'
test_flag = True

sk_model = SkModel(MODEL_NAME, MODEL_VERSION, start_date, end_date, mock_flag, test_flag)

df = sk_model.create_predict_data()

import pandas as pd
pd.set_option('display.max_rows', 500)
print(df.iloc[0])
