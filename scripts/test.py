from scripts.lb_v2 import SkModel
import modules.util as mu

import pickle


MODEL_NAME = 'raceuma_ens'
start_date = '2018/01/01'
end_date = '2018/01/11'
mock_flag = False
MODEL_VERSION = 'lb_v2'
test_flag = True

sk_model = SkModel(MODEL_NAME, MODEL_VERSION, start_date, end_date, mock_flag, test_flag)

sk_model.create_learning_data()
learning_df = sk_model.learning_df
predict_df = sk_model.create_predict_data()

# 血統登録番号                           2015105636 ヲLearningカラ削除
print(learning_df.shape)
print(predict_df.shape)
mu.check_df(learning_df)
mu.check_df(predict_df)

