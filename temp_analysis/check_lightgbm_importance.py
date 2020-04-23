import pickle
import shap
import pandas as pd

import matplotlib.font_manager as fm

pd.set_option('display.max_columns', 3000)
pd.set_option('display.max_rows', 3000)

fonts = fm.findSystemFonts()
l = []
for f in fonts:
    font = fm.FontProperties(fname=f)
    l.append((f, font.get_name(), font.get_family()))
df = pd.DataFrame(l, columns=['path', 'name', 'family'])
check = df[df['path'].apply(lambda s: 'IPA' in s)]
print(df.head())
print(check)

model_folder = "E:\python/for_test_model/lb_v4/raceuma_lgm/"
this_model_name = "raceuma_lgm_主催者コード_2_１着"

with open(model_folder + this_model_name + '.pickle', 'rb') as f:
    model = pickle.load(f)
    importance = pd.DataFrame({"特徴": model.feature_name(), "重要度": model.feature_importance()}).sort_values(["重要度", "特徴"], ascending=False)
    print(importance)

from scripts.lb_v4 import SkModel as LBv4SkModel
intermediate_folder = "E:\python/for_test_intermediate/lb_v4_predict/raceuma_lgm/"
this_file_name = "20191231_exp_data"
exp_data = pd.read_pickle(intermediate_folder + this_file_name + ".pkl")
model_version = 'lb_v4'
model_name = 'raceuma_lgm'
table_name = '地方競馬レース馬V4'
cls_val = "主催者コード"
val = "2"
target = "１着"
start_date = '2019/01/01'
end_date = '2019/12/31'
mock_flag = False
test_flag = True
mode = "predict"
skmodel = LBv4SkModel(model_name, model_version, start_date, end_date, mock_flag,test_flag, mode)

# pred_df = skmodel.proc_predict_sk_model(exp_data, cls_val, val)
exp_data.drop(["RACE_KEY", "NENGAPPI", "主催者コード"], axis=1, inplace=True)
#print(exp_data)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(exp_data)

from IPython.core.display import display
for i in range(1,17):
    print("UMABAN:", i)
    shap.force_plot(explainer.expected_value[i], shap_values[i][0,:], exp_data.iloc[0, :])
