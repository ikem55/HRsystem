from scripts.lb_v1 import SkModel as LBv1SkModel
from scripts.lb_v2 import SkModel as LBv2SkModel
import sys

from datetime import datetime as dt
from datetime import timedelta
import my_config as mc
import luigi
from distutils.util import strtobool
import pickle

from modules.base_task_predict import End_baoz_predict

# 呼び出し方
# python init_predict_export.py lb_v1 True
# ====================================== パラメータ　要変更 =====================================================

model_name = 'raceuma_ens'

start_date = '2019/01/01'
end_date = (dt.now() + timedelta(days=0)).strftime('%Y/%m/%d')
mode = "predict"
mock_flag = False
test_flag = False

args = sys.argv
print("------------- start luigi tasks ----------------")
print(args)
print("model_version：" + args[1])  # lb_v1, lb_v2
print("export：" + args[2])  # True or False

model_version = args[1]
export_mode = strtobool(args[2])

dict_path = mc.return_base_path(test_flag)
intermediate_folder = dict_path + 'intermediate/' + model_version + '_' + args[1] + '/' + model_name + '/'
print("intermediate_folder:" + intermediate_folder)

if model_version == "lb_v1":
    sk_model = LBv1SkModel(model_name, model_version, start_date, end_date, mock_flag, test_flag, mode)
elif model_version == "lb_v2":
    sk_model = LBv2SkModel(model_name, model_version, start_date, end_date, mock_flag, test_flag, mode)
else:
    print("----------- error ---------------")
    sk_model = ''

if export_mode:
    luigi.build([End_baoz_predict(start_date=start_date, end_date=end_date, skmodel=sk_model,
                              intermediate_folder=intermediate_folder, export_mode=True)], local_scheduler=True)
else:
    print("import mode")
    with open(intermediate_folder + 'export_data.pkl', 'rb') as f:
        import_df = pickle.load(f)
        sk_model.import_data(import_df)
