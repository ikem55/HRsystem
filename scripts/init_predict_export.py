from scripts.lb_v1 import SkModel as LBv1SkModel
from scripts.lb_v2 import SkModel as LBv2SkModel
from scripts.lb_v3 import SkModel as LBv3SkModel
from scripts.lb_v4 import SkModel as LBv4SkModel
from scripts.lb_v5 import SkModel as LBv5SkModel
from scripts.lbr_v1 import SkModel as LBRv1SkModel
import sys

from datetime import datetime as dt
from datetime import timedelta
import my_config as mc
import luigi
from distutils.util import strtobool
import pickle
import pandas as pd

from modules.base_task_predict import End_baoz_predict
from modules.lb_extract import LBExtract

# 呼び出し方
# python init_predict_export.py lb_v1 True
# exportがFalseの場合import処理を実施
# ====================================== パラメータ　要変更 =====================================================

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
check_flag = True
dict_path = mc.return_base_path(test_flag)

if model_version == "lb_v1":
    model_name = 'raceuma_ens'
    sk_model = LBv1SkModel(model_name, model_version, start_date, end_date, mock_flag, test_flag, mode)
    table_name = '地方競馬レース馬V1'
elif model_version == "lb_v2":
    model_name = 'raceuma_ens'
    sk_model = LBv2SkModel(model_name, model_version, start_date, end_date, mock_flag, test_flag, mode)
    table_name = '地方競馬レース馬V2'
elif model_version == "lb_v3":
    model_name = 'raceuma_ens'
    sk_model = LBv3SkModel(model_name, model_version, start_date, end_date, mock_flag, test_flag, mode)
    table_name = '地方競馬レース馬V3'
elif model_version == "lb_v4":
    model_name = 'raceuma_lgm'
    sk_model = LBv4SkModel(model_name, model_version, start_date, end_date, mock_flag, test_flag, mode)
    table_name = '地方競馬レース馬V4'
elif model_version == "lb_v5":
    model_name = 'raceuma_lgm'
    sk_model = LBv5SkModel(model_name, model_version, start_date, end_date, mock_flag, test_flag, mode)
    table_name = '地方競馬レース馬V5'
elif model_version == "lbr_v1":
    model_name = 'race_lgm'
    sk_model = LBRv1SkModel(model_name, model_version, start_date, end_date, mock_flag, test_flag, mode)
    table_name = '地方競馬レースV1'
    check_flag = False
    ######################### exportにindexが入っているので要修正
else:
    print("----------- error ---------------")
    model_name = ''
    sk_model = ''
    table_name = ''


intermediate_folder = dict_path + 'intermediate/' + model_version + '_' + args[1] + '/' + model_name + '/'
print("intermediate_folder:" + intermediate_folder)

if test_flag:
    print("set test table")
    table_name = table_name + "_test"
    sk_model.set_test_table(table_name)

if export_mode:
    luigi.build([End_baoz_predict(start_date=start_date, end_date=end_date, skmodel=sk_model,
                              intermediate_folder=intermediate_folder, export_mode=True)], local_scheduler=True)
    if check_flag:
        print(" --- check result --- ")
        with open(intermediate_folder + 'export_data.pkl', 'rb') as f:
            import_df = pickle.load(f)
            ext = LBExtract(start_date, end_date, False)
            result_raceuma_df = ext.get_raceuma_table_base()
            raceuma_df = result_raceuma_df[["競走コード", "馬番", "確定着順", "単勝配当", "複勝配当"]].rename(columns={"競走コード": "RACE_KEY" , "馬番": "UMABAN"})
            raceuma_df.loc[:, "ck1"] = raceuma_df["確定着順"].apply(lambda x: 1 if x == 1 else 0)
            raceuma_df.loc[:, "ck2"] = raceuma_df["確定着順"].apply(lambda x: 1 if x == 2 else 0)
            raceuma_df.loc[:, "ck3"] = raceuma_df["確定着順"].apply(lambda x: 1 if x == 3 else 0)
            raceuma_df.loc[:, "ckg"] = raceuma_df["確定着順"].apply(lambda x: 1 if x > 3 else 0)
            if model_version == "lb_v4":
                win_df = import_df.query('target == "１着" and predict_rank == 1')
                win_df = pd.merge(win_df, raceuma_df, on=["RACE_KEY", "UMABAN"])
                print("----- 1ck -----")
                print(win_df.describe())
                ren_df = import_df.query('target == "２着" and predict_rank == 1')
                ren_df = pd.merge(ren_df, raceuma_df, on=["RACE_KEY", "UMABAN"])
                print("----- 2ck -----")
                print(ren_df.describe())
                fuku_df = import_df.query('target == "３着" and predict_rank == 1')
                fuku_df = pd.merge(fuku_df, raceuma_df, on=["RACE_KEY", "UMABAN"])
                print("----- 3ck -----")
                print(fuku_df.describe())
            else:
                win_df = import_df.query('target == "WIN_FLAG" and predict_rank == 1')
                win_df = pd.merge(win_df, raceuma_df, on=["RACE_KEY", "UMABAN"])
                print("----- win_df -----")
                print(win_df.describe())
                jiku_df = import_df.query('target == "JIKU_FLAG" and predict_rank == 1')
                jiku_df = pd.merge(jiku_df, raceuma_df, on=["RACE_KEY", "UMABAN"])
                print("----- jiku_df -----")
                print(jiku_df.describe())
                ana_df = import_df.query('target == "ANA_FLAG" and predict_rank == 1')
                ana_df = pd.merge(ana_df, raceuma_df, on=["RACE_KEY", "UMABAN"])
                print("----- ana_df -----")
                print(ana_df.describe())

else:
    print("import mode")
    sk_model.create_mydb_table(table_name)
    with open(intermediate_folder + 'export_data.pkl', 'rb') as f:
        import_df = pickle.load(f)
        sk_model.import_data(import_df)
