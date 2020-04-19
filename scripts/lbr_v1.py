from modules.lb_extract import LBExtract
from modules.lb_transform import LBTransform
from modules.lb_load import LBLoad
from modules.lb_sk_model import LBSkModel
from modules.lb_sk_proc import LBSkProc
import modules.util as mu
import my_config as mc

import luigi
from modules.base_task_learning import End_baoz_learning
from modules.base_task_predict import End_baoz_predict

from datetime import datetime as dt
from datetime import timedelta
import sys
import pandas as pd
import numpy as np
import os
from distutils.util import strtobool

from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import accuracy_score
import featuretools as ft
import pyodbc
import pickle

basedir = os.path.dirname(__file__)[:-8]
print(basedir)
sys.path.append(basedir)

# 呼び出し方
# python lbr_v1.py learning True True
# ====================================== パラメータ　要変更 =====================================================
# 荒れるレースかどうかを判定。LightGBMを使う

MODEL_VERSION = 'lbr_v1'
MODEL_NAME = 'race_lgm'
TABLE_NAME = '地方競馬レースV1'

# ============================================================================================================

CONN_STR = (
    r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
    r'DBQ=C:\BaoZ\DB\MasterDB\MyDB.MDB;'
)

# ====================================== クラス　要変更 =========================================================

class Ext(LBExtract):
    pass

class Tf(LBTransform):

    def drop_columns_raceuma_df(self, raceuma_df):
        return raceuma_df.drop(["データ作成年月日", "予想オッズ", "血統距離評価", "血統トラック評価", "血統成長力評価",
                                           "血統総合評価", "血統距離評価B", "血統トラック評価B", "血統成長力評価B", "血統総合評価B", "騎手コード",
                                           "調教師コード", "前走着差", "前走トラック種別コード", "前走馬体重",
                                           "タイム指数上昇係数", "タイム指数回帰推定値", "タイム指数回帰標準偏差", "前走休養週数",
                                           "得点V1順位", "得点V2順位", "デフォルト得点順位", "得点V3順位"], axis=1)

    def choose_horse_column(self, horse_df):
        """ 馬データから必要な列に絞り込む。対象は血統登録番号、繁殖登録番号１、繁殖登録番号５、東西所属コード、生産者コード、馬主コード

        :param dataframe raceuma_df:
        :return: dataframe
        """
        temp_horse_df = horse_df[['血統登録番号', '繁殖登録番号1', '繁殖登録番号5', '東西所属コード', '生産者コード', '馬主コード']]
        return temp_horse_df

class Ld(LBLoad):
    def _get_extract_object(self, start_date, end_date, mock_flag):
        """ 利用するExtクラスを指定する """
        ext = Ext(start_date, end_date, mock_flag)
        return ext

    def _get_transform_object(self, start_date, end_date):
        """ 利用するTransformクラスを指定する """
        tf = Tf(start_date, end_date)
        return tf

    def _proc_raceuma_df(self, raceuma_base_df):
        raceuma_df = self.tf.normalize_raceuma_df(raceuma_base_df)
        raceuma_df = self.tf.standardize_raceuma_df(raceuma_df)
        raceuma_df = self.tf.create_feature_raceuma_df(raceuma_df)
        raceuma_df = self.tf.drop_columns_raceuma_df(raceuma_df)
        return raceuma_df.copy()

    def set_prev_df(self):
        """  prev_dfを作成するための処理。prev1_raceuma_dfに処理がされたデータをセットする。過去1走のデータと過去走を集計したデータをセットする  """
        print("skip prev_df")

    def set_result_df(self):
        """ result_dfを作成するための処理。result_dfに処理がされたデータをセットする """
        print("set_result_df")
        haraimodoshi_df = self.ext.get_haraimodoshi_table_base()
        self.dict_haraimodoshi = mu.get_haraimodoshi_dict(haraimodoshi_df)


class SkProc(LBSkProc):
    """
    地方競馬の機械学習処理プロセスを取りまとめたクラス。
    """
    index_list = ["RACE_KEY", "NENGAPPI"]
    # LightGBM のハイパーパラメータ
    lgbm_params = {
        # 2値分類問題
        'objective': 'binary'
    }

    def _get_load_object(self, version_str, start_date, end_date, mock_flag, test_flag):
        ld = Ld(version_str, start_date, end_date, mock_flag, test_flag)
        return ld

    def learning_sk_model(self, df, cls_val, val, target):
        """ 指定された場所・ターゲットに対しての学習処理を行う

        :param dataframe df: dataframe
        :param str val: str
        :param str target: str
        """
        this_model_name = self.model_name + "_" + cls_val + '_' + val + '_' + target
        if os.path.exists(self.model_folder + this_model_name + '.pickle'):
            print("\r\n -- skip create learning model -- \r\n")
        else:
            self.set_target_flag(target)
            print(df.shape)
            if df.empty:
                print("--------- alert !!! no data")
            else:
                self.set_learning_data(df, target)
                self.divide_learning_data()
                if self.y_train.sum() == 0:
                    print("---- wrong data --- skip learning")
                else:
                    self.load_learning_target_encoding()
                    self.learning_race_lgb(this_model_name)

    def learning_race_lgb(self, this_model_name):
        # テスト用のデータを評価用と検証用に分ける
        X_eval, X_valid, y_eval, y_valid = train_test_split(self.X_test, self.y_test, random_state=42)

        # データセットを生成する
        lgb_train = lgb.Dataset(self.X_train, self.y_train)
        lgb_eval = lgb.Dataset(X_eval, y_eval, reference=lgb_train)

        # 上記のパラメータでモデルを学習する
        model = lgb.train(self.lgbm_params, lgb_train,
                          # モデルの評価用データを渡す
                          valid_sets=lgb_eval,
                          # 最大で 1000 ラウンドまで学習する
                          num_boost_round=1000,
                          # 10 ラウンド経過しても性能が向上しないときは学習を打ち切る
                          early_stopping_rounds=10)

        self._save_learning_model(model, this_model_name)
        # 学習したモデルでホールドアウト検証する
        y_pred_proba = model.predict(X_valid, num_iteration=model.best_iteration)


    def _merge_df(self):
        self.base_df = pd.merge(self.ld.raceuma_df, self.ld.horse_df, on="血統登録番号")

    def _create_feature(self):
        """ マージしたデータから特徴量を生成する """
        raceuma_df = self.base_df.copy()[["競走コード", "馬番", "予想タイム指数", "予想展開", "クラス変動", "騎手評価", "調教師評価", "枠順評価", "脚質評価", "馬齢", "前走着順", "前走人気", "前走頭数", "騎手ランキング", "調教師ランキング"]]
        raceuma_df.loc[:, "競走馬コード"] = raceuma_df["競走コード"].astype(str).str.cat(raceuma_df["馬番"].astype(str))
        raceuma_df.drop("馬番", axis=1, inplace=True)
        # https://qiita.com/daigomiyoshi/items/d6799cc70b2c1d901fb5
        es = ft.EntitySet(id="race")
        es.entity_from_dataframe(entity_id='race', dataframe=raceuma_df, index="競走馬コード")
        es.normalize_entity(base_entity_id='race', new_entity_id='raceuma', index="競走コード")
        # 集約関数
        aggregation_list = ['count', 'min', 'max', 'mean']
        transform_list = []
        # run dfs
        feature_matrix, features_dfs = ft.dfs(entityset= es, target_entity= 'race', agg_primitives = aggregation_list , trans_primitives=transform_list, max_depth=2)
        print(feature_matrix.shape)
        feature_matrix.head(3)

        # 予想１番人気のデータを取得
        ninki_df = self.base_df.query("予想人気==1")[["競走コード", "枠番", "性別コード", "予想タイム指数順位", "見習区分", "キャリア", "馬齢", "予想展開", "距離増減", "前走頭数", "前走人気", "テン乗り"]].add_prefix("人気_").rename(columns={"人気_競走コード":"競走コード"})
        # 逃げ予想馬のデータを取得
        nige_df = self.base_df.query("予想展開==1")[["競走コード", "先行指数", "距離増減", "前走人気", "前走頭数", "テン乗り"]].add_prefix("逃げ_").rename(columns={"逃げ_競走コード":"競走コード"})
        self.base_df = pd.merge(feature_matrix, nige_df, on="競走コード")
        self.base_df = pd.merge(self.base_df, ninki_df, on="競走コード")
        self.base_df = pd.merge(self.base_df, self.ld.race_df, on="競走コード")

    def _drop_columns_base_df(self):
        print("-- check! this is LBSkProc class: " + sys._getframe().f_code.co_name)
        self.base_df.drop("発走時刻", axis=1, inplace=True)


    def _scale_df(self):
        print("-- check! this is LBSkProc class: " + sys._getframe().f_code.co_name)

    def _rename_key(self, df):
        """ キー名を競走コード→RACE_KEY、馬番→UMABANに変更 """
        return_df = df.rename(columns={"競走コード": "RACE_KEY", "月日": "NENGAPPI"})
        return return_df

    def proc_create_learning_data(self):
        self._proc_create_base_df()
        self._drop_unnecessary_columns()
        self._set_target_variables()
        learning_df = pd.merge(self.base_df, self.result_df, on ="RACE_KEY")
        return learning_df

    def _drop_unnecessary_columns(self):
        """ predictに不要な列を削除してpredict_dfを作成する。削除する列は血統登録番号、確定着順、タイム指数、単勝オッズ、単勝人気  """
        pass
#        self.base_df.drop(['血統登録番号'], axis=1, inplace=True)

    def _set_target_variables(self):
        self.ld.set_result_df()
        umaren_df = self._create_target_variable_umaren()
        umatan_df = self._create_target_variable_umatan()
#        wide_df = self._create_target_variable_wide()
        sanrenpuku_df = self._create_target_variable_sanrenpuku()
        self.result_df = pd.merge(umaren_df, umatan_df, on ="競走コード")
        self.result_df = pd.merge(self.result_df, sanrenpuku_df, on ="競走コード").rename(columns={"競走コード": "RACE_KEY"})

    def _create_target_variable_umaren(self):
        """  UMAREN_AREの目的変数を作成してresult_dfにセットする。条件は< 5, 5 to 20, 20 to 50, 50 over。 """
        umaren_df = self.ld.dict_haraimodoshi["umaren_df"]
        umaren_df.loc[:, "UMAREN_ARE"] = umaren_df["払戻"].apply(lambda x: 0 if x < 2000 else 1)
        return umaren_df[["競走コード", "UMAREN_ARE"]]

    def _create_target_variable_umatan(self):
        """  UMATAN_AREの目的変数を作成してresult_dfにセットする。条件は< 10,10 to 30, 30 to 70, 70 over。 """
        umatan_df = self.ld.dict_haraimodoshi["umatan_df"]
        umatan_df.loc[:, "UMATAN_ARE"] = umatan_df["払戻"].apply(lambda x: 0 if x < 3000 else 1)
        return umatan_df[["競走コード", "UMATAN_ARE"]]

    def _create_target_variable_wide(self):
        """  WIDE_AREの目的変数を作成してresult_dfにセットする。条件は< 3, 3 to 9, 9 to 20, 20 over。 """
        wide_df = self.ld.dict_haraimodoshi["wide_df"]
        wide_df.loc[:, "WIDE_ARE"] = wide_df["払戻"].apply(lambda x: 0 if x < 900 else 1)
        return wide_df[["競走コード", "WIDE_ARE"]]

    def _create_target_variable_sanrenpuku(self):
        """  SANRENPUKU_AREの目的変数を作成してresult_dfにセットする。条件は< 10, 10 to 30, 30 to 80, 80 over。 """
        sanrenpuku_df = self.ld.dict_haraimodoshi["sanrenpuku_df"]
        sanrenpuku_df.loc[:, "SANRENPUKU_ARE"] = sanrenpuku_df["払戻"].apply(lambda x: 0 if x < 3000 else 1)
        return sanrenpuku_df[["競走コード", "SANRENPUKU_ARE"]]

    def _sub_distribute_predict_model(self, cls_val, val, target, temp_df):
        """ model_nameに応じたモデルを呼び出し予測を行う

        :param str val: 場所名
        :param str target: 目的変数名
        :param dataframe temp_df: 予測するデータフレーム
        :return dataframe: pred_df
        """
        this_model_name = self.model_name + "_" + cls_val + '_' + val + '_' + target
        pred_df = self.predict_race_lgm(this_model_name, temp_df)
        return pred_df

    def predict_race_lgm(self, this_model_name, temp_df):
        print("======= this_model_name: " + this_model_name + " ==========")

        temp_df = temp_df.replace(np.inf,np.nan).fillna(temp_df.replace(np.inf,np.nan).mean())
        exp_df = temp_df.drop(self.index_list, axis=1).to_numpy()
        print(self.model_folder)
        with open(self.model_folder + this_model_name + '.pickle', 'rb') as f:
            model = pickle.load(f)
        y_pred = model.predict(exp_df)
        pred_df = pd.DataFrame({"RACE_KEY": temp_df["RACE_KEY"],"NENGAPPI": temp_df["NENGAPPI"], "prob": y_pred})
        pred_df.loc[:, "pred"] = pred_df.apply(lambda x: 1 if x["prob"] >= 0.5 else 0, axis=1)
        return pred_df


class SkModel(LBSkModel):
    class_list = ['競走種別コード', '場コード']
    table_name = TABLE_NAME
    obj_column_list = ['UMAREN_ARE', 'UMATAN_ARE', 'SANRENPUKU_ARE']

    def _get_skproc_object(self, version_str, start_date, end_date, model_name, mock_flag, test_flag):
        proc = SkProc(version_str, start_date, end_date, model_name, mock_flag, test_flag, self.obj_column_list)
        return proc

    def proc_learning_sk_model(self, df, cls_val, val):
        """  説明変数ごとに、指定された場所の学習を行う

        :param dataframe df: dataframe
        :param str basho: str
        """
        for target in self.obj_column_list:
            print(target)
            self.proc.learning_sk_model(df, cls_val, val, target)

    def create_import_data(self, all_df):
        """ データフレームをアンサンブル化（Vote）して格納 """
        grouped_all_df = all_df.groupby(["RACE_KEY", "target"], as_index=False).mean().reset_index()
        date_df = all_df[["RACE_KEY", "target_date"]].drop_duplicates()
        import_df = pd.merge(grouped_all_df, date_df, on="RACE_KEY").round(3)
        return import_df


    def import_data(self, df):
        """ 計算した予測値のdataframeを地方競馬DBに格納する

        :param dataframe df: dataframe
        """
        cnxn = pyodbc.connect(self.conn_str)
        crsr = cnxn.cursor()
        re_df = df.replace([np.inf, -np.inf], np.nan).dropna()
        date_list = df['target_date'].drop_duplicates()
        for date in date_list:
            print(date)
            target_df = re_df[re_df['target_date'] == date]
            crsr.execute("DELETE FROM " + self.table_name + " WHERE target_date ='" + date + "'")
            crsr.executemany(
                f"INSERT INTO " + self.table_name + " (競走コード, 予測フラグ, 予測値, target, target_date) VALUES (?,?,?,?,?)",
                target_df.itertuples(index=False)
            )
            cnxn.commit()

    def create_mydb_table(self, table_name):
        """ mydbに予測データを作成する """
        cnxn = pyodbc.connect(self.conn_str)
        create_table_sql = 'CREATE TABLE ' + table_name + ' (' \
            '競走コード DOUBLE, 予測フラグ SINGLE, 予測値 SINGLE, target VARCHAR(255), target_date VARCHAR(255),' \
            ' PRIMARY KEY(競走コード, target));'
        crsr = cnxn.cursor()
        table_list = []
        for talble_info in crsr.tables(tableType='TABLE'):
            table_list.append(talble_info.table_name)
        print(table_list)
        if not table_name in table_list:
            print(create_table_sql)
            crsr.execute(create_table_sql)
            crsr.commit()
        crsr.close()
        cnxn.close()

# ============================================================================================================

if __name__ == "__main__":
    args = sys.argv
    print("------------- start luigi tasks ----------------")
    print(args)
    print("mode：" + args[1])  # learning or predict
    print("mock flag：" + args[2])  # True or False
    print("test mode：" + args[3])  # True or False
    mode = args[1]
    mock_flag = strtobool(args[2])
    test_flag = strtobool(args[3])
    dict_path = mc.return_base_path(test_flag)
    INTERMEDIATE_FOLDER = dict_path + 'intermediate/' + MODEL_VERSION + '_' + args[1] + '/' + MODEL_NAME + '/'
    print("intermediate_folder:" + INTERMEDIATE_FOLDER)

    if mode == "learning":
        if test_flag:
            print("Test mode")
            start_date = '2018/01/01'
            end_date = '2018/01/31'
        else:
            start_date = '2015/01/01'
            end_date = '2018/12/31'
        if mock_flag:
            print("use mock data")
        print("MODE:learning mock_flag: " + str(args[2]) + "  start_date:" + start_date + " end_date:" + end_date)

        sk_model = SkModel(MODEL_NAME, MODEL_VERSION, start_date, end_date, mock_flag, test_flag, mode)

        luigi.build([End_baoz_learning(start_date=start_date, end_date=end_date, skmodel=sk_model,
                                       intermediate_folder=INTERMEDIATE_FOLDER)], local_scheduler=True)

    elif mode == "predict":
        if test_flag:
            print("Test mode")
            start_date = '2018/01/10'
            end_date = '2018/01/11'
        else:
            base_start_date = '2019/01/01'
            start_date = SkModel.get_recent_day(base_start_date)
            end_date = (dt.now() + timedelta(days=0)).strftime('%Y/%m/%d')
            if start_date > end_date:
                print("change start_date")
                start_date = end_date
        if mock_flag:
            print("use mock data")
        print("MODE:predict mock_flag:" + str(args[2]) + "  start_date:" + start_date + " end_date:" + end_date)

        sk_model = SkModel(MODEL_NAME, MODEL_VERSION, start_date, end_date, mock_flag, test_flag, mode)
        if test_flag:
            print("set test table")
            table_name = TABLE_NAME + "_test"
            sk_model.set_test_table(table_name)
            sk_model.create_mydb_table(table_name)

        luigi.build([End_baoz_predict(start_date=start_date, end_date=end_date, skmodel=sk_model,
                                      intermediate_folder=INTERMEDIATE_FOLDER, export_mode=False)], local_scheduler=True)

    else:
        print("error")