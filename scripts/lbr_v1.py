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
import optuna.integration.lightgbm as lgb
import lightgbm as lgb_original
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

    def create_feature_race_df(self, race_df):
        """ 特徴となる値を作成する。ナイター、季節、非根幹、距離グループ、頭数グループを作成して列として付与する。

        :param dataframe race_df:
        :return: dataframe
        """
        temp_race_df = race_df.copy()
        temp_race_df.loc[:, 'ナイター'] = race_df['発走時刻'].apply(lambda x: True if x.hour >= 17 else False)
        temp_race_df.loc[:, '季節'] = (race_df['月日'].apply(lambda x: x.month) - 1) // 3
        temp_race_df['季節'].astype('str')
        temp_race_df.loc[:, "非根幹"] = race_df["距離"].apply(lambda x: True if x % 400 == 0 else False)
        temp_race_df.loc[:, "距離グループ"] = race_df["距離"] // 400
        temp_race_df.loc[:, "頭数グループ"] = race_df["頭数"] // 5
        temp_race_df.loc[:, "コース"] = race_df["場コード"].astype(str) + race_df["トラックコード"].astype(str)
        return temp_race_df

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
            print("learning_sk_model: df", df.shape)
            if df.empty:
                print("--------- alert !!! no data")
            else:
                self.set_learning_data(df, target)
                self.divide_learning_data()
                if self.y_train.sum() == 0:
                    print("---- wrong data --- skip learning")
                else:
                    self.load_learning_target_encoding()
                    imp_features = self.learning_base_race_lgb(this_model_name)
                    self.x_df = self.x_df[imp_features]
                    self.divide_learning_data()
                    self._set_label_list(self.x_df) # 項目削除されているから再度ターゲットエンコーディングの対象リストを更新する
                    self.load_learning_target_encoding()
                    self.learning_race_lgb(this_model_name)

    def learning_base_race_lgb(self, this_model_name):
        print("learning_base_race_lgb")
        # テスト用のデータを評価用と検証用に分ける
        X_eval, X_valid, y_eval, y_valid = train_test_split(self.X_test, self.y_test, random_state=42)

        # データセットを生成する
        lgb_train = lgb.Dataset(self.X_train, self.y_train)
        lgb_eval = lgb.Dataset(X_eval, y_eval, reference=lgb_train)

        # 上記のパラメータでモデルを学習する
        model = lgb_original.train(self.lgbm_params, lgb_train,
                          # モデルの評価用データを渡す
                          valid_sets=lgb_eval,
                          # 最大で 1000 ラウンドまで学習する
                          num_boost_round=50,
                          # 10 ラウンド経過しても性能が向上しないときは学習を打ち切る
                          early_stopping_rounds=10)

        # 特徴量の重要度を含むデータフレームを作成
        imp_df = pd.DataFrame()
        imp_df["feature"] = X_eval.columns
        imp_df["importance"] = model.feature_importance()
        print(imp_df)
        imp_df = imp_df.sort_values("importance")

        # 比較用のランダム化したモデルを学習する
        N_RUM = 10 #30くらいに数上げてよさそう
        null_imp_df = pd.DataFrame()
        for i in range(N_RUM):
            print(i)
            ram_lgb_train = lgb.Dataset(self.X_train, np.random.permutation(self.y_train))
            ram_lgb_eval = lgb.Dataset(X_eval, np.random.permutation(y_eval), reference=lgb_train)
            ram_model = lgb_original.train(self.lgbm_params, ram_lgb_train,
                              # モデルの評価用データを渡す
                              valid_sets=ram_lgb_eval,
                              # 最大で 1000 ラウンドまで学習する
                              num_boost_round=100,
                              # 10 ラウンド経過しても性能が向上しないときは学習を打ち切る
                              early_stopping_rounds=6)
            ram_imp_df = pd.DataFrame()
            ram_imp_df["feature"] = X_eval.columns
            ram_imp_df["importance"] = ram_model.feature_importance()
            ram_imp_df = ram_imp_df.sort_values("importance")
            ram_imp_df["run"] = i + 1
            null_imp_df = pd.concat([null_imp_df, ram_imp_df])

        # 閾値を設定
        THRESHOLD = 30

        # 閾値を超える特徴量を取得
        imp_features = []
        for feature in imp_df["feature"]:
            actual_value = imp_df.query(f"feature=='{feature}'")["importance"].values
            null_value = null_imp_df.query(f"feature=='{feature}'")["importance"].values
            percentage = (null_value < actual_value).sum() / null_value.size * 100
            if percentage >= THRESHOLD:
                imp_features.append(feature)

        print(len(imp_features))
        print(imp_features)

        self._save_learning_model(imp_features, this_model_name + "_feat_columns")
        return imp_features


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

    def _merge_df(self):
        self.base_df = pd.merge(self.ld.race_df, self.ld.raceuma_df, on="競走コード")
        self.base_df = pd.merge(self.base_df, self.ld.horse_df, on="血統登録番号")

    def _create_feature(self):
        """ マージしたデータから特徴量を生成する """
        print("_create_feature")
        raceuma_df = self.base_df.drop(["近走競走コード1", "近走競走コード2", "近走競走コード3", "近走競走コード4", "近走競走コード5", "近走馬番1", "近走馬番2", "近走馬番3", "近走馬番4", "近走馬番5",
                                        "馬番グループ", "休養後出走回数", "休養週数", "枠順評価", "脚質評価", "調教師評価", "騎手評価", "先行指数順位", "予想人気グループ", "予想人気", "予想タイム指数順位",
                                        "距離", "予想決着指数", "日次", "頭数グループ", "回次", "枠番", "頭数", "予想勝ち指数", "初出走頭数", "季節", "登録頭数", "ナイター", "混合", "非根幹"], axis=1)
        raceuma_df.loc[:, "競走馬コード"] = raceuma_df["競走コード"].astype(str).str.cat(raceuma_df["馬番"].astype(str))
        raceuma_df.drop("馬番", axis=1, inplace=True)
        # https://qiita.com/daigomiyoshi/items/d6799cc70b2c1d901fb5
        es = ft.EntitySet(id="race")
        #es.entity_from_dataframe(entity_id='race', dataframe=raceuma_df, index="競走馬コード")
        #es.normalize_entity(base_entity_id='race', new_entity_id='raceuma', index="競走コード")
        es.entity_from_dataframe(entity_id='race', dataframe=self.ld.race_df, index="競走コード")
        es.entity_from_dataframe(entity_id='raceuma', dataframe=raceuma_df, index="競走馬コード")
        relationship = ft.Relationship(es['race']["競走コード"], es['raceuma']["競走コード"])
        es = es.add_relationship(relationship)
        print(es)
        # 集約関数
        aggregation_list = ['min', 'max', 'mean', 'skew', 'percent_true']
        transform_list = []
        # run dfs
        print("un dfs")
        feature_matrix, features_dfs = ft.dfs(entityset=es, target_entity='race', agg_primitives=aggregation_list,
                                              trans_primitives=transform_list, max_depth=2)
        print("_create_feature: feature_matrix", feature_matrix.shape)

        # 予想１番人気のデータを取得
        ninki_df = self.base_df.query("予想人気==1")[["競走コード", "枠番", "性別コード", "予想タイム指数順位", "見習区分", "キャリア", "馬齢", "予想展開", "距離増減", "前走頭数", "前走人気", "テン乗り",
                                                  "繁殖登録番号1", "繁殖登録番号5", "東西所属コード", "生産者コード", "馬主コード", "騎手名", "調教師名", "所属", "転厩"]].add_prefix("人気_").rename(columns={"人気_競走コード":"競走コード"})
        # 逃げ予想馬のデータを取得
        nige_df = self.base_df.query("予想展開==1")[["競走コード", "先行指数", "距離増減", "前走人気", "前走頭数", "テン乗り", "繁殖登録番号1", "騎手名", "調教師名"]].add_prefix("逃げ_").rename(columns={"逃げ_競走コード":"競走コード"})
        self.base_df = pd.merge(feature_matrix, nige_df, on="競走コード", how="left")
        self.base_df = pd.merge(self.base_df, ninki_df, on="競走コード")
        self.base_df = pd.merge(self.base_df, self.ld.race_df[["競走コード", "月日"]], on="競走コード")
        mu.check_df(self.base_df)

    def _drop_columns_base_df(self):
        self.base_df.drop(["場名", "競走条件コード", "競走条件コード", "距離グループ", "登録頭数"], axis=1, inplace=True)

    def _scale_df(self):
        pass

    def _rename_key(self, df):
        """ キー名を競走コード→RACE_KEY、月日→NENGAPPIに変更 """
        return_df = df.rename(columns={"競走コード": "RACE_KEY", "月日": "NENGAPPI"})
        return return_df

    def proc_create_learning_data(self):
        self._proc_create_base_df()
        self._drop_unnecessary_columns()
        self._set_target_variables()
        learning_df = pd.merge(self.base_df, self.result_df, on ="RACE_KEY")
        return learning_df

    def _drop_unnecessary_columns(self):
        """ 特に処理がないのでパス  """
        pass

    def _set_target_variables(self):
        self.ld.set_result_df()
        umaren_df = self._create_target_variable_umaren()
        umatan_df = self._create_target_variable_umatan()
#        wide_df = self._create_target_variable_wide()
        sanrenpuku_df = self._create_target_variable_sanrenpuku()
        self.result_df = pd.merge(umaren_df, umatan_df, on ="競走コード")
        self.result_df = pd.merge(self.result_df, sanrenpuku_df, on ="競走コード").rename(columns={"競走コード": "RACE_KEY"})

    def _create_target_variable_umaren(self):
        """  UMAREN_AREの目的変数を作成してresult_dfにセットする。配当が２０００円をこえているものにフラグをセットする。 """
        umaren_df = self.ld.dict_haraimodoshi["umaren_df"]
        umaren_df.loc[:, "UMAREN_ARE"] = umaren_df["払戻"].apply(lambda x: 0 if x < 2000 else 1)
        return umaren_df[["競走コード", "UMAREN_ARE"]]

    def _create_target_variable_umatan(self):
        """  UMATAN_AREの目的変数を作成してresult_dfにセットする。配当が３０００円をこえているものにフラグをセットする。。 """
        umatan_df = self.ld.dict_haraimodoshi["umatan_df"]
        umatan_df.loc[:, "UMATAN_ARE"] = umatan_df["払戻"].apply(lambda x: 0 if x < 3000 else 1)
        return umatan_df[["競走コード", "UMATAN_ARE"]]

    def _create_target_variable_wide(self):
        """  WIDE_AREの目的変数を作成してresult_dfにセットする。配当が９００円をこえているものにフラグをセットする。 """
        wide_df = self.ld.dict_haraimodoshi["wide_df"]
        wide_df.loc[:, "WIDE_ARE"] = wide_df["払戻"].apply(lambda x: 0 if x < 900 else 1)
        return wide_df[["競走コード", "WIDE_ARE"]]

    def _create_target_variable_sanrenpuku(self):
        """  SANRENPUKU_AREの目的変数を作成してresult_dfにセットする。配当が３０００円をこえているものにフラグをセットする。 """
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
        print(pred_df.head())
        return pred_df

    def predict_race_lgm(self, this_model_name, temp_df):
        print("======= this_model_name: " + this_model_name + " ==========")
        with open(self.model_folder + this_model_name + '_feat_columns.pickle', 'rb') as f:
            imp_features = pickle.load(f)
        temp_df = temp_df.replace(np.inf,np.nan).fillna(temp_df.replace(np.inf,np.nan).mean())
        exp_df = temp_df.drop(self.index_list, axis=1)
        exp_df = exp_df[imp_features].to_numpy()
        print(self.model_folder)
        if os.path.exists(self.model_folder + this_model_name + '.pickle'):
            with open(self.model_folder + this_model_name + '.pickle', 'rb') as f:
                model = pickle.load(f)
            y_pred = model.predict(exp_df)
            pred_df = pd.DataFrame({"RACE_KEY": temp_df["RACE_KEY"],"NENGAPPI": temp_df["NENGAPPI"], "prob": y_pred})
            pred_df.loc[:, "pred"] = pred_df.apply(lambda x: 1 if x["prob"] >= 0.5 else 0, axis=1)
            return pred_df
        else:
            return pd.DataFrame()

    def create_eval_prd_data(self, df):
        """ 予測されたデータの精度をチェック """
        self._set_target_variables()
        check_df = pd.merge(df[df["pred"] == 1], self.result_df, on="RACE_KEY")
        return check_df.copy()

class SkModel(LBSkModel):
    class_list = ['主催者コード']
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
        """ 特に処理はないので桁数だけそろえてリターン"""
        import_df = all_df.round(3)
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
            target_df = re_df[re_df['target_date'] == date][["RACE_KEY", "pred", "prob", "target", "target_date"]]
            print(target_df.head())
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

    def eval_pred_data(self, df):
        """ 予測されたデータの精度をチェック """
        check_df = self.proc.create_eval_prd_data(df)
        for target in self.obj_column_list:
            print(target)
            target_df = check_df[check_df["target"] == target]
            target_df = target_df.query("pred == 1")
            if len(target_df.index) != 0:
                target_df.loc[:, "的中"] = target_df.apply(lambda x: 1 if x[target] == 1 else 0, axis=1)
                print(target_df)
                avg_rate = target_df["的中"].mean()
                print(round(avg_rate*100, 1))
            else:
                print("skip eval")

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