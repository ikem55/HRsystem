from modules.jra_extract import JRAExtract
from modules.jra_transform import JRATransform
from modules.jra_load import JRALoad
from modules.jra_sk_model import JRASkModel
from modules.jra_sk_proc import JRASkProc
import modules.util as mu
import my_config as mc

import luigi
from modules.jra_task_learning import End_baoz_learning
from modules.jra_task_predict import End_baoz_predict

from datetime import datetime as dt
from datetime import timedelta
import sys
import pandas as pd
import numpy as np
import scipy
import os
from distutils.util import strtobool
from sklearn.model_selection import train_test_split
import optuna.integration.lightgbm as lgb

import featuretools as ft
import pickle


# 呼び出し方
# python jra_race_haito.py learning True True
# ====================================== パラメータ　要変更 =====================================================
# 配当の予測を行う。LightGBMを使う

MODEL_VERSION = 'jra_rc_haito'
MODEL_NAME = 'race_lgm'

# ====================================== クラス　要変更 =========================================================

class Ext(JRAExtract):
    pass

class Tf(JRATransform):
    pass

class Ld(JRALoad):
    def _get_extract_object(self, start_date, end_date, mock_flag):
        """ 利用するExtクラスを指定する """
        ext = Ext(start_date, end_date, mock_flag)
        return ext

    def _get_transform_object(self, start_date, end_date):
        """ 利用するTransformクラスを指定する """
        tf = Tf(start_date, end_date)
        return tf

    def set_race_df(self):
        """  race_dfを作成するための処理。race_dfに処理がされたデータをセットする """
        race_base_df = self.ext.get_race_before_table_base()
        pred_rap_type_df = self._get_max_value_df(self.ext.get_pred_rap_type_df(), "RAP_TYPE")
        pred_track_bias_uchisoto_df = self._get_max_value_df(self.ext.get_pred_track_bias_uchisoto_df(), "TRACK_BIAS_UCHISOTO")
        pred_track_bias_zengo_df = self._get_max_value_df(self.ext.get_pred_track_bias_zengo_df(), "TRACK_BIAS_ZENGO")
        pred_pace_df = self._get_max_value_df(self.ext.get_pred_pace_df(), "PRED_PACE")
        race_base_df = pd.merge(race_base_df, pred_rap_type_df, on="RACE_KEY")
        race_base_df = pd.merge(race_base_df, pred_track_bias_uchisoto_df, on="RACE_KEY")
        race_base_df = pd.merge(race_base_df, pred_track_bias_zengo_df, on="RACE_KEY")
        race_base_df = pd.merge(race_base_df, pred_pace_df, on="RACE_KEY")
        self.race_df = self._proc_race_df(race_base_df)
        print("set_race_df: race_df", self.race_df.shape)

    def _get_max_value_df(self, df, value):
        ddf = df.groupby("RACE_KEY")
        return_df = df.loc[ddf["prob"].idmax(), :]
        return return_df[["RACE_KEY", "CLASS"]].rename(columns={"CLASS": value})

    def set_raceuma_df(self):
        """ raceuma_dfを作成するための処理。raceuma_dfに処理がされたデータをセットする """
        raceuma_base_df = self.ext.get_raceuma_before_table_base()
        pred_nigeuma_df = self.ext.get_pred_nigeuma_df()[["RACE_KEY", "UMABAN", "prob"]].rename(columns={"prob": "NIGE_prob", "predict_rank": "NIGE_pred_rank"})
        pred_agari_saisoku_df = self.ext.get_pred_agari_saisoku_df()[["RACE_KEY", "UMABAN", "prob"]].rename(columns={"prob": "AGARI_prob", "predict_rank": "AGARI_pred_rank"})
        pred_ten_saisoku_df = self.ext.get_pred_ten_saisoku_df()[["RACE_KEY", "UMABAN", "prob"]].rename(columns={"prob": "TEN_prob", "predict_rank": "TEN_pred_rank"})
        raceuma_base_df = pd.merge(raceuma_base_df, pred_nigeuma_df, on=["RACE_KEY", "UMABAN"])
        raceuma_base_df = pd.merge(raceuma_base_df, pred_agari_saisoku_df, on=["RACE_KEY", "UMABAN"])
        raceuma_base_df = pd.merge(raceuma_base_df, pred_ten_saisoku_df, on=["RACE_KEY", "UMABAN"])
        self.raceuma_df = self._proc_raceuma_df(raceuma_base_df)
        print("set_raceuma_df: raceuma_df", self.raceuma_df.shape)

    def set_result_df(self):
        """ result_dfを作成するための処理。result_dfに処理がされたデータをセットする """
        print("set_result_df")
        haraimodoshi_df = self.ext.get_haraimodoshi_table_base()
        self.dict_haraimodoshi = self.ext.get_haraimodoshi_dict(haraimodoshi_df)

class SkProc(JRASkProc):
    """
    地方競馬の機械学習処理プロセスを取りまとめたクラス。
    """
    index_list = ["RACE_KEY", "NENGAPPI"]
    # LightGBM のハイパーパラメータ
    obj_column_list = ['UMAREN_ARE', 'UMATAN_ARE', 'SANRENPUKU_ARE']
    lgbm_params = {
        # 2値分類問題
        'objective': 'binary'
    }

    def _get_load_object(self, version_str, start_date, end_date, mock_flag, test_flag):
        ld = Ld(version_str, start_date, end_date, mock_flag, test_flag)
        return ld

    def _create_feature(self):
        """ マージしたデータから特徴量を生成する """
        print("_create_feature")
        raceuma_df = self.base_df[["RACE_KEY", "UMABAN", "脚質", "距離適性", "父馬産駒連対平均距離", "母父馬産駒連対平均距離", "IDM", "テン指数",
                                   "ペース指数", "上がり指数", "位置指数", "ＩＤＭ結果_1", "テン指数結果_1", "上がり指数結果_1", "ペース指数結果_1", "レースＰ指数結果_1",
                                   "先行率_1", "追込率_1", "fa_1_1", "fa_2_1", "fa_3_1", "fa_4_1", "fa_5_1"]]
        raceuma_df.loc[:, "RACE_UMA_KEY"] = raceuma_df["RACE_KEY"] + raceuma_df["UMABAN"]
        raceuma_df.drop("UMABAN", axis=1, inplace=True)
        # https://qiita.com/daigomiyoshi/items/d6799cc70b2c1d901fb5
        es = ft.EntitySet(id="race")
        es.entity_from_dataframe(entity_id='race', dataframe=self.ld.race_df.drop("NENGAPPI", axis=1), index="RACE_KEY")
        es.entity_from_dataframe(entity_id='raceuma', dataframe=raceuma_df, index="RACE_UMA_KEY")
        relationship = ft.Relationship(es['race']["RACE_KEY"], es['raceuma']["RACE_KEY"])
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
        ninki_df = self.base_df.query("基準人気順位==1")[["RACE_KEY", "脚質", "距離適性", "上昇度", "激走指数", "蹄コード", "見習い区分", "枠番", "総合印", "ＩＤＭ印", "情報印", "騎手印",
                                                  "厩舎印", "調教印", "激走印", "展開記号", "輸送区分", "騎手期待単勝率", "騎手期待３着内率", "激走タイプ", "休養理由分類コード", "芝ダ障害フラグ",
                                                    "距離フラグ", "クラスフラグ", "転厩フラグ", "去勢フラグ", "乗替フラグ", "放牧先ランク", "厩舎ランク", "調教量評価", "仕上指数変化", "調教評価",
                                                    "IDM", "騎手指数", "情報指数", "総合指数", "人気指数", "調教指数", "厩舎指数", "テン指数", "ペース指数", "上がり指数", "位置指数", "追切指数", "仕上指数",
                                                    "ＩＤＭ結果_1", "ＩＤＭ結果_2"]].add_prefix("人気_").rename(columns={"人気_RACE_KEY":"RACE_KEY"})
        # 逃げ予想馬のデータを取得
        nige_df = self.base_df.query("NIGE_pred_rank=='1'")[["RACE_KEY", "脚質", "距離適性", "上昇度", "激走指数", "蹄コード", "見習い区分", "枠番", "総合印", "ＩＤＭ印", "基準人気順位", "輸送区分", "激走タイプ", "休養理由分類コード", "芝ダ障害フラグ",
                                                    "距離フラグ", "クラスフラグ", "転厩フラグ", "去勢フラグ", "乗替フラグ", "IDM", "騎手指数", "テン指数", "ペース指数", "上がり指数", "位置指数", "追切指数", "仕上指数",
                                                    "斤量_1", "テン指数結果_1", "上がり指数結果_1", "ペース指数結果_1", "レースＰ指数結果_1", "斤量_2", "テン指数結果_2", "上がり指数結果_2", "ペース指数結果_2", "レースＰ指数結果_2",
                                                    "先行率_1", "先行率_2"]].add_prefix("逃げ_").rename(columns={"逃げ_RACE_KEY":"RACE_KEY"})
        # 上がり最速予想馬のデータを取得
        agari_df = self.base_df.query("AGARI_pred_rank=='1'")[["RACE_KEY", "脚質", "距離適性", "上昇度", "激走指数", "蹄コード", "見習い区分", "枠番", "総合印", "ＩＤＭ印", "基準人気順位", "輸送区分", "激走タイプ", "休養理由分類コード", "芝ダ障害フラグ",
                                                    "距離フラグ", "クラスフラグ", "転厩フラグ", "去勢フラグ", "乗替フラグ", "IDM", "騎手指数", "テン指数", "ペース指数", "上がり指数", "位置指数", "追切指数", "仕上指数",
                                                    "斤量_1", "テン指数結果_1", "上がり指数結果_1", "ペース指数結果_1", "レースＰ指数結果_1", "斤量_2", "テン指数結果_2", "上がり指数結果_2", "ペース指数結果_2", "レースＰ指数結果_2",
                                                    "先行率_1", "先行率_2"]].add_prefix("上り_").rename(columns={"上り_RACE_KEY":"RACE_KEY"})

        self.base_df = pd.merge(feature_matrix, nige_df, on="RACE_KEY", how="left")
        self.base_df = pd.merge(self.base_df, agari_df, on="RACE_KEY", how="left")
        self.base_df = pd.merge(self.base_df, ninki_df, on="RACE_KEY")
        self.base_df = pd.merge(self.base_df, self.ld.race_df[["RACE_KEY", "NENGAPPI"]], on="RACE_KEY")

    def _drop_columns_base_df(self):
        pass
        #print(self.base_df.iloc[0])
        #self.base_df.drop(["発走時間"], axis=1, inplace=True)

    def proc_create_learning_data(self):
        self._proc_create_base_df()
        self._drop_unnecessary_columns()
        self._set_target_variables()
        # いらない？
        #base_df = self.base_df
        #obj_column = base_df.select_dtypes(include=['object']).columns.tolist()
        #obj_column = [s for s in obj_column if s not in ["RACE_KEY", "NENGAPPI"]]
        #base_df[obj_column] = base_df[obj_column].fillna(0).astype(int)
        learning_df = pd.merge(self.base_df, self.result_df, on ="RACE_KEY")
        return learning_df

    def _set_target_variables(self):
        self.ld.set_result_df()
        umaren_df = self._create_target_variable_umaren()
        umatan_df = self._create_target_variable_umatan()
        sanrenpuku_df = self._create_target_variable_sanrenpuku()
        self.result_df = pd.merge(umaren_df, umatan_df, on="競走コード")
        self.result_df = pd.merge(self.result_df, sanrenpuku_df, on="競走コード").rename(columns={"競走コード": "RACE_KEY"})

    def _create_target_variable_umaren(self):
        """  UMAREN_AREの目的変数を作成してresult_dfにセットする。配当が3000円をこえているものにフラグをセットする。 """
        umaren_df = self.ld.dict_haraimodoshi["umaren_df"]
        umaren_df.loc[:, "UMAREN_ARE"] = umaren_df["払戻"].apply(lambda x: 0 if x < 3000 else 1)
        return umaren_df[["RACE_KEY", "UMAREN_ARE"]]

    def _create_target_variable_umatan(self):
        """  UMATAN_AREの目的変数を作成してresult_dfにセットする。配当が5000円をこえているものにフラグをセットする。。 """
        umatan_df = self.ld.dict_haraimodoshi["umatan_df"]
        umatan_df.loc[:, "UMATAN_ARE"] = umatan_df["払戻"].apply(lambda x: 0 if x < 5000 else 1)
        return umatan_df[["RACE_KEY", "UMATAN_ARE"]]

    def _create_target_variable_sanrenpuku(self):
        """  SANRENPUKU_AREの目的変数を作成してresult_dfにセットする。配当が5000円をこえているものにフラグをセットする。 """
        sanrenpuku_df = self.ld.dict_haraimodoshi["sanrenpuku_df"]
        sanrenpuku_df.loc[:, "SANRENPUKU_ARE"] = sanrenpuku_df["払戻"].apply(lambda x: 0 if x < 5000 else 1)
        return sanrenpuku_df[["RACE_KEY", "SANRENPUKU_ARE"]]

    def load_learning_target_encoding(self):
        pass

    def _set_predict_target_encoding(self, df):
        return df

    def learning_race_lgb(self, this_model_name, target):
        # テスト用のデータを評価用と検証用に分ける
        X_eval, X_valid, y_eval, y_valid = train_test_split(self.X_test, self.y_test, random_state=42)

        # データセットを生成する
        lgb_train = lgb.Dataset(self.X_train, self.y_train)
        lgb_eval = lgb.Dataset(X_eval, y_eval, reference=lgb_train)

        # 上記のパラメータでモデルを学習する
        best_params, history = {}, []
        this_param = self.lgbm_params[target]
        model = lgb.train(this_param, lgb_train,valid_sets=lgb_eval,
                          verbose_eval=False,
                          num_boost_round=100,
                          early_stopping_rounds=5,
                          best_params=best_params,
                          tuning_history=history)
        print("Bset Paramss:", best_params)
        print('Tuning history:', history)

        self._save_learning_model(model, this_model_name)

    def _sub_create_pred_df(self, temp_df, y_pred):
        pred_df = pd.DataFrame(y_pred, columns=range(y_pred.shape[1]))
        base_df = pd.DataFrame({"RACE_KEY": temp_df["RACE_KEY"], "NENGAPPI": temp_df["NENGAPPI"]})
        pred_df = pd.concat([base_df, pred_df], axis=1)
        pred_df = pred_df.set_index(["RACE_KEY", "NENGAPPI"])
        pred_df = pred_df.stack().reset_index().rename(columns={"level_2": "CLASS", 0: "prob"})
        return pred_df


class SkModel(JRASkModel):
    obj_column_list = ['UMAREN_ARE', 'UMATAN_ARE', 'SANRENPUKU_ARE']

    def _get_skproc_object(self, version_str, start_date, end_date, model_name, mock_flag, test_flag):
        proc = SkProc(version_str, start_date, end_date, model_name, mock_flag, test_flag, self.obj_column_list)
        return proc


    def create_featrue_select_data(self, learning_df):
        pass


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
            start_date = '2010/01/01'
            end_date = '2018/12/31'
        if mock_flag:
            print("use mock data")
        print("MODE:learning mock_flag: " + str(args[2]) + "  start_date:" + start_date + " end_date:" + end_date)

        sk_model = SkModel(MODEL_NAME, MODEL_VERSION, start_date, end_date, mock_flag, test_flag, mode)

        luigi.build([End_baoz_learning(start_date=start_date, end_date=end_date, skmodel=sk_model, test_flag=test_flag,
                                       intermediate_folder=INTERMEDIATE_FOLDER)], local_scheduler=True)

    elif mode == "predict":
        if test_flag:
            print("Test mode")
            start_date = '2018/02/01'
            end_date = '2018/02/11'
        else:
            base_start_date = '2019/01/01'
            start_date = SkModel.get_recent_day(base_start_date)
            end_date = (dt.now() + timedelta(days=0)).strftime('%Y/%m/%d')
            if start_date > end_date:
                print("change start_date")
                start_date = end_date

        print("MODE:predict mock_flag:" + str(args[2]) + "  start_date:" + start_date + " end_date:" + end_date)

        sk_model = SkModel(MODEL_NAME, MODEL_VERSION, start_date, end_date, mock_flag, test_flag, mode)

        luigi.build([End_baoz_predict(start_date=start_date, end_date=end_date, skmodel=sk_model,test_flag=test_flag,
                                      intermediate_folder=INTERMEDIATE_FOLDER, export_mode=False)], local_scheduler=True)

    else:
        print("error")