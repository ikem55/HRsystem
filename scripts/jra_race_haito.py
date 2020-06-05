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

    def set_result_df(self):
        """ result_dfを作成するための処理。result_dfに処理がされたデータをセットする """
        print("set_result_df")
        haraimodoshi_df = self.ext.get_haraimodoshi_table_base()
        self.dict_haraimodoshi = self.ext.get_haraimodoshi_dict(haraimodoshi_df)

class SkProc(JRASkProc):
    """
    地方競馬の機械学習処理プロセスを取りまとめたクラス。
    """
    index_list = ["RACE_KEY", "target_date"]
    # LightGBM のハイパーパラメータ
    obj_column_list = ['UMAREN_ARE', 'UMATAN_ARE', 'SANRENPUKU_ARE']
    lgbm_params = {
        'UMAREN_ARE':{'objective': 'binary'},
        'UMATAN_ARE':{'objective': 'binary'},
        'SANRENPUKU_ARE':{'objective': 'binary'},
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
        nige_df = self.base_df.query("展開記号=='1'")[["RACE_KEY", "脚質", "距離適性", "上昇度", "激走指数", "蹄コード", "見習い区分", "枠番", "総合印", "ＩＤＭ印", "基準人気順位", "輸送区分", "激走タイプ", "休養理由分類コード", "芝ダ障害フラグ",
                                                    "距離フラグ", "クラスフラグ", "転厩フラグ", "去勢フラグ", "乗替フラグ", "IDM", "騎手指数", "テン指数", "ペース指数", "上がり指数", "位置指数", "追切指数", "仕上指数",
                                                    "斤量_1", "テン指数結果_1", "上がり指数結果_1", "ペース指数結果_1", "レースＰ指数結果_1", "斤量_2", "テン指数結果_2", "上がり指数結果_2", "ペース指数結果_2", "レースＰ指数結果_2",
                                                    "先行率_1", "先行率_2"]].add_prefix("逃げ_").rename(columns={"逃げ_RACE_KEY":"RACE_KEY"})
        # 上がり最速予想馬のデータを取得
        agari_df = self.base_df.query("展開記号=='2'")[["RACE_KEY", "脚質", "距離適性", "上昇度", "激走指数", "蹄コード", "見習い区分", "枠番", "総合印", "ＩＤＭ印", "基準人気順位", "輸送区分", "激走タイプ", "休養理由分類コード", "芝ダ障害フラグ",
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
        learning_df = pd.merge(self.base_df, self.result_df, on ="RACE_KEY")
        return learning_df

    def _drop_unnecessary_columns(self):
        """ predictに不要な列を削除してpredict_dfを作成する。削除する列は血統登録番号、確定着順、タイム指数、単勝オッズ、単勝人気  """
        self.base_df.drop(["NENGAPPI", "COURSE_KEY", "発走時間"], axis=1, inplace=True)

    def _set_target_variables(self):
        self.ld.set_result_df()
        umaren_df = self._create_target_variable_umaren()
        umatan_df = self._create_target_variable_umatan()
        sanrenpuku_df = self._create_target_variable_sanrenpuku()
        self.result_df = pd.merge(umaren_df, umatan_df, on="RACE_KEY")
        self.result_df = pd.merge(self.result_df, sanrenpuku_df, on="RACE_KEY")

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

class SkModel(JRASkModel):
    obj_column_list = ['UMAREN_ARE', 'UMATAN_ARE', 'SANRENPUKU_ARE']

    def _get_skproc_object(self, version_str, start_date, end_date, model_name, mock_flag, test_flag):
        proc = SkProc(version_str, start_date, end_date, model_name, mock_flag, test_flag, self.obj_column_list)
        return proc

    def create_featrue_select_data(self, learning_df):
        pass


    def proc_predict_sk_model(self, df):
        """ predictする処理をまとめたもの。指定されたbashoのターゲットフラグ事の予測値を作成して連結したものをdataframeとして返す

        :param dataframe df: dataframe
        :param str val: str
        :return: dataframe
        """
        all_df = pd.DataFrame()
        if not df.empty:
            for target in self.obj_column_list:
                pred_df = self.proc._predict_sk_model(df, target)
                print(pred_df.shape)
                if not pred_df.empty:
                    pred_df["target"] = target
                    pred_df["model_name"] = self.model_name
                    date_list = sorted(pred_df["target_date"].drop_duplicates().tolist())
                    for date in date_list:
                        target_df = pred_df[pred_df["target_date"] == date]
                        target_df = target_df.sort_values(["RACE_KEY", "target"])
                        target_df.to_pickle(self.pred_folder + target + "_" + date + ".pickle")
                    all_df = pd.concat([all_df, pred_df]).round(3)
        return all_df


    def _eval_check_df(self, result_df, target_df, target):
        temp_df = result_df[["RACE_KEY", target]].rename(columns={target: "result"})
        check_df = pd.merge(target_df, temp_df, on="RACE_KEY")
        check_df.loc[:, "的中"] = check_df["result"].apply(lambda x: 1 if x == 1 else 0)
        return check_df


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
            pred_folder = dict_path + 'pred/' + MODEL_VERSION
            start_date = SkModel.get_recent_day(base_start_date, pred_folder)
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