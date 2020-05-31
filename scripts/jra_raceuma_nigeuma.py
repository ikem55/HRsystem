from modules.jra_extract import JRAExtract
from modules.jra_transform import JRATransform
from modules.jra_load import JRALoad
from modules.jra_sk_model import JRASkModel
from modules.jra_sk_proc import JRASkProc
import my_config as mc
import modules.util as mu

import luigi
from modules.jra_task_learning import End_baoz_learning
from modules.jra_task_predict import End_baoz_predict

from datetime import datetime as dt
from datetime import timedelta
import sys
import pandas as pd
import numpy as np
import pickle
import os
from distutils.util import strtobool



# 呼び出し方
# python jra_raceuma_nigeuma.py learning True True
# ====================================== パラメータ　要変更 =====================================================
# 逃げ馬、上がり最速馬を予測する（レース馬単位)

MODEL_VERSION = 'jra_ru_nigeuma'
MODEL_NAME = 'raceuma_lgm'

# ====================================== クラス　要変更 =========================================================

class Ext(JRAExtract):
    pass

class Tf(JRATransform):
    def drop_columns_raceuma_result_df(self, raceuma_df):
        """ 過去レースで不要な項目を削除する """
        raceuma_df = raceuma_df.drop(["NENGAPPI", "馬名", "レース名", "レース名略称", "騎手名", "調教師名", "1(2)着馬名", "パドックコメント", "脚元コメント",
                                      "素点", "馬場差", "ペース", "出遅", "位置取", "不利", "前不利", "中不利", "後不利", "レース", "コース取り", "上昇度コード",
                                      "クラスコード", "馬体コード", "気配コード", "確定複勝オッズ下", "10時単勝オッズ", "10時複勝オッズ",
                                      "天候コード", "本賞金", "収得賞金", "レース馬コメント", "KAISAI_KEY", "ハロンタイム０１", "ハロンタイム０２", "ハロンタイム０３",
                                      "ハロンタイム０４", "ハロンタイム０５", "ハロンタイム０６", "ハロンタイム０７", "ハロンタイム０８", "ハロンタイム０９", "ハロンタイム１０",
                                      "ハロンタイム１１", "ハロンタイム１２", "ハロンタイム１３", "ハロンタイム１４", "ハロンタイム１５", "ハロンタイム１６", "ハロンタイム１７",
                                      "ハロンタイム１８", "１コーナー", "２コーナー", "３コーナー", "４コーナー", "１角１", "１角２", "１角３", "２角１", "２角２", "２角３",
                                      "向正１", "向正２", "向正３", "３角１", "３角２", "３角３", "４角０", "４角１", "４角２", "４角３", "４角４", "直線０", "直線１",
                                      "直線２", "直線３", "直線４", "１着算入賞金", "１ハロン平均_mean", "ＩＤＭ結果_mean", "テン指数結果_mean", "上がり指数結果_mean",
                                      "ペース指数結果_mean", "前３Ｆタイム_mean", "後３Ｆタイム_mean", "コーナー順位１_mean", "コーナー順位２_mean", "コーナー順位３_mean",
                                      "コーナー順位４_mean", "前３Ｆ先頭差_mean", "後３Ｆ先頭差_mean", "追走力_mean", "追上力_mean", "後傾指数_mean", "１ハロン平均_std",
                                      "上がり指数結果_std", "ペース指数結果_std", "COURSE_KEY",
                                      "馬具(その他)コメント", "レースコメント", "異常区分", "血統登録番号", "単勝", "複勝", "馬体重増減", "KYOSO_RESULT_KEY"], axis=1)
        return raceuma_df


class Ld(JRALoad):
    def _get_extract_object(self, start_date, end_date, mock_flag):
        """ 利用するExtクラスを指定する """
        ext = Ext(start_date, end_date, mock_flag)
        return ext

    def _get_transform_object(self, start_date, end_date):
        """ 利用するTransformクラスを指定する """
        tf = Tf(start_date, end_date)
        return tf

    def _proc_prev_df(self, raceuma_5_prev_df):
        """  prev_dfを作成するための処理。prev1_raceuma_df,prev2_raceuma_dfに処理がされたデータをセットする。過去２走のデータと過去走を集計したデータをセットする  """
        raceuma_5_prev_df = self.tf.cluster_course_df(raceuma_5_prev_df, self.dict_path)
        raceuma_5_prev_df = self.tf.cluster_raceuma_result_df(raceuma_5_prev_df, self.dict_path)
        raceuma_5_prev_df = self.tf.factory_analyze_race_result_df(raceuma_5_prev_df, self.dict_path)
        self.prev5_raceuma_df = self._get_prev_df(5, raceuma_5_prev_df, "")
        self.prev5_raceuma_df.rename(columns=lambda x: x + "_5", inplace=True)
        self.prev5_raceuma_df.rename(columns={"RACE_KEY_5": "RACE_KEY", "UMABAN_5": "UMABAN", "target_date_5": "target_date"}, inplace=True)
        self.prev4_raceuma_df = self._get_prev_df(4, raceuma_5_prev_df, "")
        self.prev4_raceuma_df.rename(columns=lambda x: x + "_4", inplace=True)
        self.prev4_raceuma_df.rename(columns={"RACE_KEY_4": "RACE_KEY", "UMABAN_4": "UMABAN", "target_date_4": "target_date"}, inplace=True)
        self.prev3_raceuma_df = self._get_prev_df(3, raceuma_5_prev_df, "")
        self.prev3_raceuma_df.rename(columns=lambda x: x + "_3", inplace=True)
        self.prev3_raceuma_df.rename(columns={"RACE_KEY_3": "RACE_KEY", "UMABAN_3": "UMABAN", "target_date_3": "target_date"}, inplace=True)
        self.prev2_raceuma_df = self._get_prev_df(2, raceuma_5_prev_df, "")
        self.prev2_raceuma_df.rename(columns=lambda x: x + "_2", inplace=True)
        self.prev2_raceuma_df.rename(columns={"RACE_KEY_2": "RACE_KEY", "UMABAN_2": "UMABAN", "target_date_2": "target_date"}, inplace=True)
        self.prev1_raceuma_df = self._get_prev_df(1, raceuma_5_prev_df, "")
        self.prev1_raceuma_df.rename(columns=lambda x: x + "_1", inplace=True)
        self.prev1_raceuma_df.rename(columns={"RACE_KEY_1": "RACE_KEY", "UMABAN_1": "UMABAN", "target_date_1": "target_date"}, inplace=True)
        self.prev_feature_raceuma_df = self._get_prev_feature_df(raceuma_5_prev_df)

    def _get_prev_feature_df(self, raceuma_5_prev_df):
        max_columns = ['血統登録番号', 'target_date', 'fa_1', 'fa_2', 'fa_3', 'fa_4', 'fa_5', 'ＩＤＭ結果', 'テン指数結果', '上がり指数結果', 'ペース指数結果']
        min_columns = ['血統登録番号', 'target_date', 'fa_4', 'テン指数結果順位', '上がり指数結果順位', 'ペース指数結果順位']
        max_score_df = raceuma_5_prev_df[max_columns].groupby(['血統登録番号', 'target_date']).max().add_prefix("max_").reset_index()
        min_score_df = raceuma_5_prev_df[min_columns].groupby(['血統登録番号', 'target_date']).min().add_prefix("min_").reset_index()
        feature_df = pd.merge(max_score_df, min_score_df, on=["血統登録番号", "target_date"])
        race_df = self.race_df[["RACE_KEY", "course_cluster"]].copy()
        raceuma_df = self.raceuma_df[["RACE_KEY", "UMABAN", "血統登録番号", "target_date"]].copy()
        raceuma_df = pd.merge(race_df, raceuma_df, on="RACE_KEY")
        filtered_df = pd.merge(raceuma_df, raceuma_5_prev_df.drop(["RACE_KEY", "UMABAN"], axis=1), on=["血統登録番号", "target_date", "course_cluster"])[["RACE_KEY", "UMABAN", "ru_cluster"]]
        filtered_df_c1 = filtered_df.query("ru_cluster == '1'").groupby(["RACE_KEY", "UMABAN"]).count().reset_index()
        filtered_df_c1.columns = ["RACE_KEY", "UMABAN", "c1_cnt"]
        filtered_df_c2 = filtered_df.query("ru_cluster == '2'").groupby(["RACE_KEY", "UMABAN"]).count().reset_index()
        filtered_df_c2.columns = ["RACE_KEY", "UMABAN", "c2_cnt"]
        filtered_df_c3 = filtered_df.query("ru_cluster == '3'").groupby(["RACE_KEY", "UMABAN"]).count().reset_index()
        filtered_df_c3.columns = ["RACE_KEY", "UMABAN", "c3_cnt"]
        filtered_df_c4 = filtered_df.query("ru_cluster == '4'").groupby(["RACE_KEY", "UMABAN"]).count().reset_index()
        filtered_df_c4.columns = ["RACE_KEY", "UMABAN", "c4_cnt"]
        filtered_df_c7 = filtered_df.query("ru_cluster == '7'").groupby(["RACE_KEY", "UMABAN"]).count().reset_index()
        filtered_df_c7.columns = ["RACE_KEY", "UMABAN", "c7_cnt"]
        raceuma_df = pd.merge(raceuma_df, filtered_df_c1, on=["RACE_KEY", "UMABAN"], how="left")
        raceuma_df = pd.merge(raceuma_df, filtered_df_c2, on=["RACE_KEY", "UMABAN"], how="left")
        raceuma_df = pd.merge(raceuma_df, filtered_df_c3, on=["RACE_KEY", "UMABAN"], how="left")
        raceuma_df = pd.merge(raceuma_df, filtered_df_c4, on=["RACE_KEY", "UMABAN"], how="left")
        raceuma_df = pd.merge(raceuma_df, filtered_df_c7, on=["RACE_KEY", "UMABAN"], how="left")
        raceuma_df = raceuma_df.fillna(0)
        raceuma_df = pd.merge(raceuma_df, feature_df, on=["血統登録番号", "target_date"], how="left").drop(["course_cluster", "血統登録番号", "target_date"], axis=1)
        print(raceuma_df.head(30))
        return raceuma_df

    def set_result_df(self):
        """ result_dfを作成するための処理。result_dfに処理がされたデータをセットする """
        return self.ext.get_raceuma_table_base()[["RACE_KEY", "UMABAN", "レース脚質", "上がり指数結果順位", "テン指数結果順位"]]

class SkProc(JRASkProc):
    """
    地方競馬の機械学習処理プロセスを取りまとめたクラス。
    """
    index_list = ["RACE_KEY", "UMABAN", "target_date"]
    # LightGBM のハイパーパラメータ
    obj_column_list = ['NIGEUMA', 'AGARI_SAISOKU', 'TEN_SAISOKU']
    lgbm_params = {
        'NIGEUMA':{'objective': 'binary'},
        'AGARI_SAISOKU':{'objective': 'binary'},
        'TEN_SAISOKU':{'objective': 'binary'},
                   }

    def _get_load_object(self, version_str, start_date, end_date, mock_flag, test_flag):
        ld = Ld(version_str, start_date, end_date, mock_flag, test_flag)
        return ld

    def _merge_df(self):
        self.base_df = pd.merge(self.ld.race_df, self.ld.raceuma_df, on=["RACE_KEY", "target_date", "NENGAPPI"])
        self.base_df = pd.merge(self.base_df, self.ld.horse_df, on=["血統登録番号", "target_date"])
        self.base_df = pd.merge(self.base_df, self.ld.prev1_raceuma_df, on=["RACE_KEY", "UMABAN", "target_date"], how='left')
        self.base_df = pd.merge(self.base_df, self.ld.prev2_raceuma_df, on=["RACE_KEY", "UMABAN", "target_date"], how='left')
        self.base_df = pd.merge(self.base_df, self.ld.prev3_raceuma_df, on=["RACE_KEY", "UMABAN", "target_date"], how='left')
        self.base_df = pd.merge(self.base_df, self.ld.prev4_raceuma_df, on=["RACE_KEY", "UMABAN", "target_date"], how='left')
        self.base_df = pd.merge(self.base_df, self.ld.prev5_raceuma_df, on=["RACE_KEY", "UMABAN", "target_date"], how='left')
        self.base_df = pd.merge(self.base_df, self.ld.prev_feature_raceuma_df, on=["RACE_KEY", "UMABAN"], how='left')

    def _create_feature(self):
        """ マージしたデータから特徴量を生成する """
        self.base_df.loc[:, "継続騎乗"] = (self.base_df["騎手コード"] == self.base_df["騎手コード_1"]).astype(int)
        self.base_df.loc[:, "距離増減"] = self.base_df["距離"] - self.base_df["距離_1"]
        self.base_df.loc[:, "同根幹"] = (self.base_df["非根幹"] == self.base_df["非根幹_1"]).astype(int)
        self.base_df.loc[:, "同距離グループ"] = (self.base_df["距離グループ"] == self.base_df["距離グループ_1"]).astype(int)
        self.base_df.loc[:, "前走凡走"] = self.base_df.apply(lambda x: 1 if (x["人気率_1"] < 0.3 and x["着順率_1"] > 0.5) else 0, axis=1)
        self.base_df.loc[:, "前走激走"] = self.base_df.apply(lambda x: 1 if (x["人気率_1"] > 0.5 and x["着順率_1"] < 0.3) else 0, axis=1)
        self.base_df.loc[:, "前走逃げそびれ"] = self.base_df.apply(lambda x: 1 if (x["展開記号"] == '1' and x["先行率_1"] > 0.5) else 0, axis=1)
        self.base_df.drop(
            ["非根幹_1", "非根幹_2", "距離グループ_1", "距離グループ_2"],
            axis=1)

    def _drop_columns_base_df(self):
        self.base_df.drop(["場名", "ZENSO1_KYOSO_RESULT", "ZENSO2_KYOSO_RESULT", "ZENSO3_KYOSO_RESULT", "ZENSO4_KYOSO_RESULT", "ZENSO5_KYOSO_RESULT",
                           "ZENSO1_RACE_KEY", "ZENSO2_RACE_KEY", "ZENSO3_RACE_KEY", "ZENSO4_RACE_KEY", "ZENSO5_RACE_KEY", "発走時間", "COURSE_KEY",
                           "血統登録番号", "NENGAPPI", "参考前走", "登録抹消フラグ", "入厩何日前", "入厩年月日", "転圧_1", "凍結防止剤_1",

                           "距離_2", "芝ダ障害コード_2", "右左_2", "内外_2", "馬場状態_2", "種別_2", "条件_2", "記号_2", "重量_2", "グレード_2", "頭数_2", "着順_2",
                           "タイム_2", "確定単勝オッズ_2", "確定単勝人気順位_2", "レースペース_2", "馬ペース_2", "コーナー順位１_2", "コーナー順位２_2", "コーナー順位３_2",
                           "コーナー順位４_2", "コース_2", "レースペース流れ_2", "馬ペース流れ_2", "４角コース取り_2", "IDM_2", "ペースアップ位置_2", "ラスト５ハロン_2",
                           "ラスト４ハロン_2", "ラスト３ハロン_2", "ラスト２ハロン_2", "ラップ差４ハロン_2", "ラップ差３ハロン_2", "ラップ差２ハロン_2", "ラップ差１ハロン_2",
                           "連続何日目_2", "芝種類_2", "草丈_2", "転圧_2", "凍結防止剤_2", "中間降水量_2", "ハロン数_2",

                           "距離_3", "芝ダ障害コード_3", "右左_3", "内外_3", "馬場状態_3", "種別_3", "条件_3", "記号_3", "重量_3", "グレード_3", "頭数_3", "着順_3",
                           "タイム_3", "確定単勝オッズ_3", "確定単勝人気順位_3", "レースペース_3", "馬ペース_3", "コーナー順位１_3", "コーナー順位２_3", "コーナー順位３_3",
                           "コーナー順位４_3", "コース_3", "レースペース流れ_3", "馬ペース流れ_3", "４角コース取り_3", "IDM_3", "ペースアップ位置_3", "ラスト５ハロン_3",
                           "ラスト４ハロン_3", "ラスト３ハロン_3", "ラスト２ハロン_3", "ラップ差４ハロン_3", "ラップ差３ハロン_3", "ラップ差２ハロン_3", "ラップ差１ハロン_3",
                           "連続何日目_3", "芝種類_3", "草丈_3", "転圧_3", "凍結防止剤_3", "中間降水量_3", "ハロン数_3",

                           "距離_4", "芝ダ障害コード_4", "右左_4", "内外_4", "馬場状態_4", "種別_4", "条件_4", "記号_4", "重量_4", "グレード_4", "頭数_4", "着順_4",
                           "タイム_4", "確定単勝オッズ_4", "確定単勝人気順位_4", "レースペース_4", "馬ペース_4", "コーナー順位１_4", "コーナー順位２_4", "コーナー順位３_4",
                           "コーナー順位４_4", "コース_4", "レースペース流れ_4", "馬ペース流れ_4", "４角コース取り_4", "IDM_4", "ペースアップ位置_4", "ラスト５ハロン_4",
                           "ラスト４ハロン_4", "ラスト３ハロン_4", "ラスト２ハロン_4", "ラップ差４ハロン_4", "ラップ差３ハロン_4", "ラップ差２ハロン_4", "ラップ差１ハロン_4",
                           "連続何日目_4", "芝種類_4", "草丈_4", "転圧_4", "凍結防止剤_4", "中間降水量_4", "ハロン数_4",

                           "距離_5", "芝ダ障害コード_5", "右左_5", "内外_5", "馬場状態_5", "種別_5", "条件_5", "記号_5", "重量_5", "グレード_5", "頭数_5", "着順_5",
                           "タイム_5", "確定単勝オッズ_5", "確定単勝人気順位_5", "レースペース_5", "馬ペース_5", "コーナー順位１_5", "コーナー順位２_5", "コーナー順位３_5",
                           "コーナー順位４_5", "コース_5", "レースペース流れ_5", "馬ペース流れ_5", "４角コース取り_5", "IDM_5", "ペースアップ位置_5", "ラスト５ハロン_5",
                           "ラスト４ハロン_5", "ラスト３ハロン_5", "ラスト２ハロン_5", "ラップ差４ハロン_5", "ラップ差３ハロン_5", "ラップ差２ハロン_5", "ラップ差１ハロン_5",
                           "連続何日目_5", "芝種類_5", "草丈_5", "転圧_5", "凍結防止剤_5", "中間降水量_5", "ハロン数_5"
                           ], axis=1, inplace=True)

    def _set_label_list(self, df):
        """ label_listの値にわたされたdataframeのデータ型がobjectのカラムのリストをセットする。TargetEncodingを行わないカラムを除く

        :param dataframe df: dataframe
        """
        label_list = df.select_dtypes(include=object).columns.tolist()
        except_list = ["距離", "芝ダ障害コード", "右左", "内外", "種別", "記号", "重量", "グレード", "コース", "開催区分", "曜日", "天候コード", "芝馬場状態コード", "芝馬場状態内", "芝馬場状態中", "芝馬場状態外",
                       "直線馬場差最内", "直線馬場差内", "直線馬場差中", "直線馬場差外", "直線馬場差大外", "ダ馬場状態コード", "ダ馬場状態内", "ダ馬場状態中", "ダ馬場状態外", "芝種類", "転圧", "凍結防止剤", "場コード",
                       "距離グループ", "UMABAN", "RACE_KEY", "target_date", "グレード_1", "コース_1", "ペースアップ位置_1", "レースペース流れ_1", "右左_1", "参考前走騎手コード", "芝種類_1", "種別_1", "重量_1",
                       "条件クラス"]
        self.label_list = [i for i in label_list if i not in except_list]

    def _set_target_variables(self):
        self.result_df = self.ld.set_result_df()
        self._create_target_variable_nigeuma()
        self._create_target_variable_agari_saisoku()
        self._create_target_variable_ten_saisoku()
        self.result_df.drop(["レース脚質", "上がり指数結果順位", "テン指数結果順位"], axis=1, inplace=True)

    def _create_target_variable_nigeuma(self):
        """  ラップタイプをフラグとしてセットする """
        self.result_df['NIGEUMA'] = self.result_df['レース脚質'].apply(lambda x: 1 if x == '1' else 0)

    def _create_target_variable_agari_saisoku(self):
        """  トラックバイアスをフラグとしてセットする """
        self.result_df['AGARI_SAISOKU'] = self.result_df['上がり指数結果順位'].apply(lambda x: 1 if x == 1 else 0)

    def _create_target_variable_ten_saisoku(self):
        """  トラックバイアスをフラグとしてセットする """
        self.result_df['TEN_SAISOKU'] = self.result_df['テン指数結果順位'].apply(lambda x: 1 if x == 1 else 0)

    def _sub_create_pred_df(self, temp_df, y_pred):
        pred_df = pd.DataFrame(
            {"RACE_KEY": temp_df["RACE_KEY"], "UMABAN": temp_df["UMABAN"], "target_date": temp_df["target_date"],
             "prob": y_pred})
        pred_df = self._calc_grouped_data(pred_df)
        #pred_df.loc[:, "pred"] = pred_df.apply(lambda x: 1 if x["prob"] >= 0.5 else 0, axis=1)
        return pred_df

class SkModel(JRASkModel):
    obj_column_list = ['NIGEUMA', 'AGARI_SAISOKU', 'TEN_SAISOKU']

    def _get_skproc_object(self, version_str, start_date, end_date, model_name, mock_flag, test_flag):
        proc = SkProc(version_str, start_date, end_date, model_name, mock_flag, test_flag, self.obj_column_list)
        return proc


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

    pd.set_option('display.max_rows', 300)

    if mode == "learning":
        if test_flag:
            print("Test mode")
            start_date = '2014/01/01'
            end_date = '2014/01/31'
        else:
            start_date = '2012/01/01'
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
            start_date = '2019/01/01'
            end_date = '2019/01/31'
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


        luigi.build([End_baoz_predict(start_date=start_date, end_date=end_date, skmodel=sk_model, test_flag=test_flag,
                                      intermediate_folder=INTERMEDIATE_FOLDER, export_mode=False)], local_scheduler=True)

    else:
        print("error")