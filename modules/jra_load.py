from modules.base_load import BaseLoad
from modules.jra_extract import JRAExtract
from modules.jra_transform import JRATransform as Tf
import modules.util as mu

import sys
import pandas as pd

class JRALoad(BaseLoad):
    """
    地方競馬のデータロードクラス
    """

    def _get_extract_object(self, start_date, end_date, mock_flag):
        """ 利用するExtクラスを指定する """
        ext = JRAExtract(start_date, end_date, mock_flag)
        return ext

    def _get_transform_object(self, start_date, end_date):
        """ 利用するTransformクラスを指定する """
        tf = Tf(start_date, end_date)
        return tf

    def set_race_base_df(self, race_base_df):
        self.race_base_df = race_base_df

    def set_raceuma_base_df(self, raceuma_base_df):
        self.raceuma_base_df = raceuma_base_df

    def set_horse_base_df(self, horse_base_df):
        self.horse_base_df = horse_base_df

    def set_raceuma_5_prev_df(self, raceuma_5_prev_df):
        self.raceuma_5_prev_df = raceuma_5_prev_df

    def set_race_df(self):
        """  race_dfを作成するための処理。race_dfに処理がされたデータをセットする """
        self.race_base_df = self.ext.get_race_before_table_base()
        print(self.race_base_df.shape)
        self.race_df = self._proc_race_df(self.race_base_df)
        print("set_race_df: race_df", self.race_df.shape)

    def _proc_race_df(self, race_base_df):
        race_df = self.tf.cluster_course_df(race_base_df, self.dict_path)
        race_df = self.tf.drop_columns_race_df(race_df)
        race_df = self.tf.encode_race_before_df(race_df, self.dict_folder)
        race_df = self.tf.create_feature_race_df(race_df)
        return race_df

    def set_raceuma_df(self):
        """ raceuma_dfを作成するための処理。raceuma_dfに処理がされたデータをセットする """
        self.raceuma_base_df = self.ext.get_raceuma_before_table_base()
        self.raceuma_df = self._proc_raceuma_df(self.raceuma_base_df)
        print("set_raceuma_df: raceuma_df", self.raceuma_df.shape)

    def set_horse_df(self):
        """  horse_dfを作成するための処理。horse_dfに処理がされたデータをセットする """
        self.horse_base_df = self.ext.get_horse_table_base()
        self.horse_df = self._proc_horse_df(self.horse_base_df)
        print("set_horse_df: horse_df", self.horse_df.shape)

    def _proc_horse_df(self, horse_base_df):
        horse_df = self.tf.drop_columns_horse_df(horse_base_df)
        horse_df = self.tf.encode_horse_df(horse_df, self.dict_folder)
        return horse_df.copy()

    def set_prev_df(self):
        """  prev_dfを作成するための処理。prev1_raceuma_df,prev2_raceuma_dfに処理がされたデータをセットする。過去２走のデータと過去走を集計したデータをセットする  """
        self.raceuma_5_prev_df = self.ext.get_raceuma_zenso_table_base()
        self._proc_prev_df(self.raceuma_5_prev_df)
        print("set_prev_df: prev1_raceuma_df", self.prev1_raceuma_df.shape)
        print("set_prev_df: prev_feature_raceuma_df", self.prev_feature_raceuma_df.shape)

    def _proc_prev_df(self, raceuma_5_prev_df):
        raceuma_5_prev_df = self.tf.cluster_course_df(raceuma_5_prev_df, self.dict_path)
        raceuma_5_prev_df = self.tf.cluster_raceuma_result_df(raceuma_5_prev_df, self.dict_path)
        raceuma_5_prev_df = self.tf.factory_analyze_race_result_df(raceuma_5_prev_df, self.dict_path)
        self.prev2_raceuma_df = self._get_prev_df(2, raceuma_5_prev_df, "")
        self.prev2_raceuma_df.rename(columns=lambda x: x + "_2", inplace=True)
        self.prev2_raceuma_df.rename(columns={"RACE_KEY_2": "RACE_KEY", "UMABAN_2": "UMABAN", "target_date_2": "target_date"}, inplace=True)
        self.prev1_raceuma_df = self._get_prev_df(1, raceuma_5_prev_df, "")
        self.prev1_raceuma_df.rename(columns=lambda x: x + "_1", inplace=True)
        self.prev1_raceuma_df.rename(columns={"RACE_KEY_1": "RACE_KEY", "UMABAN_1": "UMABAN", "target_date_1": "target_date"}, inplace=True)
        self.prev_feature_raceuma_df = self._get_prev_feature_df(raceuma_5_prev_df)

    def _get_prev_df(self, num, raceuma_5_prev_df, empty_df):
        """ numで指定した過去走のデータを取得して、過去走として返す。変数を合わせるためempty_dfを用意するが使わない

        :param int num: number(過去１走前の場合は1)
        """
        prev_race_key = f"ZENSO{num}_KYOSO_RESULT"
        raceuma_base_df = self.raceuma_df[["RACE_KEY", "UMABAN", "target_date", prev_race_key]].rename(columns={prev_race_key: "KYOSO_RESULT_KEY"})
        this_raceuma_df = pd.merge(raceuma_base_df, raceuma_5_prev_df.drop(["RACE_KEY", "UMABAN"], axis=1), on=["KYOSO_RESULT_KEY", "target_date"])
        this_raceuma_df = self.tf.encode_raceuma_result_df(this_raceuma_df, self.dict_folder)
        this_raceuma_df = self.tf.normalize_raceuma_result_df(this_raceuma_df)
        this_raceuma_df = self.tf.create_feature_raceuma_result_df(this_raceuma_df)
        this_raceuma_df = self.tf.drop_columns_raceuma_result_df(this_raceuma_df)
        print("_proc_raceuma_result_df: raceuma_df", this_raceuma_df.shape)

        return this_raceuma_df

    def _get_prev_feature_df(self, raceuma_5_prev_df):
        """ 馬券術を参考にフラグを作る """
        return pd.DataFrame()

    def set_result_df(self):
        """ 目的変数作成用のresult_dfを作成するための処理。result_dfに処理がされたデータをセットする """
        result_race_df = self.ext.get_race_table_base()
        result_raceuma_df = self.ext.get_raceuma_table_base()
        self.result_race_df, self.result_raceuma_df = self._proc_result_df(result_race_df, result_raceuma_df)

    def set_prepred_df(self, phase):
        # リークしそうなので使わない。。
        """ 事前に予測したデータを取得する。ＲＡＰＴＹＰＥ→レース逃げ馬→レース馬逃げ馬→レース馬印→配当の順 """
        if phase ==0:
            return pd.DataFrame()
        # RAPTYPE: ['RAP_TYPE', 'TRACK_BIAS_ZENGO', 'TRACK_BIAS_UCHISOTO', 'PRED_PACE']
        raptype_df = self.ext.get_pred_df("jra_rc_raptype", "RAP_TYPE").pivot_table(values=['prob'], index=['RACE_KEY'], columns=['CLASS']).reset_index()
        raptype_df.columns = ["RACE_KEY", "RT_一貫", "RT_L4加", "RT_L3加", "RT_L2加", "RT_L1加", "RT_L4失", "RT_L3失", "RT_L2失", "RT_L1失", "RT_他"]
        track_bias_zengo_df = self.ext.get_pred_df("jra_rc_raptype", "TRACK_BIAS_ZENGO").pivot_table(values=['prob'], index=['RACE_KEY'], columns=['CLASS']).reset_index()
        track_bias_zengo_df.columns = ["RACE_KEY", "TB_超前", "TB_前", "TB_ZG", "TB_後", "TB_超後"]
        track_bias_uchiso_df = self.ext.get_pred_df("jra_rc_raptype", "TRACK_BIAS_UCHISOTO").pivot_table(values=['prob'], index=['RACE_KEY'], columns=['CLASS']).reset_index()
        track_bias_uchiso_df.columns = ["RACE_KEY", "TB_超内", "TB_内", "TB_US", "TB_外", "TB_超外"]
        pred_pace_df = self.ext.get_pred_df("jra_rc_raptype", "PRED_PACE").pivot_table(values=['prob'], index=['RACE_KEY'], columns=['CLASS']).reset_index()
        pred_pace_df.columns = ["RACE_KEY", "Pなし", "P後半型", "PU早持続", "PU早失速", "P上速", "P平均", "Pテン飛失速", "P中弛み", "Pテン飛", "P前飛"]
        prepred_df = pd.merge(raptype_df, track_bias_zengo_df, on="RACE_KEY", how="left")
        prepred_df = pd.merge(prepred_df, track_bias_uchiso_df, on="RACE_KEY", how="left")
        prepred_df = pd.merge(prepred_df, pred_pace_df, on="RACE_KEY", how="left")
        if phase == 1:
            return prepred_df
        # レース逃げ馬: ['R_NIGEUMA', 'R_AGARI_SAISOKU', 'R_TEN_SAISOKU']
        r_nigeuma_df = self.ext.get_pred_df("jra_rc_nigeuma", "R_NIGEUMA")[["RACE_KEY", "UMABAN", "prob"]].rename(columns={"prob": "R_NIGEUMA"})
        r_agari_saisoku = self.ext.get_pred_df("jra_rc_nigeuma", "R_AGARI_SAISOKU")[["RACE_KEY", "UMABAN", "prob"]].rename(columns={"prob": "R_AGARI_SAISOKU"})
        r_ten_saisoku = self.ext.get_pred_df("jra_rc_nigeuma", "R_TEN_SAISOKU")[["RACE_KEY", "UMABAN", "prob"]].rename(columns={"prob": "R_TEN_SAISOKU"})
        prepred_df = pd.merge(prepred_df, r_nigeuma_df, on="RACE_KEY", how="left")
        prepred_df = pd.merge(prepred_df, r_agari_saisoku, on=["RACE_KEY", "UMABAN"], how="left")
        prepred_df = pd.merge(prepred_df, r_ten_saisoku, on=["RACE_KEY", "UMABAN"], how="left")
        if phase == 2:
            return prepred_df
        # レース馬逃げ馬：['NIGEUMA', 'AGARI_SAISOKU', 'TEN_SAISOKU']
        nigeuma_df = self.ext.get_pred_df("jra_ru_nigeuma", "NIGEUMA")[["RACE_KEY", "UMABAN", "predict_std"]].rename(columns={"predict_std": "NIGEUMA"})
        agari_saisoku_df = self.ext.get_pred_df("jra_ru_nigeuma", "AGARI_SAISOKU")[["RACE_KEY", "UMABAN", "predict_std"]].rename(columns={"predict_std": "AGARI_SAISOKU"})
        ten_saisoku_df = self.ext.get_pred_df("jra_ru_nigeuma", "TEN_SAISOKU")[["RACE_KEY", "UMABAN", "predict_std"]].rename(columns={"predict_std": "TEN_SAISOKU"})
        prepred_df = pd.merge(prepred_df, nigeuma_df, on=["RACE_KEY", "UMABAN"], how="left")
        prepred_df = pd.merge(prepred_df, agari_saisoku_df, on=["RACE_KEY", "UMABAN"], how="left")
        prepred_df = pd.merge(prepred_df, ten_saisoku_df, on=["RACE_KEY", "UMABAN"], how="left")
        if phase == 2:
            return prepred_df
        # レース馬印: ['WIN_FLAG', 'JIKU_FLAG', 'ANA_FLAG']
        win_flag_df = self.ext.get_pred_df("jra_ru_mark", "WIN_FLAG")[["RACE_KEY", "UMABAN", "predict_std"]].rename(columns={"predict_std": "WIN_FLAG"})
        jiku_flag_df = self.ext.get_pred_df("jra_ru_mark", "JIKU_FLAG")[["RACE_KEY", "UMABAN", "predict_std"]].rename(columns={"predict_std": "JIKU_FLAG"})
        ana_flag_df = self.ext.get_pred_df("jra_ru_mark", "ANA_FLAG")[["RACE_KEY", "UMABAN", "predict_std"]].rename(columns={"predict_std": "ANA_FLAG"})
        prepred_df = pd.merge(prepred_df, win_flag_df, on=["RACE_KEY", "UMABAN"], how="left")
        prepred_df = pd.merge(prepred_df, jiku_flag_df, on=["RACE_KEY", "UMABAN"], how="left")
        prepred_df = pd.merge(prepred_df, ana_flag_df, on=["RACE_KEY", "UMABAN"], how="left")
        self.prepred_df = prepred_df