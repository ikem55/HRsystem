from modules.base_sk_proc import BaseSkProc
from modules.lb_load import LBLoad
import modules.util as mu

import pandas as pd
import sys

class LBSkProc(BaseSkProc):
    """
    地方競馬の機械学習処理プロセスを取りまとめたクラス。
    """
    dict_folder = './for_test_dict/lb/'
    model_path = './for_test_model/lb/'
    index_list = ["RACE_KEY", "UMABAN", "NENGAPPI"]

    def _get_load_object(self, version_str, start_date, end_date, mock_flag, test_flag):
        print("-- check! this is LBSkProc class: " + sys._getframe().f_code.co_name)
        ld = LBLoad(version_str, start_date, end_date, mock_flag, test_flag)
        return ld

    def _drop_unnecessary_columns(self):
        """ predictに不要な列を削除してpredict_dfを作成する。削除する列は血統登録番号、確定着順、タイム指数、単勝オッズ、単勝人気  """
        self.base_df.drop(['血統登録番号'], axis=1, inplace=True)

    def _merge_df(self):
        print("-- check! this is LBSkProc class: " + sys._getframe().f_code.co_name)
        self.base_df = pd.merge(self.ld.race_df, self.ld.raceuma_df, on="競走コード")
        self.base_df = pd.merge(self.base_df, self.ld.horse_df, on="血統登録番号")
        self.base_df = pd.merge(self.base_df, self.ld.prev1_raceuma_df, on=["競走コード", "馬番"], how='left')
        self.base_df = pd.merge(self.base_df, self.ld.prev2_raceuma_df, on=["競走コード", "馬番"], how='left')
        self.base_df = pd.merge(self.base_df, self.ld.grouped_raceuma_prev_df, on=["競走コード", "馬番"], how='left')

    def _create_feature(self):
        """ マージしたデータから特徴量を生成する """
        print("-- check! this is LBSkProc class: " + sys._getframe().f_code.co_name)
        self.base_df.loc[:, "継続騎乗"] = (self.base_df["騎手名"] == self.base_df["騎手名_1"]).astype(int)
        self.base_df.loc[:, "同場騎手"] = (self.base_df["騎手所属場コード"] == self.base_df["場コード"]).astype(int)
        self.base_df.loc[:, "同所属場"] = (self.base_df["調教師所属場コード"] == self.base_df["場コード"]).astype(int)
        self.base_df.loc[:, "同所属騎手"] = (self.base_df["騎手所属場コード"] == self.base_df["調教師所属場コード"]).astype(int)
        self.base_df.loc[:, "同主催者"] = (self.base_df["主催者コード"] == self.base_df["主催者コード_1"]).astype(int)
        self.base_df.loc[:, "同場コード"] = (self.base_df["場コード"] == self.base_df["場コード_1"]).astype(int)
        self.base_df.loc[:, "同根幹"] = (self.base_df["非根幹"] == self.base_df["非根幹_1"]).astype(int)
        self.base_df.loc[:, "同距離グループ"] = (self.base_df["距離グループ"] == self.base_df["距離グループ_1"]).astype(int)
        self.base_df.loc[:, "前走位置取り変化"] = self.base_df["コーナー順位4_2"] -self. base_df["コーナー順位4_1"]
        self.base_df.loc[:, "休み明け"] = self.base_df["休養週数"].apply(lambda x: True if x >= 20 else False)
        self.base_df.loc[:, "前走凡走"] = self.base_df.apply(lambda x: 1 if (x["単勝人気_1"] < 0.3 and x["確定着順_1"] > 0.3) else 0, axis=1)
        self.base_df.loc[:, "前走激走"] = self.base_df.apply(lambda x: 1 if (x["単勝人気_1"] > 0.5 and x["確定着順_1"] < 0.3) else 0, axis=1)
        self.base_df.loc[:, "前走逃げそびれ"] = self.base_df.apply(lambda x: 1 if (x["予想展開"] == 1 and x["コーナー順位4_1"] > 0.5) else 0, axis=1)
        self.base_df.drop(
            ["非根幹_1", "非根幹_2", "主催者コード_1", "主催者コード_2", "場コード_1", "場コード_2", "距離グループ_1", "距離グループ_2"],
            axis=1)

    def _drop_columns_base_df(self):
        print("-- check! this is LBSkProc class: " + sys._getframe().f_code.co_name)
        print(self.base_df.iloc[0])
        self.base_df.drop(["場名", "発走時刻", "登録頭数", "回次", "日次", "データ作成年月日", "年月日", "近走競走コード1",
                           "近走馬番1", "近走競走コード2", "近走馬番2", "近走競走コード3", "近走馬番3", "近走競走コード4", "近走馬番4", "近走競走コード5", "近走馬番5",
                        "調教師所属場コード_1", "調教師所属場コード_2"], axis=1, inplace=True)

    def _scale_df(self):
        print("-- check! this is LBSkProc class: " + sys._getframe().f_code.co_name)
        mmsc_columns = ["距離", "競走番号", "頭数", "初出走頭数", "枠番", "予想タイム指数順位", "休養後出走回数", "予想人気", "先行指数順位", "馬齢", "距離増減"
            , "前走着順", "前走人気", "前走頭数", "騎手ランキング", "調教師ランキング", "得点V1順位", "得点V2順位", "デフォルト得点順位", "得点V3順位"]
        mmsc_dict_name = "sc_base_mmsc"
        stdsc_columns = ["予想勝ち指数", "予想決着指数", "休養週数", "予想オッズ", "血統距離評価", "血統トラック評価", "血統成長力評価", "先行指数"
            , "血統総合評価", "血統距離評価B", "血統トラック評価B", "血統成長力評価B", "血統総合評価B", "騎手評価", "調教師評価", "枠順評価", "脚質評価", "キャリア", "前走着差"
            , "前走馬体重", "タイム指数上昇係数", "タイム指数回帰推定値", "タイム指数回帰標準偏差", "斤量比", "前走休養週数", "負担重量", "予想タイム指数", "デフォルト得点", "得点V1"
            , "得点V2", "得点V3", "fa_1_1", "fa_2_1", "fa_3_1", "fa_1_2", "fa_2_2", "fa_3_2", "同場_max", "同場_mean", "同距離_max", "同距離_mean"
            , "同根幹_max", "同根幹_mean", "同距離グループ_max", "同距離グループ_mean", "同馬番グループ_max", "同馬番グループ_mean"]
        stdsc_dict_name = "sc_base_stdsc"
        self.base_df = mu.scale_df_for_fa(self.base_df, mmsc_columns, mmsc_dict_name, stdsc_columns, stdsc_dict_name, self.dict_folder)
        oh_columns = ["月", "グレードコード", "クラス変動", "前走トラック種別コード", "月_1", "クラス変動_1", "月_2", "クラス変動_2"]
        oh_dict_name = "sc_base_hash_month"
        self.base_df = mu.hash_eoncoding(self.base_df, oh_columns, 20, oh_dict_name, self.dict_folder)
        self.base_df.loc[:, "競走種別コード_h"] = self.base_df["競走種別コード"]
        self.base_df.loc[:, "場コード_h"] = self.base_df["場コード"]
        hash_track_columns = ["トラック種別コード", "主催者コード", "競走種別コード_h", "場コード_h", "競走条件コード", "トラックコード", "混合"]
        hash_track_dict_name = "sc_base_hash_track"
        self.base_df = mu.hash_eoncoding(self.base_df, hash_track_columns, 10, hash_track_dict_name, self.dict_folder)
        hash_kishu_columns = ["騎手コード", "予想展開", "馬番グループ", "騎手所属場コード", "見習区分", "テン乗り"]
        hash_kishu_dict_name = "sc_base_hash_kishu"
        self.base_df = mu.hash_eoncoding(self.base_df, hash_kishu_columns, 30, hash_kishu_dict_name, self.dict_folder)
        hash_chokyoshi_columns = ["調教師コード", "調教師所属場コード", "所属", "転厩", "東西所属コード"]
        hash_chokyoshi_dict_name = "sc_base_hash_chokyoshi"
        self.base_df = mu.hash_eoncoding(self.base_df, hash_chokyoshi_columns,30, hash_chokyoshi_dict_name, self.dict_folder)
        hash_horse_columns = ["生産者コード", "繁殖登録番号1", "繁殖登録番号5", "馬主コード", "性別コード"]
        hash_horse_dict_name = "sc_base_hash_horse"
        self.base_df = mu.hash_eoncoding(self.base_df, hash_horse_columns,30, hash_horse_dict_name, self.dict_folder)
        hash_prev1_columns = ["騎手名_1", "ペース_1", "馬場状態コード_1", "競走種別コード_1", "展開コード_1", "騎手所属場コード_1", "テン乗り_1"]
        hash_prev1_dict_name = "sc_base_hash_prev1"
        self.base_df = mu.hash_eoncoding(self.base_df, hash_prev1_columns, 10, hash_prev1_dict_name, self.dict_folder)
        hash_prev2_columns = ["騎手名_2", "ペース_2", "馬場状態コード_2", "競走種別コード_2", "展開コード_2", "騎手所属場コード_2", "テン乗り_2"]
        hash_prev2_dict_name = "sc_base_hash_prev2"
        self.base_df = mu.hash_eoncoding(self.base_df, hash_prev2_columns, 10, hash_prev2_dict_name, self.dict_folder)

    def _rename_key(self, df):
        """ キー名を競走コード→RACE_KEY、馬番→UMABANに変更 """
        return_df = df.rename(columns={"競走コード": "RACE_KEY", "馬番": "UMABAN", "月日": "NENGAPPI"})
        return return_df

    def _create_target_variable_win(self):
        """  WIN_FLAGの目的変数を作成してbase_dfにセットする。条件は確定着順＝１  """
        self.result_df['WIN_FLAG'] = self.result_df['確定着順'].apply(lambda x: 1 if x == 1 else 0)

    def _create_target_variable_jiku(self):
        """  JIKU_FLAGの目的変数を作成してbase_dfにセットする。条件は確定着順＝１，２ """
        self.result_df['JIKU_FLAG'] = self.result_df['確定着順'].apply(lambda x: 1 if x in (1, 2) else 0)

    def _create_target_variable_ana(self):
        """ ANA_FLAGの目的変数を作成してbase_dfにセットする、条件は確定着順＜＝３かつ単勝オッズ10倍以上  """
        self.result_df['ANA_FLAG'] = self.result_df.apply(lambda x: 1 if x['確定着順'] in (1, 2, 3) and x['単勝オッズ'] >= 10 else 0, axis=1)

    def _drop_columns_result_df(self):
        self.result_df.drop(["確定着順", "単勝オッズ"], axis=1, inplace=True)

