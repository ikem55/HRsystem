from modules.base_sk_proc import BaseSkProc
import my_config as mc

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier

import pandas as pd
import sys

class BaseSkModel(object):
    """
    モデルに関する情報を定義する。モデル名、フォルダパス、目的変数等
    """
    version_str = 'base'
    """ モデルのバージョン名 """
    model_name = ''
    """ 学習モデルの名前（XGBoostとか）。init時の引数で定義される """
    model_path = ""
    """ モデルデータが格納される親フォルダ。 """
    class_list = ['競走種別コード', '場コード']
    """ 分類軸のリスト。このリスト毎に学習モデルを生成 """
    obj_column_list = ['WIN_FLAG', 'JIKU_FLAG', 'ANA_FLAG']
    """ 説明変数のリスト。このリストの説明変数毎に処理を実施する """
    ens_folder_path = ""
    """ モデルデータが格納される親フォルダ。 """
    dict_folder = ""
    """ 辞書フォルダのパス """
    index_list = ["RACE_KEY", "UMABAN", "NENGAPPI"]
    """ 対象データの主キー。ModeがRaceの場合はRACEにする """
    clfs = [RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
            ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
            GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50),
            KNeighborsClassifier(n_neighbors=10, n_jobs=-1),
            GaussianNB(),
            XGBClassifier(learning_rate=0.1, n_estimators=1000, max_depth=5, min_child_weight=1,
                          gamma=0, subsample=0.8, colsample_bytree=0.5, objective='binary:logistic',
                          scale_pos_weight=1, seed=0
                          )
            ]
    """ アンサンブル学習時に利用するクラス """
    learning_df = ""

    def __init__(self, model_name, version_str, start_date, end_date, mock_flag, test_flag, mode):
        self.model_name = model_name
        self.version_str = version_str
        self.start_date = start_date
        self.end_date = end_date
        self.dict_path = mc.return_base_path(test_flag)
        self._set_folder_path(mode)
        self.model_folder = self.model_path + model_name + '/'
        self.proc = self._get_skproc_object(version_str, start_date, end_date, model_name, mock_flag, test_flag)

    def _set_folder_path(self, mode):
        self.model_path = self.dict_path + 'model/' + self.version_str + '/'
        self.dict_folder = self.dict_path + 'dict/' + self.version_str + '/'
        self.ens_folder_path = self.dict_path + 'intermediate/' + self.version_str + '_' + mode + '/'

    def _get_skproc_object(self, version_str, start_date, end_date, model_name, mock_flag, test_flag):
        print("-- check! this is BaseSkModel class: " + sys._getframe().f_code.co_name)
        proc = BaseSkProc(version_str, start_date, end_date, model_name, mock_flag, test_flag, self.obj_column_list)
        return proc

    def create_learning_data(self):
        """ 学習用データを作成。処理はprocを呼び出す """
        self.learning_df = self.proc.proc_create_learning_data()

    def get_all_learning_df_for_save(self):
        save_learning_df = self.learning_df.drop(self.class_list, axis=1)
        return save_learning_df

    def get_val_list(self, df, cls_val):
        val_list = df[cls_val].drop_duplicates().astype(str)
        return val_list

    def get_filter_df(self, df, cls_val, val):
        if cls_val == "コース":
            query_str = cls_val + " == '" + str(val) + "'"
        else:
            query_str = cls_val + " == " + val
        print(query_str)
        filter_df = df.query(query_str)
        # 分類対象のデータを削除
        filter_df.drop(self.class_list, axis=1, inplace=True)
        return filter_df

    def create_featrue_select_data(self, learning_df):
        """  説明変数ごとに特徴量作成の処理（TargetEncodingとか）の処理を実施

        :param dataframe learning_df: dataframe
        """
        self.proc.proc_create_featrue_select_data(learning_df)

    def proc_learning_sk_model(self, df, cls_val, val):
        """  説明変数ごとに、指定された場所の学習を行う

        :param dataframe df: dataframe
        :param str basho: str
        """
        if not df.dropna().empty:
            if len(df.index) >= 30:
                print("----- アンサンブル学習用のクラスをセット -----")
                self.proc.set_ensemble_params(self.clfs, self.index_list, self.ens_folder_path)
                print("proc_learning_sk_model: df", df.shape)
                for target in self.obj_column_list:
                    print(target)
                    self.proc.learning_sk_model(df, cls_val, val, target)
            else:
                print("---- 少数レコードのため学習スキップ -- " + str(len(df.index)))
        else:
            print("---- NaNデータが含まれているため学習をスキップ")

    def create_predict_data(self):
        """ 予測用データを作成。処理はprocを呼び出す """
        predict_df = self.proc.proc_create_predict_data()
        return predict_df


    def proc_predict_sk_model(self, df, cls_val, val):
        """ predictする処理をまとめたもの。指定されたbashoのターゲットフラグ事の予測値を作成して連結したものをdataframeとして返す

        :param dataframe df: dataframe
        :param str val: str
        :return: dataframe
        """
        all_df = pd.DataFrame()
        if not df.empty:
            for target in self.obj_column_list:
                pred_df = self.proc._predict_sk_model(df, cls_val, val, target)
                if not pred_df.empty:
                    grouped_df = pred_df  #self._calc_grouped_data(pred_df)
                    grouped_df["target"] = target
                    grouped_df["target_date"] = pred_df["NENGAPPI"].dt.strftime('%Y/%m/%d')
                    grouped_df["model_name"] = self.model_name
                    all_df = pd.concat([all_df, grouped_df]).round(3)
        return all_df

    def create_import_data(self, all_df):
        """ データフレームをアンサンブル化（Vote）して格納 """
        all_df.dropna(inplace=True)
        grouped_all_df = all_df.groupby(["RACE_KEY", "UMABAN", "target"], as_index=False).mean()
        date_df = all_df[["RACE_KEY", "target_date"]].drop_duplicates()
        temp_grouped_df = pd.merge(grouped_all_df, date_df, on="RACE_KEY")
        grouped_df = self._calc_grouped_data(temp_grouped_df)
        import_df = grouped_df[["RACE_KEY", "UMABAN", "pred", "prob", "predict_std", "predict_rank", "target", "target_date"]].round(3)
        print(import_df)
        return import_df


    def eval_pred_data(self, df):
        """ 予測されたデータの精度をチェック """
        check_df = self.proc.create_eval_prd_data(df)
        for target in self.obj_column_list:
            print(target)
            target_df = check_df[check_df["target"] == target]
            target_df = target_df.query("predict_rank == 1")
            target_df.loc[:, "的中"] = target_df.apply(lambda x: 1 if x[target] == 1 else 0, axis=1)
            print(target_df)
            avg_rate = target_df["的中"].mean()
            print(round(avg_rate*100, 1))

    def import_data(self, df):
        print("-- check! this is BaseSkModel class: " + sys._getframe().f_code.co_name)

    @classmethod
    def get_recent_day(cls, start_date):
        print("-- check! this is BaseSkModel class: " + sys._getframe().f_code.co_name)

    def set_target_date(self, start_date, end_date):
        """ 学習等データ作成の対象期間をセットする

        :param str start_date: 開始日（文字列）
        :param str end_date: 終了日（文字列）
        """
        self.start_date = start_date
        self.end_date = end_date

    def set_test_table(self, table_name):
        """ test用のテーブルをセットする """
        self.table_name = table_name


    def _calc_grouped_data(self, df):
        """ 与えられたdataframe(予測値）に対して偏差化とランク化を行ったdataframeを返す

        :param dataframe df: dataframe
        :return: dataframe
        """
        grouped = df.groupby(["RACE_KEY", "target"])
        grouped_df = grouped.describe()['prob'].reset_index()
        merge_df = pd.merge(df, grouped_df, on=["RACE_KEY", "target"])
        merge_df['predict_std'] = (
            merge_df['prob'] - merge_df['mean']) / merge_df['std'] * 10 + 50
        df['predict_rank'] = grouped['prob'].rank("dense", ascending=False)
        merge_df = pd.merge(merge_df, df[["RACE_KEY", "UMABAN", "predict_rank", "target"]], on=["RACE_KEY", "UMABAN", "target"])
        return_df = merge_df[['RACE_KEY', 'UMABAN',
                              'pred', 'prob', 'predict_std', 'predict_rank', "target", "target_date"]]
        return return_df