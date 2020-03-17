from modules.base_load import BaseLoad
import modules.util as mu

import pandas as pd
import os
import numpy as np
import pickle
import json
import sys
import category_encoders as ce

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import BorderlineSMOTE
from xgboost import XGBClassifier

from sklearn import model_selection
from sklearn.metrics import log_loss, accuracy_score

class BaseSkProc(object):
    """
    機械学習を行うのに必要なプロセスを定義する。learning/predictデータの作成プロセスや
    """
    mock_flag = False
    """ mockデータを利用する場合はTrueにする """
    dict_folder = './for_test_dict/base/'
    """ 辞書フォルダのパス """
    model_path = './for_test_model/base/'
    """ モデルデータが格納される親フォルダ。set_model_pathメソッドを使うことで変更可能。 """
    ens_folder_path = ""
    """ モデルデータが格納される親フォルダ。。SkModelクラスから値をセットする """
    index_list = ["RACE_KEY", "UMABAN", "NENGAPPI"]
    """ 対象データの主キー。。SkModelクラスから値をセットする """
    clfs = []
    """ アンサンブル学習用のクラスセット。SkModelクラスから値をセットする """
    base_df = ""
    result_df = ""
    learning_df = ""
    obj_column_list = ""
    target_flag = ""
    x_df = ""
    y_df = ""
    X_train = ""
    X_test = ""
    y_train = ""
    y_test = ""
    label_list = ""
    folder_path = ""

    def __init__(self, version_str, start_date, end_date, model_name, mock_flag, test_flag, obj_column_list):
        print(__class__.__name__)
        self.start_date = start_date
        self.end_date = end_date
        self.model_name = model_name
        self._set_folder_path(version_str, test_flag)
        self.model_folder = self.model_path + model_name + '/'
        self.ld = self._get_load_object(version_str, start_date, end_date, mock_flag, version_str)
        self.mock_flag = mock_flag
        self.obj_column_list = obj_column_list

    def _set_folder_path(self, version_str, test_flag):
        if test_flag:
            dict_path = 'C:\HRsystem/HRsystem/for_test_'
        else:
            dict_path = 'E:\python/'
        self.dict_folder = dict_path + 'dict/' + version_str + '/'
        self.model_path = dict_path + 'model/' + version_str + '/'
        self.ens_folder_path = dict_path + 'model/' + version_str + '/'

    def _get_load_object(self, version_str, start_date, end_date, mock_flag, test_flag):
        print("-- check! this is BaseSkProc class: " + sys._getframe().f_code.co_name)
        ld = BaseLoad(version_str,start_date, end_date, mock_flag, test_flag)
        return ld

    def proc_create_learning_data(self):
        self._proc_create_base_df()
        self._drop_unnecessary_columns()
        self._set_target_variables()
        learning_df = pd.merge(self.base_df, self.result_df, on =["RACE_KEY","UMABAN"])
        return learning_df

    def proc_create_predict_data(self):
        self._proc_create_base_df()
        print("nullレコード削除：" + str(self.base_df.isnull().any().sum()))
        self.base_df.dropna(how='any', axis=0, inplace=True)
        self._drop_unnecessary_columns()
        return self.base_df

    def _drop_unnecessary_columns(self):
        print("-- check! this is BaseSkProc class: " + sys._getframe().f_code.co_name)

    def _proc_create_base_df(self):
        self._set_ld_data()
        self._merge_df()
        self._create_feature()
        self._drop_columns_base_df()
        self._scale_df()
        self.base_df = self._rename_key(self.base_df)

    def _set_ld_data(self):
        """  Loadオブジェクトにデータをセットする処理をまとめたもの。Race,Raceuma,Horse,Prevのデータフレームをセットする

        :param object ld: データロードオブジェクト(ex.LocalBaozLoad)
        """
        self.ld.set_race_df()  # データ取得
        self.ld.set_raceuma_df()
        self.ld.set_horse_df()
        self.ld.set_prev_df()

    def _merge_df(self):
        print("-- check! this is BaseSkProc class: " + sys._getframe().f_code.co_name)

    def _create_feature(self):
        print("-- check! this is BaseSkProc class: " + sys._getframe().f_code.co_name)

    def _drop_columns_base_df(self):
        print("-- check! this is BaseSkProc class: " + sys._getframe().f_code.co_name)

    def _scale_df(self):
        print("-- check! this is BaseSkProc class: " + sys._getframe().f_code.co_name)

    def _rename_key(self, df):
        print("-- check! this is BaseSkProc class: " + sys._getframe().f_code.co_name)

    def _set_target_variables(self):
        self.ld.set_result_df()
        self.result_df = self.ld.result_df
        self._create_target_variable_win()
        self._create_target_variable_jiku()
        self._create_target_variable_ana()
        self._drop_columns_result_df()
        self.result_df = self._rename_key(self.result_df)

    def _create_target_variable_win(self):
        print("-- check! this is BaseSkProc class: " + sys._getframe().f_code.co_name)

    def _create_target_variable_jiku(self):
        print("-- check! this is BaseSkProc class: " + sys._getframe().f_code.co_name)

    def _create_target_variable_ana(self):
        print("-- check! this is BaseSkProc class: " + sys._getframe().f_code.co_name)

    def _drop_columns_result_df(self):
        print("-- check! this is BaseSkProc class: " + sys._getframe().f_code.co_name)

    def proc_create_featrue_select_data(self, learning_df):
        self.learning_df = learning_df
        for target_flag in self.obj_column_list:
            print(target_flag)
            self._set_target_flag(target_flag)
            self._create_feature_select_data(target_flag)

    def _set_target_flag(self, target_flag):
        """ 目的変数となるターゲットフラグの値をセットする

        :param str target_flag: ターゲットフラグ名(WIN_FLAG etc.)
        """
        self.target_flag = target_flag

    def _create_feature_select_data(self, target_flag):
        """  指定した説明変数に対しての特徴量作成の処理を行う。TargetEncodingや、Borutaによる特徴選択（DeepLearning以外）を行う

        :param str target_flag:
        """
        print("create_feature_select_data")
        self._set_learning_data(self.learning_df, target_flag)
        self._divide_learning_data()
        self._create_learning_target_encoding()

    def _set_learning_data(self, df, target_column):
        """ 与えられたdfからexp_data,obj_dataを作成する。目的変数が複数ある場合は除外する

        :param dataframe df: dataframe
        :param str target_column: 目的変数のカラム名
        """
        df = df.drop(self.index_list, axis=1)
        self.y_df = df[target_column].copy()
        print(self.obj_column_list)
        self.x_df = df.drop(self.obj_column_list, axis=1).copy()
        print(self.x_df.shape)
        print(self.y_df.shape)
        self._set_label_list(self.x_df)


    def _set_label_list(self, df):
        """ label_listの値にわたされたdataframeのデータ型がobjectのカラムのリストをセットする

        :param dataframe df: dataframe
        """
        self.label_list = df.select_dtypes(include=object).columns.tolist()

    def _divide_learning_data(self):
        """ 学習データをtrainとtestに分ける。オブジェクトのx_df,y_dfからX_train,X_test,y_train,y_testに分けてセットする """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.x_df, self.y_df, test_size=0.25, random_state=None)
        print("obj data check: y_train")
        print(self.y_train.value_counts())
        print("obj data check: y_test")
        print(self.y_test.value_counts())

    def _create_learning_target_encoding(self):
        """ TargetEncodeの計算を行い、計算結果のエンコード値をセットする """
        self.X_train = self._set_target_encoding(self.X_train, True, self.y_train).copy()
        self.X_test = self._set_target_encoding(self.X_test, False, self.y_test).copy()

    def _set_target_encoding(self, x_df, fit, y_df):
        """ TargetEncodingの処理をコントロールする処理。label_list（dataframeでデータ型がobjectとなったカラムのリスト）に対してtarget_encodingの処理を行う。
        fitの値がTrueの場合はエンコーダーを作り、Falseの場合は作成されたエンコーダーを適用し、エンコードした値のdataframeを返す。

        :param dataframe x_df: dataframe
        :param bool fit: bool
        :param dataframe y_df: dataframe
        :return: dataframe
        """
        for label in self.label_list:
            x_df.loc[:, label] = self._target_encoding(x_df[label], label, self.target_flag + '_tr_' + label, fit, y_df)
        return x_df

    def _target_encoding(self, sr, label, dict_name, fit, y):
        """ srに対してtarget encodingした結果を返す。fitがTrueの場合はエンコーディングする。ただし、すでに辞書がある場合はそれを使う。

        :param series sr: エンコード対象のSeries
        :param str label: エンコードするラベル名
        :param str dict_name: エンコード辞書名
        :param bool fit: エンコード実施 or 実施済み辞書から変換
        :param series y: 目的変数のSeries
        :return:
        """
        tr = ce.TargetEncoder(cols=label)
        dict_file = self.dict_folder + '/' + dict_name + '.pkl'
        if fit and not os.path.exists(dict_file):
            tr = tr.fit(sr, y)
            mu.save_dict(tr, dict_name, self.dict_folder)
        else:
            tr = mu.load_dict(dict_name, self.dict_folder)
        sr_tr = tr.transform(sr)
        return sr_tr

    def set_ensemble_params(self, clfs, index_list, ens_folder_path):
        self.clfs = clfs
        self.index_list = index_list
        self.ens_folder_path = ens_folder_path

    def learning_sk_model(self, df, cls_val, val, target):
        """ 指定された場所・ターゲットに対しての学習処理を行う

        :param dataframe df: dataframe
        :param str val: str
        :param str target: str
        """
        this_model_name = self.model_name + "_" + cls_val + '_' + val + '_' + target
        self._set_target_flag(target)
        df = df.fillna(df.median())
        df = df.dropna() #SMOTEでNaNがあると処理できないため
        print(df.shape)
        if df.empty:
            print("--------- alert !!! no data")
        else:
            self._set_learning_data(df, target)
            self._divide_learning_data()
            self._load_learning_target_encoding()
            self._set_smote_data()
            self._learning_raceuma_ens(this_model_name)

    def _load_learning_target_encoding(self):
        """ TargetEncodeを行う。すでに作成済みのエンコーダーから値をセットする  """
        self.X_train = self._set_target_encoding(self.X_train, False, self.y_train).copy()
        self.X_test = self._set_target_encoding(self.X_test, False, self.y_test).copy()

    def _set_smote_data(self):
        """ 学習データのSMOTE処理を行い学習データを更新する  """
        # 対象数が少ない場合はサンプリングレートを下げる
        if self.y_train[self.y_train == 1].shape[0] >= 6:
            smote = BorderlineSMOTE()
            self.X_train, self.y_train = smote.fit_sample(self.X_train, self.y_train)
        else:
            ros = RandomOverSampler(
                ratio={1: self.X_train.shape[0], 0: self.X_train.shape[0] // 3}, random_state=71)
            # 学習用データに反映
            self.X_train, self.y_train = ros.fit_sample(self.X_train, self.y_train)
        print("-- after sampling: " + str(np.unique(self.y_train, return_counts=True)))


    def _learning_raceuma_ens(self, this_model_name):
        print("this_model_name: " + this_model_name)
        self.folder_path = self.ens_folder_path + self.model_name + '/'
        self._transform_test_df_to_np()
        self._train_1st_layer(this_model_name)

        first_train, first_test = self._read_npy('first', this_model_name)
        self._train_2nd_layer(first_train, first_test, this_model_name)

        second_train, second_test = self._read_npy('second', this_model_name)
        self._train_3rd_layer(second_train, second_test, this_model_name)

    def _predict_raceuma_ens(self, this_model_name, temp_df):
        print("this_model_name: " + this_model_name)

        temp_df = temp_df.replace(np.inf,np.nan).fillna(temp_df.replace(np.inf,np.nan).mean())
        exp_df = temp_df.drop(self.index_list, axis=1).to_numpy()
        print(exp_df.shape)
        first_np = self._calc_pred_layer('first', this_model_name, exp_df)
        if len(first_np) != 0:
            second_np = self._calc_pred_layer('second', this_model_name, first_np)
            y = self._calc_pred_layer('third', this_model_name, second_np)
            print("check!!!")
            print(self.index_list)
            if self.index_list == ["RACE_KEY", "NENGAPPI"]:
                print("check2")
                pred_df = pd.DataFrame({"RACE_KEY": temp_df["RACE_KEY"],"NENGAPPI": temp_df["NENGAPPI"], "prob": y[:, 1]})
            else:
                pred_df = pd.DataFrame({"RACE_KEY" : temp_df["RACE_KEY"], "UMABAN": temp_df["UMABAN"],"NENGAPPI": temp_df["NENGAPPI"], "prob": y[:, 1]})
            pred_df.loc[:, "pred"] = pred_df.apply(lambda x: 1 if x["prob"] >= 0.5 else 0, axis=1)
            return pred_df
        else:
            return pd.DataFrame()

    def read_pred_npy(self, layer, this_model_name):
        tr_p = self.folder_path + layer + '/'
        tr_list = [s for s in os.listdir(tr_p) if str(self.end_date).replace('/', '') + '_' + this_model_name in s]
        train_file_names = map(lambda x: tr_p + x, tr_list)
        print(tr_p)
        print(tr_list)

        list_train = []
        for path_train in train_file_names:
            frame_train = np.load(path_train)
            list_train.append(frame_train)
        print(len(list_train))
        if len(list_train) != 0:
            l_train = list_train[0]
            for train_ in list_train[1:]:
                l_train = np.concatenate([l_train, train_], axis=1)
            return l_train
        else:
            return []

    def _calc_pred_layer(self, layer, this_model_name, exp_df):
        model_p = self.model_path + self.model_name + '/' + layer + '/'
        model_list = [s for s in os.listdir(model_p) if this_model_name in s]
        np_list = []
        print(model_p)
        for file in model_list:
            print(file)
            with open(model_p + file, 'rb') as f:
                clf = pickle.load(f)
            temp_y = clf.predict_proba(exp_df)
            np_list.append(temp_y)
        print(np_list)
        if len(np_list) != 0:
            l_train = np_list[0]
            for train_ in np_list[1:]:
                l_train = np.concatenate([l_train, train_], axis=1)
            return l_train
        else:
            return []

    def _transform_train_df_to_np(self):
        self.X_train = self.X_train.to_numpy()
        self.y_train = self.y_train.to_numpy()

    def _transform_test_df_to_np(self):
        self.X_test = self.X_test.to_numpy()
        self.y_test = self.y_test.to_numpy()

    def _train_1st_layer(self, this_model_name):
        print("---------------- start train_1st_layer ------------------")
        for clf in self.clfs:
            clf_name = "first/" + str(clf)[1:3] + '_' + this_model_name
            print(clf_name)
            folder_path = self.folder_path + 'first/'
            save_clf = self._blend_proba(clf, X_train=self.X_train, y=self.y_train, X_test=self.X_test, save_preds="1", n_splits=3, folder_path=folder_path, this_model_name = this_model_name)
            self._save_learning_model(save_clf, clf_name)


    def _train_2nd_layer(self, first_train, first_test, this_model_name):
        print("---------------- start train_2nd_layer ------------------")
        for clf in self.clfs:
            clf_name = "second/" + str(clf)[1:3] + '_' + this_model_name
            print(clf_name)
            folder_path = self.folder_path + 'second/'
            save_clf = self._blend_proba(clf, X_train=first_train, y=self.y_train, X_test=first_test, save_preds="2", n_splits=3, folder_path=folder_path, this_model_name = this_model_name)
            self._save_learning_model(save_clf, clf_name)

    def _train_3rd_layer(self, second_train, second_test, this_model_name):
        print("---------------- start train_3rd_layer ------------------")
        clf = XGBClassifier(learning_rate=0.1, n_estimators=1000, max_depth=5, min_child_weight=1,
                            gamma=0, subsample=0.8, colsample_bytree=0.5, objective='binary:logistic',
                            scale_pos_weight=1, seed=0
                            )

        clf_name = "third/" + this_model_name
        folder_path = self.folder_path + 'third/'
        save_clf = self._blend_proba(clf, X_train=second_train, y=self.y_train, X_test=second_test, save_test_only="3", n_splits=3, folder_path=folder_path, this_model_name = this_model_name)
        self._save_learning_model(save_clf, clf_name)

    def _save_learning_model(self, model, model_name):
        """ 渡されたmodelをmodel_nameで保存する。

        :param object model: 学習モデル
        :param str model_name: モデル名
        """
        with open(self.model_folder + model_name + '.pickle', 'wb') as f:
            pickle.dump(model, f)

    def _read_npy(self, layer, this_model_name):
        tr_p = self.folder_path + layer + '/train/'
        te_p = self.folder_path + layer + '/test/'
        tr_list = [s for s in os.listdir(tr_p) if this_model_name in s]
        te_list = [s for s in os.listdir(te_p) if this_model_name in s]
        train_file_names = map(lambda x: tr_p + x, tr_list)
        test_file_names = map(lambda x: te_p + x,te_list)

        list_train, list_test = [], []
        for path_train, path_test in zip(train_file_names, test_file_names):
            frame_train, frame_test = np.load(path_train), np.load(path_test)
            list_train.append(frame_train)
            list_test.append(frame_test)
        l_train, l_test = list_train[0], list_test[0]
        for train_, test_ in zip(list_train[1:], list_test[1:]):
            l_train = np.concatenate([l_train, train_], axis=1)
            l_test = np.concatenate([l_test, test_], axis=1)
        return l_train, l_test

    def _blend_proba(self, clf, X_train, y, X_test, n_splits=5, save_preds="",
                     save_test_only="", seed=300373, save_params="",
                     clf_name="XX", generalizers_params=[], minimal_loss=0,
                     return_score=False, minimizer="log_loss", folder_path='', this_model_name=''):
        folds = list(model_selection.StratifiedKFold(n_splits,shuffle=True,random_state=seed).split(X_train, y))
        dataset_blend_train = np.zeros((X_train.shape[0],np.unique(y).shape[0]))

        loss = 0

        for i, (train_index, test_index) in enumerate( folds ):
            fold_X_train = X_train[train_index]
            fold_y_train = y[train_index]
            fold_X_test = X_train[test_index]
            fold_y_test = y[test_index]
            clf.fit(fold_X_train, fold_y_train)

            fold_preds = clf.predict_proba(fold_X_test)
            dataset_blend_train[test_index] = fold_preds
            if minimizer == "log_loss":
                loss += log_loss(fold_y_test,fold_preds)
            if minimizer == "accuracy":
                fold_preds_a = np.argmax(fold_preds, axis=1)
                loss += accuracy_score(fold_y_test,fold_preds_a)

            if minimal_loss > 0 and loss > minimal_loss and i == 0:
                return False, False
            fold_preds = np.argmax(fold_preds, axis=1)
        avg_loss = loss / float(i+1)
        clf.fit(X_train, y)
        dataset_blend_test = clf.predict_proba(X_test)

        if clf_name == "XX":
            clf_name = str(clf)[1:3]

        if len(save_preds)>0:
            id = ''
            np.save(folder_path + "train/{}_{}{}_{}_{}_train.npy".format(this_model_name, save_preds,clf_name,avg_loss,id),dataset_blend_train)
            np.save(folder_path + "test/{}_{}{}_{}_{}_test.npy".format(this_model_name, save_preds,clf_name,avg_loss,id),dataset_blend_test)

        if len(save_test_only)>0:
            id = ''
            dataset_blend_test = clf.predict(X_test)
            np.savetxt(folder_path + "test/{}_{}{}_{}_{}_test.txt".format(this_model_name, save_test_only,clf_name,avg_loss,id),dataset_blend_test)
            d = {}
            d["stacker"] = clf.get_params()
            d["generalizers"] = generalizers_params
            with open(folder_path + "param/{}_{}{}_{}_{}_params.json".format(this_model_name, save_test_only,clf_name,avg_loss, id), 'w',) as f:
                json.dump(d, f)

        if len(save_params)>0:
            id = ''
            d = {}
            d["name"] = clf_name
            d["params"] = { k:(v.get_params() if "\n" in str(v) or "<" in str(v) else v) for k,v in clf.get_params().items()}
            d["generalizers"] = generalizers_params
            with open(folder_path + "param/{}_{}{}_{}_{}_params.json".format(this_model_name, save_params,clf_name,avg_loss, id), 'wb') as f:
                json.dump(d, f)

        return clf

    def _predict_sk_model(self, df, cls_val, val, target):
        """ 指定された場所・ターゲットに対しての予測データの作成を行う

        :param dataframe  df: dataframe
        :param str val: str
        :param str target: str
        :return: dataframe
        """
        self._set_target_flag(target)
        temp_df = self._set_predict_target_encoding(df)
        # temp_df.drop("NENGAPPI", axis=1, inplace=True)
        date_df = df[["RACE_KEY", "NENGAPPI"]].drop_duplicates()
        pred_df = self._sub_distribute_predict_model(cls_val, val, target, temp_df)
        if pred_df.empty:
            return pd.DataFrame()
        else:
            return pd.merge(pred_df, date_df, on="RACE_KEY")


    def _sub_distribute_predict_model(self, cls_val, val, target, temp_df):
        """ model_nameに応じたモデルを呼び出し予測を行う

        :param str val: 場所名
        :param str target: 目的変数名
        :param dataframe temp_df: 予測するデータフレーム
        :return dataframe: pred_df
        """
        this_model_name = self.model_name + "_" + cls_val + '_' + val + '_' + target
        pred_df = self._predict_raceuma_ens(this_model_name, temp_df)
        return pred_df

    def _set_predict_target_encoding(self, df):
        """ 渡されたdataframeに対してTargetEncodeを行いエンコードした値をセットしたdataframeを返す

        :param dataframe df: dataframe
        :return: dataframe
        """
        self._set_label_list(df)
        df_temp = self._set_target_encoding(df, False, "").copy()
        return df_temp

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
                              'pred', 'prob', 'predict_std', 'predict_rank', 'target', 'target_date']]
        return return_df
