from modules.base_sk_proc import BaseSkProc
from modules.jra_load import JRALoad
import modules.util as mu

import pandas as pd
import numpy as np
import sys
import os
import pickle

from sklearn.model_selection import train_test_split
import optuna.integration.lightgbm as lgb
import lightgbm as lgb_original

class JRASkProc(BaseSkProc):
    """
    地方競馬の機械学習処理プロセスを取りまとめたクラス。
    """
    dict_folder = ""
    model_path = ""
    index_list = ["RACE_KEY", "UMABAN", "target_date"]
    class_dict = [{"name": "芝", "code": "1", "except_list": ["芝ダ障害コード", "ダ馬場状態コード", "ダ馬場状態内", "ダ馬場状態中", "ダ馬場状態外", "ダ馬場差", "転圧", "凍結防止剤"]},
                  {"name": "ダ", "code": "2", "except_list": ["芝ダ障害コード", "芝馬場状態コード", "芝馬場状態内", "芝馬場状態中", "芝馬場状態外", "芝馬場差", "直線馬場差最内",
                                                             "直線馬場差内", "直線馬場差中", "直線馬場差外", "直線馬場差大外", "芝種類", "草丈"]}]

    def _get_load_object(self, version_str, start_date, end_date, mock_flag, test_flag):
        print("-- check! this is LBSkProc class: " + sys._getframe().f_code.co_name)
        ld = JRALoad(version_str, start_date, end_date, mock_flag, test_flag)
        return ld

    def _merge_df(self):
        self.base_df = pd.merge(self.ld.race_df, self.ld.raceuma_df, on=["RACE_KEY", "target_date", "NENGAPPI"])
        self.base_df = pd.merge(self.base_df, self.ld.horse_df, on=["血統登録番号", "target_date"])
        self.base_df = pd.merge(self.base_df, self.ld.prev1_raceuma_df, on=["RACE_KEY", "UMABAN", "target_date"], how='left')
        self.base_df = pd.merge(self.base_df, self.ld.prev2_raceuma_df, on=["RACE_KEY", "UMABAN", "target_date"], how='left')

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
                           "ZENSO1_RACE_KEY", "ZENSO2_RACE_KEY", "ZENSO3_RACE_KEY", "ZENSO4_RACE_KEY", "ZENSO5_RACE_KEY"], axis=1, inplace=True)

    def _scale_df(self):
        pass

    def _rename_key(self, df):
        return df

    def _drop_unnecessary_columns(self):
        """ predictに不要な列を削除してpredict_dfを作成する。削除する列は血統登録番号、確定着順、タイム指数、単勝オッズ、単勝人気  """
        pass

    def _drop_columns_result_df(self):
        self.result_df.drop(["確定着順", "単勝オッズ"], axis=1, inplace=True)

    def learning_sk_model(self, df, target):
        """ 指定された場所・ターゲットに対しての学習処理を行う

        :param dataframe df: dataframe
        :param str target: str
        """
        for dict in self.class_dict:
            this_model_name = self.model_name + "_" + target + "_" + dict["name"]
            temp_df = df.query(f"芝ダ障害コード == '{dict['code']}'")
            temp_df.drop(dict["except_list"], axis=1, inplace=True)
            if os.path.exists(self.model_folder + this_model_name + '.pickle'):
                print("\r\n -- skip create learning model -- \r\n")
            else:
                self.set_target_flag(target)
                print("learning_sk_model: df", temp_df.shape)
                if temp_df.empty:
                    print("--------- alert !!! no data")
                else:
                    self.set_learning_data(temp_df, target)
                    self.divide_learning_data()
                    if self.y_train.sum() == 0:
                        print("---- wrong data --- skip learning")
                    else:
                        self.load_learning_target_encoding()
                        self.X_train = self.change_obj_to_int(self.X_train)
                        self.X_test = self.change_obj_to_int(self.X_test)
                        imp_features = self.learning_base_race_lgb(this_model_name, target)
                        # 抽出した説明変数でもう一度Ｌｅａｒｎｉｎｇを実施
                        self.x_df = self.x_df[imp_features]
                        self.divide_learning_data()
                        self._set_label_list(self.x_df) # 項目削除されているから再度ターゲットエンコーディングの対象リストを更新する
                        self.load_learning_target_encoding()
                        self.X_train = self.change_obj_to_int(self.X_train)
                        self.X_test = self.change_obj_to_int(self.X_test)
                        self.learning_race_lgb(this_model_name, target)

    def change_obj_to_int(self, df):
        """ objのデータ項目をint型に変更する """
        label_list = df.select_dtypes(include=object).columns.tolist()
        df[label_list] = df[label_list].astype(float) #NaNがあるとintにできない
        return df

    def learning_base_race_lgb(self, this_model_name, target):
        """ null importanceにより有効な説明変数を抽出する """
        print("learning_base_race_lgb")
        # テスト用のデータを評価用と検証用に分ける
        X_eval, X_valid, y_eval, y_valid = train_test_split(self.X_test, self.y_test, random_state=42)

        if self.test_flag:
            num_boost_round = 5
            n_rum = 3
            threshold = 5
            ram_imp_num_boost_round = 5
            early_stopping_rounds = 3
        else:
            num_boost_round = 100
            n_rum = 15
            threshold = 30
            ram_imp_num_boost_round = 100
            early_stopping_rounds = 50

        # データセットを生成する
        lgb_train = lgb.Dataset(self.X_train, self.y_train)
        lgb_eval = lgb.Dataset(X_eval, y_eval, reference=lgb_train)

        # 上記のパラメータでモデルを学習する
        this_param = self.lgbm_params[target]
        model = lgb_original.train(this_param, lgb_train,
                          # モデルの評価用データを渡す
                          valid_sets=lgb_eval,
                          # 最大で 1000 ラウンドまで学習する
                          num_boost_round=num_boost_round,
                          # 10 ラウンド経過しても性能が向上しないときは学習を打ち切る
                          early_stopping_rounds=early_stopping_rounds)

        # 特徴量の重要度を含むデータフレームを作成
        imp_df = pd.DataFrame()
        imp_df["feature"] = X_eval.columns
        imp_df["importance"] = model.feature_importance()
        print(imp_df)
        imp_df = imp_df.sort_values("importance")

        # 比較用のランダム化したモデルを学習する
        null_imp_df = pd.DataFrame()
        for i in range(n_rum):
            print(i)
            ram_lgb_train = lgb.Dataset(self.X_train, np.random.permutation(self.y_train))
            ram_lgb_eval = lgb.Dataset(X_eval, np.random.permutation(y_eval), reference=lgb_train)
            ram_model = lgb_original.train(this_param, ram_lgb_train,
                              # モデルの評価用データを渡す
                              valid_sets=ram_lgb_eval,
                              # 最大で 1000 ラウンドまで学習する
                              num_boost_round=ram_imp_num_boost_round,
                              # 10 ラウンド経過しても性能が向上しないときは学習を打ち切る
                              early_stopping_rounds=early_stopping_rounds)
            ram_imp_df = pd.DataFrame()
            ram_imp_df["feature"] = X_eval.columns
            ram_imp_df["importance"] = ram_model.feature_importance()
            ram_imp_df = ram_imp_df.sort_values("importance")
            ram_imp_df["run"] = i + 1
            null_imp_df = pd.concat([null_imp_df, ram_imp_df])

        # 閾値を超える特徴量を取得
        imp_features = []
        for feature in imp_df["feature"]:
            actual_value = imp_df.query(f"feature=='{feature}'")["importance"].values
            null_value = null_imp_df.query(f"feature=='{feature}'")["importance"].values
            percentage = (null_value < actual_value).sum() / null_value.size * 100
            if percentage >= threshold:
                imp_features.append(feature)

        print(len(imp_features))
        print(imp_features)

        self._save_learning_model(imp_features, this_model_name + "_feat_columns")
        return imp_features


    def learning_race_lgb(self, this_model_name, target):
        # テスト用のデータを評価用と検証用に分ける
        X_eval, X_valid, y_eval, y_valid = train_test_split(self.X_test, self.y_test, random_state=42)

        # データセットを生成する
        lgb_train = lgb.Dataset(self.X_train, self.y_train)
        lgb_eval = lgb.Dataset(X_eval, y_eval, reference=lgb_train)

        if self.test_flag:
            num_boost_round=5
            early_stopping_rounds = 3
        else:
            num_boost_round=1000
            early_stopping_rounds = 50

        # 上記のパラメータでモデルを学習する
        best_params, history = {}, []
        this_param = self.lgbm_params[target]
        model = lgb.train(this_param, lgb_train,
                          valid_sets=lgb_eval,
                          verbose_eval=False,
                          num_boost_round=num_boost_round,
                          early_stopping_rounds=early_stopping_rounds,
                          best_params=best_params,
                          tuning_history=history)
        print("Bset Paramss:", best_params)
        print('Tuning history:', history)

        self._save_learning_model(model, this_model_name)


    def _predict_sk_model(self, df, target):
        """ 指定された場所・ターゲットに対しての予測データの作成を行う

        :param dataframe  df: dataframe
        :param str val: str
        :param str target: str
        :return: dataframe
        """
        self.set_target_flag(target)
        all_pred_df = pd.DataFrame()
        for dict in self.class_dict:
            this_model_name = self.model_name + "_" + target + "_" + dict["name"]
            print("======= this_model_name: " + this_model_name + " ==========")
            temp_df = df.query(f"芝ダ障害コード == '{dict['code']}'")
            temp_df.drop(dict["except_list"], axis=1, inplace=True)
            temp_df = self._set_predict_target_encoding(temp_df)
            print(temp_df.shape)
            with open(self.model_folder + this_model_name + '_feat_columns.pickle', 'rb') as f:
                imp_features = pickle.load(f)
            exp_df = temp_df.drop(self.index_list, axis=1)
            exp_df = exp_df[imp_features].to_numpy()
            # print(self.model_folder)
            if os.path.exists(self.model_folder + this_model_name + '.pickle'):
                with open(self.model_folder + this_model_name + '.pickle', 'rb') as f:
                    model = pickle.load(f)
                y_pred = model.predict(exp_df)
                pred_df = self._sub_create_pred_df(temp_df, y_pred)
                all_pred_df = pd.concat([all_pred_df, pred_df])
        return all_pred_df

    def _sub_create_pred_df(self, temp_df, y_pred):
        pred_df = pd.DataFrame({"RACE_KEY": temp_df["RACE_KEY"], "target_date": temp_df["target_date"], "prob": y_pred})
        pred_df.loc[:, "pred"] = pred_df.apply(lambda x: 1 if x["prob"] >= 0.5 else 0, axis=1)
        return pred_df

    def _calc_grouped_data(self, df):
        """ 与えられたdataframe(予測値）に対して偏差化とランク化を行ったdataframeを返す

        :param dataframe df: dataframe
        :return: dataframe
        """
        grouped = df.groupby("RACE_KEY")
        grouped_df = grouped.describe()['prob'].reset_index()
        merge_df = pd.merge(df, grouped_df, on="RACE_KEY")
        merge_df['predict_std'] = (
            merge_df['prob'] - merge_df['mean']) / merge_df['std'] * 10 + 50
        df['predict_rank'] = grouped['prob'].rank("dense", ascending=False)
        merge_df = pd.merge(merge_df, df[["RACE_KEY", "UMABAN", "predict_rank"]], on=["RACE_KEY", "UMABAN"])
        return_df = merge_df[['RACE_KEY', 'UMABAN', 'target_date', 'prob', 'predict_std', 'predict_rank']]
        return return_df