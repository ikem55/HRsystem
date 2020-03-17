# データ生成の基礎クラス
# データ取得、データ保存
from modules.jrdb_transform_before_race import BEFORE_RACE
from modules.jrdb_transform_after_race import AFTER_RACE
from modules.jrdb_transform_after_raceuma_ml import AFTER_RACEUMA_ML
from modules.jrdb_sql import JrdbSql


class JrdbTransform(object):
    def __init__(self):
        self.sql_proc = JrdbSql()

    def get_unprocessed_list(self, jrdb_table, trans_table, start_date):
        """  JRDB側のデータを取り込んでいるが変換処理がされていない日付のリストを作成

        :param str jrdb_table:
        :param str trans_table:
        :param str start_date:
        :return:
        """
        jrdb_list = self.sql_proc.get_jrdb_target_date_list(
            jrdb_table, start_date)
        target_list = self.sql_proc.get_target_date_list(
            trans_table, start_date)
        def_list = list(set(jrdb_list) - set(target_list))
        return sorted(def_list)

    def procedure_all_transformed_data(self):
        """ すべてのモデルに対してデータを生成して保存する手順 """
        self.procedure_import_data("AFTER_RACE")
        self.procedure_import_data("BEFORE_RACE")

    def procedure_import_data(self, model):
        """ 指定したモデルに対してインポートする処理をまとめる

        :param model:
        """
        print("----" + model + "-----")
        if model == 'BEFORE_RACE':
            jrdb_table = '[TALEND].[jrdb].[ORG_KYI]'
            trans_table = '[TALEND].[dbo].[BEFORE_RACE]'
            transform_model = BEFORE_RACE()
            target_date_list = self.__get_unprocessed_list(
                jrdb_table, trans_table, '120101')
            # target_date_list = ['181006','181002']
        elif model == 'AFTER_RACE':
            jrdb_table = '[TALEND].[jrdb].[ORG_SED]'
            trans_table = '[TALEND].[dbo].[AFTER_RACE]'
            transform_model = AFTER_RACE()
            target_date_list_all = self.__get_unprocessed_list(
                jrdb_table, trans_table, '100101')
            target_date_list = list(
                set(target_date_list_all) - set(self.sql_proc.get_unfinished_sed_list()))
            # target_date_list = ['181006','181002']
        elif model == 'AFTER_RACEUMA_ML':
            jrdb_table = '[TALEND].[jrdb].[ORG_SED]'
            trans_table = '[TALEND].[dbo].[AFTER_RACEUMA_ML]'
            transform_model = AFTER_RACEUMA_ML()
            target_date_list_all = self.__get_unprocessed_list(
                jrdb_table, trans_table, '100101')
            target_date_list = list(
                set(target_date_list_all) - set(self.sql_proc.get_unfinished_sed_list()))
            # target_date_list = ['181006','181002']

        # データ取得・保存処理を繰り返し実施
        for target_date in target_date_list:
            print("target date:" + target_date)
            transform_model.set_target_date(target_date)
            transform_model.procedure_create_data()

    def set_BEFORE_RACE_model(self):
        return BEFORE_RACE()

    def set_AFTER_RACE_model(self):
        return AFTER_RACE()

    def set_AFTER_RACEUMA_ML_model(self):
        return AFTER_RACEUMA_ML()

    def procedure_model_target_date_data(self, model, target_date):
        """  指定したモデルと処理日に対してインポートする処理をまとめる

        :param model:
        """
        if model == "BEFORE_RACE":
            transform_model = BEFORE_RACE()
        elif model == "AFTER_RACE":
            transform_model = AFTER_RACE()
        transform_model.set_target_date(target_date)
        transform_model.procedure_create_data()

    # 対象の処理日を取得する
#    def get_target_datelist(self):
#        target_datelist = ['181020' , '181021']
#        return target_datelist
#
#    # レース前の分析データを作成する
#    def procedure_before_race(self, target_date):
#        proc_before_race = BEFORE_RACE(target_date)
#        proc_before_race.procedure_create_data()
#
#    # レース前の分析データを作成する
#    def procedure_after_race(self, target_date):
#        proc_after_race = AFTER_RACE(target_date)
#        proc_after_race.procedure_create_data()
