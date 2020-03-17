import luigi
from luigi.mock import MockTarget

from modules.jrdb_download import JrdbDownload
from modules.jrdb_import import JrdbImport
from modules.jrdb_transform_procedure import TRANSFORM_PROCEDURE

import modules.jrdb_sql as msf
import pandas.io.sql as psql

import modules.my_utility as mu

class End_Download_JRDB(luigi.Task):
    """
    | JRDBファイルをダウンロードするLuigiタスク
    | Sub_download_JRDB_data　→　End_Download_JRDB
    """
    task_namespace = 'jrdb'

    def requires(self):
        """
        | Sub_download_JRDB_data を実行した後で開始
        """
        print("----" + __class__.__name__ + ": requires")
        return Sub_download_JRDB_data()

    def run(self):
        """
        | 処理の最後
        """
        print("----" + __class__.__name__ + ": run")
        with self.output().open("w") as target:
            print(__class__.__name__ + " says: task finished".format(task=self.__class__.__name__))

    def output(self):
        """
        :return: MockのOutputを返す
        """
        return MockTarget("output")

class Sub_download_JRDB_data(luigi.Task):
    """
    | JRDBファイルのダウンロードを管理するタスク
    """
    task_namespace = 'jrdb'

    def run(self):
        """
        | DOWNLOAD_JRDBのモジュールを使ってJRDBサイトからファイルをダウンロードする。
        | PACI,SED,SKB,HJC,TYBに対して、get_downloaded_listでダウンロード済ファイルにないファイルをダウンロードする
        """
        print("----" + __class__.__name__ + ": run")
        obj = JrdbDownload()
        print("============== DOWNLOAD JRDB ====================")
        typelist = ["PACI","SED","SKB","HJC","TYB"]
        for type in typelist:
            print("----------------" + type + "---------------")
            filelist = mu.get_downloaded_list(type, obj.archive_path)
            target_df = obj.get_jrdb_page(type)
            for index, row in target_df.iterrows():
                if row['filename'] not in filelist:
                    obj.download_jrdb_file(type.title(),row['url'],row['filename'])

        with self.output().open("w") as target:
            print(__class__.__name__ + " says: task finished".format(task=self.__class__.__name__))


    def output(self):
        """
        :return: MockのOutputを返す
        """
        return MockTarget("output")


class End_import_JRDB(luigi.Task):
    """
    | ダウンロードしたJRDBファイルをDBに取り込むLuigiタスク
    | Sub_import_JRDB_data　→　Sub_retrieve_JRDB_data　→　End_import_JRDB
    """
    task_namespace = 'jrdb'

    def requires(self):
        """
        | Sub_retrieve_JRDB_data を実行した後で開始
        """
        print("----" + __class__.__name__ + ": requires")
        return Sub_retrieve_JRDB_data()

    def run(self):
        """
        | 処理の最後
        """
        print("----" + __class__.__name__ + ": run")
        with self.output().open("w") as target:
            print(__class__.__name__ + " says: task finished".format(task=self.__class__.__name__))

    def output(self):
        """
        :return: MockのOutputを返す
        """
        return MockTarget("output")

class Sub_retrieve_JRDB_data(luigi.Task):
    """
    | 中途半端な状態のデータ取り込みを除外するタスク
    """
    task_namespace = 'jrdb'

    def requires(self):
        """
        | Sub_import_JRDB_data を実行した後で開始
        """
        print("----" + __class__.__name__ + ": requires")
        return Sub_import_JRDB_data()

    def run(self):
        """
        | IMPORT_JRDB のクラスからPACIデータとSEDデータの再取り込みタスクを呼び出して実行
        """
        print("----" + __class__.__name__ + ": run")
        import_procedure = JrdbImport()
        import_procedure.procedure_retrieve_data_paci()
        import_procedure.procedure_retrieve_data_sed()

        with self.output().open("w") as target:
            print(__class__.__name__ + " says: task finished".format(task=self.__class__.__name__))

    def output(self):
        """
        :return: MockのOutputを返す
        """
        return MockTarget("output")

class Sub_import_JRDB_data(luigi.Task):
    """
    JRDBファイルをDBにインポートするタスク
    """
    task_namespace = 'jrdb'

    def run(self):
        """
        | IMPORT_JRDB のクラスに定義されたファイルタイプのリストに対してデータの取り込み処理を実行
        """
        print("----" + __class__.__name__ + ": run")
        import_procedure = JrdbImport()
        for filetype in import_procedure.filetype_list:
            print("=========" + filetype + "========")
            import_procedure.import_jrdb_data(filetype)

        with self.output().open("w") as target:
            print(__class__.__name__ + " says: task finished".format(task=self.__class__.__name__))

    def output(self):
        """
        :return: MockのOutputを返す
        """
        return MockTarget("output")


class End_Transform_JRDB(luigi.Task):
    """
    | DBに取り込んだデータを加工するLuigiタスク
    | Sub_import_after_race_data　→　Sub_import_before_race_data　→　Sub_import_after_ml_race_data　→　End_Transform_JRDB
    """
    task_namespace = 'jrdb'

    def requires(self):
        """
        | Sub_import_after_ml_race_data を実行した後で開始
        """
        print("----" + __class__.__name__ + ": requires")
        return Sub_import_after_ml_race_data()

    def run(self):
        """
        | 処理の最後
        """
        print("----" + __class__.__name__ + ": run")
        with self.output().open("w") as target:
            print(__class__.__name__ + " says: task finished".format(task=self.__class__.__name__))

    def output(self):
        """
        :return: MockのOutputを返す
        """
        return MockTarget("output")

class Sub_import_before_race_data(luigi.Task):
    """
    レース開始前のデータの処理をするタスク
    """
    task_namespace = 'jrdb'

    def requires(self):
        """
        | Sub_import_after_race_data を実行した後で開始
        """
        print("----" + __class__.__name__ + ": requires")
        return Sub_import_after_race_data()

    def run(self):
        """
        | TRANSFORM_PROCEDURE を呼び出して、未処理日付のリストを取得する。
        | 対象日毎に以下のデータを作成してDBにとりこむ
        | - BEFORE_RACEの procedure_create_data
        | - ANALYZE_CORRECTION_TIMEの proc_correction_time →　別処理に切り出し（予定）
        """
        print("----" + __class__.__name__ + ": run")
        obj = TRANSFORM_PROCEDURE()
        jrdb_table = '[TALEND].[jrdb].[ORG_KYI]'
        trans_table = '[TALEND].[dbo].[BEFORE_RACE]'
        transform_model = obj.set_BEFORE_RACE_model()
        target_date_list = obj.get_unprocessed_list(jrdb_table, trans_table, '120101')
        target_date_list.sort()
        for target_date in target_date_list:
            transform_model.set_target_date(target_date)
            transform_model.procedure_create_data()

        with self.output().open("w") as target:
            print(__class__.__name__ + " says: task finished".format(task=self.__class__.__name__))

    def output(self):
        """
        :return: MockのOutputを返す
        """
        return MockTarget("output")

class Sub_import_after_race_data(luigi.Task):
    """
    レース終了後のデータの処理をするタスク
    """
    task_namespace = 'jrdb'

    def run(self):
        """
        | TRANSFORM_PROCEDURE を呼び出して、未処理日付のリストを取得する。
        | 対象日毎に以下のデータを作成してDBにとりこむ
        | - AFTER_RACEの procedure_create_data
        | - ANALYZE_CORRECTION_TIMEの proc_correction_time →　別処理に切り出し（予定）
        """
        print("----" + __class__.__name__ + ": run")
        obj = TRANSFORM_PROCEDURE()
        jrdb_table = '[TALEND].[jrdb].[ORG_SED]'
        trans_table = '[TALEND].[dbo].[AFTER_RACE]'
        transform_model = obj.set_AFTER_RACE_model()
        target_date_list_all = obj.get_unprocessed_list(jrdb_table, trans_table, '100101')
        target_date_list = list(set(target_date_list_all) - set(obj.sql_proc.get_unfinished_sed_list()))
        target_date_list.sort()
        if target_date_list != None:
            for target_date in target_date_list:
                transform_model.set_target_date(target_date)
                transform_model.procedure_create_data()
        with self.output().open("w") as target:
            print(__class__.__name__ + " says: task finished".format(task=self.__class__.__name__))

    def output(self):
        """
        :return: MockのOutputを返す
        """
        return MockTarget("output")


class Sub_import_after_ml_race_data(luigi.Task):
    """
    MLデータ計算後のレース終了後のデータの処理をするタスク
    """
    task_namespace = 'jrdb'

    def requires(self):
        """
        | Sub_import_before_race_data を実行した後で開始
        """
        print("----" + __class__.__name__ + ": requires")
        return Sub_import_before_race_data()

    def run(self):
        """
        | TRANSFORM_PROCEDURE を呼び出して、未処理日付のリストを取得する。
        | 対象日毎に以下のデータを作成してDBにとりこむ
        | - AFTER_RACEの procedure_create_data
        | - ANALYZE_CORRECTION_TIMEの proc_correction_time →　別処理に切り出し（予定）
        """
        print("----" + __class__.__name__ + ": run")
        obj = TRANSFORM_PROCEDURE()
        jrdb_table = '[TALEND].[jrdb].[ORG_SED]'
        trans_table = '[TALEND].[dbo].[AFTER_RACEUMA_ML]'
        target_date_list_all = obj.get_unprocessed_list(jrdb_table, trans_table, '140101')
        target_date_list = list(set(target_date_list_all) - set(obj.sql_proc.get_unfinished_sed_list()))
        target_date_list.sort()
        if target_date_list != None:
            for target_date in target_date_list:
                import_df = self.get_df(target_date)
                self.import_data(import_df, target_date)
        with self.output().open("w") as target:
            print(__class__.__name__ + " says: task finished".format(task=self.__class__.__name__))

    def get_df(self, target_date):
        sql_proc = msf.JrdbSql()
        sql_text = """
SELECT a.[RACE_KEY]
      ,a.[UMABAN]
      ,a.[IDM] - b.predict_IDM as diff_IDM
	  ,a.target_date
  FROM [TALEND].[jrdb].[ORG_SED] a , [TALEND].[sony].[predict_idm] b 
  where a.IDM is not null
  and  a.RACE_KEY = b.RACE_KEY and a.UMABAN = b.UMABAN
  and a.target_date = '""" + target_date + """'
  order by a.RACE_KEY , a.UMABAN
        """
        df = psql.read_sql(sql_text, sql_proc.cnn)
        return df

    def import_data(self, df, target_date):
        sql_proc = msf.JrdbSql()
        table_name = '[TALEND].[dbo].[AFTER_RACEUMA_ML]'
        columns = ['%s','%s','%s','%s']
        create_sql = """if object_id('""" + table_name + """') is null
        CREATE TABLE """ + table_name + """ (
        RACE_KEY nvarchar(8),
        UMABAN nvarchar(2),
        diff_IDM real,
        target_date          nvarchar(6)
        );
        """
        msf.JrdbSql.class_exec_create_sql(create_sql)
        sql_proc.proc_insert_by_target_date(target_date, table_name, columns, df)

    def output(self):
        """
        :return: MockのOutputを返す
        """
        return MockTarget("output")