import pymssql
import pandas.io.sql as psql
import os
import pandas as pd

import modules.util as mu
import my_config


class JrdbSql(object):
    """
    SQL Serverとのデータやり取りに関する処理をまとめる
    """

    def __init__(self):
        self.cnn = pymssql.connect(host=os.environ["DATABASE_HOST"], user=os.environ["DATABASE_USER"],
                                   password=os.environ["DATABASE_PASSWORD"], database="TALEND")
        self.cur = self.cnn.cursor()

    def proc_insert_by_target_date(self, target_date, table_name, columns, df):
        """ 指定したtarget_dateとtableに対してdataframeをinsertする処理

        :param str target_date: yyMMdd
        :param str table_name: '[TALEND].[predict].[RACE_UMA_BASHO_KEY]'
        :param list columns: ['%s','%s','%s','%s','%s']
        :param dataframe df: データフレーム
        """
        self.__exec_delete_sql_by_target_date(target_date, table_name)
        self.__exec_many_insert_sql(table_name, columns, df)
        self.__exec_commit()

    def proc_insert_by_filename(self, filename, table_name, columns, df):
        """  指定したfilenameとtableに対してdataframeをinsertする処理

        :param str filename:
        :param str table_name:
        :param list columns:
        :param dataframe df:
        :return:
        """
        self.__exec_delete_sql_by_filename(filename, table_name)
        self.__exec_many_insert_sql(table_name, columns, df)
        self.__exec_commit()

    def get_sql_data(self, table_name, target_date):
        """  指定したtableのfilenameに対してtarget_dateで絞り込んだデータを取得

        :param str table_name:
        :param str target_date:
        :return:
        """
        check_value = self.__check_table_exists(table_name)
        if check_value != 'null':
            sql = "SELECT * from " + table_name + \
                " where filename like '%" + target_date + "%'"
            df = psql.read_sql(sql, self.cnn)
        else:
            df = pd.DataFrame(columns=[])
        return df

    def get_sql_data_by_psql(self, query):
        """ 指定したクエリでデータを取得

        :param str query:
        :return:
        """
        df = psql.read_sql(query, self.cnn)
        return df

    def get_sql_data_by_target_date(self, table_name, target_date):
        """  指定したtableに対してtarget_dateで絞り込んだデータを取得

        :param str table_name:
        :param str target_date:
        :return:
        """
        check_value = self.__check_table_exists(table_name)
        if check_value != 'null':
            sql = "SELECT * from " + table_name + " where target_date='" + target_date + "'"
            df = psql.read_sql(sql, self.cnn)
        else:
            df = pd.DataFrame(columns=[])
        return df

    def get_sql_data_between_target_date(self, table_name, start_date, end_date):
        """ 指定したtableに対してtarget_dateで絞り込んだデータを取得

        :param str table_name:
        :param str start_date:
        :param str end_date:
        :return:
        """
        check_value = self.__check_table_exists(table_name)
        if check_value != 'null':
            sql = "SELECT * from " + table_name + " where target_date>='" + \
                start_date + "' and target_date < '" + end_date + "'"
            df = psql.read_sql(sql, self.cnn)
        else:
            df = pd.DataFrame(columns=[])
        return df

    def get_previous_sql_data(self, target_date):
        """ 指定したtarget_dateのデータの前走データを取得

        :param str target_date:
        :return:
        """
        sql = "SELECT a.RACE_KEY as THIS_RACE_KEY , a.UMABAN as THIS_UMABAN , b.*  from [TALEND].[jrdb].[ORG_KYI] a , [TALEND].[jrdb].[ORG_SED] b where a.filename like '%" + \
            target_date + \
            "%' and a.ZENSO1_KYOSO_RESULT = concat(b.KETTO_TOROKU_BANGO , b.NENGAPPI)"
        df = psql.read_sql(sql, self.cnn)
        return df.drop(['RACE_KEY', 'UMABAN'], axis=1).rename(columns={'THIS_RACE_KEY': 'RACE_KEY', 'THIS_UMABAN': 'UMABAN'})

    def get_kyi_target_data_record(self, filename, retrieve_check_table_name):
        """  Retrieve処理用、kyiの最新のレコードファイルを取得

        :param str filename:
        :param str retrieve_check_table_name:
        :return:
        """
        cur = self.cnn.cursor()
        sql_select = """
        select count(*) from """ + retrieve_check_table_name + """ where [filename] = '""" + filename + """'
        """
        cur.execute(sql_select)
        row = cur.fetchall()
        for r in row:
            row_count = r[0]
        return row_count

    def get_sed_target_data_row(self, sed_table_name):
        """  Retrieve処理用、sedの最新のレコードファイルを取得

        :param str sed_table_name:
        :return:
        """
        cur = self.cnn.cursor()
        sql_select = """
        SELECT [filename]  FROM """ + sed_table_name + """ where [IDM] is null  group by [filename]  having count(*) >= 100
        """
        cur.execute(sql_select)
        row = cur.fetchall()
        return row

    def get_jrdb_target_date_list(self, table_name, start_date):
        """  JRDBの取り込みORGテーブルに対して指定したtableの取り込み済み対象日のリストを取得

        :param str table_name:
        :return:
        """
        check_value = self.__check_table_exists(table_name)
        if check_value != 'null':
            sql = "SELECT distinct target_date FROM " + table_name + \
                " where target_date >= '" + start_date + "'"
            df = psql.read_sql(sql, self.cnn)
            date_list = df['target_date'].values.tolist()
        else:
            date_list = []
        return date_list

    @classmethod
    def class_get_target_date_list(cls, table_name, predictor_name):
        """  classmethodでget_target_date_listを呼び出すための関数

        :param str table_name:
        :param str predictor_name:
        :return:
        """
        ins_sql = JrdbSql()
        check_value = ins_sql.__check_table_exists(table_name)
        if check_value != 'null':
            sql = "SELECT * from " + table_name + \
                " where PREDICTOR_NAME = '" + predictor_name + "'"
            date_list = psql.read_sql(sql, ins_sql.cnn)
        else:
            date_list = pd.DataFrame(columns=[])
        return date_list

    def get_target_date_list(self, table_name, start_date):
        """  指定したtableの取り込み済み対象日のリストを取得

        :param str table_name:
        :return:
        """
        check_value = self.__check_table_exists(table_name)
        if check_value != 'null':
            sql = "SELECT distinct [target_date] FROM " + table_name + " where target_date >= '" + \
                start_date + "' group by [target_date] having count(*) >= 5"
            df = psql.read_sql(sql, self.cnn)
            date_list = df['target_date'].values.tolist()
        else:
            date_list = []
        return date_list

    def get_unfinished_sed_list(self):
        """ JRDBのSEDデータで未完成のものを取得

        :return:
        """
        sql = "SELECT [target_date]  FROM [TALEND].[jrdb].[ORG_SED] group by [target_date]  having max([IDM]) is NULL"
        df = psql.read_sql(sql, self.cnn)
        return df['target_date'].values.tolist()

    def __check_table_exists(self, table_name):
        """ 指定したtableが存在するか確認

        :param str table_name:
        :return:
        """
        sql = "if OBJECT_ID('" + table_name + \
            "') is null SELECT 'null' as val ELSE SELECT ' not null' as val"
        df = psql.read_sql(sql, self.cnn)
        return df['val'].values[0]

    def import_race_df(self, table_name, race_df, target_date):
        """ RACE_KEYをIDとしたテーブルのインサート

        :param str table_name:
        :param dataframe race_df:
        :param str target_date:
        :return:
        """
        create_sql = """if object_id('""" + table_name + """') is null
        CREATE TABLE """ + table_name + """(
        RACE_KEY nvarchar(8) """
        columns = ['%s']

        for column_name, item in race_df.iteritems():
            if column_name != "RACE_KEY":
                create_sql += ', ' + \
                              mu.escape_create_text(
                        column_name) + ' ' + mu.convert_python_to_sql_type(item.dtype)
                columns.append('%s')
        create_sql += ", target_date nvarchar(6));"
        columns.append('%s')
        race_df['target_date'] = target_date

        # SQLの実行
        self.cur.execute(create_sql)
        self.proc_insert_by_target_date(
            target_date, table_name, columns, race_df)

    def import_raceuma_df(self, table_name, race_df, target_date):
        """ RACE_KEYとUMABANをIDとしたテーブルのインサート

        :param str table_name:
        :param dataframe race_df:
        :param str target_date:
        :return:
        """
        create_sql = """if object_id('""" + table_name + """') is null
        CREATE TABLE """ + table_name + """(
        RACE_KEY nvarchar(8) , UMABAN nvarchar(2)"""
        columns = ['%s', '%s']

        for column_name, item in race_df.iteritems():
            if not(column_name == "RACE_KEY" or column_name == "UMABAN" or column_name == "target_date"):
                create_sql += ', ' + \
                              mu.escape_create_text(
                        column_name) + ' ' + mu.convert_python_to_sql_type(item.dtype)
                columns.append('%s')
        create_sql += ", target_date nvarchar(6));"
        columns.append('%s')
        race_df['target_date'] = target_date

        # SQLの実行
        self.cur.execute(create_sql)
        self.proc_insert_by_target_date(
            target_date, table_name, columns, race_df)

    def import_course_df(self, table_name, race_df, target_date):
        """  COURSE_KEYをIDとしたテーブルのインサート

        :param str table_name:
        :param dataframe race_df:
        :param str target_date:
        :return:
        """
        create_sql = """if object_id('""" + table_name + """') is null
        CREATE TABLE """ + table_name + """(
        [COURSE_KEY] nvarchar(8) """
        columns = ['%s']

        for column_name, item in race_df.iteritems():
            if column_name != "COURSE_KEY":
                create_sql += ", [" + mu.escape_create_text(
                    column_name) + "] " + mu.convert_python_to_sql_type(item.dtype)
                columns.append('%s')
        create_sql += ", [target_date] nvarchar(6));"
        columns.append('%s')
        race_df['target_date'] = target_date

        # SQLの実行
        self.cur.execute(create_sql)
        self.proc_insert_by_target_date(
            target_date, table_name, columns, race_df.fillna(0))

    def import_basho_df(self, table_name, race_df, target_date):
        """  BASHO_KEYをIDとしたテーブルのインサート

        :param str table_name:
        :param dataframe race_df:
        :param str target_date:
        :return:
        """
        create_sql = """if object_id('""" + table_name + """') is null
        CREATE TABLE """ + table_name + """(
        [BASHO_KEY] nvarchar(4) """
        columns = ['%s']

        for column_name, item in race_df.iteritems():
            if column_name != "BASHO_KEY":
                create_sql += ", [" + mu.escape_create_text(
                    column_name) + "] " + mu.convert_python_to_sql_type(item.dtype)
                columns.append('%s')
        create_sql += ", [target_date] nvarchar(6));"
        columns.append('%s')
        race_df['target_date'] = target_date

        # SQLの実行
        self.cur.execute(create_sql)
        self.proc_insert_by_target_date(
            target_date, table_name, columns, race_df.fillna(0))

    def import_kaisai_df(self, table_name, race_df, target_date):
        """ KAISAI_KEYをIDとしたテーブルのインサート

        :param str table_name:
        :param dataframe race_df:
        :param str target_date:
        :return:
        """
        create_sql = """if object_id('""" + table_name + """') is null
        CREATE TABLE """ + table_name + """(
        [KAISAI_KEY] nvarchar(3) """
        columns = ['%s']

        for column_name, item in race_df.iteritems():
            if column_name != "KAISAI_KEY":
                create_sql += ", [" + mu.escape_create_text(
                    column_name) + "] " + mu.convert_python_to_sql_type(item.dtype)
                columns.append('%s')
        create_sql += ", [target_date] nvarchar(6));"
        columns.append('%s')
        race_df['target_date'] = target_date

        # SQLの実行
        self.cur.execute(create_sql)
        self.proc_insert_by_target_date(
            target_date, table_name, columns, race_df.fillna(0))

    def import_pm_method_raceuma_df(self, table_name, method_df, target_date, predictor_name, method_name):
        """ PMの計算済みチェックをテーブルに格納する

        :param str table_name:
        :param dataframe check_df:
        :param str target_date:
        :param str predictor_name:
        :return:
        """
        columns = ['%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s']
        delete_sql = "delete " + table_name + " where target_date = '" + target_date + \
            "' and PREDICTOR_NAME = '" + predictor_name + \
            "' and METHOD_NAME ='" + method_name + "'"
        self.cur.execute(delete_sql)
        self.__exec_many_insert_sql(table_name, columns, method_df)
        self.__exec_commit()

    def import_pm_method_race_df(self, table_name, method_df, target_date, predictor_name, method_name):
        """ PMの計算済みチェックをテーブルに格納する

        :param str table_name:
        :param dataframe check_df:
        :param str target_date:
        :param str predictor_name:
        :return:
        """
        columns = ['%s', '%s', '%s', '%s', '%s', '%s', '%s']
        delete_sql = "delete " + table_name + " where target_date = '" + target_date + \
            "' and PREDICTOR_NAME = '" + predictor_name + \
            "' and METHOD_NAME ='" + method_name + "'"
        self.cur.execute(delete_sql)
        self.__exec_many_insert_sql(table_name, columns, method_df)
        self.__exec_commit()

    def import_checktable_df(self, table_name, check_df, target_date, predictor_name, method_name):
        """ PMの計算済みチェックをテーブルに格納する

        :param str table_name:
        :param dataframe check_df:
        :param str target_date:
        :param str predictor_name:
        :return:
        """
        create_sql = "if object_id('" + table_name + "')is null CREATE TABLE " + table_name + \
            " (PREDICTOR_NAME nvarchar(10) , METHOD_NAME nvarchar(100) , target_date nvarchar(6))"
        columns = ['%s', '%s', '%s']
        self.cur.execute(create_sql)
#        self.__exec_class_sql(create_sql)
        delete_sql = "delete " + table_name + " where target_date = '" + target_date + \
            "' and PREDICTOR_NAME = '" + predictor_name + \
            "' and METHOD_NAME ='" + method_name + "'"
        self.cur.execute(delete_sql)
        self.__exec_many_insert_sql(table_name, columns, check_df)
        self.__exec_commit()

    def __exec_delete_sql_by_target_date(self, target_date, table_name):
        """ 指定したtarget_dateとtableに対してdelete sqlの実行

        :param str target_date:
        :param str table_name:
        :return:
        """
        delete_sql = "delete " + table_name + \
            " where target_date = '" + target_date + "'"
        self.cur.execute(delete_sql)

    def __exec_delete_sql_by_filename(self, filename, table_name):
        """ 指定したfilenameとtableに対してdelete sqlの実行

        :param str filename:
        :param str table_name:
        :return:
        """
        delete_sql = "delete " + table_name + " where filename = '" + filename + "'"
        self.cur.execute(delete_sql)

    def __exec_commit(self):
        """ SQL文のcommit"""
        self.cnn.commit()

#    def delete_sql(self, filename , table_name):
#        delete_sql = """
#        delete """ + table_name + """ where filename = '"""+ filename + """'
#        """
#        return delete_sql

#    def sql_insert(self , table_name , columns):
#        args = dict(table=table_name, columns=', '.join(columns))
#        insert_sql = 'INSERT INTO {table} VALUES ({columns})'.format(**args)
#        return insert_sql

    @classmethod
    def class_exec_create_sql(cls, create_sql):
        """ classmethodでexec_create_sqlを呼び出すための関数

        :param str create_sql:
        """
        ins_sql = JrdbSql()
        ins_sql.__exec_class_sql(create_sql)

    @classmethod
    def class_exec_pm_delete_sql(cls,  table_name, predictor_name, method_name):
        """ 指定したメソッドの結果を削除するためのdelete sqlの実行

        :param str filename:
        :param str table_name:
        """
        ins_sql = JrdbSql()
        delete_sql = "delete " + table_name + " where PREDICTOR_NAME = '" + \
            predictor_name + "' and METHOD_NAME ='" + method_name + "'"
        ins_sql.__exec_class_sql(delete_sql)

    def __exec_class_sql(self, sql):
        """ classmethodで呼びされたSQLの実行

        :param str sql:
        """
        self.cur.execute(sql)
        self.cnn.commit()

    def __exec_many_insert_sql(self, table_name, columns, df):
        """ 指定したtableに対してdata frameのinsert sql の実行

        :param str table_name:
        :param list columns:
        :param dataframe df:
        """
        params = [tuple(x) for x in df.values]
        args = dict(table=table_name, columns=', '.join(columns))
        insert_sql = 'INSERT INTO {table} VALUES ({columns})'.format(**args)
        self.cur.executemany(insert_sql, params)

    def import_mv_course_df(self, df, target_date, type):
        """ c_transform_moving_averageで計算したコース毎の移動平均のデータを格納する

        :param dataframe df:
        :param str target_date:
        """
        if type == "COURSE_WAKU":
            table_name = "[TALEND].[dbo].[MA_COURSE_RESULT_WAKU]"
            create_sql_column = ", [WAKUBAN] nvarchar(1) "
        elif type == "RACE_KYAKUSHITSU":
            table_name = "[TALEND].[dbo].[MA_COURSE_RESULT_KYAKUSHITSU]"
            create_sql_column = ", [RACE_KYAKUSHITSU] nvarchar(1) "
        elif type == "COURSE_4KAKU":
            table_name = "[TALEND].[dbo].[MA_COURSE_RESULT_4KAKU]"
            create_sql_column = ", [COURSE_4KAKU] nvarchar(1) "
        elif type == "KISHU_CODE":
            table_name = "[TALEND].[dbo].[MA_COURSE_RESULT_KISHU]"
            create_sql_column = ", [KISHU_CODE] nvarchar(5) "
        elif type == "CHICHI_NAME":
            table_name = "[TALEND].[dbo].[MA_COURSE_RESULT_CHICHI_NAME]"
            create_sql_column = ", [CHICHI_NAME]  nvarchar(36) "
        elif type == "CHICHI_KEITO":
            table_name = "[TALEND].[dbo].[MA_COURSE_RESULT_CHICHI_KEITO]"
            create_sql_column = ", [CHICHI_KEITO]  nvarchar(5) "
        elif type == "HAHA_CHICHI_KEITO":
            table_name = "[TALEND].[dbo].[MA_COURSE_RESULT_HAHA_CHICHI_KEITO]"
            create_sql_column = ", [HAHA_CHICHI_KEITO]  nvarchar(5) "

        create_sql = "if object_id('" + table_name + "') is null CREATE TABLE " + \
            table_name + "( [COURSE_KEY] nvarchar(8)"
        create_sql += create_sql_column
        create_sql += ", [ALL] real , [WIN] real , [REN] real ,[FUKU] real , [TANSHO] real , [FUKUSHO] real , [WIN_RATE] real , [REN_RATE] real , [FUKU_RATE] real , [target_date] nvarchar(6));"
        columns = ['%s', '%s', '%s', '%s', '%s',
                   '%s', '%s', '%s', '%s', '%s', '%s', '%s']
        # SQLの実行
        self.cur.execute(create_sql)
        self.proc_insert_by_target_date(
            target_date, table_name, columns, df.fillna(0).reset_index())

    def import_mv_base_df(self, df, target_date, type):
        """ c_transform_moving_averageで計算したコース毎の移動平均のデータを格納する

        :param dataframe df:
        :param str target_date:
        """
        if type == "KISHU_CODE":
            table_name = "[TALEND].[dbo].[MA_BASE_RESULT_KISHU]"
            create_sql_column = "[KISHU_CODE] nvarchar(5) "
        elif type == "CHICHI_NAME":
            table_name = "[TALEND].[dbo].[MA_BASE_RESULT_CHICHI_NAME]"
            create_sql_column = "[CHICHI_NAME]  nvarchar(36) "
        elif type == "CHICHI_KEITO":
            table_name = "[TALEND].[dbo].[MA_BASE_RESULT_CHICHI_KEITO]"
            create_sql_column = "[CHICHI_KEITO]  nvarchar(5) "
        elif type == "HAHA_CHICHI_KEITO":
            table_name = "[TALEND].[dbo].[MA_BASE_RESULT_HAHA_CHICHI_KEITO]"
            create_sql_column = "[HAHA_CHICHI_KEITO]  nvarchar(5) "

        create_sql = "if object_id('" + table_name + \
            "') is null CREATE TABLE " + table_name + "( "
        create_sql += create_sql_column
        create_sql += ", [ALL] real , [WIN] real , [REN] real ,[FUKU] real , [TANSHO] real , [FUKUSHO] real , [WIN_RATE] real , [REN_RATE] real , [FUKU_RATE] real , [target_date] nvarchar(6));"
        columns = ['%s', '%s', '%s', '%s', '%s',
                   '%s', '%s', '%s', '%s', '%s', '%s']
        # SQLの実行
        self.cur.execute(create_sql)
        self.proc_insert_by_target_date(
            target_date, table_name, columns, df.fillna(0).reset_index())
