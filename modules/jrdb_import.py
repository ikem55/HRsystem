import os

from modules.jrdb_schema import BAC, CHA, CYB, JOA, KAB, KKA, KYI, UKC, SED, SRB, SKB, HJC, TYB, OZ, OT
import modules.util as mu
import modules.jrdb_sql as msf


class JrdbImport(object):
    """ JRDBファイルをSQL Serverに取り込むのに利用

      :param str folder_path: 取り込み元JRDBファイルが存在するフォルダ
      :param str filetype_list: 取り込み対象のJRDBファイルの種類

      Example::

          from dog import Dog
          dog = Dog()
    """

    def __init__(self):
        self.folder_path = os.environ["PROGRAM_PATH"] + "import_JRDB/data_org/"
        self.jrdb_class = ""
        self.filetype_list = ['BAC', 'CHA', 'CYB', 'JOA', 'KAB', 'KKA',
                              'KYI', 'UKC', 'SED', 'SRB', 'SKB', 'HJC', 'TYB']  # ,'OT','OZ']
        self.finish_path = os.environ["PROGRAM_PATH"] + \
            "import_JRDB/data_org/done/"
        self.archive_path = os.environ["PROGRAM_PATH"] + \
            "import_JRDB/data_archive/"
        self.retrieve_check_table_name = '[TALEND].[jrdb].[ORG_KYI]'
        self.sed_table_name = '[TALEND].[jrdb].[ORG_SED]'
        #self.cnn = pymssql.connect(host=os.environ["DATABASE_HOST"] , user=os.environ["DATABASE_USER"] , password=os.environ["DATABASE_PASSWORD"] , database="TALEND")

    def procedure_import_jrdb_data(self):
        """ 通常データの取り込み処理をまとめた手順 """
        print("============== IMPORT JRDB ====================")
        for filetype in self.filetype_list:
            print("=========" + filetype + "========")
            self.import_jrdb_data(filetype)
        print("============== CHECK RETRIEVE DATA =============")
        self.procedure_retrieve_data_paci()
        self.procedure_retrieve_data_sed()

    def procedure_import_jrdb_sokuho_data(self):
        """ 速報データの取り込み処理をまとめた手順  """
        print("============== IMPORT SOKUHO JRDB ====================")
        sokuho_typelist = ["TYB"]
        for filetype in sokuho_typelist:
            print("=========" + filetype + "========")
            self.import_jrdb_data(filetype)

    def import_jrdb_data(self, filetype):
        """ 指定されたJRDBのタイプのファイルをSQLServerに取り込む

        :param str filetype:
        """
        self.set_file_type(filetype)
        filelist = mu.get_file_list(filetype, self.folder_path)
        sql_proc = msf.MY_SQL_CLASS()
        msf.MY_SQL_CLASS.class_exec_create_sql(self.jrdb_class.create_sql)
#        sql_proc.exec_create_sql(self.jrdb_class.create_sql)
        for file in filelist:
            print(file)
            sql_proc.proc_insert_by_filename(
                file, self.jrdb_class.table_name, self.jrdb_class.columns, self.jrdb_class.set_df(file))
            mu.move_file(file, self.finish_path + file)

    def set_file_type(self, filetype):
        """ 指定したJRDBのタイプにあった取り込みクラスを設定する

        :param str filetype:
        """
        if filetype == "BAC":
            self.jrdb_class = BAC()
        elif filetype == "CHA":
            self.jrdb_class = CHA()
        elif filetype == "CYB":
            self.jrdb_class = CYB()
        elif filetype == "JOA":
            self.jrdb_class = JOA()
        elif filetype == "KAB":
            self.jrdb_class = KAB()
        elif filetype == "KKA":
            self.jrdb_class = KKA()
        elif filetype == "KYI":
            self.jrdb_class = KYI()
        elif filetype == "UKC":
            self.jrdb_class = UKC()
        elif filetype == "SED":
            self.jrdb_class = SED()
        elif filetype == "SRB":
            self.jrdb_class = SRB()
        elif filetype == "SKB":
            self.jrdb_class = SKB()
        elif filetype == "HJC":
            self.jrdb_class = HJC()
        elif filetype == "TYB":
            self.jrdb_class = TYB()
        elif filetype == "OZ":
            self.jrdb_class = OZ()
        elif filetype == "OT":
            self.jrdb_class = OT()

        else:
            print("NA")
        print(self.jrdb_class)

    def procedure_retrieve_data_paci(self):
        """  PACIファイルの取り込みに対して、一部しか取り込みができていないファイルに対して再取り込みの処理を実施する """
        latest_file = mu.get_latest_file(self.finish_path)
        sql_proc = msf.MY_SQL_CLASS()
        row_count = sql_proc.get_kyi_target_data_record(
            latest_file, self.retrieve_check_table_name)
        if row_count < 80:
            target_date = latest_file[3:9]
            del_file_name = self.archive_path + "PACI" + target_date + ".zip"
            if os.path.exists(del_file_name):
                print("delete " + del_file_name)
                os.remove(del_file_name)

    def procedure_retrieve_data_sed(self):
        """  SEDファイルの取り込みに対して、一部しか取り込みができていないファイルに対しての再取り込みの処理を実施する  """
        sql_proc = msf.MY_SQL_CLASS()
        row = sql_proc.get_sed_target_data_row(self.sed_table_name)
        target_list = []
        for r in row:
            del_file = r[0][0:9] + ".zip"
            del_path = self.archive_path + del_file
            if os.path.exists(del_path):
                print("delete " + del_file + " file")
                os.remove(del_path)
