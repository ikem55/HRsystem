from modules.base_sk_model import BaseSkModel
from modules.lb_sk_proc import LBSkProc

import pandas as pd
import numpy as np
import pyodbc
import sys
from datetime import datetime as dt
from datetime import timedelta



class LBSkModel(BaseSkModel):
    """
    地方競馬の機械学習モデルを定義
    """
    version_str = 'lb'
    model_path = './for_test_model/lb/'
    class_list = ['競走種別コード' , '場コード']
    ens_folder_path = './for_test_model/lb/'
    table_name = '地方競馬レース馬'
    conn_str = (
        r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
        r'DBQ=C:\BaoZ\DB\MasterDB\MyDB.MDB;'
    )

    def set_table_name(self, table_name):
        """ test用のテーブルをセットする """
        self.table_name = table_name
        self.conn_str = (
            r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
            r'DBQ=C:\BaoZ\DB\MasterDB\test_MyDB.MDB;'
        )

    def _get_skproc_object(self, version_str, start_date, end_date, model_name, mock_flag, test_flag):
        print("-- check! this is BaseSkModel class: " + sys._getframe().f_code.co_name)
        proc = LBSkProc(version_str, start_date, end_date, model_name, mock_flag, test_flag, self.obj_column_list)
        return proc

    def _import_data(self, df):
        """ 計算した予測値のdataframeを地方競馬DBに格納する

        :param dataframe df: dataframe
        """
        cnxn = pyodbc.connect(self.conn_str)
        crsr = cnxn.cursor()
        re_df = df.replace([np.inf, -np.inf], np.nan).dropna()
        date_list = df['target_date'].drop_duplicates()
        for date in date_list:
            target_df = re_df[re_df['target_date'] == date]
            crsr.execute("DELETE FROM " + self.table_name + " WHERE target_date ='" + date + "'")
            crsr.executemany(
                f"INSERT INTO " + self.table_name + " (競走コード, 馬番, 予測フラグ, 予測値, 予測値偏差, 予測値順位, target, target_date) VALUES (?,?,?,?,?,?,?,?)",
                target_df.itertuples(index=False)
            )
        cnxn.commit()

    @classmethod
    def get_recent_day(cls, start_date):
        cnxn = pyodbc.connect(cls.conn_str)
        select_sql = "SELECT target_date from " + cls.table_name
        df = pd.read_sql(select_sql, cnxn)
        if not df.empty:
            recent_date = df['target_date'].max()
            dt_recent_date = dt.strptime(recent_date, '%Y/%m/%d') + timedelta(days=1)
            print(dt_recent_date)
            changed_start_date = dt_recent_date.strftime('%Y/%m/%d')
            return changed_start_date
        else:
            return start_date