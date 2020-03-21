import pyodbc

end_date = '2012/12/31'

print("baoZ-O6　削除")
baoz_06_conn_str = (
    r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
    r'DBQ=C:\BaoZ\DB\MasterDB\BaoZ-O6.MDB;'
)
baoz_06_cnxn = pyodbc.connect(baoz_06_conn_str)
baoz_06_cnxn.autocommit = True
baoz_06_crsr = baoz_06_cnxn.cursor()

baoz_06_crsr.execute("DELETE FROM 三連単オッズT WHERE データ作成年月日 <= #" + end_date + "#")
baoz_06_crsr.commit()

print("baoZ-O6W　削除")
baoz_06W_conn_str = (
    r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
    r'DBQ=C:\BaoZ\DB\MasterDB\BaoZ-O6W.MDB;'
)
baoz_06W_cnxn = pyodbc.connect(baoz_06W_conn_str)
baoz_06W_cnxn.autocommit = True
baoz_06W_crsr = baoz_06W_cnxn.cursor()

baoz_06W_crsr.execute("DELETE FROM 三連単オッズWT WHERE データ作成年月日 <=#" + end_date + "#")
baoz_06W_crsr.commit()

print("baoZ-SE　削除")
baoz_SE_conn_str = (
    r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
    r'DBQ=C:\BaoZ\DB\MasterDB\BaoZ-SE.MDB;'
)
baoz_SE_cnxn = pyodbc.connect(baoz_SE_conn_str)
baoz_SE_cnxn.autocommit = True
baoz_SE_crsr = baoz_SE_cnxn.cursor()

baoz_SE_crsr.execute("DELETE FROM 出走馬マスタ WHERE 開催年月日 <=#" + end_date + "#")
baoz_SE_crsr.commit()


print("baoZSys2　削除")
baoz_Sys2_conn_str = (
    r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
    r'DBQ=C:\BaoZ\DB\MasterDB\BaoZSys2.MDB;'
)
baoz_Sys2_cnxn = pyodbc.connect(baoz_Sys2_conn_str)
baoz_Sys2_cnxn.autocommit = True
baoz_Sys2_crsr = baoz_Sys2_cnxn.cursor()

baoz_Sys2_crsr.execute("DELETE FROM コース成績T WHERE 年月日 <=#" + end_date + "#")
baoz_Sys2_crsr.commit()
baoz_Sys2_crsr.execute("DELETE FROM 騎手成績T WHERE 年月日 <=#" + end_date + "#")
baoz_Sys2_crsr.commit()
baoz_Sys2_crsr.execute("DELETE FROM 血統成績T WHERE 年月日 <=#" + end_date + "#")
baoz_Sys2_crsr.commit()
baoz_Sys2_crsr.execute("DELETE FROM 調教師成績T WHERE 年月日 <=#" + end_date + "#")
baoz_Sys2_crsr.commit()


print("baoZ-O6K　削除")
baoz_06K_conn_str = (
    r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
    r'DBQ=C:\BaoZ\DB\MasterDB\BaoZ-O6K.MDB;'
)
baoz_06K_cnxn = pyodbc.connect(baoz_06K_conn_str)
baoz_06K_cnxn.autocommit = True
baoz_06K_crsr = baoz_06K_cnxn.cursor()

baoz_06K_crsr.execute("DELETE FROM 三連単オッズKT WHERE データ作成年月日 <=#" + end_date + "#")
baoz_06K_crsr.commit()

print("baoZ-RA　削除")
baoz_ra_conn_str = (
    r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
    r'DBQ=C:\BaoZ\DB\MasterDB\baoZ-RA.MDB;'
)
baoz_ra_cnxn = pyodbc.connect(baoz_ra_conn_str)
baoz_ra_cnxn.autocommit = True
baoz_ra_crsr = baoz_ra_cnxn.cursor()


baoz_ra_crsr.execute("DELETE FROM レースマスタ WHERE 開催月日 <= #" + end_date + "#")
baoz_ra_crsr.commit()
baoz_ra_crsr.execute("DELETE FROM WIN5T WHERE 月日 <= #" + end_date + "#")
baoz_ra_crsr.commit()
baoz_ra_crsr.execute("DELETE FROM 競走馬マスタ WHERE 競走馬抹消年月日 <= #2000/01/01#")
baoz_ra_crsr.commit()
baoz_ra_crsr.execute("DELETE FROM 票数T WHERE データ作成年月日 <= #" + end_date + "#")
baoz_ra_crsr.commit()
baoz_ra_crsr.execute("DELETE FROM 払戻T WHERE データ作成年月日 <= #" + end_date + "#")
baoz_ra_crsr.commit()

print("baoZ　削除")
baoz_conn_str = (
    r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
    r'DBQ=C:\BaoZ\DB\BaoZ.MDB;'
)
baoz_cnxn = pyodbc.connect(baoz_conn_str)
baoz_cnxn.autocommit = True
baoz_crsr = baoz_cnxn.cursor()

baoz_crsr.execute("DELETE FROM レースT WHERE 月日 <= #" + end_date + "#")
baoz_crsr.commit()
baoz_crsr.execute("DELETE FROM 仮想投票記録T WHERE 日付 <= #" + end_date + "#")
baoz_crsr.commit()

print("baoZ ex　削除")
baoz_ex_conn_str = (
    r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
    r'DBQ=C:\BaoZ\DB\BaoZ.ex.MDB;'
)
baoz_ex_cnxn = pyodbc.connect(baoz_ex_conn_str)
baoz_ex_cnxn.autocommit = True
baoz_ex_crsr = baoz_ex_cnxn.cursor()

baoz_ex_crsr.execute("DELETE FROM 出走馬T WHERE 年月日 <= #" + end_date + "#")
baoz_ex_crsr.commit()