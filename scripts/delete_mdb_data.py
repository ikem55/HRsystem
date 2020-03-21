conn_str = (
    r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
    r'DBQ=C:\BaoZ\DB\MasterDB\MyDB.MDB;'
)

import pyodbc

start_date = '2020/3/15'
end_date = '2020/3/22'
cnxn = pyodbc.connect(conn_str)
crsr = cnxn.cursor()

crsr.execute("DELETE FROM 地方競馬レース馬V1 WHERE target_date>= #" + start_date + "# AND target_date <= #" + end_date + "#")
crsr.execute("DELETE FROM 地方競馬レース馬V2 WHERE target_date>= #" + start_date + "# AND target_date <= #" + end_date + "#")

cnxn.commit()