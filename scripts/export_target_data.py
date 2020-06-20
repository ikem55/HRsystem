from modules.jra_extract import JRAExtract
from modules.jra_transform import JRATransform
import modules.util as mu
import my_config as mc
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta

start_date = '2019/01/01'
#end_date = '2020/05/31'
#start_date = (dt.now() + timedelta(days=-90)).strftime('%Y/%m/%d')
end_date = (dt.now() + timedelta(days=1)).strftime('%Y/%m/%d')
mock_flag = False
test_flag = False
ext = JRAExtract(start_date, end_date, mock_flag)
tf = JRATransform(start_date, end_date)
dict_path = mc.return_jra_path(test_flag)
target_path = mc.TARGET_PATH
ext_score_path = target_path + 'ORIGINAL_DATA/'

race_base_df = ext.get_race_before_table_base()[["RACE_KEY", "NENGAPPI", "距離", "芝ダ障害コード", "内外", "条件"]]
race_base_df.loc[:, "RACE_ID"] = race_base_df.apply(lambda x: mu.convert_jrdb_id(x["RACE_KEY"] , x["NENGAPPI"]), axis=1)
race_base_df.loc[:, "file_id"] = race_base_df["RACE_KEY"].apply(lambda x: mu.convert_target_file(x))
race_base_df.loc[:, "nichiji"] = race_base_df["RACE_KEY"].apply(lambda x: mu.convert_kaiji(x[5:6]))
race_base_df.loc[:, "race_no"] = race_base_df["RACE_KEY"].str[6:8]
race_base_df.loc[:, "rc_file_id"] = race_base_df["RACE_KEY"].apply(lambda x: "RC" + x[0:5])
race_base_df.loc[:, "kc_file_id"] = "KC" + race_base_df["RACE_KEY"].str[0:6]

update_start_date = '20190101'
#update_end_date = '20200601'

#update_start_date = (dt.now() + timedelta(days=-9)).strftime('%Y%m%d')
update_end_date = (dt.now() + timedelta(days=1)).strftime('%Y%m%d')

update_term_df = race_base_df.query(f"NENGAPPI >= '{update_start_date}' and NENGAPPI <= '{update_end_date}'")
print(update_term_df.shape)
file_list = update_term_df["file_id"].drop_duplicates()
date_list = update_term_df["NENGAPPI"].drop_duplicates()
rc_file_list = update_term_df["rc_file_id"].drop_duplicates()
kc_file_list = update_term_df["kc_file_id"].drop_duplicates()

def return_mark(num):
    if num == 1: return "◎"
    if num == 2: return "○"
    if num == 3: return "▲"
    if num == 4: return "△"
    if num == 5: return "×"
    else: return "  "

def create_rm_file(df, pred_df, folder_path):
    """ valの値をレース印としてファイルを作成 """
    for file in file_list:
        print(file)
        file_text = ""
        temp_df = pred_df.query(f"file_id == '{file}'")
        nichiji_list = temp_df["nichiji"].drop_duplicates().sort_values()
        for nichiji in nichiji_list:
            line_text = ""
            temp2_df = temp_df.query(f"nichiji == '{nichiji}'").sort_values("race_no")
            race_list = sorted(temp2_df["RACE_KEY"].drop_duplicates())
            for race in race_list:
                temp3_df = df.query(f"RACE_KEY =='{race}'")
                if temp3_df.empty:
                    temp3_df = temp2_df.query(f"RACE_KEY =='{race}'")
                temp3_sr = temp3_df.iloc[0]
                if temp3_sr["val"] == temp3_sr["val"]:
                    line_text += temp3_sr["val"]
                else:
                    line_text += "      "
            file_text += line_text + "\r\n"
        with open(folder_path + "RM" + file + ".DAT", mode='w', encoding="shift-jis") as f:
            f.write(file_text.replace('\r', ''))

def create_rc_file(df, folder_path):
    """ レースコメントを作成 """
    for file in rc_file_list:
        print(file)
        file_text = ""
        temp_df = df.query(f"rc_file_id == '{file}'")[["RACE_KEY", "レースコメント"]].sort_values("RACE_KEY")
        nichiji_list = temp_df["nichiji"].drop_duplicates().sort_values()
        for nichiji in nichiji_list:
            line_text = ""
            temp2_df = temp_df.query(f"nichiji == '{nichiji}'").sort_values("race_no")
            race_list = sorted(temp2_df["RACE_KEY"].drop_duplicates())
            for race in race_list:
                temp3_sr = temp2_df.query(f"RACE_KEY =='{race}'").iloc[0]
                if temp3_sr["val"] == temp3_sr["val"]:
                    line_text += temp3_sr["val"]
                else:
                    line_text += "      "
            file_text += line_text + "\r\n"
        with open(folder_path + "RM" + file + ".DAT", mode='w', encoding="shift-jis") as f:
            f.write(file_text.replace('\r', ''))


def create_um_mark_file(df, folder_path):
    """ ランクを印にして馬印ファイルを作成 """
    df.loc[:, "RACEUMA_ID"] = df.apply(
        lambda x: mu.convert_jrdb_id(x["RACE_KEY"], x["target_date"]) + x["UMABAN"], axis=1)
    df.loc[:, "predict_std"] = df["predict_std"].round(2)
    df.loc[:, "predict_rank"] = df["predict_rank"].astype(int)
    df = pd.merge(race_base_df[["RACE_KEY", "file_id", "nichiji", "race_no"]], df, on="RACE_KEY", how="left")
    #file_list = df["file_id"].drop_duplicates()
    for file in file_list:
        print(file)
        file_text = ""
        temp_df = df.query(f"file_id == '{file}'")
        nichiji_list = sorted(temp_df["nichiji"].drop_duplicates())
        for nichiji in nichiji_list:
            temp2_df = temp_df.query(f"nichiji == '{nichiji}'")
            race_list = sorted(temp2_df["RACE_KEY"].drop_duplicates())
            for race in race_list:
                line_text = "      "
                temp3_df = temp2_df.query(f"RACE_KEY == '{race}'").sort_values("UMABAN")
                i = 0
                for idx, val in temp3_df.iterrows():
                    line_text += return_mark(val["predict_rank"])
                    i += 1
                if i != 18:
                    for j in range(i, 18):
                        line_text += "  "
                file_text += line_text + '\r\n'
        with open(folder_path + "UM" + file + ".DAT", mode='w', encoding="shift-jis") as f:
            f.write(file_text.replace('\r', ''))



def create_main_mark_file(race_df, raceuma_df, folder_path):
    """ 馬１、レース１用のファイルを作成 """
    for file in file_list:
        print(file)
        file_text = ""
        temp_df = race_base_df.query(f"file_id == '{file}'")
        nichiji_list = sorted(temp_df["nichiji"].drop_duplicates())
        for nichiji in nichiji_list:
            temp2_df = temp_df.query(f"nichiji == '{nichiji}'")
            race_list = sorted(temp2_df["RACE_KEY"].drop_duplicates())
            for race in race_list:
                line_text = ""
                temp_race_df = race_df.query(f"RACE_KEY =='{race}'")
                if not temp_race_df.empty:
                    temp3_sr = temp_race_df.iloc[0]
                    if temp3_sr["val"] == temp3_sr["val"]:
                        line_text += temp3_sr["val"]
                    else:
                        line_text += "      "
                else:
                    line_text += "      "
                temp3_df = raceuma_df.query(f"RACE_KEY == '{race}'").sort_values("UMABAN")
                i = 0
                for idx, val in temp3_df.iterrows():
                    line_text += return_mark(val["predict_rank"])
                    i += 1
                if i != 18:
                    for j in range(i, 18):
                        line_text += "  "
                file_text += line_text + '\r\n'
        with open(folder_path + "UM" + file + ".DAT", mode='w', encoding="shift-jis") as f:
            f.write(file_text.replace('\r', ''))

def create_win_flag(course_cluster, joken, win, ana, nige, ten, agari):
    """ コースクラスタ、条件ごとに勝ち率の高い（３割以上）馬に数値ラベルをつける """
    if win >= 60 or ana >= 60:
        if course_cluster == 0:
            if joken == '10' and win >= 60 and agari >= 50: return "▲"
            if joken == '16' and agari >= 50: return "☆"
            if joken == 'A1':
                if win >= 60 and (nige >= 50 or agari >= 60): return "▲"
                if agari >= 70: return "▲"
            if joken == 'OP' and win >= 60 and agari >= 60: return "▲"
        if course_cluster == 2:
            if joken == '05' and win >= 60 and agari >= 50: return "▲"
            if joken == '10' and win >= 60 and agari >= 50: return "▲"
            if joken == '16':
                if agari >= 50 or (win >= 60 and ten >= 50): return "☆"
                if win >= 60 and nige >= 50: return "○"
                if win >= 60: return "▲"
            if joken == 'A1':
                if nige <= 50 and ten <= 60 and agari <= 60: return "消"
                if nige >= 50: return "☆"
                if win >= 60 and ten >= 50 and agari >= 60: return "▲"
            if joken == 'A3' and nige <= 50 and ten <= 50 and agari <= 50: return "消"
            if joken == 'OP' and win >= 60 and agari >= 60: return "▲"
        if course_cluster == 3:
            if joken == '16':
                if ana >= 60 and ten >= 50: return "☆"
                if win >= 60 and agari >= 50: return "▲"
            if joken == 'A1':
                if nige <= 60 and ten <= 50 and agari <= 60: return "消"
                if agari >= 50 and (nige >= 50 or ten > 60): return "☆"
                if win >= 60: return "▲"
            if joken == 'A3' and nige >= 70: return "☆"
            if joken == 'OP':
                if nige >= 70 or (nige >= 60 and ten >= 70): return "☆"
                if win >= 60 and (agari >= 50 or ten >= 50): return "▲"
        if course_cluster == 4:
            if joken == '10' and (ten >= 60 or nige >= 50): return "☆"
            if joken == 'A3' and win >= 60 and (nige >= 60 or (ten >= 50 and agari >= 50)): return "▲"
        if course_cluster == 5:
            if joken == '05' and win >= 60 and agari >= 60: return  "▲"
            if joken == "16" and (agari >= 50 or (nige >= 50 and ten >= 60) or (nige >= 60 and ten >= 50)): return "☆"
            if joken == "A1":
                if nige <= 60 and ten <= 60 and agari <= 60: return "消"
                if ten >= 60 or (nige >= 50 and agari >= 50): return "☆"
                if win >= 60 or (agari >= 60 and (ten >= 50 or nige >= 50)): return "▲"
            if joken == "A3":
                if agari >= 70: return "☆"
                if win >= 60 and (nige >= 50 or ten >= 50 or agari >= 60): return "▲"
        if course_cluster == 6:
            if joken == '16' and (ten >= 60 or nige >= 60 or (nige >= 70 and ten >= 50) or (nige >= 50 and ten >= 70)): return "☆"
            if joken == 'A1' and nige <= 60 and ten <= 60 and agari <= 60: return "消"
            if joken == 'A3':
                if nige <= 50 and ten <= 50 and agari <= 50: return "消"
                if win >= 60 and agari >= 60: return "▲"
            if joken == 'OP' and win >= 60 and agari >= 50: return "▲"
        if course_cluster == 8:
            if joken == 'A1':
                if nige <= 50 and ten <= 60 and agari <= 50: return "消"
                if ten >= 60 and (nige >= 50 or agari >= 50): return "☆"
            if joken == 'A3':
                if nige <= 50 and ten <= 50 and agari <= 50: return "消"
                if nige >=60 or ten >= 70: return "☆"
            if joken == 'OP' and ana >= 60: return "☆"
        if course_cluster == 9:
            if joken == '05' and win >= 60 and agari >= 50: return "☆"
            if joken == 'A3':
                if nige <= 50 and ten <= 50 and agari <= 50: return "消"
                if agari >= 60 or nige >= 60 or ten >= 60: return "☆"
            if joken == 'OP':
                if nige >= 60 and ten >= 50: return "☆"
                if win >= 60: return "▲"
        if course_cluster == 10:
            if joken == '10' and (agari >= 70 or nige >= 70 or (nige >= 60 and ten >= 60)): return "☆"
            if joken == '16' and agari >= 70: return "☆"
            if joken == 'A1':
                if nige <= 50 and ten <= 70 and agari <= 50: return "消"
                if agari >= 50: return "☆"
                if (win >= 60 and ten >= 60) or ((nige >= 50 or ten >= 50) and agari >= 70): return "▲"
            if joken == 'A3':
                if win >= 60 and (nige >= 50 or ten >= 50 or agari >= 50): return "▲"
                if nige >= 50 and ten >= 50 and agari >= 50: return "▲"
            if joken == 'OP':
                if ten >= 50 or ten >= 60 or (ten >= 50 and nige >= 50): return "☆"
        if course_cluster == 11:
            if joken == '05' and win >= 60 and agari >= 50: return "▲"
            if joken == '10' and nige >= 50 or agari >= 60: return "☆"
            if joken == 'A1':
                if ten >= 50 or nige >= 50: return "☆"
                if win >= 60: return "◎"
                if agari >= 60 or (ten >= 50 and agari >= 50): return "▲"
            if joken == 'A3' and ((win >= 60 and agari >= 60) or agari >= 70): return "▲"
            if joken == 'OP' and agari >= 50: return "☆"
        if course_cluster == 12:
            if joken == '10' and nige >= 60: return "☆"
            if joken == '16':
                if win >= 60 and agari >= 60: return "☆"
                if win >= 60 and (nige >= 50 or ten >= 50 or agari >= 50): return "☆"
            if joken == 'A1':
                if win >= 60 and ten >= 50 and agari >= 60: return "○"
                if (win >= 60 and (nige >= 50 or ten >= 50)) or agari >= 70: return "▲"
            if joken == 'A3':
                if nige <= 50 and ten <= 50 and agari <= 50: return "消"
                if win >= 60 or (nige >= 50 and agari >= 60): return "▲"
            if joken == 'OP' and win >= 60 and agari >= 60: return "▲"
        if course_cluster == 13:
            if joken == '05' and (agari >= 70 or (win >= 60 and (agari >= 50 or nige >= 50))): return "▲"
            if joken == '10':
                if ten >= 70 or agari >= 70 or (win >= 60 and agari >= 60): return "☆"
                if win >= 60 and nige >= 50: return "▲"
            if joken == '16' and (agari >= 50 or ana >= 60): return "☆"
            if joken == 'A1' and ten >= 50: return "☆"
            if joken == 'A3':
                if nige <= 60 and ten <= 50 and agari <= 50: return "消"
                if nige >= 50 or ten >= 70: return "☆"
            if joken == 'OP':
                if nige >= 60: return "☆"
                if win >= 60: return "▲"
        if course_cluster == 14:
            if joken == '05' and win >= 60 and agari >= 60: return "▲"
            if joken == '10':
                if nige >= 60 or (nige >= 50 and ten >= 50): return "☆"
                if win >= 60: return "▲"
            if joken == '16' and (nige >= 50 or ten >= 50): return "☆"
            if joken == 'A1':
                if nige <= 60 and ten <= 60 and agari <= 60: return "消"
                if agari >= 70: return "○"
                if win >= 60: return "▲"
            if joken == 'A3' and (win >=60 or agari >= 70): return "▲"
    return "  "


def create_jiku_flag(course_cluster, joken, jiku, ana, nige, ten, agari):
    """ コースクラスタ、条件ごとに連対率の高い（５割以上）馬に数値ラベルをつける """
    if jiku >= 60 or ana >= 60:
        if course_cluster == 0:
            if joken == '05':
                if jiku >= 60 and agari >= 70: return "○"
                if jiku >= 60 or agari >= 70: return "▲"
            if joken == '10' and jiku >= 60: return "▲"
            if joken == '16' and jiku >= 60 and agari >= 50: return "▲"
            if joken == 'A1':
                if (jiku >= 60 and agari >= 60) or (nige >= 50 and agari >= 60) or agari >= 70: return "○"
                if jiku >= 60 or agari >= 60: return "▲"
            if joken == 'A3' and (jiku >= 60 or agari >= 70 or ((nige >= 50 or ten >= 50) and agari >= 60)): return "▲"
            if joken == 'OP' and jiku >= 60: return "▲"
        if course_cluster == 2:
            if joken == '05':
                if jiku >= 60 and agari >= 60: return "○"
                if jiku >= 60 or agari >= 60 or (nige >= 60 and agari >= 60): return "▲"
            if joken == '10':
                if jiku >= 60 and agari >= 70: return "○"
                if jiku >= 60 and (agari >= 50 or nige >= 60): return "▲"
            if joken == '16' and jiku >= 60: return "▲"
            if joken == 'A3':
                if ana >= 60 and agari >= 50: return "☆"
                if agari >= 50 and jiku >= 70 and ten >= 50: return "◎"
                if jiku >= 60 and agari >= 50 and (nige >= 50 or ten >= 70): return "◎"
                if jiku >= 60 and agari >= 70 and (nige >= 50 or ten >= 50): return "◎"
                if jiku >= 60 and agari >= 50 and (nige >= 50 or ten >= 50): return "○"
                if jiku >= 60 and nige >= 70: return "○"
                if (ten >= 70 and agari >= 50) or (nige >= 70 and agari >= 50): return "○"
                if jiku >= 60 or agari >= 70 or (agari >= 50 and ten >= 60): return "▲"
            if joken == 'OP' and jiku >= 60: return "▲"
        if course_cluster == 3:
            if joken == '05':
                if ten >= 70 or (agari >= 50 and (ana >= 60 or nige >= 50)): return "☆"
                if jiku >= 60 and agari >= 70: return "○"
                if jiku >= 60 and agari >= 60: return "◎"
            if joken == 'A1':
                if nige <= 60 and ten <= 50 and agari <= 50: return "消"
                if ten >= 50: return "☆"
                if jiku >= 60 or agari >= 60 or (nige >= 60 and agari >= 50): return "▲"
            if joken == 'A3':
                if jiku >= 60 and nige >= 60 and agari >= 50: return "○"
                if jiku >= 60 and (nige >= 60 or ten >= 60 or (nige >= 50 and agari >= 50)): return "▲"
                if agari >= 50 and (nige >= 60 or ten >= 60): return "▲"
            if joken == 'OP':
                if jiku >= 60 and agari >= 50: return "○"
                if jiku >= 60: return "▲"
        if course_cluster == 4:
            if joken == 'A3' and jiku >= 60 and (nige >= 50 or ten >= 50): return "▲"
        if course_cluster == 5:
            if joken == '05' and jiku >= 60 and agari >= 60: return "▲"
            if joken == '10':
                if ten >= 60: return "☆"
                if jiku >= 60: return "○"
            if joken == 'A1':
                if nige <= 50 and ten <= 60 and agari <= 50: return "消"
                if agari >= 60 and (nige >= 50 or ten >= 50): return "☆"
                if jiku >= 60 and agari >= 60: return "○"
                if jiku >= 60 or agari >= 60 or (agari >=50 and (nige >= 60 or ten >= 60)): return "▲"
            if joken == 'A3':
                if (jiku >= 60 or nige >= 50) and agari >= 60: return "○"
                if jiku >= 60 and agari >= 60 and (ten >= 50 or nige >= 50): return "○"
                if jiku >= 60 or (nige >= 60 and agari >= 50): return "▲"
        if course_cluster == 6:
            if joken == '05' and jiku >= 60 and (agari >= 70 or (agari >= 50 and (nige >= 50 or ten >= 50))): return "▲"
            if joken == '10' and jiku >= 60 and nige >= 50: return "▲"
            if joken == '16' and nige >= 60 or ten >= 60: return "☆"
            if joken == 'A1':
                if ten >= 50 and agari >= 60: return "○"
                if jiku >= 60 or agari >= 60: return "▲"
            if joken == 'A3':
                if nige <= 60 and ten <= 60 and agari <= 50: return "消"
                if jiku >= 60 and agari >= 50 and (nige >= 70 or (nige >= 60 and ten >= 60)): return "◎"
                if (nige >= 60 or ten >= 60) and agari >= 50: return "○"
                if jiku >= 60 and nige >= 60 and ten >= 60: return "○"
                if jiku >= 60 or (nige >=50 and ten >= 50 and agari >= 50): return "○"
            if joken == 'OP' and agari >= 50: return "☆"
        if course_cluster == 8:
            if joken == 'A3' and jiku >= 60 and agari >= 50: return "▲"
        if course_cluster == 10:
            if joken == '05':
                if jiku >= 60 and (nige >= 50 or ten >= 50 or agari >= 60): return "○"
                if nige >= 50 and ten >= 50 and agari >= 50: return "○"
            if joken == '10' and agari >= 60: return "☆"
            if joken == '16' and agari >= 50: return "☆"
            if joken == 'A1':
                if nige <= 60 and ten <= 60 and agari <= 60: return "消"
                if jiku >= 60 or (ten >= 50 and agari >= 60) or (ten >= 60 or agari >= 50): return "▲"
            if joken == 'A3':
                if ten >= 70 and agari >= 50: return "☆"
                if jiku >= 60 and (nige >= 50 or ten >= 50) and agari >= 60: return "◎"
                if (nige >= 50 or ten >= 50) and agari >= 70: return "◎"
                if jiku >= 60 and (nige >= 70 or ten >= 70) and agari >= 50: return "◎"
                if jiku >= 60 and (nige >= 60 or ten >= 60): return "○"
                if ((nige >= 60 or ten >= 60) and agari >= 50) or ((nige >=50 or ten >= 50) and agari >= 60): return "○"
                if jiku >= 60 or (ten >= 50 and agari >= 50) or nige >= 70: return "▲"
        if course_cluster == 11:
            if joken == '05':
                if jiku >= 60 and agari >= 60: return "○"
                if jiku >= 60: return "▲"
            if joken == 'A3':
                if (jiku >= 60 and agari >= 60) or agari >= 70: return "○"
                if jiku >= 60 or agari >= 60: return "▲"
        if course_cluster == 12:
            if joken == '05':
                if nige >= 50 and ten >= 50 and agari >= 50: return "○"
                if jiku >= 60 or (agari >= 50 and (nige >= 50 or ten >= 50)): return "▲"
            if joken == 'A3':
                if nige <= 60 and ten <= 60 and agari <= 50: return "消"
                if jiku >= 60 and nige >= 50 and agari >= 60: return "◎"
                if ((nige >= 50 or ten >= 50) and agari >= 70) or (nige >= 60 and agari >= 60): return "◎"
                if jiku >= 50 and (nige >= 50 or ten >= 50 or agari >=60): return "○"
                if agari >= 70 or (agari >= 60 and (nige >= 50 or ten >= 50)): return "○"
                if jiku >= 60 or agari >= 60 or ((agari >= 50 and (nige >= 50 or ten >= 50))): return "▲"
            if joken == 'OP' and agari >= 50: return "☆"
        if course_cluster == 13:
            if joken == '05':
                if ten >= 60: return "☆"
                if jiku >= 60 and agari >= 60: return "◎"
                if (jiku >= 60 and agari >= 50) or agari >= 70: return "○"
                if jiku >= 60 or agari >= 60: return "▲"
            if joken == 'A3':
                if nige <= 50 and ten <= 50 and agari <= 50: return "消"
                if ten >= 50: return "☆"
                if jiku >= 60 or agari >= 70 or (nige >=50 and agari >= 60): return "▲"
        if course_cluster == 14:
            if joken == '05' and jiku >= 60 or agari >= 70: return "▲"
            if joken == '10':
                if nige >= 70: return "☆"
                if jiku >= 60: return "▲"
            if joken == '16' and ten >= 50 or nige >= 60: return "☆"
            if joken == 'A1':
                if jiku >= 60 and nige >= 50: return "○"
                if agari >= 60 or jiku >= 60: return "▲"
            if joken == 'A3':
                if ana >= 60 and nige >= 60: return "☆"
                if jiku >= 60 and agari >= 50 and (ten >= 50 or nige >= 50): return "◎"
                if nige >= 50 and agari >= 60: return "◎"
                if jiku >= 60 and (agari >= 70 or nige >= 60 or (nige >= 50 and agari >= 50)): return "○"
                if jiku >= 60 or agari >=60 or (ten >= 50 and agari > 50): return "▲"
            if joken == 'OP' and jiku >= 60 and agari >= 50: return "▲"
    return "  "

def create_tb_zg_flag(course_cluster, tb_zg, jiku, ana, nige, ten, agari):
    """ コースクラスタ、条件ごとに連対率の高い（５割以上）馬に数値ラベルをつける """
    if jiku >= 60 or ana >= 60:
        if course_cluster == 0:
            if tb_zg == '01　前':
                if jiku >= 60: return "◎"
                if agari >= 60: return "○"
        if course_cluster == 2:
            if tb_zg == '01　前':
                if ana >= 60 and nige >= 60: return "☆"
                if jiku >= 60 or agari >= 70 or (nige >= 60 and ten >= 60 and agari >= 60): return "○"
            if tb_zg == '02超後' and jiku >= 60 and agari >= 60: return "☆"
        if course_cluster == 3:
            if tb_zg == '01　前':
                if nige >= 60: return "☆"
                if jiku >= 60 or agari >= 60: return "◎"
        if course_cluster == 6:
            if tb_zg == '01　前':
                if nige <= 60 and ten <= 60 and agari <= 60: return "消"
                if nige >= 60 and agari >= 60: return "◎"
                if (jiku >= 60 and (nige >= 60 or agari >= 60)) or (ten >= 60 and agari >= 60): return "○"
            if tb_zg == '02超後' and agari >= 60: return "☆"
        if course_cluster == 10:
            if tb_zg == '00超前':
                if nige <= 60 and ten <= 60 and agari <= 60: return "消"
                if jiku >= 60 and (nige >= 60 or ten >= 60 and agari >= 60): return "◎"
                if ten >= 60 and (nige >= 70 or agari >= 60): return "◎"
                if ten >= 70 or nige >= 70: return "○"
            if tb_zg == '01　前':
                if nige <= 60 and ten <= 60 and agari <= 60: return "消"
                if jiku >= 60: return "○"
            if tb_zg == '02超後' and agari >= 60: return "☆"
        if course_cluster == 11:
            if tb_zg == '01　前': return "☆"
        if course_cluster == 12:
            if tb_zg == '01　前':
                if nige >= 60 and ten >= 60 and agari >= 60: return "消"
                if jiku >= 60 or agari >= 70: return "○"
        if course_cluster == 13:
            if tb_zg == '01　前' and jiku >= 60 and agari >= 60: return "○"
        if course_cluster == 14:
            if tb_zg == '01　前':
                if jiku >= 60: return "◎"
                if agari >= 60: return "○"
            if tb_zg == '02超後' and (nige >= 60 and ten >= 60): return "☆"
    return "  "

def create_tb_us_flag(course_cluster, tb_us, jiku, ana, nige, ten, agari):
    """ コースクラスタ、条件ごとに連対率の高い（５割以上）馬に数値ラベルをつける """
    if jiku >= 60 or ana >= 60:
        if course_cluster == 0:
            if tb_us == '01　内' and jiku >= 60 and agari >= 60: return "○"
            if tb_us == '03　外':
                if jiku >= 60 and agari >= 60: return "◎"
                if jiku >= 60 or agari >= 60: return "○"
        if course_cluster == 2:
            if tb_us == '01　内' and (jiku >= 60 or agari >= 60) and (nige >= 60 or ten >= 60): return "○"
            if tb_us == '03　外' and (jiku >= 60 or agari >= 60): return "◎"
        if course_cluster == 4:
            if tb_us == '01　内' and jiku >= 60 and nige >= 60: return "◎"
        if course_cluster == 5:
            if tb_us == '01　内' and ana >= 60 and ten >= 60: return "☆"
        if course_cluster == 6:
            if tb_us == '01　内':
                if agari >= 60 and (nige >= 60 or (ten >= 60 and jiku >= 60)): return "◎"
                if jiku >= 60 and (ten >= 60 or nige >= 70 or agari >= 70): return "○"
                if ten >= 60 and agari >= 60: return "○"
        if course_cluster == 10:
            if tb_us == '01　内':
                if ana >= 60 and ten >= 60: return "☆"
                if nige >= 60 and agari >= 60: return "◎"
                if jiku >= 60 and (nige >= 60 or agari >= 60 or ten >= 70): return "○"
                if ten >= 60 and agari >= 60: return "○"
        if course_cluster == 11:
            if tb_us == '01　内':
                if agari >= 60: return "☆"
                if jiku >= 60 and agari >= 60: return "○"
        if course_cluster == 12:
            if tb_us == '01　内':
                if nige <= 60 and ten <= 60 and agari <= 60: return "消"
                if jiku >= 60 and (nige >= 60 or ten >= 60): return "○"
                if nige >= 60 and agari >= 60: return "○"
        if course_cluster == 13:
            if tb_us == '01　内' and ten >= 60: return "☆"
        if course_cluster == 14:
            if tb_us == '01　内':
                if (nige >= 70 and ten >= 70) or (jiku >= 60 and nige >= 60 and ten >= 60) or (ana >= 60 and nige >= 70 and ten >= 60): return "☆"
                if jiku >= 60 and (ten >= 60 or nige >= 60 or agari >= 60): return "○"
            if tb_us == '03　外':
                if jiku >= 60 or agari >= 60: return "○"
    return "  "


def create_ana_flag(course_cluster, joken, ana_std, nige_std, ten_std, agari_std):
    """ コースクラスタ、条件ごとに複勝率の高い（２割以上）穴馬に数値ラベルをつける """
    if ana_std > 65:
        if course_cluster == 0:
            if joken == '05':
                if ten_std >= 50: return " 2"
            if joken == 'A3':
                if agari_std >= 50: return " 2"
        if course_cluster == 2:
            if joken == '05' and ten_std >= 50: return " 2"
            if joken == '10': return " 2"
            if joken == 'A3':
                if agari_std >= 50 or ten_std >= 60: return " 2"
        if course_cluster == 3:
            if joken == '10': return " 2"
            if joken == 'A3': return " 2"
        if course_cluster == 6 and joken == 'A3' and ten_std >= 50: return " 2"
        if course_cluster == 10 and joken == '05' and (nige_std >= 50 or ten_std >= 50): return " 2"
        if course_cluster == 11 and joken in ('05', 'A3') : return " 2"
        if course_cluster == 12 and joken == '05': return " 2"
        if course_cluster == 13 and joken == '05': return " 2"
        if course_cluster == 14 and joken == '05': return " 2"
        if course_cluster == 14 and joken == 'A3' and (nige_std >= 50 or ten_std >= 50): return " 2"
    return "  "


def create_um_mark_file_for_pickup(df, folder_path, target):
    """ ランクを印にして馬印ファイルを作成。targetは勝、軸、穴 """
    df.loc[:, "RACEUMA_ID"] = df.apply(
        lambda x: mu.convert_jrdb_id(x["RACE_KEY"], x["target_date"]) + x["UMABAN"], axis=1)
    df = pd.merge(race_base_df[["RACE_KEY", "file_id", "nichiji", "race_no"]], df, on="RACE_KEY", how="left")
    for file in file_list:
        print(file)
        file_text = ""
        temp_df = df.query(f"file_id == '{file}'")
        nichiji_list = sorted(temp_df["nichiji"].drop_duplicates())
        for nichiji in nichiji_list:
            temp2_df = temp_df.query(f"nichiji == '{nichiji}'")
            race_list = sorted(temp2_df["RACE_KEY"].drop_duplicates())
            for race in race_list:
                line_text = "      "
                temp3_df = temp2_df.query(f"RACE_KEY == '{race}'").sort_values("UMABAN")
                i = 0
                for idx, val in temp3_df.iterrows():
                    if target == "勝":
                        line_text += create_win_flag(val["course_cluster"], val["条件"], val["win_std"], val["ana_std"], val["nige_std"], val["ten_std"], val["agari_std"])
                    elif target == "軸":
                        line_text += create_jiku_flag(val["course_cluster"], val["条件"], val["jiku_std"], val["ana_std"], val["nige_std"], val["ten_std"], val["agari_std"])
                    elif target == "内外":
                        line_text += create_tb_us_flag(val["course_cluster"], val["tb_us"], val["jiku_std"], val["ana_std"], val["nige_std"], val["ten_std"], val["agari_std"])
                    elif target == "前後":
                        line_text += create_tb_zg_flag(val["course_cluster"], val["tb_zg"], val["jiku_std"], val["ana_std"], val["nige_std"], val["ten_std"], val["agari_std"])
                    i += 1
                if i != 18:
                    for j in range(i, 18):
                        line_text += "  "
                file_text += line_text + '\r\n'
        with open(folder_path + "UM" + file + ".DAT", mode='w', encoding="shift-jis") as f:
            f.write(file_text.replace('\r', ''))


############ 予想データ作成：レース ###############
raptype_df = ext.get_pred_df("jra_rc_raptype", "RAP_TYPE")[["RACE_KEY", "CLASS", "predict_rank"]].rename(columns={"CLASS": "val"})
raptype_df.loc[:, "val"] = raptype_df["val"].apply(lambda x: mu.decode_rap_type(x))
raptype_df_1st = raptype_df.query("predict_rank == 1").groupby("RACE_KEY").first().reset_index().drop("predict_rank", axis=1)
raptype_df_1st = pd.merge(race_base_df, raptype_df_1st, on="RACE_KEY", how="left")

tb_uchisoto_df = ext.get_pred_df("jra_rc_raptype", "TRACK_BIAS_UCHISOTO")[["RACE_KEY", "CLASS", "predict_rank"]].rename(columns={"CLASS": "val"})
tb_uchisoto_df.loc[:, "val"] = tb_uchisoto_df["val"].apply(lambda x: mu._decode_uchisoto_bias(x))
tb_uchisoto_df_1st = tb_uchisoto_df.query("predict_rank == 1").groupby("RACE_KEY").first().reset_index().drop("predict_rank", axis=1)
#tb_uchisoto_df_1st = pd.merge(race_base_df, tb_uchisoto_df_1st, on="RACE_KEY", how="left")

tb_zengo_df = ext.get_pred_df("jra_rc_raptype", "TRACK_BIAS_ZENGO")[["RACE_KEY", "CLASS", "predict_rank"]].rename(columns={"CLASS": "val"})
tb_zengo_df.loc[:, "val"] = tb_zengo_df["val"].apply(lambda x: mu._decode_zengo_bias(x))
tb_zengo_df_1st = tb_zengo_df.query("predict_rank == 1").groupby("RACE_KEY").first().reset_index().drop("predict_rank", axis=1)
#tb_zengo_df_1st = pd.merge(race_base_df, tb_zengo_df_1st, on="RACE_KEY", how="left")

tb_df = pd.merge(tb_uchisoto_df_1st.rename(columns={"val": "uc"}), tb_zengo_df_1st.rename(columns={"val": "zg"}), on="RACE_KEY")
tb_df = pd.merge(race_base_df, tb_df, on="RACE_KEY", how="left")
tb_df.loc[:, "val"] = tb_df.apply(lambda x: mu.convert_bias(x["uc"], x["zg"]), axis=1)

umaren_are_df = ext.get_pred_df("jra_rc_haito", "UMAREN_ARE")[["RACE_KEY", "pred"]].rename(columns={"pred": "umaren_are"})
umatan_are_df = ext.get_pred_df("jra_rc_haito", "UMATAN_ARE")[["RACE_KEY", "pred"]].rename(columns={"pred": "umatan_are"})
sanrenpuku_are_df = ext.get_pred_df("jra_rc_haito", "SANRENPUKU_ARE")[["RACE_KEY", "pred"]].rename(columns={"pred": "sanrenpuku_are"})
are_df = pd.merge(umaren_are_df, umatan_are_df, on="RACE_KEY")
are_df = pd.merge(are_df, sanrenpuku_are_df, on="RACE_KEY")
are_df = pd.merge(race_base_df, are_df, on="RACE_KEY", how="left")
are_df.loc[:, "val"] = are_df.apply(lambda x: mu.convert_are_flag(x["umaren_are"], x["umatan_are"], x["sanrenpuku_are"]), axis=1)

########## 予想データ作成：レース馬指数 ##################
win_df = ext.get_pred_df("jra_ru_mark", "WIN_FLAG")
win_df.loc[:, "RACEUMA_ID"] = win_df.apply(lambda x: mu.convert_jrdb_id(x["RACE_KEY"], x["target_date"]) + x["UMABAN"], axis=1)
win_df.loc[:, "predict_std"] = round(win_df["predict_std"],2)
win_df.loc[:, "predict_rank"] = win_df["predict_rank"].astype(int)

jiku_df = ext.get_pred_df("jra_ru_mark", "JIKU_FLAG")
jiku_df.loc[:, "RACEUMA_ID"] = jiku_df.apply(lambda x: mu.convert_jrdb_id(x["RACE_KEY"], x["target_date"]) + x["UMABAN"], axis=1)
jiku_df.loc[:, "predict_std"] = round(jiku_df["predict_std"],2)
jiku_df.loc[:, "predict_rank"] = jiku_df["predict_rank"].astype(int)

ana_df = ext.get_pred_df("jra_ru_mark", "ANA_FLAG")
ana_df.loc[:, "RACEUMA_ID"] = ana_df.apply(lambda x: mu.convert_jrdb_id(x["RACE_KEY"], x["target_date"]) + x["UMABAN"], axis=1)
ana_df.loc[:, "predict_std"] = round(ana_df["predict_std"],2)
ana_df.loc[:, "predict_rank"] = ana_df["predict_rank"].astype(int)

nigeuma_df = ext.get_pred_df("jra_ru_nigeuma", "NIGEUMA")
nigeuma_df.loc[:, "RACEUMA_ID"] = nigeuma_df.apply(lambda x: mu.convert_jrdb_id(x["RACE_KEY"], x["target_date"]) + x["UMABAN"], axis=1)
nigeuma_df.loc[:, "predict_std"] = round(nigeuma_df["predict_std"],2)
nigeuma_df.loc[:, "predict_rank"] = nigeuma_df["predict_rank"].astype(int)
agari_df = ext.get_pred_df("jra_ru_nigeuma", "AGARI_SAISOKU")
agari_df.loc[:, "RACEUMA_ID"] = agari_df.apply(lambda x: mu.convert_jrdb_id(x["RACE_KEY"], x["target_date"]) + x["UMABAN"], axis=1)
agari_df.loc[:, "predict_std"] = round(agari_df["predict_std"],2)
agari_df.loc[:, "predict_rank"] = agari_df["predict_rank"].astype(int)
ten_df = ext.get_pred_df("jra_ru_nigeuma", "TEN_SAISOKU")
ten_df.loc[:, "RACEUMA_ID"] = ten_df.apply(lambda x: mu.convert_jrdb_id(x["RACE_KEY"], x["target_date"]) + x["UMABAN"], axis=1)
ten_df.loc[:, "predict_std"] = round(ten_df["predict_std"],2)
ten_df.loc[:, "predict_rank"] = ten_df["predict_rank"].astype(int)

score_df = pd.merge(win_df[["RACE_KEY", "UMABAN", "RACEUMA_ID", "predict_std", "target_date"]].rename(columns={"predict_std": "win_std"}), jiku_df[["RACEUMA_ID", "predict_std"]].rename(columns={"predict_std": "jiku_std"}), on="RACEUMA_ID")
score_df = pd.merge(score_df, ana_df[["RACEUMA_ID", "predict_std"]].rename(columns={"predict_std": "ana_std"}), on="RACEUMA_ID")
score_df.loc[:, "predict_std"] = score_df["win_std"] * 0.25 + score_df["jiku_std"] * 0.30 + score_df["ana_std"] * 0.45
grouped_score_df = score_df.groupby("RACE_KEY")
score_df.loc[:, "predict_rank"] = grouped_score_df["predict_std"].rank("dense", ascending=False)
score_df.loc[:, "predict_std"] = round(score_df["predict_std"],2)
score_df.loc[:, "predict_rank"] = score_df["predict_rank"].astype(int)

uma_mark_df = pd.merge(win_df[["RACE_KEY", "UMABAN", "predict_std", "target_date"]].rename(columns={"predict_std": "win_std"}), jiku_df[["RACE_KEY", "UMABAN", "predict_std"]].rename(columns={"predict_std": "jiku_std"}), on =["RACE_KEY", "UMABAN"])
uma_mark_df = pd.merge(uma_mark_df, ana_df[["RACE_KEY", "UMABAN", "predict_std"]].rename(columns={"predict_std": "ana_std"}), on =["RACE_KEY", "UMABAN"])
uma_mark_df = pd.merge(uma_mark_df, nigeuma_df[["RACE_KEY", "UMABAN", "predict_std"]].rename(columns={"predict_std": "nige_std"}), on =["RACE_KEY", "UMABAN"])
uma_mark_df = pd.merge(uma_mark_df, agari_df[["RACE_KEY", "UMABAN", "predict_std"]].rename(columns={"predict_std": "agari_std"}), on =["RACE_KEY", "UMABAN"])
uma_mark_df = pd.merge(uma_mark_df, ten_df[["RACE_KEY", "UMABAN", "predict_std"]].rename(columns={"predict_std": "ten_std"}), on =["RACE_KEY", "UMABAN"])
race_course_df = tf.cluster_course_df(race_base_df, dict_path)[["RACE_KEY", "course_cluster", "条件"]].copy()
uma_mark_df = pd.merge(uma_mark_df, race_course_df, on="RACE_KEY")
uma_mark_df = pd.merge(uma_mark_df, tb_uchisoto_df_1st.rename(columns={"val": "tb_us"}), on="RACE_KEY")
uma_mark_df = pd.merge(uma_mark_df, tb_zengo_df_1st.rename(columns={"val": "tb_zg"}), on="RACE_KEY")

######### 結果データ作成 ####################
race_table_base_df = ext.get_race_table_base().drop(["馬場状態", "target_date", "距離", "芝ダ障害コード", "頭数"], axis=1)
race_table_base_df.loc[:, "RACE_ID"] = race_table_base_df.apply(lambda x: mu.convert_jrdb_id(x["RACE_KEY"] , x["NENGAPPI"]), axis=1)
race_table_base_df.loc[:, "file_id"] = race_table_base_df["RACE_KEY"].apply(lambda x: mu.convert_target_file(x))
race_table_base_df.loc[:, "nichiji"] = race_table_base_df["RACE_KEY"].apply(lambda x: mu.convert_kaiji(x[5:6]))
race_table_base_df.loc[:, "race_no"] = race_table_base_df["RACE_KEY"].str[6:8]
raceuma_table_base_df = ext.get_raceuma_table_base()
result_df = pd.merge(race_table_base_df, raceuma_table_base_df, on="RACE_KEY")
result_df.loc[:, "距離"] = result_df["距離"].astype(int)

cluster_raceuma_result_df = tf.cluster_raceuma_result_df(result_df, dict_path)
factory_analyze_race_result_df = tf.factory_analyze_race_result_df(result_df, dict_path)

raceuma_result_df = cluster_raceuma_result_df[["RACE_KEY", "UMABAN", "ru_cluster", "ＩＤＭ結果", "レース馬コメント"]]
race_result_df = factory_analyze_race_result_df[["RACE_KEY", "target_date", "fa_1", "fa_2", "fa_3", "fa_4", "fa_5", "RAP_TYPE", "TRACK_BIAS_ZENGO", "TRACK_BIAS_UCHISOTO", "レースペース流れ", "レースコメント"]]

race_result_df.loc[:, "val"] = race_result_df["RAP_TYPE"].apply(lambda x: mu.decode_rap_type(int(mu.encode_rap_type(x))))
race_result_df.loc[: ,"TB_ZENGO"] = race_result_df["TRACK_BIAS_ZENGO"].apply(lambda x: mu._decode_zengo_bias(int(mu._encode_zengo_bias(x))))
race_result_df.loc[: ,"TB_UCHISOTO"] = race_result_df["TRACK_BIAS_UCHISOTO"].apply(lambda x: mu._decode_uchisoto_bias(int(mu._calc_uchisoto_bias(x))))
race_result_df.loc[: ,"RACE_PACE"] = race_result_df["レースペース流れ"].apply(lambda x: mu._decode_race_pace(int(mu._encode_race_pace(x))))
race_result_df.loc[: ,"TB"] = race_result_df.apply(lambda x: mu.convert_bias(x["TB_UCHISOTO"], x["TB_ZENGO"]), axis=1)
race_result_df = race_result_df.groupby("RACE_KEY").first().reset_index()
race_result_df = pd.merge(race_result_df, race_base_df, on="RACE_KEY")

result_uchisoto_df = race_result_df[["RACE_KEY", "TB_UCHISOTO", "file_id", "nichiji", "race_no"]].rename(columns={"TB_UCHISOTO": "val"})
result_zengo_df = race_result_df[["RACE_KEY", "TB_ZENGO", "file_id", "nichiji", "race_no"]].rename(columns={"TB_ZENGO": "val"})
result_tb_df = race_result_df[["RACE_KEY", "TB", "file_id", "nichiji", "race_no"]].rename(columns={"TB": "val"})

raceuma_result_df = pd.merge(raceuma_result_df, race_result_df, on ="RACE_KEY")


raceuma_result_df.loc[:, "RACEUMA_ID"] = raceuma_result_df.apply(lambda x: mu.convert_jrdb_id(x["RACE_KEY"], x["target_date"]) + x["UMABAN"], axis=1)
fa_df = raceuma_result_df[["RACEUMA_ID", "fa_1", "fa_2", "fa_3", "fa_4", "fa_5", "target_date"]]


########## 予想ファイル作成：レース馬指数 ##################
print("---- 勝マーク --------")
win_mark_path = target_path + "UmaMark2/"
#create_um_mark_file(win_df, win_mark_path)
create_um_mark_file_for_pickup(uma_mark_df, win_mark_path, "勝")
print("---- 軸マーク --------")
jiku_mark_path = target_path + "UmaMark3/"
create_um_mark_file_for_pickup(uma_mark_df, jiku_mark_path, "軸")
#create_um_mark_file(jiku_df, jiku_mark_path)
print("---- バイアス（内外）マーク --------")
tb_us_mark_path = target_path + "UmaMark4/"
create_um_mark_file_for_pickup(uma_mark_df, tb_us_mark_path, "内外")
#create_um_mark_file(ana_df, ana_mark_path)
print("---- バイアス（前後）マーク --------")
tb_zg_mark_path = target_path + "UmaMark5/"
create_um_mark_file_for_pickup(uma_mark_df, tb_zg_mark_path, "前後")
#create_um_mark_file(ana_df, ana_mark_path)
print("---- nigeuma_df --------")
nigeuma_mark_path = target_path + "UmaMark6/"
create_um_mark_file(nigeuma_df, nigeuma_mark_path)
print("---- agari_df --------")
agari_mark_path = target_path + "UmaMark7/"
create_um_mark_file(agari_df, agari_mark_path)
#print("---- ten_df --------")
#ten_mark_path = target_path + "UmaMark7/"
#create_um_mark_file(ten_df, ten_mark_path)

########## 予想ファイル作成：メイン指数 ##################
print("---- score_df --------")
score_mark_path = target_path
main_raceuma_df = pd.merge(score_df, race_base_df, on ="RACE_KEY")
create_main_mark_file(are_df, main_raceuma_df, score_mark_path)


print("---- 予想外部指数作成 --------")
for date in date_list:
    print(date)
    win_temp_df = win_df.query(f"target_date == '{date}'")[["RACEUMA_ID", "predict_std", "predict_rank"]].sort_values("RACEUMA_ID")
    win_temp_df.to_csv(ext_score_path + "pred_win/" + date + ".csv", header=False, index=False)
    jiku_temp_df = jiku_df.query(f"target_date == '{date}'")[["RACEUMA_ID", "predict_std", "predict_rank"]].sort_values("RACEUMA_ID")
    jiku_temp_df.to_csv(ext_score_path + "pred_jiku/" + date + ".csv", header=False, index=False)
    ana_temp_df = ana_df.query(f"target_date == '{date}'")[["RACEUMA_ID", "predict_std", "predict_rank"]].sort_values("RACEUMA_ID")
    ana_temp_df.to_csv(ext_score_path + "pred_ana/" + date + ".csv", header=False, index=False)
    score_temp_df = score_df.query(f"target_date == '{date}'")[["RACEUMA_ID", "predict_std", "predict_rank"]].sort_values("RACEUMA_ID")
    score_temp_df.to_csv(ext_score_path + "pred_score/" + date + ".csv", header=False, index=False)
    nigeuma_temp_df = nigeuma_df.query(f"target_date == '{date}'")[["RACEUMA_ID", "predict_std", "predict_rank"]].sort_values("RACEUMA_ID")
    nigeuma_temp_df.to_csv(ext_score_path + "pred_nige/" + date + ".csv", header=False, index=False)
    agari_temp_df = agari_df.query(f"target_date == '{date}'")[["RACEUMA_ID", "predict_std", "predict_rank"]].sort_values("RACEUMA_ID")
    agari_temp_df.to_csv(ext_score_path + "pred_agari/" + date + ".csv", header=False, index=False)
    ten_temp_df = ten_df.query(f"target_date == '{date}'")[["RACEUMA_ID", "predict_std", "predict_rank"]].sort_values("RACEUMA_ID")
    ten_temp_df.to_csv(ext_score_path + "pred_ten/" + date + ".csv", header=False, index=False)


######### レース印ファイル作成 ####################
print("---- tb_df --------")
tb_mark_folder = target_path + "RMark2/"
create_rm_file(result_tb_df, tb_df, tb_mark_folder)
print("---- result_rap_type --------")
raptype_mark_folder = target_path + "RMark3/"
create_rm_file(race_result_df, raptype_df_1st, raptype_mark_folder)

print("---- 結果外部指数作成 --------")
for date in date_list:
    print(date)
    temp_df = fa_df.query(f"target_date == '{date}'")
    fa1_df = temp_df[["RACEUMA_ID", "fa_1"]]
    fa1_df.loc[:, "fa_1"] = round(fa1_df["fa_1"] * 10, 2)
    fa1_df.to_csv(ext_score_path + "fa_1/" + date + ".csv", header=False, index=False)
    fa2_df = temp_df[["RACEUMA_ID", "fa_2"]]
    fa2_df.loc[:, "fa_2"] = round(fa2_df["fa_2"] * 10, 2)
    fa2_df.to_csv(ext_score_path + "fa_2/" + date + ".csv", header=False, index=False)
    fa3_df = temp_df[["RACEUMA_ID", "fa_3"]]
    fa3_df.loc[:, "fa_3"] = round(fa3_df["fa_3"] * 10, 2)
    fa3_df.to_csv(ext_score_path + "fa_3/" + date + ".csv", header=False, index=False)
    fa4_df = temp_df[["RACEUMA_ID", "fa_4"]]
    fa4_df.loc[:, "fa_4"] = round(fa4_df["fa_4"] * 10, 2)
    fa4_df.to_csv(ext_score_path + "fa_4/" + date + ".csv", header=False, index=False)
    fa5_df = temp_df[["RACEUMA_ID", "fa_5"]]
    fa5_df.loc[:, "fa_5"] = round(fa5_df["fa_5"] * 10, 2)
    fa5_df.to_csv(ext_score_path + "fa_5/" + date + ".csv", header=False, index=False)

print("---- result ru_cluster --------")
ru_cluster_path = target_path + "UmaMark8/"
for file in file_list:
    print(file)
    file_text = ""
    temp_df = raceuma_result_df.query(f"file_id == '{file}'")
    nichiji_list = sorted(temp_df["nichiji"].drop_duplicates())
    for nichiji in nichiji_list:
        temp2_df = temp_df.query(f"nichiji == '{nichiji}'")
        race_list = sorted(temp2_df["RACE_KEY"].drop_duplicates())
        for race in race_list:
            line_text = "      "
            temp3_df = temp2_df.query(f"RACE_KEY == '{race}'").sort_values("UMABAN")
            i = 0
            for idx, val in temp3_df.iterrows():
                if len(str(val["ru_cluster"])) == 1:
                    line_text += ' ' + str(val["ru_cluster"])
                else:
                    line_text += '  '
                i += 1
            if i != 18:
                for j in range(i, 18):
                    line_text += "  "
            file_text += line_text + '\r\n'
    with open(ru_cluster_path + "UM" + file + ".DAT", mode='w', encoding="shift-jis") as f:
        f.write(file_text.replace('\r', ''))


print("---- コメントファイル作成 --------")
######### コメントファイル作成：レース ####################
for file in rc_file_list:
    print(file)
    race_comment_df = race_result_df.query(f"rc_file_id == '{file}'")[["RACE_KEY", "レースコメント"]].sort_values("RACE_KEY")
    race_comment_df.to_csv(target_path + "RACE_COM/20" + file[4:6] + "/" + file + ".dat", header=False, index=False, encoding="cp932")


######### コメントファイル作成：レース馬 ####################
for file in kc_file_list:
    print(file)
    race_comment_df = raceuma_result_df.query(f"kc_file_id == '{file}'")[["RACE_KEY", "UMABAN", "レース馬コメント"]]
    race_comment_df.loc[:, "RACE_UMA_KEY"] = race_comment_df["RACE_KEY"] + race_comment_df["UMABAN"]
    race_comment_df = race_comment_df[["RACE_UMA_KEY", "レース馬コメント"]].sort_values("RACE_UMA_KEY")
    race_comment_df.to_csv(target_path + "KEK_COM/20" + file[4:6] + "/" + file + ".dat", header=False, index=False, encoding="cp932")
