from modules.jra_extract import JRAExtract
from modules.jra_transform import JRATransform
import modules.util as mu
import my_config as mc
import pandas as pd

start_date = '2020/01/01'
end_date = '2020/05/31'
mock_flag = False
test_flag = False
ext = JRAExtract(start_date, end_date, mock_flag)
tf = JRATransform(start_date, end_date)
dict_path = mc.return_base_path(test_flag)
target_path = mc.TARGET_PATH
ext_score_path = target_path + 'ORIGINAL_DATA/'

race_base_df = ext.get_race_before_table_base()[["RACE_KEY", "NENGAPPI"]]
race_base_df.loc[:, "RACE_ID"] = race_base_df.apply(lambda x: mu.convert_jrdb_id(x["RACE_KEY"] , x["NENGAPPI"]), axis=1)
race_base_df.loc[:, "file_id"] = race_base_df["RACE_KEY"].apply(lambda x: mu.convert_target_file(x))
race_base_df.loc[:, "nichiji"] = race_base_df["RACE_KEY"].apply(lambda x: mu.convert_kaiji(x[5:6]))
race_base_df.loc[:, "race_no"] = race_base_df["RACE_KEY"].str[6:8]
race_base_df.loc[:, "rc_file_id"] = race_base_df["RACE_KEY"].apply(lambda x: "RC" + x[0:5])

update_start_date = '20200521'
update_end_date = '20200601'
update_term_df = race_base_df.query(f"NENGAPPI >= '{update_start_date}' and NENGAPPI <= '{update_end_date}'")
print(update_term_df.shape)
file_list = update_term_df["file_id"].drop_duplicates()
date_list = update_term_df["NENGAPPI"].drop_duplicates()
rc_file_list = update_term_df["rc_file_id"].drop_duplicates()

def return_mark(num):
    if num == 1: return "◎"
    if num == 2: return "○"
    if num == 3: return "▲"
    if num == 4: return "△"
    if num == 5: return "×"
    else: return "  "

def create_rm_file(df, folder_path):
    """ valの値をレース印としてファイルを作成 """
    for file in file_list:
        print(file)
        file_text = ""
        temp_df = df.query(f"file_id == '{file}'")
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
    raceuma_df.loc[:, "predict_std"] = raceuma_df["predict_std"].round(2)
    raceuma_df.loc[:, "predict_rank"] = raceuma_df["predict_rank"].astype(int)
    for file in file_list:
        print(file)
        file_text = ""
        temp_df = race_df.query(f"file_id == '{file}'")
        nichiji_list = sorted(temp_df["nichiji"].drop_duplicates())
        for nichiji in nichiji_list:
            temp2_df = temp_df.query(f"nichiji == '{nichiji}'")
            race_list = sorted(temp2_df["RACE_KEY"].drop_duplicates())
            for race in race_list:
                line_text = ""
                temp3_sr = temp2_df.query(f"RACE_KEY =='{race}'").iloc[0]
                if temp3_sr["val"] == temp3_sr["val"]:
                    line_text += temp3_sr["val"]
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

############ 予想データ作成：レース ###############
raptype_df = ext.get_pred_df("jra_rc_raptype", "RAP_TYPE")[["RACE_KEY", "CLASS", "predict_rank"]].rename(columns={"CLASS": "val"})
raptype_df.loc[:, "val"] = raptype_df["val"].apply(lambda x: mu.decode_rap_type(x))
raptype_df_1st = raptype_df.query("predict_rank == 1").groupby("RACE_KEY").first().reset_index().drop("predict_rank", axis=1)
raptype_df_1st = pd.merge(race_base_df, raptype_df_1st, on="RACE_KEY", how="left")

tb_uchisoto_df = ext.get_pred_df("jra_rc_raptype", "TRACK_BIAS_UCHISOTO")[["RACE_KEY", "CLASS", "predict_rank"]].rename(columns={"CLASS": "val"})
tb_uchisoto_df.loc[:, "val"] = tb_uchisoto_df["val"].apply(lambda x: mu._decode_uchisoto_bias(x))
tb_uchisoto_df_1st = tb_uchisoto_df.query("predict_rank == 1").groupby("RACE_KEY").first().reset_index().drop("predict_rank", axis=1)
tb_uchisoto_df_1st = pd.merge(race_base_df, tb_uchisoto_df_1st, on="RACE_KEY", how="left")

tb_zengo_df = ext.get_pred_df("jra_rc_raptype", "TRACK_BIAS_ZENGO")[["RACE_KEY", "CLASS", "predict_rank"]].rename(columns={"CLASS": "val"})
tb_zengo_df.loc[:, "val"] = tb_zengo_df["val"].apply(lambda x: mu._decode_zengo_bias(x))
tb_zengo_df_1st = tb_zengo_df.query("predict_rank == 1").groupby("RACE_KEY").first().reset_index().drop("predict_rank", axis=1)
tb_zengo_df_1st = pd.merge(race_base_df, tb_zengo_df_1st, on="RACE_KEY", how="left")


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
agari_df = ext.get_pred_df("jra_ru_nigeuma", "AGARI_SAISOKU")
ten_df = ext.get_pred_df("jra_ru_nigeuma", "TEN_SAISOKU")

score_df = pd.merge(win_df.rename(columns={"predict_std": "win_std"}), jiku_df.rename(columns={"predict_std": "jiku_std"}), on="RACEUMA_ID")
score_df = pd.merge(score_df, ana_df.rename(columns={"predict_std": "ana_std"}), on="RACEUMA_ID")
score_df.loc[:, "predict_std"] = score_df["win_std"] * 0.25 + score_df["jiku_std"] * 0.30 + score_df["ana_std"] * 0.45
grouped_score_df = score_df.groupby("RACE_KEY")
score_df.loc[:, "predict_rank"] = grouped_score_df["predict_std"].rank("dense", ascending=False)


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
race_result_df.loc[:, "val"] = race_result_df["RAP_TYPE"].apply(lambda x: mu.decode_rap_type(x))
race_result_df.loc[: ,"TB_ZENGO"] = race_result_df["TRACK_BIAS_ZENGO"].apply(lambda x: mu._decode_zengo_bias(int(mu._encode_zengo_bias(x))))
race_result_df.loc[: ,"TB_UCHISOTO"] = race_result_df["TRACK_BIAS_UCHISOTO"].apply(lambda x: mu._decode_uchisoto_bias(int(mu._calc_uchisoto_bias(x))))
race_result_df.loc[: ,"RACE_PACE"] = race_result_df["レースペース流れ"].apply(lambda x: mu._decode_race_pace(int(mu._encode_race_pace(x))))
race_result_df = race_result_df.drop_duplicates()
race_result_df = pd.merge(race_result_df, race_base_df, on="RACE_KEY")

result_uchisoto_df = race_result_df[["RACE_KEY", "TB_UCHISOTO", "file_id", "nichiji", "race_no"]].rename(columns={"TB_UCHISOTO": "val"})
result_zengo_df = race_result_df[["RACE_KEY", "TB_ZENGO", "file_id", "nichiji", "race_no"]].rename(columns={"TB_ZENGO": "val"})

raceuma_result_df = pd.merge(raceuma_result_df, race_result_df, on ="RACE_KEY")
raceuma_result_df.loc[:, "RACEUMA_ID"] = raceuma_result_df.apply(lambda x: mu.convert_jrdb_id(x["RACE_KEY"], x["target_date"]) + x["UMABAN"], axis=1)
fa_df = raceuma_result_df[["RACEUMA_ID", "fa_1", "fa_2", "fa_3", "fa_4", "fa_5", "target_date"]]


############ 予想ファイル作成：レース ###############
print("---- raptype_df --------")
raptype_mark_folder = target_path + "RMark2/"
create_rm_file(raptype_df_1st, raptype_mark_folder)
print("---- tb_uchisoto_df --------")
tb_uchisoto_mark_folder = target_path + "RMark2/"
create_rm_file(tb_uchisoto_df_1st, tb_uchisoto_mark_folder)
print("---- tb_zengo_df --------")
tb_zengo_mark_folder = target_path + "RMark3/"
create_rm_file(tb_zengo_df_1st, tb_zengo_mark_folder)

########## 予想ファイル作成：レース馬指数 ##################
print("---- win_df --------")
win_mark_path = target_path + "UmaMark2/"
create_um_mark_file(win_df, win_mark_path)
print("---- jiku_df --------")
jiku_mark_path = target_path + "UmaMark3/"
create_um_mark_file(jiku_df, jiku_mark_path)
print("---- ana_df --------")
ana_mark_path = target_path + "UmaMark4/"
create_um_mark_file(ana_df, ana_mark_path)
print("---- nigeuma_df --------")
nigeuma_mark_path = target_path + "UmaMark5/"
create_um_mark_file(nigeuma_df, nigeuma_mark_path)
print("---- agari_df --------")
agari_mark_path = target_path + "UmaMark6/"
create_um_mark_file(agari_df, agari_mark_path)
print("---- ten_df --------")
ten_mark_path = target_path + "UmaMark7/"
create_um_mark_file(ten_df, ten_mark_path)

########## 予想ファイル作成：メイン指数 ##################
print("---- score_df --------")
score_mark_path = target_path
create_main_mark_file(race_result_df, score_df, score_mark_path)

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

######### 結果ファイル作成：レース ####################
print("---- result_uchisoto_df --------")
create_rm_file(result_uchisoto_df, tb_zengo_mark_folder)
print("---- result_uchisoto_df --------")
create_rm_file(result_zengo_df, tb_zengo_mark_folder)

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

######### コメントファイル作成：レース ####################
for file in rc_file_list:
    print(file)
    race_comment_df = race_result_df.query(f"rc_file_id == '{file}'")[["RACE_KEY", "レースコメント"]].sort_values("RACE_KEY")
    race_comment_df.to_csv(target_path + "RACE_COM/20" + file[4:6] + "/" + file + ".dat", header=False, index=False, encoding="shift_jis")