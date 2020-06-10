import os
import shutil
import urllib.request
import zipfile
import glob
import numpy as np
import pandas as pd
import pickle
import math

from sklearn.preprocessing import MinMaxScaler, StandardScaler
import category_encoders as ce
import bisect
from sklearn.preprocessing import LabelEncoder

def scale_df_for_fa(df, mmsc_columns, mmsc_dict_name, stdsc_columns, stdsc_dict_name, dict_folder):

    mmsc_filename = dict_folder + mmsc_dict_name + '.pkl'
    if os.path.exists(mmsc_filename):
        mmsc = load_dict(mmsc_dict_name, dict_folder)
        stdsc = load_dict(stdsc_dict_name, dict_folder)
    else:
        mmsc = MinMaxScaler()
        stdsc = StandardScaler()
        mmsc.fit(df[mmsc_columns])
        stdsc.fit(df[stdsc_columns])
        save_dict(mmsc, mmsc_dict_name, dict_folder)
        save_dict(stdsc, stdsc_dict_name, dict_folder)
    mmsc_norm = pd.DataFrame(mmsc.transform(df[mmsc_columns]), columns=mmsc_columns)
    stdsc_norm = pd.DataFrame(stdsc.transform(df[stdsc_columns]), columns=stdsc_columns)
    other_df = df.drop(mmsc_columns, axis=1)
    other_df = other_df.drop(stdsc_columns, axis=1)
    norm_df = pd.concat([mmsc_norm, stdsc_norm, other_df], axis=1)
    return norm_df

def save_dict(dict, dict_name, dict_folder):
    """ エンコードした辞書を保存する

    :param dict dict: エンコード辞書
    :param str dict_name: 辞書名
    """
    if not os.path.exists(dict_folder):
        os.makedirs(dict_folder)
    with open(dict_folder + dict_name + '.pkl', 'wb') as f:
        print("save dict:" + dict_folder + dict_name)
        pickle.dump(dict, f)

def load_dict(dict_name, dict_folder):
    """ エンコードした辞書を呼び出す

    :param str dict_name: 辞書名
    :return: encodier
    """
    with open(dict_folder + dict_name + '.pkl', 'rb') as f:
        return pickle.load(f)

def onehot_eoncoding(df, oh_columns, dict_name, dict_folder):
    """ dfを指定したEncoderでdataframeに置き換える

    :param dataframe df_list_oh: エンコードしたいデータフレーム
    :param str dict_name: 辞書の名前
    :param str encode_type: エンコードのタイプ。OneHotEncoder or HashingEncoder
    :param int num: HashingEncoder時のコンポーネント数
    :return: dataframe
    """
    encoder = ce.OneHotEncoder(cols=oh_columns, handle_unknown='impute')
    filename = dict_folder + dict_name + '.pkl'
    oh_df = df[oh_columns].astype('str')
    if os.path.exists(filename):
        ce_fit = load_dict(dict_name, dict_folder)
    else:
        ce_fit = encoder.fit(oh_df)
        save_dict(ce_fit, dict_name, dict_folder)
    df_ce = ce_fit.transform(oh_df)
    other_df = df.drop(oh_columns, axis=1)
    return_df = pd.concat([other_df, df_ce], axis=1)
    return return_df

def hash_eoncoding(df, oh_columns, num, dict_name, dict_folder):
    """ dfを指定したEncoderでdataframeに置き換える

    :param dataframe df_list_oh: エンコードしたいデータフレーム
    :param str dict_name: 辞書の名前
    :param str encode_type: エンコードのタイプ。OneHotEncoder or HashingEncoder
    :param int num: HashingEncoder時のコンポーネント数
    :return: dataframe
    """
    encoder = ce.HashingEncoder(cols=oh_columns, n_components=num)
    filename = dict_folder + dict_name + '.pkl'
    oh_df = df[oh_columns].astype('str')
    if os.path.exists(filename):
        ce_fit = load_dict(dict_name, dict_folder)
    else:
        ce_fit = encoder.fit(oh_df)
        save_dict(ce_fit, dict_name, dict_folder)
    #print(dir(ce_fit))
    df_ce = ce_fit.transform(oh_df)
    df_ce.columns = [dict_name + '_' + str(x) for x in list(range(num))]
    other_df = df.drop(oh_columns, axis=1)
    return_df = pd.concat([other_df, df_ce.astype(str)], axis=1)
    return return_df



def setup_basic_auth(url, id, pw):
    """ ベーシック認証のあるURLにアクセス

    :param str url:
    """
    password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    password_mgr.add_password(
        realm=None, uri = url, user = id, passwd = pw
    )
    auth_handler = urllib.request.HTTPBasicAuthHandler(password_mgr)
    opener = urllib.request.build_opener(auth_handler)
    urllib.request.install_opener(opener)


def unzip_file(filename, download_path, archive_path):
    """  ZIPファイルを解凍する。解凍後のZIPファイルはarvhice_pathに移動

    :param str filename:
    :param str download_path:
    :param str archive_path:
    """
    with zipfile.ZipFile(download_path + filename) as existing_zip:
        existing_zip.extractall(download_path)
    shutil.move(download_path + filename, archive_path + filename)


def get_downloaded_list(type, archive_path):
    """ ARCHIVE_FOLDERに存在するTYPEに対してのダウンロード済みリストを取得する

    :param str type:
    :param str archive_path:
    :return:
    """
    os.chdir(archive_path)
    filelist = glob.glob(type + "*")
    return filelist


def get_file_list(filetype, folder_path):
    """ 該当フォルダに存在するTYPE毎のファイル一覧を取得する

    :param str filetype:
    :param str folder_path:
    :return:
    """
    os.chdir(folder_path)
    filelist = glob.glob(filetype + "*")
    return filelist


def get_latest_file(finish_path):
    """ 最新のダウンロードファイルを取得する（KYIで判断）

    :param str finish_path:
    :return:
    """
    os.chdir(finish_path)
    latest_file = glob.glob("KYI*")
    return sorted(latest_file)[-1]


def int_null(str):
    """ 文字列をintにし、空白の場合はNoneに変換する

    :param str str:
    :return:
    """
    cnt = len(str)
    empty_val = ''
    for i in range(cnt):
        empty_val += ' '
    if str == empty_val:
        return None
    else:
        return int(str)


def float_null(str):
    """ 文字列をfloatにし、空白の場合はNoneに変換する

    :param str str:
    :return:
    """
    cnt = len(str)
    empty_val = ''
    for i in range(cnt):
        empty_val += ' '
    if str == empty_val:
        return None
    else:
        return float(str)


def int_bataiju_zogen(str):
    """ 文字列の馬体重増減を数字に変換する

    :param str str:
    :return:
    """
    fugo = str[0:1]
    if fugo == "+":
        return int(str[1:3])
    elif fugo == "-":
        return int(str[1:3])*(-1)
    else:
        return 0


def convert_time(str):
    """ 文字列のタイム(ex.1578)を秒数(ex.1178)に変換する

    :param str str:
    :return:
    """
    min = str[0:1]
    if min != ' ':
        return int(min) * 600 + int(str[1:4])
    else:
        return int_null(str[1:4])


def int_haito(str):
    """ 文字列の配当を数字に変換する。空白の場合は0にする

    :param str str:
    :return:
    """
    cnt = len(str)
    empty_val = ''
    for i in range(cnt):
        empty_val += ' '
    if str == empty_val:
        return 0
    else:
        return int(str)


def get_kaisai_date(filename):
    """ ファイル名から開催年月日を取得する(ex.20181118)

    :param str filename:
    :return:
    """
    return '20' + filename[3:9]


def move_file(file, folder_path):
    """ ファイルをフォルダに移動する

    :param str file:
    :param str folder_path:
    :return:
    """
    shutil.move(file, folder_path)


def escape_create_text(text):
    """ CREATE SQL文の生成時に含まれる記号をエスケープする

    :param str text:
    :return:
    """
    new_text = text.replace('%', '')
    return new_text


def convert_python_to_sql_type(dtype):
    """ Pythonのデータ型(ex.float)からSQL serverのデータ型(ex.real)に変換する

    :param str dtype:
    :return:
    """
    if dtype == 'float64':
        return 'real'
    elif dtype == 'object':
        return 'nvarchar(10)'
    elif dtype == 'int64':
        return 'real'


def convert_date_to_str(date):
    """ yyyy/MM/ddの文字列をyyyyMMddの文字型に変換する

    :param str date: date yyyy/MM/dd
    :return: str yyyyMMdd
    """
    return date.replace('/', '')


def convert_str_to_date(date):
    """ yyyyMMddの文字列をyyyy/MM/ddの文字型に変換する

    :param str date: date yyyyMMdd
    :return: str yyyy/MM/dd
    """
    return date[0:4] + '/' + date[4:6] + '/' + date[6:8]


def check_df(df):
    """ 与えられたdfの値チェック

    :param dataframe df:
    """
    pd.set_option('display.max_columns', 3000)
    pd.set_option('display.max_rows', 3000)

    print("------------ データサンプル ----------------------------")
    print(df.iloc[0])
    print(df.shape)

    print("----------- データ統計量確認 ---------------")
    print(df.describe())

    print("----------- Null個数確認 ---------------")
    df_null = df.isnull().sum()
    print(df_null[df_null != 0])

    print("----------- infinity存在確認 ---------------")
    temp_df_inf = df.replace([np.inf, -np.inf], np.nan).isnull().sum()
    df_inf = temp_df_inf - df_null
    print(df_inf[df_inf != 0])

    print("----------- 重複データ確認 ---------------")
    # print(df[df[["RACE_KEY","UMABAN"]].duplicated()].shape)

    print("----------- データ型確認 ---------------")
    print("object型")
    print(df.select_dtypes(include=object).columns)
    # print(df.select_dtypes(include=object).columns.tolist())
    print("int型")
    print(df.select_dtypes(include=int).columns)
    print("float型")
    print(df.select_dtypes(include=float).columns)
    print("datetime型")
    print(df.select_dtypes(include='datetime').columns)


def trans_baken_type(type):
    if type == 1:
        return '単勝　'
    elif type == 2:
        return '複勝　'
    elif type == 3:
        return '枠連　'
    elif type == 4:
        return '枠単　'
    elif type == 5:
        return '馬連　'
    elif type == 6:
        return '馬単　'
    elif type == 7:
        return 'ワイド'
    elif type == 8:
        return '三連複'
    elif type == 9:
        return '三連単'
    elif type == 0:
        return '合計　'

def label_encoding(sr, dict_name, dict_folder):
    """ srに対してlabel encodingした結果を返す

    :param series sr: エンコードしたいSeries
    :param str dict_name: 辞書の名前
    :param str dict_folder: 辞書のフォルダ
    :return: Series
    """
    le = LabelEncoder()
    filename = dict_folder + dict_name + '.pkl'
    if os.path.exists(filename):
        le = load_dict(dict_name, dict_folder)
    else:
        le = le.fit(sr.astype('str'))
        save_dict(le, dict_name, dict_folder)
    sr = sr.map(lambda s: 'other' if s not in le.classes_ else s)
    le_classes = le.classes_.tolist()
    bisect.insort_left(le_classes, 'other')
    le.classes_ = le_classes
    sr_ce = le.transform(sr.astype('str'))
    return sr_ce


def get_tansho_df(df):
    """ 単勝配当のデータフレームを作成する。同着対応のため横になっているデータを縦に結合する。

    :param dataframe df:
    :return: dataframe
    """
    tansho_df1 = df[["競走コード", "単勝馬番1", "単勝払戻金1"]]
    tansho_df2 = df[["競走コード", "単勝馬番2", "単勝払戻金2"]]
    tansho_df3 = df[["競走コード", "単勝馬番3", "単勝払戻金3"]]
    df_list = [tansho_df1, tansho_df2, tansho_df3]
    return_df = arrange_return_df(df_list)
    return return_df

def get_fukusho_df(df):
    """ 複勝配当のデータフレームを作成する。同着対応のため横になっているデータを縦に結合する。

    :param dataframe df:
    :return: dataframe
    """
    fukusho_df1 = df[["競走コード", "複勝馬番1", "複勝払戻金1"]]
    fukusho_df2 = df[["競走コード", "複勝馬番2", "複勝払戻金2"]]
    fukusho_df3 = df[["競走コード", "複勝馬番3", "複勝払戻金3"]]
    fukusho_df4 = df[["競走コード", "複勝馬番4", "複勝払戻金4"]]
    fukusho_df5 = df[["競走コード", "複勝馬番5", "複勝払戻金5"]]
    df_list = [fukusho_df1, fukusho_df2, fukusho_df3, fukusho_df4, fukusho_df5]
    return_df = arrange_return_df(df_list)
    return return_df

def get_wide_df( df):
    """ ワイド配当のデータフレームを作成する。同着対応のため横になっているデータを縦に結合する。

    :param dataframe df:
    :return: dataframe
    """
    wide_df1 = df[["競走コード", "ワイド連番1", "ワイド払戻金1"]]
    wide_df2 = df[["競走コード", "ワイド連番2", "ワイド払戻金2"]]
    wide_df3 = df[["競走コード", "ワイド連番3", "ワイド払戻金3"]]
    wide_df4 = df[["競走コード", "ワイド連番4", "ワイド払戻金4"]]
    wide_df5 = df[["競走コード", "ワイド連番5", "ワイド払戻金5"]]
    wide_df6 = df[["競走コード", "ワイド連番6", "ワイド払戻金6"]]
    wide_df7 = df[["競走コード", "ワイド連番7", "ワイド払戻金7"]]
    df_list = [wide_df1, wide_df2, wide_df3, wide_df4, wide_df5, wide_df6, wide_df7]
    return_df = arrange_return_df(df_list)
    return_df.loc[:, "馬番"] = return_df["馬番"].map(separate_umaban)
    return return_df

def get_umaren_df(df):
    """ 馬連配当のデータフレームを作成する。同着対応のため横になっているデータを縦に結合する。

    :param dataframe df:
    :return: dataframe
    """
    umaren_df1 = df[["競走コード", "馬連連番1", "馬連払戻金1"]]
    umaren_df2 = df[["競走コード", "馬連連番2", "馬連払戻金2"]]
    umaren_df3 = df[["競走コード", "馬連連番3", "馬連払戻金3"]]
    df_list = [umaren_df1, umaren_df2, umaren_df3]
    return_df = arrange_return_df(df_list)
    return_df.loc[:, "馬番"] = return_df["馬番"].map(separate_umaban)
    return return_df

def get_umatan_df(df):
    """ 馬単配当のデータフレームを作成する。同着対応のため横になっているデータを縦に結合する。

    :param dataframe df:
    :return: dataframe
    """
    umatan_df1 = df[["競走コード", "馬単連番1", "馬単払戻金1"]]
    umatan_df2 = df[["競走コード", "馬単連番2", "馬単払戻金2"]]
    umatan_df3 = df[["競走コード", "馬単連番3", "馬単払戻金3"]]
    umatan_df4 = df[["競走コード", "馬単連番4", "馬単払戻金4"]]
    umatan_df5 = df[["競走コード", "馬単連番5", "馬単払戻金5"]]
    umatan_df6 = df[["競走コード", "馬単連番6", "馬単払戻金6"]]
    df_list = [umatan_df1, umatan_df2, umatan_df3, umatan_df4, umatan_df5, umatan_df6]
    return_df = arrange_return_df(df_list)
    return_df.loc[:, "馬番"] = return_df["馬番"].map(separate_umaban)
    return return_df

def get_sanrenpuku_df(df):
    """  三連複配当のデータフレームを作成する。同着対応のため横になっているデータを縦に結合する。

    :param dataframe df:
    :return: dataframe
    """
    sanrenpuku1 = df[["競走コード", "三連複連番1", "三連複払戻金1"]]
    sanrenpuku2 = df[["競走コード", "三連複連番2", "三連複払戻金2"]]
    sanrenpuku3 = df[["競走コード", "三連複連番3", "三連複払戻金3"]]
    df_list = [sanrenpuku1, sanrenpuku2, sanrenpuku3]
    return_df = arrange_return_df(df_list)
    return_df.loc[:, "馬番"] = return_df["馬番"].map(separate_umaban)
    return return_df

def get_sanrentan_df(df):
    """ 三連単配当のデータフレームを作成する。同着対応のため横になっているデータを縦に結合する。

    :param dataframe df:
    :return: dataframe
    """
    sanrentan1 = df[["競走コード", "三連単連番1", "三連単払戻金1"]]
    sanrentan2 = df[["競走コード", "三連単連番2", "三連単払戻金2"]]
    sanrentan3 = df[["競走コード", "三連単連番3", "三連単払戻金3"]]
    sanrentan4 = df[["競走コード", "三連単連番4", "三連単払戻金4"]]
    sanrentan5 = df[["競走コード", "三連単連番5", "三連単払戻金5"]]
    sanrentan6 = df[["競走コード", "三連単連番6", "三連単払戻金6"]]
    df_list = [sanrentan1, sanrentan2, sanrentan3, sanrentan4, sanrentan5, sanrentan6]
    return_df = arrange_return_df(df_list)
    return_df.loc[:, "馬番"] = return_df["馬番"].map(separate_umaban)
    return return_df

def arrange_return_df(df_list):
    """ 内部処理用、配当データの列を競走コード、馬番、払戻に統一する

    :param list df_list: dataframeのリスト
    :return: dataframe
    """
    for df in df_list:
        df.columns = ["競走コード", "馬番", "払戻"]
    return_df = pd.concat(df_list)
    temp_return_df = return_df[return_df["払戻"] != 0]
    return temp_return_df

def separate_umaban(x):
    """ 内部処理用。馬番結果の文字をリスト(4,6),(12,1,3)とかに変換する.0,00といった値の場合は０のリストを返す

    :param str x: str
    :return: list
    """
    umaban = str(x)
    if len(umaban) <= 2:
        return [0, 0, 0]
    if len(umaban) % 2 != 0:
        umaban = '0' + umaban
    if len(umaban) == 6:
        list_umaban = [int(umaban[0:2]), int(umaban[2:4]), int(umaban[4:6])]
    else:
        list_umaban = [int(umaban[0:2]), int(umaban[2:4])]
    return list_umaban

def get_haraimodoshi_dict(haraimodoshi_df):
    """ 払戻用のデータを作成する。extオブジェクトから各払戻データを取得して辞書化して返す。

    :return: dict {"tansho_df": tansho_df, "fukusho_df": fukusho_df}
    """
    tansho_df = get_tansho_df(haraimodoshi_df)
    fukusho_df = get_fukusho_df(haraimodoshi_df)
    umaren_df = get_umaren_df(haraimodoshi_df)
    wide_df = get_wide_df(haraimodoshi_df)
    umatan_df = get_umatan_df(haraimodoshi_df)
    sanrenpuku_df = get_sanrenpuku_df(haraimodoshi_df)
    sanrentan_df = get_sanrentan_df(haraimodoshi_df)
    dict_haraimodoshi = {"tansho_df": tansho_df, "fukusho_df": fukusho_df, "umaren_df": umaren_df,
                         "wide_df": wide_df, "umatan_df": umatan_df, "sanrenpuku_df": sanrenpuku_df, "sanrentan_df": sanrentan_df}
    return dict_haraimodoshi

def create_folder(folder_path):
    try:
        os.makedirs(folder_path)
    except FileExistsError:
        pass

def encode_rap_type(type):
    if type == "一貫":
        return "0"
    elif type == "L4加速":
        return "1"
    elif type == "L3加速":
        return "2"
    elif type == "L2加速":
        return "3"
    elif type == "L1加速":
        return "4"
    elif type == "L4失速":
        return "5"
    elif type == "L3失速":
        return "6"
    elif type == "L2失速":
        return "7"
    elif type == "L1失速":
        return "8"
    else:
        return "9"

def _encode_zengo_bias(num):
    if num < -3:
        return "0" #"超前有利"
    elif num < -1.2:
        return "1" #"前有利"
    elif num > 3:
        return "4" ##"超後有利"
    elif num > 1.2:
        return "3" #"後有利"
    else:
        return "2" #"フラット"

def _calc_uchisoto_bias(num):
    if num < -1.8:
        return "0" #"超内有利"
    elif num < -0.6:
        return "1" #"内有利"
    elif num > 1.8:
        return "4" #"超外有利"
    elif num > 0.6:
        return "3" #"外有利"
    else:
        return "2" #"フラット"

def _encode_race_pace(val):
    if val == "11": return "1"
    elif val == "12": return "2"
    elif val == "13": return "3"
    elif val == "21": return "4"
    elif val == "22": return "5"
    elif val == "23": return "6"
    elif val == "31": return "7"
    elif val == "32": return "8"
    elif val == "33": return "9"
    else: return "0"


def decode_rap_type(num):
    if num == 0:
        return "00一貫"
    elif num == 1:
        return "01L4加"
    elif num == 2:
        return "02L3加"
    elif num ==3:
        return "03L2加"
    elif num == 4:
        return "04L1加"
    elif num == 5:
        return "05L4失"
    elif num == 6:
        return "06L3失"
    elif num == 7:
        return "07L2失"
    elif num == 8:
        return "08L1失"
    else:
        return "09其他"

def _decode_zengo_bias(num):
    if num == 0:
        return "00超前" #"超前有利"
    elif num == 1:
        return "01　前" #"前有利"
    elif num == 4:
        return "02超後" ##"超後有利"
    elif num  == 3:
        return "03　後" #"後有利"
    else:
        return "04なし" #"フラット"

def _decode_uchisoto_bias(num):
    if num == 0:
        return "00超内" #"超内有利"
    elif num == 1:
        return "01　内" #"内有利"
    elif num == 4:
        return "02超外" #"超外有利"
    elif num == 3:
        return "03　外" #"外有利"
    else:
        return "04なし" #"フラット"

def convert_bias(uc, zg):
    if uc == "04なし" and zg == "04なし": return "08なし"
    elif uc == "04なし": return zg
    elif zg == "04なし": return uc
    elif zg == "00超前":
        if uc == "00超内": return "05CUCZ"
        if uc == "02超外": return "05CSCZ"
        if uc == "01　内": return "07内CZ"
        if uc == "03　外": return "07外CZ"
    elif zg == "02超後":
        if uc == "00超内": return "05CUCG"
        if uc == "02超外": return "05CSCG"
        if uc == "01　内": return "07内CG"
        if uc == "03　外": return "07外CG"
    elif zg == "01　前":
        if uc == "00超内": return "06CU前"
        if uc == "02超外": return "06CS前"
        if uc == "01　内": return "09内前"
        if uc == "03　外": return "09外前"
    elif zg == "03　後":
        if uc == "00超内": return "06CU後"
        if uc == "02超外": return "06CS後"
        if uc == "01　内": return "09内後"
        if uc == "03　外": return "09外後"
    else:
        return "08不明"

def convert_are_flag(are1, are2, are3):
    are1 = int(are1) if are1 == are1 else 0
    are2 = int(are2) if are2 == are2 else 0
    are3 = int(are3) if are3 == are3 else 0
    text_are = str(are1) + str(are2) + str(are3)
    val_are = are1 + are2 + are3
    return '0' + str(val_are) + str(val_are) + text_are

def _decode_race_pace(val):
    if val == 1: return "00／　"
    elif val == 2: return "01／￣"
    elif val == 3: return "02／＼"
    elif val == 4: return "03＿／"
    elif val == 5: return "04ーー"
    elif val == 6: return "05￣＼"
    elif val == 7: return "06＼／"
    elif val == 8: return "07＼＿"
    elif val == 9: return "08＼　"
    else: return "09　　"

def convert_jrdb_id(jrdb_race_key, nengappi):
    """ JRDBのRACE_KEYからTarget用のRaceIDに変換する """
    ba_code = jrdb_race_key[0:2]
    kai = jrdb_race_key[4:5]
    nichi = jrdb_race_key[5:6]
    raceno = jrdb_race_key[6:8]
    return nengappi + ba_code + convert_kaiji(kai) + convert_kaiji(nichi) + raceno


def convert_kaiji(kai):
    if kai == 'a': return '10'
    if kai == 'b': return '11'
    if kai == 'c': return '12'
    if kai == 'd': return '13'
    if kai == 'e': return '14'
    if kai == 'f': return '15'
    else: return '0' + kai

def convert_target_file(jrdb_race_key):
    """ JRDBのRaceKeyからtargetのレース、馬印２用のファイル名を生成する """
    yearkai = jrdb_race_key[2:5]
    ba_code = jrdb_race_key[0:2]
    if ba_code == '01': ba = "札"
    if ba_code == '02': ba = "函"
    if ba_code == '03': ba = "福"
    if ba_code == '04': ba = "新"
    if ba_code == '05': ba = "東"
    if ba_code == '06': ba = "中"
    if ba_code == '07': ba = "名"
    if ba_code == '08': ba = "京"
    if ba_code == '09': ba = "阪"
    if ba_code == '10': ba = "小"
    return yearkai + ba

def convert_kyakushitsu(cd):
    dict = {"1":"逃げ", "2":"先行", "3":"差し", "4":"追込", "5":"好位差し", "6":"自在", "0": ""}
    return dict.get(cd, '')

def convert_kyori_tekisei(cd):
    dict = {"1":"短距離", "2":"中距離", "3":"長距離", "5":"哩（マイル）", "6":"万能", "0": ""}
    return dict.get(cd, '')

def convert_joshodo(cd):
    dict = {"1":"上昇度AA", "2":"上昇度A", "3":"上昇度B", "4":"上昇度C", "5":"上昇度?", "0": ""}
    return dict.get(cd, '')

def convert_chokyo_yajirushi(cd):
    dict = {"1":"デキ抜群", "2":"上昇", "3":"平行線", "4":"やや下降気味", "5":"デキ落ち", "0": ""}
    return dict.get(cd, '')

def convert_kyusha_hyouka(cd):
    dict = {"1":"超強気", "2":"強気", "3":"現状維持", "4":"弱気", "0": ""}
    return dict.get(cd, '')

def convert_hidume(cd):
    dict = {"01":"大ベタ", "02":"中ベタ", "03":"小ベタ", "04":"細ベタ", "05":"大立", "06":"中立", "07":"小立", "08":"細立", "09":"大標準", "10":"中標準",
            "11":"小標準", "12":"細標準", "13":"", "14":"", "15":"", "16":"", "17":"大標起", "18":"中標起", "19":"小標起", "20":"細標起",
            "21":"大標ベ", "22":"中標ベ", "23":"小標ベ", "24":"細標ベ", "0": ""}
    return dict.get(cd, '')

def convert_omotekisei(cd):
    dict = {"1":"◎ 得意", "2":"○ 普通", "3":"△ 苦手", "0": ""}
    return dict.get(cd, '')

def convert_course(cd):
    dict = {'1': '(Ｇ１)', '2': '(Ｇ２)', '3': '(Ｇ３)', '4': '(重賞)', '5': '特別', '0': ''}
    return dict.get(cd, '')

def convert_track(cd):
    dict = {'139': '芝直', '121': '芝左', '122': '芝左外', '121': '芝左', '121': '芝左', '121': '芝左', '121': '芝左', '111': '芝右', '112': '芝右外', '112': '芝右外', '111': '芝右', '111': '芝右', '112': '芝右外',
            '221': 'ダ左', '211': 'ダ右', '212': 'ダ右外', '399': '障芝', '393': '障直ダ', '0': ''}
    return dict.get(cd, '')

def convert_umakigo3digit(cd):
    dict = {'000': '', '001': '(指）', '004': '見習', '002': '[指]', '003': '(特指)', '020': '牝', '021': '牝 （指）', '022': '牝 [指]', '023': '牝 （特指）', '030': '牡・せん', '031': '牡・せん(指)', '032': '牡・せん[指]',
            '040': '牡・牝', '041': '牡・牝(指)', '042': '牡・牝[指]', '043': '牡・牝(特指)', '100': '(混)', '101': '(混) (指)', '104': '(混) 見習', '102': '(混) [指]', '103': '(混) （特指）', '110': '(混) 牡', '111': '(混) 牡(指)',
            '112': '(混) 牡 [指]', '113': '(混) 牡（特指）', '120': '(混) 牝', '121': '(混) 牝(指)', '122': '(混) 牝[指]', '123': '(混) 牝（特指）', '130': '(混) 牡・せん', '131': '(混) 牡・せん(指)', '132': '(混) 牡・せん[指]',
            '133': '(混) 牡・せん（特指）', '140': '(混) 牡・牝', '141': '(混) 牡・牝(指)', '200': '(父)', '201': '(父)(指)', '202': '(父)[指]', '203': '(父)（特指）', '': '(市)', '300': '(抽)', '301': '(抽)(指）', '302': '(抽)[指]',
            '300': '(抽)', '301': '(抽)(指)', '302': '(抽)[指]', '300': '(抽)(市)', '301': '(抽)(市)(指)', '302': '(市)(抽)[指]', '303': '(市)(抽)（特指）', '300': '(抽)', '301': '(抽)', '302': '', '300': '(抽)', '301': '', '400': '九州',
            '401': '九州 (指)', '402': '九州 [指]', '403': '九州（特指）', '500': '(国際)', '501': '(国際)(指)', '502': '(国際)[指]', '520': '(国際) 牝', '521': '(国際)牝（指）', '530': '(国際)牡・せん', '531': '(国際)牡・せん(指)',
            '540': '(国際)牡・牝', '541': '(国際)牡・牝(指)',  '0': ''}
    return dict.get(cd, '')

def convert_shubetsu(cd):
    dict = {'11': '２歳', '12': '３歳', '13': '３歳以上', '14': '４歳以上', '20': '', '99': '', '0': ''}
    return dict.get(cd, '')

def convert_joken(cd):
    dict = {'04': '400万下', '05': '500万下', '08': '800万下', '09': '900万下', '10': '1000万下', '15': '1500万下', '16': '1600万下', 'A1': '新馬', 'A2': '未出走', 'A3': '未勝利', 'OP': 'オープン',  '0': ''}
    return dict.get(cd, '')

def convert_class(cd):
    dict = {'01': '芝Ｇ１', '02': '芝Ｇ２', '03': '芝Ｇ３', '04': '芝ＯＰ A', '05': '芝ＯＰ B', '06': '芝ＯＰ C', '07': '芝３勝A', '08': '芝３勝B', '09': '芝３勝C', '10': '芝２勝A', '11': '芝２勝B', '12': '芝２勝C',
            '13': '芝１勝A', '14': '芝１勝B', '15': '芝１勝C', '16': '芝未 A', '17': '芝未 B', '18': '芝未 C', '21': 'ダＧ１', '22': 'ダＧ２', '23': 'ダＧ３', '24': 'ダＯＰ Ａ', '25': 'ダＯＰ Ｂ', '26': 'ダＯＰ Ｃ',
            '27': 'ダ３勝Ａ', '28': 'ダ３勝Ｂ', '29': 'ダ３勝Ｃ', '30': 'ダ２勝Ａ', '31': 'ダ２勝Ｂ', '32': 'ダ２勝Ｃ', '33': 'ダ１勝Ａ', '34': 'ダ１勝Ｂ', '35': 'ダ１勝Ｃ', '36': 'ダ未 Ａ', '37': 'ダ未 Ｂ', '38': 'ダ未 Ｃ',
            '51': '障Ｇ１', '52': '障Ｇ２', '53': '障Ｇ３', '54': '障ＯＰ Ａ', '55': '障ＯＰ Ｂ', '56': '障ＯＰ Ｃ', '57': '障１勝Ａ', '58': '障１勝Ｂ', '59': '障１勝Ｃ', '60': '障未 Ａ',
            '61': '障未 Ｂ', '62': '障未 Ｃ', '0': ''}
    return dict.get(cd, '')

def convert_grade(cd):
    dict = {'1': 'Ｇ１', '2': 'Ｇ２', '3': 'Ｇ３', '4': '重賞', '5': '特別', '6': 'Ｌ', '0': ''}
    return dict.get(cd, '')

def convert_course_dori(cd):
    dict = {'1': '最内', '2': '内', '3': '中', '4': '外', '5': '大外', '0': ''}
    return dict.get(cd, '')

def convert_ijo_kubun(cd):
    dict = {'0': '', '1': '取消', '2': '除外', '3': '中止', '4': '失格', '5': '降着', '6': '再騎乗'}
    return dict.get(cd, '')

def convert_mark(cd):
    dict = {'1': '◎', '2': '○', '3': '▲', '4': '注', '5': '△', '6': '△', '0': ''}
    return dict.get(cd, '')

def convert_kehai(cd):
    dict = {'1': '状態良', '2': '平凡', '3': '不安定', '4': 'イレ込', '5': '気合良', '6': '気不足', '7': 'チャカ', '8': 'イレチ', '0': ''}
    return dict.get(cd, '')

def convert_ashimoto(cd):
    dict = {'000': 'バンテージ腕節まで', '001': '蹄汚い', '002': '球節腫れる', '003': '交突バ繋部分', '004': '交突防止帯', '005': '蹄鉄浮く', '006': '蹄冠部全面裂蹄防止テープ', '007': '蹄壁部裂蹄防止テープ', '008': 'ブーツ',
            '009': 'バンテージ外す', '010': 'バンテージ巻く（変更）', '011': 'ソエ焼き', '012': '半鉄', '013': '連尾鉄', '014': '曲鉄', '015': '鉄橋', '017': '四分の三鉄', '018': '脚腫れる', '019': '繋ぎキズ', '020': 'ソエ傷',
            '021': 'アダプターパッド', '022': '調教バンテージ', '023': '左前球節キズ', '024': 'バンテージ巻き跡', '025': '交突蹄冠部分', '026': '蹄冠部キズ', '027': '裏筋腫れる', '028': '骨瘤', '029': '骨瘤腫れ小', '030': 'ソエ腫れる',
            '031': 'ソエ腫れ小', '032': '膝焼く', '033': '膝キズ', '034': '膝裏焼く', '035': 'エクイロックス', '036': '裏筋傷', '037': '骨瘤焼く', '038': '水ブリスター', '040': '裸足', '041': '追突防止パッド', '042': '新エクイロックス',
            '043': '蹄欠損', '045': 'スプーンヒール鉄', '046': '柿元鉄', '047': '飛節焼き治療', '048': '蹄切り込み線', '049': '着地時に蹄をひねる', '050': '着地時に蹄踵が浮く', '051': '着地時に球節が沈む', '052': '鉄唇横', '053': '鉄唇なし',
            '054': '鉄唇３つ', '055': '内に入る歩様', '056': '高く上げて歩く', '057': '球節バンテージ', '058': '蹄踵部交突防止帯', '059': '繋腫れ', '060': '繋弾く', '061': '２度踏み', '062': '球節の毛剃る', '063': '繋の毛剃る', '064': '飛節傷',
            '065': '球節に小さい瘤', '066': '裏筋に小さい瘤', '067': '繋に小さい瘤', '068': '管骨に透明の巻物', '069': '繋に透明の巻物', '070': '球節に透明の巻物', '071': '球節焼き治療', '072': '裏筋イボ', '073': '着地時に飛節傾く', '074': '肘腫（髄液溜まる）',
            '075': '球節腫れ（髄液溜まる）', '076': 'ヒールパッド', '077': 'リバーシブル鉄', '078': '脇の肘腫', '079': '裏筋の毛を剃る', '080': '外に振って歩く', '081': '蹄亀裂', '082': '蹄底パッド', '100': '痛そうな歩様', '101': '周回中躓く', '102': '蹄汚い',
            '103': '裏筋腫れる', '104': 'トモ流れる', '105': '骨瘤', '106': '前脚腫れ大', '107': '前脚腫れ小', '108': '両前バンテージ外す（変更）', '109': '両前バンテージ巻く（変更）', '110': '後脚腫れる', '111': 'ソエ腫れる', '112': '膝焼く', '113': '脚部不安',
            '114': '後肢踏み込み甘い', '115': '着地時に膝が震える', '116': '着地時に飛節がブレる', '117': '膝硬い', '118': '後肢躓く', '119': '交差する歩様', '120': '肩の出窮屈', '0': ''}
    return dict.get(cd, '')

def convert_kyoso_kigoA(cd):
    dict = {'0': '', '1': '○混', '2': '○父', '3': '○市○抽', '4': '九州産限定', '5': '○国際混'}
    return dict.get(cd, '')

def convert_kyoso_kigoB(cd):
    dict = {'0': '', '1': '牡馬限定', '2': '牝馬限定', '3': '牡・せん馬限定', '4': '牡・牝馬限定'}
    return dict.get(cd, '')

def convert_kyoso_kigoC(cd):
    dict = {'0': '', '1': '○指', '2': '□指', '3': '○特指', '4': '若手',}
    return dict.get(cd, '')

def convert_kyoso_shubetsu(cd):
    dict = {'11': '２歳', '12': '３歳', '13': '３歳以上', '14': '４歳以上', '20': '障害', '99': 'その他', '0': ''}
    return dict.get(cd, '')

def convert_kyoso_joken(cd):
    dict = {'04': '１勝クラス', '05': '１勝クラス', '08': '２勝クラス', '09': '２勝クラス', '10': '２勝クラス', '15': '３勝クラス', '16': '３勝クラス', 'A1': '新馬', 'A2': '未出走', 'A3': '未勝利', 'OP': 'オープン', '0': ''}
    return dict.get(cd, '')

def convert_keito(cd):
    dict = {'1101': 'ノーザンダンサー系', '1102': 'ニジンスキー系', '1103': 'ヴァイスリージェント系', '1104': 'リファール系', '1105': 'ノーザンテースト系', '1106': 'ダンジグ系', '1107': 'ヌレイエフ系', '1108': 'ストームバード系', '1109': 'サドラーズウェルズ系',
            '1201': 'ロイヤルチャージャー系', '1202': 'ターントゥ系', '1203': 'ヘイルトゥリーズン系', '1204': 'サーゲイロード系', '1205': 'ハビタット系', '1206': 'ヘイロー系', '1207': 'ロベルト系', '1301': 'ナスルーラ系', '1302': 'グレイソヴリン系', '1303': 'ネヴァーベンド系',
            '1304': 'プリンスリーギフト系', '1305': 'ボールドルーラー系', '1306': 'レッドゴッド系', '1307': 'ゼダーン系', '1308': 'カロ系', '1309': 'ミルリーフ系', '1310': 'リヴァーマン系', '1311': 'シアトルスルー系', '1312': 'ブラッシンググルーム系', '1401': 'ネアルコ系',
            '1402': 'ニアークティック系', '1403': 'デリングドゥ系', '1501': 'ネイティヴダンサー系', '1502': 'シャーペンアップ系', '1503': 'ミスタープロスペクター系', '1601': 'フェアウェイ系', '1602': 'バックパサー系', '1603': 'ファラリス系', '1701': 'ダマスカス系',
            '1702': 'テディ系', '1801': 'ハイペリオン系', '1802': 'オリオール系', '1803': 'ロックフェラ系', '1804': 'テューダーミンストレル系', '1805': 'オーエンテューダー系', '1806': 'スターキングダム系', '1807': 'フォルリ系', '1901': 'エクリプス系',
            '1902': 'ブランドフォード系', '1903': 'ドンカスター系', '1904': 'ドミノ系', '1905': 'ヒムヤー系', '1906': 'エルバジェ系', '1907': 'ダークロナルド系', '1908': 'ファイントップ系', '1909': 'ゲインズボロー系', '1910': 'ハーミット系', '1911': 'アイシングラス系',
            '1912': 'コングリーヴ系', '1913': 'ロックサンド系', '2001': 'セントサイモン系', '2002': 'リボー系', '2003': 'ヒズマジェスティ系', '2004': 'グロースターク系', '2005': 'トムロルフ系', '2006': 'ワイルドリスク系', '2007': 'チャウサー系', '2008': 'プリンスローズ系',
            '2009': 'プリンスキロ系', '2010': 'ラウンドテーブル系', '2101': 'マッチェム系', '2102': 'フェアプレイ系', '2103': 'ハリーオン系', '2104': 'マンノウォー系', '2105': 'インリアリティ系', '2201': 'パーソロン系', '2202': 'リュティエ系', '2203': 'ジェベル系',
            '2204': 'トウルビヨン系', '2205': 'ザテトラーク系', '2206': 'ヘロド系', '2301': 'サンドリッジ系', '2401': 'スウィンフォード系', '9901': 'アラ系', '0': ''}
    return dict.get(cd, '')

def convert_juryo(cd):
    dict = {'1': 'ハンデ', '2': '別定', '3': '馬齢', '4': '定量', '0': ''}
    return dict.get(cd, '')

def convert_basho(cd):
    dict = {'01': '札幌', '02': '函館', '03': '福島', '04': '新潟', '05': '東京', '06': '中山', '07': '中京', '08': '京都', '09': '阪神', '10': '小倉', '21': '旭川', '22': '札幌', '23': '門別', '24': '函館', '25': '盛岡', '26': '水沢', '27': '上山',
            '28': '新潟', '29': '三条', '30': '足利', '31': '宇都', '32': '高崎', '33': '浦和', '34': '船橋', '35': '大井', '36': '川崎', '37': '金沢', '38': '笠松', '39': '名古', '40': '中京', '41': '園田', '42': '姫路', '43': '益田', '44': '福山',
            '45': '高知', '46': '佐賀', '47': '荒尾', '48': '中津', '61': '英国', '62': '愛国', '63': '仏国', '64': '伊国', '65': '独国', '66': '米国', '67': '加国', '68': 'UAE ', '69': '豪州', '70': '新国', '71': '香港', '72': 'チリ', '73': '星国',
            '74': '瑞国', '75': 'マカ', '76': '墺国', '0': '', '00': ''}
    return dict.get(cd, '')

def convert_chokyo_type(cd):
    dict = {'01': 'スパルタ', '02': '標準多め', '03': '乗込', '04': '一杯平均', '05': '標準', '06': '馬ナリ平均', '07': '急仕上げ', '08': '標準少め', '09': '軽目', '10': '連闘', '11': '調教せず', '0': ''}
    return dict.get(cd, '')

def convert_tenko(cd):
    dict = {'1': '晴', '2': '曇', '3': '小雨', '4': '雨', '5': '小雪', '6': '雪', '0': ''}
    return dict.get(cd, '')

def convert_tokki(cd):
    dict = {'033': '口向き悪い', '034': '放馬', '035': '落鉄', '037': '揉まれ弱い', '038': '芝向き', '039': 'ダート向き', '040': '心房細動', '041': '喉鳴り', '042': '熱発', '043': '鼻出血', '044': 'ソエ', '045': '恐がり', '046': '脚腫れる', '047': 'コズミ',
            '048': '腰フラ', '049': '気悪', '050': '素直', '051': '根性有り', '052': 'ハミ悪', '053': '頭高い（レース中）', '054': '内枠×', '055': '外枠×', '056': '内枠○', '057': '外枠○', '058': 'スタート良い', '059': 'スタート悪い', '060': '内もたれ',
            '061': '外もたれ', '062': 'ハナ条件', '063': 'ダ重い○', '064': 'ダ重い×', '065': 'ダ軽い○', '066': 'ダ軽い×', '067': 'ダ少し掛○', '068': 'ダ少し掛×', '069': '芝重い○', '070': '芝重い×', '071': '芝軽い○', '072': '芝軽い×', '073': '芝滑る○',
            '074': '芝滑る×', '075': '芝少し掛○', '076': '芝少し掛×', '077': '雨嫌い', '078': 'ズブイ', '079': 'シブトイ', '080': '折り合い○', '081': '折り合い×', '082': '先行力○', '083': '先行力×', '084': '瞬発力○', '085': '瞬発力×', '086': '調教良い',
            '087': '大トビ', '088': '調教出来ず', '089': '右回り○', '090': '左回り○', '091': 'トモ甘い', '092': '馬込み×', '093': '骨膜炎', '094': '裂蹄', '095': 'ゴトゴト', '096': '物見', '097': '鐙外れ', '098': 'ソラ使う', '099': '左回り×', '100': '右回り×',
            '101': 'ダート×', '102': '芝×', '103': '左モタレ', '104': '右モタレ', '105': 'センスある', '106': '落馬', '107': '輸送ダメ', '108': '馬込ダメ', '109': '冬ダメ', '110': '外逃解消', '111': '鞍ズレ', '112': 'ジリ脚', '113': '口向解消', '114': 'ローカル向',
            '115': '外逃げる', '116': '恐い解消', '117': 'ソエ解消', '118': 'モタレ解消', '119': 'コーナーワーク×', '120': '小回り向き', '121': '外枠発走', '122': '冬○', '123': '追って甘い', '124': '坂○', '125': '体質弱い', '126': '走る気ない', '127': '素質あり',
            '128': '鉄砲○', '129': '気が小さい', '130': '終い甘い', '131': '他馬気にする', '132': '輸送熱', '133': 'パドック転倒', '134': 'ゲート内転倒', '135': 'ハミ外す', '136': '腰悪い', '137': 'ソエ焼き', '138': '筋肉痛', '140': 'ゲート良い', '141': 'ゲート悪い',
            '142': '気が強い', '143': '斤量泣き', '144': '器用さ欠く', '145': '使い込む×', '146': '手前替え×', '147': '小回り×', '148': '追って甘い', '149': '後方から', '150': 'お終い確実', '151': '展開待ち', '152': '砂被る○', '153': '砂被る×', '154': 'コーナーワーク○',
            '155': '坂×', '156': 'ふらつく', '157': 'ダッシュ○', '158': 'ダッシュ×', '159': 'フワフワ', '160': '中山向き', '161': '府中向き', '162': '腰甘い', '163': '怯む', '164': '足元弱い', '165': '馬気', '166': '歩様悪い', '167': '気難しい', '168': '小頭数○',
            '169': 'ノメル', '170': 'トウ骨', '171': '追って頭上げる', '172': '砂被頭上げる', '173': '広いコース向き', '174': 'ラチ接触', '175': 'エビ', '176': 'フケ', '177': '追ってしっかり', '178': 'トモ落とす', '179': '外被せられる×', '180': '泥被る○', '181': '泥被る×',
            '182': '一頭だと気を抜く', '183': 'ラチ頼る', '184': '耳絞る', '185': '首使い○', '186': '首使い×', '187': '砂被○', '188': '砂被×', '189': 'フットワーク×', '190': '中間角膜炎', '191': '中間挫跖', '192': '気性成長', '193': '拍車', '194': '気性若い',
            '195': 'フレグモーネ', '196': '入線後故障', '197': '道中気を抜く', '198': '直線長いコース向き', '199': '外から被せられる×', '200': 'トモの状態悪い', '201': '飛越○', '202': '飛越×', '203': '水濠○', '204': '水濠×', '205': '飛越に気を遣う', '206': '落馬影響',
            '207': '着地○', '208': '着地×', '209': '立ち回り○', '210': '立ち回り×', '211': '飛越慎重', '212': '平地力○', '213': '平地力×', '215': 'スタミナ○', '216': 'スタミナ×', '217': '道悪○', '218': '道悪×', '219': '飛越時気を抜く', '221': '障害接触', '222': 'バンケット×',
            '223': '斜飛', '224': 'バンケット○', '225': '飛越高い', '226': '飛越△', '227': '飛越安定', '228': '着地△', '229': '水濠トモ落', '230': '障害飛ばず', '231': '障害調教再審査', '232': '終ダ×', '233': '落馬寸前', '234': '時計掛○', '235': '時計掛×', '236': '飛越◎',
            '237': '斤量応', '238': '平地力◎', '239': '障害調教再審査', '240': '終いダ○', '241': '落馬後遺症残', '242': '飛越時ブレーキ', '245': '直線芝○', '246': '直線芝×', '247': '落馬再騎乗', '249': '背が低い', '250': '脚短い', '251': '連続障害○', '252': '連続障害×',
            '253': '踏み切り○', '254': '踏み切り×', '255': '助走○', '256': '助走×', '257': '右斜飛', '258': '左斜飛', '301': '非力', '302': '腰良化', '303': 'トモ良化', '304': '高脚使う', '305': '平坦向き', '306': '遊びながら走る', '307': '尻尾振る', '308': '揉まれ弱い解消',
            '309': '他馬と接触', '310': 'レース中脚気にする', '311': '揉まれ強い', '313': 'トモ治療中', '314': '熱発明け', '315': '一頭だと走る気出す', '316': '口で息をする', '317': '口が硬い', '320': 'スピード非凡', '321': '脚を外に振って走る', '322': '喉弱い', '323': '背中が良い',
            '324': 'レース中故障(入線)', '325': '器用', '326': '次走ハミ替え○', '327': '今回ハミ替え○', '328': '今回ハミ替え×', '329': '連闘○', '330': '輸送弱い', '331': '脚元悪い', '332': '使い込む○', '333': '鉄砲×', '334': '中間順調さ欠く', '335': '中間放馬', '336': 'ゲートもたれ',
            '337': 'ベタ爪', '338': '最内枠×', '339': 'パドック落馬', '340': '夏○', '341': '夏×', '342': '歩様良化', '343': '(競争中)鼻出血', '344': '滞在競馬○', '345': 'リズム悪い', '346': '皮膚病', '347': '（馬場入場後）暴走', '348': '外傷', '349': 'タイムオーバー',
            '350': 'レース中骨折', '351': '舌がハミ越す', '352': 'ハミ頼る', '353': 'レース前落鉄', '354': 'ハミもたれ', '355': 'トモ疲れ', '356': '冬毛出てくる', '357': '声で気合', '358': '疲れ気味', '359': '砂被ハミ掛悪い', '360': 'ブリ効果あり', '361': '脚不安解消',
            '362': '水浮ダ×', '363': '躓きやすい', '364': '札幌向き', '365': '函館向き', '366': '福島向き', '367': '新潟向き', '368': '中京向き', '369': '阪神向き', '370': '京都向き', '371': '小倉向き', '372': '鳴く', '373': '夏負け気味', '374': 'タマ', '375': 'ツメ悪い',
            '376': '多頭数×', '378': 'シャド効果あり', '379': '馬場入場後転倒', '380': '膠着', '381': '体質シッカリ', '382': '喉良化', '383': '芝切れ目が飛ぶ', '384': '渋馬場○', '385': '渋馬場×', '386': 'メンコ効果あり', '387': '不利', '388': 'アオル', '389': 'ブリ外効果あり',
            '390': '京都×', '391': '距離長', '392': '距離短', '393': '鉄唇横', '394': '瞬発力非凡', '395': 'スピード有', '396': 'スピード無', '397': '集中力出る', '398': '集中力ない', '399': '下を気にする', '400': '内側に斜行', '401': '外側に斜行', '402': '返し馬入念',
            '403': '(道中)息入る', '404': '(道中)息入らず', '405': 'ハミ替え２走目', '406': 'ハミ替え３走目', '407': '今回馬具替え○', '408': '今回馬具替え×', '409': '直線手前替える', '410': '本格化', '411': 'ハミ取る', '412': 'ハミ取らず', '413': '躓く', '414': '輸送慣れ',
            '415': '大外回る', '416': 'コズミ解消', '417': '攻め軽目', '418': '急仕上げ', '419': '掛かる', '420': '掛かり気味', '421': '突っ張る', '422': '行きたがる', '423': 'デキ一息', '424': '出走停止', '425': '調教再審査', '426': '発走再審査', '427': '発走調教再審査',
            '428': '枠内駐立不良', '429': '枠入不良', '430': '阪神×', '431': '腰力つく', '432': '連闘×', '433': 'パワー○', '434': 'スタート芝×', '435': '息切れ', '436': '逆手前', '437': '気を抜く', '438': '二走ボケ', '439': '捌き硬い', '440': '追い通し', '441': '余力なし',
            '442': 'いい脚長く使う', '443': '落ち着きほしい', '444': '落ち着きでる', '445': 'レースせず', '446': 'シャド外効果アリ', '447': 'トモぶつける', '448': 'バランス崩す', '449': 'ペース速い○', '450': 'ペース速い×', '451': 'ペース遅い○', '452': 'ペース遅い×', '453': 'モタつく',
            '454': '調教注意', '455': '返し馬ウルサイ', '456': '返し馬できない', '457': '馬具変更２走目', '458': '中間フレグモーネ', '459': '口割る', '460': 'ズブイ解消', '461': '減量効果あり', '462': 'いい脚少しだけ', '463': 'ダ少し速○', '464': 'ダ少し速×', '465': '馬込平気',
            '466': 'ないら', '467': '蹄良化', '468': 'フラッシュ×', '469': 'ひざ外傷', '470': 'スネ外傷', '471': '食い細', '472': '瞬発力△', '473': 'エンジン掛速', '474': 'エンジン掛遅', '475': '裂蹄のため次走延期', '476': 'ゲート音怖がる', '477': '軟ら芝×', '478': '次走良化気配',
            '479': '次走一変可', '480': '芝大丈夫', '481': 'ダ大丈夫', '482': 'ゲート潜る', '483': '連闘ダメ', '484': '前行くとダメ', '485': '見せ場なし', '486': '後方まま', '487': '中間熱発', '488': '軟ら芝○', '489': '故障アオリ食', '490': '馬込△', '491': 'フットワーク△',
            '492': 'トビ小さい', '493': '挫跖のため次走延期', '494': '熱発のため次走延期', '495': '裂蹄のため放牧', '496': '輸送食い細', '497': '皮膚色抜ける', '498': '馬体緩める', '499': 'ブリンカー２走目', '500': 'ブリンカー３走目', '501': '感冒', '502': '疝痛', '503': '角膜炎',
            '504': '右前踏創', '505': '左前踏創', '506': '右後踏創', '507': '左後踏創', '508': '両前踏創', '509': '両後踏創', '510': '右前管骨々膜炎', '511': '左前管骨々膜炎', '512': '右後管骨々膜炎', '513': '左後管骨々膜炎', '514': '両前管骨々膜炎', '515': '両後管骨々膜炎',
            '516': '右前肢挫創', '517': '左前肢挫創', '518': '右後肢挫創', '519': '左後肢挫創', '520': '両前肢挫創', '521': '両後肢挫創', '522': '右前球節炎', '523': '左前球節炎', '524': '右後球節炎', '525': '左後球節炎', '526': '両前球節炎', '527': '両後球節炎',
            '528': '右前球節部フレグモー', '529': '左前球節部フレグモー', '530': '右後球節部フレグモー', '531': '左後球節部フレグモー', '532': '両前球節部フレグモー', '533': '右前肢フレグモーネ', '534': '左前肢フレグモーネ', '535': '右後肢フレグモーネ', '536': '左後肢フレグモーネ',
            '537': '両前肢フレグモーネ', '538': '両後肢フレグモーネ', '539': '右前屈腱炎', '540': '左前屈腱炎', '541': '両前屈腱炎', '542': '外傷性鼻出血', '543': '右前挫跖', '544': '左前挫跖', '545': '右後挫跖', '546': '左後挫跖', '547': '両後挫跖', '548': '両前挫跖',
            '549': '右前裂蹄', '550': '左前裂蹄', '551': '右後裂蹄', '552': '左後裂蹄', '553': '両前裂蹄', '554': '両後裂蹄', '555': '右寛跛行', '556': '左寛跛行', '557': '右肩跛行', '558': '左肩跛行', '559': '右前繋部挫創', '560': '左前繋部挫創', '561': '右後繋部挫創',
            '562': '左後繋部挫創', '563': '両前繋部挫創', '564': '両後繋部挫創', '565': '右第１骨々折', '572': '左第１指骨々折', '573': 'ジンマシン', '575': '左前繋靭帯断裂', '576': '左第１指関節脱臼', '578': '左前屈腱部打撲傷', '579': '右前屈腱断裂', '580': '両前浅屈腱断裂',
            '581': '寛骨々折', '582': '右前浅屈腱不全断裂', '583': '左前繋靭帯炎', '584': '右肘部フレグモーネ', '585': '右前浅屈腱炎', '586': '左前腕部蹴傷', '587': '右第１指関節脱臼', '588': 'キ甲部挫創', '589': '両腕接部挫創', '590': '両頚部打撲傷', '591': '疲労が著しいため除外',
            '592': '鞍傷', '593': '右第1指節種子骨々折', '594': '右腰角部挫創', '595': '右前繋靭帯炎', '596': '両前腕部挫創', '597': '右膝蓋部挫創', '598': '左後蹄冠部挫創', '599': '左副手根骨々折', '600': '両飛節挫創', '601': '寛跛行', '602': '便秘疝', '603': '右飛節炎',
            '604': '右上眼瞼部挫創', '605': '腰椎骨折', '606': '両前管部挫創', '607': '左腕節部挫創', '608': '事故のため除外', '609': 'パドック蹄鉄打ち直し', '610': '左前球節部挫創', '611': '右大腿骨々折', '612': '右前屈腱部打撲傷', '613': '左第３中手骨複骨折', '614': '右口角節',
            '615': '肩跛行', '616': '右前管部挫創', '617': '左第３指趾骨々折', '618': '左前繋靭帯炎', '619': '右飛節部挫創', '620': '左前蹄球炎', '621': '右下腿部挫創', '622': '左第３指骨骨折', '623': '左腰角部挫創', '624': '額部挫創', '625': '右第３中足骨々折', '626': '左飛節部挫創',
            '627': '左後繋靭帯断裂', '628': '左後管部打撲傷', '629': '右後管部挫創', '630': '左腰部打撲', '631': '左前飛節炎', '632': '頚部挫創', '633': '左第３手根骨板状骨折', '634': '右前腕骨々折', '635': '右前浅屈腱断裂', '636': '創傷性右角膜炎', '637': '外傷性右鼻出血',
            '638': '左上腕骨々折', '639': '創傷性左角膜炎', '640': '右腕節部挫創', '641': '左前浅屈腱不全断裂', '642': '右第３指骨々折', '643': '左前腕部挫創', '644': '右前繋靭帯断裂', '645': '蹄冠挫石','646': '右第１趾骨粉砕骨折','647': '左第３中手骨々折','648': '左第３中足骨々折',
            '649': '左副手根骨複骨折','650': '左橈骨遠位端骨折','651': '左前靭帯不全断裂','652': '急性心不全','653': '左肩部蹴傷','654': '左後球節部切創','655': '胸部フレグモーネ','656': '右第３中足骨複骨折','657': '両前肢打撲傷','658': '右アキレス腱脱位','659': '左下腿骨複骨折',
            '660': '食道梗塞','661': '右前蹄球炎','662': '左上腕骨々折','663': '左手根骨複骨折','664': '左第１指骨粉砕骨折','665': 'キ甲部打撲傷','666': '左膝蓋部挫創','667': '右第３中足骨開放骨折','668': '右肘腫','669': '四肢挫創','670': '左第２指関節開放性脱','671': '左前繋靭帯不全断裂',
            '672': '背部フレグモーネ','673': '右前球節部挫創','674': '左前浅屈腱断裂','675': '両後球節部挫創','676': '右上腕骨々折','677': '右中手骨開放骨折','678': '右脛骨々膜炎','679': '左第１趾骨粉砕骨折','680': '左第３中足骨開放骨折','681': '左第１指骨複骨折','682': '左前深管骨瘤',
            '683': '右第３中手骨々折','684': '右後肢蹴傷','685': '左前腕部打撲傷','686': '左結膜炎','687': '右腕節炎','688': '右後肢裂創','689': '右膝蓋部蹴傷','690': '頭部打撲症','691': '左後球節部挫創','692': '左第１趾骨々折','693': '左第１指節種子骨複骨','694': '口内炎',
            '695': '鼻部挫創','696': '左肋部蹴傷','697': '左前管部挫創','698': '左腰部挫創','699': '乳房炎','700': 'シャド２走目','701': 'シャド３走目','702': 'ブリ逆効果','703': '発進不良','704': 'つかみどころない','705': '内側に逃避','706': '外側に逃避','707': 'フットワーク○',
            '708': '落馬再騎乗','709': 'トモ傷める(ゲート内)','710': '障害練習効果','711': '空馬影響','712': '馬体検査','713': '今回蹄鉄替○','714': '今回蹄鉄替×','715': 'バドック放馬','716': '発進不良','717': 'ステッキ×','718': '道中外々','719': '雨○','720': '雨×','721': 'スタート芝○',
            '722': '勝負所モタつく','723': '適距離','724': 'ゲート練習','725': '障害練習','726': '中間鞍傷','727': '次走オッズチェック','728': '時計速○','729': '時計速×','730': '脚使処難','731': '根性なし','732': 'レース振スムーズ','733': '一本調子','734': '完歩小さい','735': '芝ダＯＫ',
            '736': 'スムーズさ欠','737': '生ズルイ','738': '絞れれば','739': '水浮ダ○','740': '追いかけられる○','741': '口切れる','742': '馬装整備','743': '芝ダ切れ目飛ぶ','744': '中間外傷','745': '中間感冒','746': '中間抜歯','747': '調教不足','748': '歯替わり','749': '馬場入り１番早い',
            '750': 'Pブリンカー効果あり','751': '舌くくり効果あり','752': 'ノーズバンド効果あり','753': '鼻逆効果','754': '舌くくりで出血','755': 'ハミで口切る','756': '馬場入り前蹄鉄打ち直','757': 'ハミ替え','758': '馬具変更','759': 'ハミ替え逆効果','760': '転厩','761': '厩務員替り',
            '762': '腰疲れ','763': '口硬い','764': '返し馬気合乗りすぎ','765': '口開けたまま走る','766': 'パドック後出し','767': '返し馬後出し','768': 'ハミ効かず','769': 'もたれる','770': '頭上げる','771': 'ブリンカー必要','772': 'フラフラ','773': '芝切れ目気にする','774': '芝切れ目躓く',
            '775': '脚部不安','779': '再ゲートやる気失う','780': '攻め駆けする','781': '直線追うのやめる','782': '異常歩様','783': '針立て直し効果○','784': 'ムキになって走る','785': '仕掛け遅れる','786': '手綱切れる','787': 'ゴチャつく','789': '併せる形○','790': 'ハミ敏感','791': 'トビ綺麗',
            '792': '硬い馬場○','793': '硬い馬場×','794': '調教注意（戒告）','795': '脚抜きいいダ○','796': '脚抜きいいダ×','797': '乾いたダ○','798': '乾いたダ×','799': '中間フレグモーネ','800': 'ソエ良化','801': '馬場入場後再検量','802': '多頭数○','803': '耳立てる','804': '直線余力あり',
            '805': 'ゲート入り嫌がる','806': 'ヨレる','807': '併せる形×','808': '右回り△','809': '荒れ馬場○','810': '荒れ馬場△','811': '荒れ馬場×','812': 'ハナこだわらず','813': 'スタート抜群','814': '蹄鉄ずれる','815': 'レース直前ボロ','816': '仕掛け早い','817': '馬場良い所通る',
            '818': '馬場悪い所通る','819': '展開厳しい','820': '展開恵まれ','821': '不正駆歩','822': '濡れたダ○','823': '返し馬落馬','824': '暖かくなってくる○','825': '水浮くダ×','826': 'パドック先頭','827': '返し馬スムーズ','828': '返し馬逆方向へ','829': 'パドック尻尾振る','830': '制御が乱暴',
            '831': '濡れたダ×','832': 'パドック立ち上がる','833': 'パドック舌出す','834': '返し馬気合つける','835': '返し馬元気','836': '返し馬のびのび','837': '返し馬口割る','838': '返し馬掛かる','839': '返し馬舌越す','840': 'ハミ気にする','841': '尻尾短い','842': '尻っ跳ね','843': 'あくび',
            '844': '馬場入り嫌がる','845': '厩務員に引かれて歩く','846': 'パドックハミ気にする','847': '軟らかい芝×','848': '背ったる','849': '馬場入場後騎乗','850': '返し馬立ち上がる','851': '返し馬尻っぱね','852': 'パドック外側から馬引','853': 'レースセンス抜群','854': '蛇行',
            '855': '(名パド)文字気にする','856': '厩務員に甘える','857': '返し馬軽目','858': '返し馬抑える','859': '器官疾患','860': '返し馬怒る','861': 'パドック柵蹴る','862': 'パドック機嫌良し','863': 'パドック口出血','864': '返し馬気負う','865': 'パドック耳動かす',
            '866': '返し馬やる気','867': '返し馬しっかり','868': '返し馬バネ利いた動き','869': '緩急苦手','870': '高速馬場○','871': '高速馬場×','872': '上がり速い○','873': '上がり速い×','874': '上がり掛かる○','875': '上がり掛かる×','876': '直線挟まる','877': '新潟外回り向き',
            '878': '新潟内回り向き','879': '競走中止','880': '完勝','881': '距離○','882': '展開向かず','883': '追って案外','884': '４角一杯','885': '出ムチ入る','886': '展開向く','887': '二の脚速い','888': '後ろから行くとダメ','889': '前に行く○','890': '後ろから行く○',
            '901': '左第１趾骨粉砕骨折','902': '右第１趾骨粉砕骨折','903': '左中手骨開放骨折','904': '右第３中手骨複骨折','905': '右第１指骨粉砕骨折','906': '左第１指骨開放骨折','907': '右第３中手骨開放骨折','908': '左第１指関節開放脱臼','909': '左下腿骨粉砕骨折','910': '左第１指節種子骨粉砕',
            '911': '右第３中手骨罅裂骨折','912': '左手根骨粉砕骨折','913': '右第１指骨開放骨折','914': '心不全','915': '左第１指骨複骨折','916': '右下腿骨開放骨折','917': '右中足骨開放骨折','918': '右第１指骨複骨折','919': '右第１指関節脱臼','920': '右第１指関節開放性脱',
            '921': '左手根骨複骨折','922': '左第１指関節開放性脱','923': '左第３中手骨開放骨折','924': '右前種子骨靱帯断裂','925': '両第１指関節開放性脱','926': '右前繋靭帯不全断裂','927': '歯ぎしり（パドック）','928': 'レース後口出血','929': '騎乗してパドック周回','930': '喉不安',
            '931': '右手根骨粉砕骨折','932': '両前浅屈腱不全断裂','933': '両後繋靭帯不全断裂','934': '両後管部挫創','935': '腹部挫創','936': '鼻梁部挫創','937': '背部挫創','938': '頭部打撲傷','939': '頭部挫傷','940': '頭部外傷','941': '舌部裂創','942': '肺充血',
            '943': '右第１指節種子骨粉砕','944': '右腸骨々折','0': ''}
    return dict.get(cd, '')

def convert_umakigo(cd):
    dict = {'01': '○抽', '02': '□抽', '03': '○父', '04': '○市', '05': '○地', '06': '○外', '07': '○父○抽', '08': '○父○市', '09': '○父○地', '10': '○市○地', '11': '○外○地', '12': '○父○市○地', '15': '○招', '16': '○招○外', '17': '○招○父', '18': '○招○市',
            '19': '○招○父○市', '20': '○父○外', '21': '□地', '22': '○外□地', '23': '○父□地', '24': '○市□地', '25': '○父○市□地', '26': '□外', '27': '○父□外', '0': '', '00': ''}
    return dict.get(cd, '')

def convert_bagu(cd):
    dict = {'001': 'ブリンカー', '002': 'シャドーロール', '003': 'リングハミ', '004': 'Dハミ', '005': 'エッグハミ', '006': '枝ハミ', '007': 'バンテージ', '008': 'メンコ', '009': 'ガムチェーン', '010': 'ハートハミ', '011': 'ハミ吊', '012': 'ビットガード', '013': 'ノートンハミ',
            '014': 'ジョウハミ', '015': 'スライド', '016': 'てこハミ', '017': 'イタイタ', '018': 'ノーズバンド', '019': 'チェーンシャンク', '020': 'パドックブリンカー', '021': '舌くくる', '022': '上唇くくる', '023': '馬気', '024': '下痢', '025': '二度汗', '026': '頭高い',
            '028': '毛艶良い', '030': '毛艶悪い', '031': 'ミックレム頭絡', '032': '引き返し', '036': 'レバーノーズバンド', '037': '保護テープ', '038': 'キネトンノーズバンド', '039': 'アダプターパッド', '040': 'ノーマルハミポチつき', '041': '皮膚病', '042': '玉腫れる', '043': 'フケ',
            '044': 'スリーリングハミ', '045': 'ソエ焼く', '047': '半鉄', '048': '連尾鉄', '049': '四分の三蹄鉄（曲）', '050': '鉄橋鉄', '051': '歩様悪い', '054': '四分の三蹄鉄', '055': '目の下黒い', '056': 'エクイロックス', '061': '骨瘤', '062': '骨瘤小', '063': 'ソエ腫れる',
            '064': 'ソエ腫れ小', '066': '鞍傷用スポンジ', '067': 'サイテーションハミ', '068': 'ネックストラップ', '069': 'ホライゾネット（レース）', '070': 'ホライゾネット（パドック）', '071': 'ハナゴム', '072': 'ユニバーサルハミ', '073': '蹄鉄なし', '074': 'チークピース',
            '075': '追突防止パッド', '076': '新エクイロックス', '077': 'スプーンヒール鉄', '078': '柿元鉄', '079': '耳当て', '080': '体毛剃る', '081': 'プラスチックカップ', '082': 'マウスネット', '083': 'ブロウピース', '084': 'ヒールパッド', '085': 'リバーシブル鉄',
            '087': '歯ぎしり', '088': 'リーグルハミ', '089': 'ホートンハミ', '090': 'トライアハミ', '091': 'シガフース蹄鉄', '092': 'ピーウィーハミ', '094': '裂蹄', '095': '蹄底パッド', '096': 'アイシールド', '097': 'eハミ', '098': 'タンプレートハミ', '099': '耳栓', '0': '', '000': ''}
    return dict.get(cd, '')

def convert_babajotai(cd):
    dict = {'10': '良', '11': '速良', '12': '遅良', '20': '稍重', '21': '速稍', '22': '遅稍', '30': '重', '31': '速重', '32': '遅重', '40': '不良', '41': '速不', '42': '遅不',  '0': ''}
    return dict.get(cd, '')

def convert_batai(cd):
    dict = {'1': '太', '2': '余', '3': '良', '4': '普', '5': '細', '6': '張', '7': '緩', '0': ''}
    return dict.get(cd, '')

def convert_keiro(cd):
    dict = {'01': '栗毛', '02': '栃栗', '03': '鹿毛', '04': '黒鹿', '05': '青鹿', '06': '青毛', '07': '芦毛', '08': '栗粕', '09': '鹿粕', '10': '青粕', '11': '白毛', '0': ''}
    return dict.get(cd, '')

def convert_shida(cd):
    dict = {'1': '芝', '2': 'ダート', '3': '障害', '0': ''}
    return dict.get(cd, '')

def convert_shibatype(cd):
    dict = {'1': '野芝', '2': '洋芝', '3': '混生', '0': ''}
    return dict.get(cd, '')

def convert_yusokubun(cd):
    dict = {'1': '滞在', '2': '通常', '3': '遠征', '4': '連闘', '0': ''}
    return dict.get(cd, '')

def convert_minaraikubun(cd):
    dict = {'1': '☆(1K減)', '2': '△(2K減)', '3': '▲(3K減)', '9': '', '0': ''}
    return dict.get(cd, '')

def convert_tekisei(cd):
    dict = {'1': '◎', '2': '○', '3': '△', '0': ''}
    return dict.get(cd, '')

def convert_tenkaimark(cd):
    dict = {'1': '展開＜', '2': '展開@', '3': '展開*', '4': '展開?', '0': ''}
    return dict.get(cd, '')

def convert_sex(cd):
    dict = {'1': '牡馬', '2': '牝馬', '3': '騙馬', '0': ''}
    return dict.get(cd, '')

def convert_taikei(cd):
    dict = {'1': '長方形', '2': '普通', '3': '正方形', '0': ''}
    return dict.get(cd, '')

def convert_taikei_big(cd):
    dict = {'1': '大きい', '2': '普通', '3': '小さい', '0': ''}
    return dict.get(cd, '')

def convert_taikei_kakudo(cd):
    dict = {'1': '大きい', '2': '普通', '3': '小さい', '0': ''}
    return dict.get(cd, '')

def convert_taikei_hohaba(cd):
    dict = {'1': '広い', '2': '普通', '3': '狭い', '0': ''}
    return dict.get(cd, '')

def convert_taikei_long(cd):
    dict = {'1': '長い', '2': '普通', '3': '短い', '0': ''}
    return dict.get(cd, '')

def convert_taikei_tsukene(cd):
    dict = {'1': '上げる', '2': '下げる',  '0': ''}
    return dict.get(cd, '')

def convert_taikei_o(cd):
    dict = {'1': '激しい', '2': '少し', '3': 'あまり振らない', '0': ''}
    return dict.get(cd, '')

def convert_chokyo_course(cd):
    dict = {
        '01': '美浦坂路', '02': '南Ｗ', '03': '南Ｄ', '04': '南芝', '05': '南Ａ', '06': '北Ｂ', '07': '北Ｃ', '08': '美浦障害芝', '09': '美浦プール', '10': '南ポリトラック',
        '11': '栗東坂路', '12': 'ＣＷ', '13': 'ＤＷ', '14': '栗Ｂ', '15': '栗Ｅ', '16': '栗芝', '17': '栗ポリトラック', '18': '栗東障害', '19': '栗東プール', '21': '札幌ダ',
        '22': '札幌芝', '23': '函館ダ', '24': '函館芝', '25': '函館Ｗ', '26': '福島芝', '27': '福島ダ', '28': '新潟芝', '29': '新潟ダ', '30': '東京芝', '31': '東京ダ',
        '32': '中山芝', '33': '中山ダ', '34': '中京芝', '35': '中京ダ', '36': '京都芝', '37': '京都ダ', '38': '阪神芝', '39': '阪神ダ', '40': '小倉芝', '41': '小倉ダ',
        '42': '福島障害', '43': '新潟障害', '44': '東京障害', '45': '中山障害', '46': '中京障害', '47': '京都障害', '48': '阪神障害', '49': '小倉障害', '50': '地方競馬',
        '61': '障害試験', '62': '北障害', '68': '美障害ダ', '70': '北A', '81': '美ゲート', '82': '栗ゲート', '88': '牧場', '93': '白井ダ', 'A1': '連闘', 'B1': 'その他', '0': ''}
    return dict.get(cd, '')

def convert_oikiri_shurui(cd):
    dict = {'1': '一杯', '2': '強目', '3': '馬なり', '0': ''}
    return dict.get(cd, '')

def convert_chokyo_course_shubetsu(cd):
    dict = {'1': '坂路調教', '2': 'コース調教', '3': '併用', '4': '障害調教', '5': '障害調教他', '0': '調教なし'}
    return dict.get(cd, '')

def convert_chokyo_kyori(cd):
    dict = {'1': '長め調教', '2': '標準調教', '3': '短め調教', '4': '調教２本', '0': ''}
    return dict.get(cd, '')

def convert_chokyo_juten(cd):
    dict = {'1': 'テン重点', '2': '中間重点', '3': '終い重点', '4': '平均的', '0': ''}
    return dict.get(cd, '')
