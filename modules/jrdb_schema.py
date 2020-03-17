import pandas as pd
import unicodedata
import datetime as dt
from collections import deque

import modules.util as mu

# memo pymssqlのインストールに苦戦
# https://anaconda.org/anaconda/pymssql conda install -c anaconda pymssqlをすればpymssql2.1.4がインストール出来る。


class JrdbSchemaBase(object):
    def __init__(self):
        self.filename = ""

    def replace_line(self, line):
        count = 0
        new_line = ''
        for c in line:
            if unicodedata.east_asian_width(c) in 'FWA':
                new_line += c + ' '
                count += 2
            else:
                new_line += c
                count += 1
        return new_line


class BAC(JrdbSchemaBase):
    def __init__(self):
        super().__init__()
        self.names = ['RACE_KEY', 'NENGAPPI', 'HASSO_JIKOKU', 'KYORI', 'SHIBA_DART', 'MIGIHIDARI', 'UCHISOTO', 'SHUBETSU', 'JOKEN', 'KIGO', 'JURYO', 'GRADE', 'RACE_NAME', 'KAISU', 'TOSU', 'COURSE', 'KAISAI_KUBUN',
                      'RACE_TANSHUKU', 'RACE_TANSHUKU_9', 'DATA_KUBUN', 'SHOKIN_1CK', 'SHOKIN_2CK', 'SHOKIN_3CK', 'SHOKIN_4CK', 'SHOKIN_5CK', 'SANNYU_SHOKIN_1CK', 'SANNYU_SHOKIN_2CK', 'BAKEN_HATSUBAI_FLAG', 'WIN5', 'file_name', 'target_date']
        self.columns = ['%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s',
                        '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s']
        self.create_sql = self.sql_create()
        self.table_name = '[TALEND].[jrdb].[ORG_BAC]'

    def set_df(self, filename):
        with open('./' + filename, 'r', encoding="ms932") as fh:
            df = pd.DataFrame(index=[], columns=self.names)
            for line in fh:
                new_line = self.replace_line(line)
                sr = pd.Series([
                    new_line[0:8]  # RACE_KEY
                    , new_line[8:16]  # NENGAPPI
                    , new_line[16:20]  # HASSO_JIKOKU
                    , mu.int_null(new_line[20:24])  # KYORI
                    , new_line[24:25]  # SHIBA_DART
                    , new_line[25:26]  # MIGIHIDARI
                    , new_line[26:27]  # UCHISOTO
                    , new_line[27:29]  # SHUBETSU
                    , new_line[29:31]  # JOKEN
                    , new_line[31:34]  # KIGO
                    , new_line[34:35]  # JURYO
                    , new_line[35:36]  # GRADE
                    , new_line[36:86].replace(" ", "")  # RACE_NAME
                    , new_line[86:94].replace(" ", "")  # KAISU
                    , mu.int_null(new_line[94:96])  # TOSU
                    , new_line[96:97]  # COURSE
                    , new_line[97:98]  # KAISAI_KUBUN
                    , new_line[98:106].replace(" ", "")  # RACE_TANSHUKU
                    , new_line[106:124].replace(" ", "")  # RACE_TANSHUKU_9
                    , new_line[124:125]  # DATA_KUBUN
                    , mu.int_null(new_line[125:130])  # SHOKIN_1CK
                    , mu.int_null(new_line[130:135])  # SHOKIN_2CK
                    , mu.int_null(new_line[135:140])  # SHOKIN_3CK
                    , mu.int_null(new_line[140:145])  # SHOKIN_4CK
                    , mu.int_null(new_line[145:150])  # SHOKIN_5CK
                    , mu.int_null(new_line[150:155])  # SANNYU_SHOKIN_1CK
                    , mu.int_null(new_line[155:160])  # SANNYU_SHOKIN_2CK
                    , new_line[160:176]  # BAKEN_HATSUBAI_FLAG
                    , new_line[176:177]  # WIN5
                    , filename, filename[3:9]
                ], index=self.names)
                df = df.append(sr, ignore_index=True)
        return df

    def sql_create(self):
        create_sql = """if object_id('[TALEND].[jrdb].[ORG_BAC]') is null
        CREATE TABLE [TALEND].[jrdb].[ORG_BAC] (
        RACE_KEY nvarchar(8),
        NENGAPPI nvarchar(8),
        HASSO_JIKOKU nvarchar(4),
        KYORI int,
        SHIBA_DART nvarchar(1),
        MIGIHIDARI nvarchar(1),
        UCHISOTO nvarchar(1),
        SHUBETSU nvarchar(2),
        JOKEN nvarchar(2),
        KIGO nvarchar(3),
        JURYO nvarchar(1),
        GRADE nvarchar(1),
        RACE_NAME nvarchar(50),
        KAISU nvarchar(8),
        TOSU int,
        COURSE nvarchar(1),
        KAISAI_KUBUN nvarchar(1),
        RACE_TANSHUKU nvarchar(8),
        RACE_TANSHUKU_9 nvarchar(18),
        DATA_KUBUN nvarchar(1),
        SHOKIN_1CK int,
        SHOKIN_2CK int,
        SHOKIN_3CK int,
        SHOKIN_4CK int,
        SHOKIN_5CK int,
        SANNYU_SHOKIN_1CK int,
        SANNYU_SHOKIN_2CK int,
        BAKEN_HATSUBAI_FLAG nvarchar(16),
        WIN5 nvarchar(1),
        filename nvarchar(20),
        target_date          nvarchar(6)
        );
        """
        return create_sql


class CHA(JrdbSchemaBase):
    def __init__(self):
        super().__init__()
        self.names = ['RACE_KEY', 'UMABAN', 'YOBI', 'CHOKYO_NENGAPPI', 'KAISU', 'CHOKYO_COURSE', 'OIKIRI_SHURUI', 'OI_JOTAI', 'NORIYAKU', 'CHOKYO_F', 'TEN_F', 'CHUKAN_F',
                      'SHIMAI_F', 'TEN_F_RECORD', 'CHUKAN_F_RECORD', 'SHIMAI_F_RECORD', 'OIKIRI_RECORD', 'AWASE_KEKKA', 'AWASE_OIKIRI_SHURUI', 'NENREI', 'CLASS', 'file_name', 'target_date']
        self.columns = ['%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s',
                        '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s']
        self.create_sql = self.sql_create()
        self.table_name = '[TALEND].[jrdb].[ORG_CHA]'

    def set_df(self, filename):
        with open('./' + filename, 'r', encoding="ms932") as fh:
            df = pd.DataFrame(index=[], columns=self.names)
            for line in fh:
                new_line = self.replace_line(line)
                sr = pd.Series([
                    new_line[0:8]  # RACE_KEY
                    , new_line[8:10]  # UMABAN
                    , new_line[10:12]  # YOBI
                    , new_line[12:20]  # CHOKYO_NENGAPPI
                    , new_line[20:21]  # KAISU
                    , new_line[21:23]  # CHOKYO_COURSE
                    , new_line[23:24]  # OIKIRI_SHURUI
                    , new_line[24:26]  # OI_JOTAI
                    , new_line[26:27]  # NORIYAKU
                    , mu.int_null(new_line[27:28])  # CHOKYO_F
                    , mu.int_null(new_line[28:31])  # TEN_F
                    , mu.int_null(new_line[31:34])  # CHUKAN_F
                    , mu.int_null(new_line[34:37])  # SHIMAI_F
                    , mu.int_null(new_line[37:40])  # TEN_F_RECORD
                    , mu.int_null(new_line[40:43])  # CHUKAN_F_RECORD
                    , mu.int_null(new_line[43:46])  # SHIMAI_F_RECORD
                    , mu.int_null(new_line[46:49])  # OIKIRI_RECORD
                    , new_line[49:50]  # AWASE_KEKKA
                    , new_line[50:51]  # AWASE_OIKIRI_SHURUI
                    , mu.int_null(new_line[51:53])  # NENREI
                    , new_line[53:55]  # CLASS
                    , filename, filename[3:9]
                ], index=self.names)
                df = df.append(sr, ignore_index=True)
        return df

    def sql_create(self):
        create_sql = """if object_id('[TALEND].[jrdb].[ORG_CHA]') is null
        CREATE TABLE [TALEND].[jrdb].[ORG_CHA] (
        RACE_KEY             nvarchar(8),
        UMABAN               nvarchar(2),
        YOBI                 nvarchar(2),
        CHOKYO_NENGAPPI      nvarchar(8),
        KAISU                nvarchar(1),
        CHOKYO_COURSE        nvarchar(2),
        OIKIRI_SHURUI        nvarchar(1),
        OI_JOTAI             nvarchar(2),
        NORIYAKU             nvarchar(1),
        CHOKYO_F             nvarchar(1),
        TEN_F                int,
        CHUKAN_F             int,
        SHIMAI_F             int,
        TEN_F_RECORD         int,
        CHUKAN_F_RECORD      int,
        SHIMAI_F_RECORD      int,
        OIKIRI_RECORD        int,
        AWASE_KEKKA          nvarchar(1),
        AWASE_OIKIRI_SHURUI  nvarchar(1),
        NENREI               int,
        CLASS                nvarchar(2),
        filename             nvarchar(20),
        target_date          nvarchar(6)
        );
        """
        return create_sql


class CYB(JrdbSchemaBase):
    def __init__(self):
        super().__init__()
        self.names = ['RACE_KEY', 'UMABAN', 'CHOKYO_TYPE', 'CHOKYO_COURSE_SHUBETSU', 'HANRO', 'WOOD', 'DART', 'SHIBA', 'POOL', 'SHOBAI', 'POLYTRACK', 'CHOKYO_KYORI', 'CHOKYO_JUTEN',
                      'OIKIRI_RECORD', 'SHIAGE_RECORD', 'CHOKYORYO_HYOUKA', 'SHIAGE_RECORD_HENKA', 'CHOKYO_COMMENT', 'COMENT_NENGAPPI', 'CHOKYO_HYOKA', 'file_name', 'target_date']
        self.columns = ['%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s',
                        '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s']
        self.create_sql = self.sql_create()
        self.table_name = '[TALEND].[jrdb].[ORG_CYB]'

    def set_df(self, filename):
        with open('./' + filename, 'r', encoding="ms932") as fh:
            df = pd.DataFrame(index=[], columns=self.names)
            for line in fh:
                new_line = self.replace_line(line)
                sr = pd.Series([
                    new_line[0:8]  # RACE_KEY
                    , new_line[8:10]  # UMABAN
                    , new_line[10:12]  # CHOKYO_TYPE
                    , new_line[12:13]  # CHOKYO_COURSE_SHUBETSU
                    , mu.int_null(new_line[13:15])  # HANRO
                    , mu.int_null(new_line[15:17])  # WOOD
                    , mu.int_null(new_line[17:19])  # DART
                    , mu.int_null(new_line[19:21])  # SHIBA
                    , mu.int_null(new_line[21:23])  # POOL
                    , mu.int_null(new_line[23:25])  # SHOBAI
                    , mu.int_null(new_line[25:27])  # POLYTRACK
                    , new_line[27:28]  # CHOKYO_KYORI
                    , new_line[28:29]  # CHOKYO_JUTEN
                    , mu.int_null(new_line[29:32])  # OIKIRI_RECORD
                    , mu.int_null(new_line[32:35])  # SHIAGE_RECORD
                    , new_line[35:36]  # CHOKYORYO_HYOUKA
                    , new_line[36:37]  # SHIAGE_RECORD_HENKA
                    , new_line[37:77]  # CHOKYO_COMMENT
                    , new_line[77:85]  # COMENT_NENGAPPI
                    , new_line[85:86]  # CHOKYO_HYOKA
                    , filename, filename[3:9]
                ], index=self.names)
                df = df.append(sr, ignore_index=True)
        return df

    def sql_create(self):
        create_sql = """if object_id('[TALEND].[jrdb].[ORG_CYB]') is null
        CREATE TABLE [TALEND].[jrdb].[ORG_CYB] (
        RACE_KEY             nvarchar(8),
        UMABAN               nvarchar(2),
        CHOKYO_TYPE                 nvarchar(2),
        CHOKYO_COURSE_SHUBETSU      nvarchar(1),
        HANRO                int,
        WOOD                 int,
        DART                 int,
        SHIBA                int,
        POOL                 int,
        SHOBAI               int,
        POLYTRACK            int,
        CHOKYO_KYORI         nvarchar(1),
        CHOKYO_JUTEN         nvarchar(1),
        OIKIRI_RECORD        int,
        SHIAGE_RECORD        int,
        CHOKYORYO_HYOUKA     nvarchar(1),
        SHIAGE_RECORD_HENKA  nvarchar(1),
        CHOKYO_COMMENT       nvarchar(40),
        COMENT_NENGAPPI      nvarchar(8),
        CHOKYO_HYOKA         nvarchar(1),
        filename             nvarchar(20),
        target_date          nvarchar(6)
        );
        """
        return create_sql


class JOA(JrdbSchemaBase):
    def __init__(self):
        super().__init__()
        self.names = ['RACE_KEY', 'UMABAN', 'KETTO_TOROUK_BANGO', 'UMA_NAME', 'KIJUN_ODDS', 'KIJUN_FUKUSHO_ODDS', 'CID_CHOKYO_SOTEN', 'CID_KYUSHA_SOTEN', 'CID_SOTEN', 'CID', 'LS_RECORD',
                      'LS_HYOKA', 'EM', 'KYUSHA_BB_MARK', 'KYUSHA_BB_TANSHO_KAISHU', 'KYUSHA_BB_RENTAIRITSU', 'KISHU_BB_MARK', 'KISHU_BB_TANSHO_KAISHU', 'TANSHO_BB_RENTAIRITSU', 'file_name', 'target_date']
        self.columns = ['%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s',
                        '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s']
        self.create_sql = self.sql_create()
        self.table_name = '[TALEND].[jrdb].[ORG_JOA]'

    def set_df(self, filename):
        with open('./' + filename, 'r', encoding="ms932") as fh:
            df = pd.DataFrame(index=[], columns=self.names)
            for line in fh:
                new_line = self.replace_line(line)
                sr = pd.Series([
                    new_line[0:8]  # RACE_KEY
                    , new_line[8:10]  # UMABAN
                    , new_line[10:18]  # KETTO_TOROUK_BANGO
                    , new_line[18:54].replace(" ", "")  # UMA_NAME
                    , mu.float_null(new_line[54:59])  # KIJUN_ODDS
                    , mu.float_null(new_line[59:64])  # KIJUN_FUKUSHO_ODDS
                    , mu.float_null(new_line[64:69])  # CID_CHOKYO_SOTEN
                    , mu.float_null(new_line[69:74])  # CID_KYUSHA_SOTEN
                    , mu.float_null(new_line[74:79])  # CID_SOTEN
                    , mu.int_null(new_line[79:82])  # CID
                    , mu.float_null(new_line[82:87])  # LS_RECORD
                    , new_line[87:88]  # LS_HYOKA
                    , new_line[88:89]  # EM
                    , new_line[89:90]  # KYUSHA_BB_MARK
                    , mu.int_null(new_line[90:95])  # KYUSHA_BB_TANSHO_KAISHU
                    , mu.int_null(new_line[95:100])  # KYUSHA_BB_RENTAIRITSU
                    , new_line[100:101]  # KISHU_BB_MARK
                    , mu.int_null(new_line[101:106])  # KISHU_BB_TANSHO_KAISHU
                    , mu.int_null(new_line[106:111])  # TANSHO_BB_RENTAIRITSU
                    , filename, filename[3:9]
                ], index=self.names)
                df = df.append(sr, ignore_index=True)
        return df

    def sql_create(self):
        create_sql = """if object_id('[TALEND].[jrdb].[ORG_JOA]') is null
        CREATE TABLE [TALEND].[jrdb].[ORG_JOA] (
        RACE_KEY             nvarchar(8),
        UMABAN               nvarchar(2),
        KETTO_TOROUK_BANGO   nvarchar(8),
        UMA_NAME             nvarchar(36),
        KIJUN_ODDS           real,
        KIJUN_FUKUSHO_ODDS   real,
        CID_CHOKYO_SOTEN     real,
        CID_KYUSHA_SOTEN     real,
        CID_SOTEN            real,
        CID                  int,
        LS_RECORD            real,
        LS_HYOKA             nvarchar(1),
        EM                   nvarchar(1),
        KYUSHA_BB_MARK       nvarchar(1),
        KYUSHA_BB_TANSHO_KAISHU  int,
        KYUSHA_BB_RENTAIRITSU    int,
        KISHU_BB_MARK            nvarchar(1),
        KISHU_BB_TANSHO_KAISHU   int,
        TANSHO_BB_RENTAIRITSU    int,
        filename             nvarchar(20),
        target_date          nvarchar(6)
        );
        """
        return create_sql


class KAB(JrdbSchemaBase):
    def __init__(self):
        super().__init__()
        self.names = ['KAISAI_KEY', 'NENGAPPI', 'KAISAI_KUBUN', 'YOUBI', 'BASHO_NAME', 'TENKO', 'SHIBA_JOTAI', 'SHIBA_JOTAI_UCHI', 'SHIBA_JOTAI_NAKA', 'SHIBA_JOTAI_SOTO', 'SHIBA_BABASA', 'CHOKUSEN_BABASA_SAIUCHI', 'CHOKUSEN_BABASA_UCHI', 'CHOKUSEN_BABASA_NAKA',
                      'CHOKUSEN_BABASA_SOTO', 'CHOKUSEN_BABASA_OOSOTO', 'DART_BABA_JOTAI', 'DART_JOTAI_UCHI', 'DART_JOTAI_NAKA', 'DART_JOTAI_SOTO', 'DART_BABASA', 'DATA_KUBUN', 'RENZOKU', 'SHIBA_SHURUI', 'KUSATAKE', 'TENATSU', 'TOKETSU_BOSHI', 'CHUKAN_KOUSUI', 'file_name', 'target_date']
        self.columns = ['%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s',
                        '%s', '%s', ' %s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s']
        self.create_sql = self.sql_create()
        self.table_name = '[TALEND].[jrdb].[ORG_KAB]'

    def set_df(self, filename):
        with open('./' + filename, 'r', encoding="ms932") as fh:
            df = pd.DataFrame(index=[], columns=self.names)
            for line in fh:
                new_line = self.replace_line(line)
                sr = pd.Series([
                    new_line[0:6]  # KAISAI_KEY
                    , new_line[6:14]  # NENGAPPI
                    , new_line[14:15]  # KAISAI_KUBUN
                    , new_line[15:17]  # YOUBI
                    , new_line[17:21].replace(" ", "")  # BASHO_NAME
                    , new_line[21:22]  # TENKO
                    , new_line[22:24]  # SHIBA_JOTAI
                    , new_line[24:25]  # SHIBA_JOTAI_UCHI
                    , new_line[25:26]  # SHIBA_JOTAI_NAKA
                    , new_line[26:27]  # SHIBA_JOTAI_SOTO
                    , new_line[27:30]  # SHIBA_BABASA
                    , new_line[30:32]  # CHOKUSEN_BABASA_SAIUCHI
                    , new_line[32:34]  # CHOKUSEN_BABASA_UCHI
                    , new_line[34:36]  # CHOKUSEN_BABASA_NAKA
                    , new_line[36:38]  # CHOKUSEN_BABASA_SOTO
                    , new_line[38:40]  # CHOKUSEN_BABASA_OOSOTO
                    , new_line[40:42]  # DART_BABA_JOTAI
                    , new_line[42:43]  # DART_JOTAI_UCHI
                    , new_line[43:44]  # DART_JOTAI_NAKA
                    , new_line[44:45]  # DART_JOTAI_SOTO
                    , new_line[45:48]  # DART_BABASA
                    , new_line[48:49]  # DATA_KUBUN
                    , mu.int_null(new_line[49:51])  # RENZOKU
                    , new_line[51:52]  # SHIBA_SHURUI
                    , mu.float_null(new_line[52:56])  # KUSATAKE
                    , new_line[56:57]  # TENATSU
                    , new_line[57:58]  # TOKETSU_BOSHI
                    , mu.float_null(new_line[58:63])  # CHUKAN_KOUSUI
                    , filename, filename[3:9]
                ], index=self.names)
                df = df.append(sr, ignore_index=True)
        return df

    def sql_create(self):
        create_sql = """if object_id('[TALEND].[jrdb].[ORG_KAB]') is null
        CREATE TABLE [TALEND].[jrdb].[ORG_KAB] (
        KAISAI_KEY               nvarchar(6),
        NENGAPPI                 nvarchar(8),
        KAISAI_KUBUN             nvarchar(1),
        YOUBI                    nvarchar(2),
        BASHO_NAME               nvarchar(4),
        TENKO                    nvarchar(1),
        SHIBA_JOTAI              nvarchar(2),
        SHIBA_JOTAI_UCHI         nvarchar(1),
        SHIBA_JOTAI_NAKA         nvarchar(1),
        SHIBA_JOTAI_SOTO         nvarchar(1),
        SHIBA_BABASA             nvarchar(3),
        CHOKUSEN_BABASA_SAIUCHI  nvarchar(2),
        CHOKUSEN_BABASA_UCHI     nvarchar(2),
        CHOKUSEN_BABASA_NAKA     nvarchar(2),
        CHOKUSEN_BABASA_SOTO     nvarchar(2),
        CHOKUSEN_BABASA_OOSOTO   nvarchar(2),
        DART_BABA_JOTAI          nvarchar(2),
        DART_JOTAI_UCHI          nvarchar(1),
        DART_JOTAI_NAKA          nvarchar(1),
        DART_JOTAI_SOTO          nvarchar(1),
        DART_BABASA              nvarchar(3),
        DATA_KUBUN               nvarchar(1),
        RENZOKU                  int,
        SHIBA_SHURUI             nvarchar(1),
        KUSATAKE                 real,
        TENATSU                  nvarchar(1),
        TOKETSU_BOSHI            nvarchar(1),
        CHUKAN_KOUSUI            real,
        filename                 nvarchar(20),
        target_date          nvarchar(6)
        );
        """
        return create_sql


class KKA(JrdbSchemaBase):
    def __init__(self):
        super().__init__()
        self.names = ['RACE_KEY', 'UMABAN', 'JRA_1CK', 'JRA_2CK', 'JRA_3CK', 'JRA_GAI', 'NRA_1CK', 'NRA_2CK', 'NRA_3CK', 'NRA_GAI', 'OTHER_1CK', 'OTHER_2CK', 'OTHER_3CK', 'OTHER_GAI', 'SHIDA_1CK', 'SHIDA_2CK', 'SHIDA_3CK', 'SHIDA_GAI', 'SHIDA_KYORI_1CK', 'SHIDA_KYORI_2CK', 'SHIDA_KYORI_3CK', 'SHIDA_KYORI_GAI', 'TRACK_1CK', 'TRACK_2CK', 'TRACK_3CK', 'TRACK_GAI', 'ROTE_1CK', 'ROTE_2CK', 'ROTE_3CK', 'ROTE_GAI', 'MAWARI_1CK', 'MAWARI_2CK', 'MAWARI_3CK', 'MAWARI_GAI', 'KISHU_1CK', 'KISHU_2CK', 'KISHU_3CK', 'KISHU_GAI', 'RYO_1CK', 'RYO_2CK', 'RYO_3CK', 'RYO_GAI', 'YAYA_1CK', 'YAYA_2CK', 'YAYA_3CK', 'YAYA_GAI', 'OMO_1CK', 'OMO_2CK', 'OMO_3CK', 'OMO_GAI', 'S_1CK', 'S_2CK', 'S_3CK', 'S_GAI', 'M_1CK', 'M_2CK', 'M_3CK', 'M_GAI', 'H_1CK', 'H_2CK', 'H_3CK', 'H_GAI', 'SEASON_1CK',
                      'SEASON_2CK', 'SEASON_3CK', 'SEASON_GAI', 'WAKU_1CK', 'WAKU_2CK', 'WAKU_3CK', 'WAKU_GAI', 'KISHU_KYORI_1CK', 'KISHU_KYORI_2CK', 'KISHU_KYORI_3CK', 'KISHU_KYORI_GAI', 'KISHU_TRACK_1CK', 'KISHU_TRACK_2CK', 'KISHU_TRACK_3CK', 'KISHU_TRACK_GAI', 'KISHU_CHOKYOSHI_1CK', 'KISHU_CHOKYOSHI_2CK', 'KISHU_CHOKYOSHI_3CK', 'KISHU_CHOKYOSHI_GAI', 'KISHU_BANUSHI_1CK', 'KISHU_BANUSHI_2CK', 'KISHU_BANUSHI_3CK', 'KISHU_BANUSHI_GAI', 'KISHU_BURINKA_1CK', 'KISHU_BURINKA_2CK', 'KISHU_BURINKA_3CK', 'KISHU_BURINKA_GAI', 'CHOKYOSHI_BANUSHI_1CK', 'CHOKYOSHI_BANUSHI_2CK', 'CHOKYOSHI_BANUSHI_3CK', 'CHOKYOSHI_BANUSHI_GAI', 'CHICHI_SHIBA_RENTAI', 'CHICHI_DART_RENTAI', 'CHICHI_RENTAI_KYORI', 'HAHACHICHI_SHIBA_RENTAI', 'HAHACHICHI_DART_RENTAI', 'HAHACHICHI_RENTAI_KYORI', 'file_name', 'target_date']
        self.columns = ['%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s',
                        '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s']
        self.create_sql = self.sql_create()
        self.table_name = '[TALEND].[jrdb].[ORG_KKA]'

    def set_df(self, filename):
        with open('./' + filename, 'r', encoding="ms932") as fh:
            df = pd.DataFrame(index=[], columns=self.names)
            for line in fh:
                new_line = self.replace_line(line)
                sr = pd.Series([
                    new_line[0:8]  # RACE_KEY
                    , new_line[8:10]  # UMABAN
                    , mu.int_null(new_line[10:13])  # JRA_1CK
                    , mu.int_null(new_line[13:16])  # JRA_2CK
                    , mu.int_null(new_line[16:19])  # JRA_3CK
                    , mu.int_null(new_line[19:22])  # JRA_GAI
                    , mu.int_null(new_line[22:25])  # NRA_1CK
                    , mu.int_null(new_line[25:28])  # NRA_2CK
                    , mu.int_null(new_line[28:31])  # NRA_3CK
                    , mu.int_null(new_line[31:34])  # NRA_GAI
                    , mu.int_null(new_line[34:37])  # OTHER_1CK
                    , mu.int_null(new_line[37:40])  # OTHER_2CK
                    , mu.int_null(new_line[40:43])  # OTHER_3CK
                    , mu.int_null(new_line[43:46])  # OTHER_GAI
                    , mu.int_null(new_line[46:49])  # SHIDA_1CK
                    , mu.int_null(new_line[49:52])  # SHIDA_2CK
                    , mu.int_null(new_line[52:55])  # SHIDA_3CK
                    , mu.int_null(new_line[55:58])  # SHIDA_GAI
                    , mu.int_null(new_line[58:61])  # SHIDA_KYORI_1CK
                    , mu.int_null(new_line[61:64])  # SHIDA_KYORI_2CK
                    , mu.int_null(new_line[64:67])  # SHIDA_KYORI_3CK
                    , mu.int_null(new_line[67:70])  # SHIDA_KYORI_GAI
                    , mu.int_null(new_line[70:73])  # TRACK_1CK
                    , mu.int_null(new_line[73:76])  # TRACK_2CK
                    , mu.int_null(new_line[76:79])  # TRACK_3CK
                    , mu.int_null(new_line[79:82])  # TRACK_GAI
                    , mu.int_null(new_line[82:85])  # ROTE_1CK
                    , mu.int_null(new_line[85:88])  # ROTE_2CK
                    , mu.int_null(new_line[88:91])  # ROTE_3CK
                    , mu.int_null(new_line[91:94])  # ROTE_GAI
                    , mu.int_null(new_line[94:97])  # MAWARI_1CK
                    , mu.int_null(new_line[97:100])  # MAWARI_2CK
                    , mu.int_null(new_line[100:103])  # MAWARI_3CK
                    , mu.int_null(new_line[103:106])  # MAWARI_GAI
                    , mu.int_null(new_line[106:109])  # KISHU_1CK
                    , mu.int_null(new_line[109:112])  # KISHU_2CK
                    , mu.int_null(new_line[112:115])  # KISHU_3CK
                    , mu.int_null(new_line[115:118])  # KISHU_GAI
                    , mu.int_null(new_line[118:121])  # RYO_1CK
                    , mu.int_null(new_line[121:124])  # RYO_2CK
                    , mu.int_null(new_line[124:127])  # RYO_3CK
                    , mu.int_null(new_line[127:130])  # RYO_GAI
                    , mu.int_null(new_line[130:133])  # YAYA_1CK
                    , mu.int_null(new_line[133:136])  # YAYA_2CK
                    , mu.int_null(new_line[136:139])  # YAYA_3CK
                    , mu.int_null(new_line[139:142])  # YAYA_GAI
                    , mu.int_null(new_line[142:145])  # OMO_1CK
                    , mu.int_null(new_line[145:148])  # OMO_2CK
                    , mu.int_null(new_line[148:151])  # OMO_3CK
                    , mu.int_null(new_line[151:154])  # OMO_GAI
                    , mu.int_null(new_line[154:157])  # S_1CK
                    , mu.int_null(new_line[157:160])  # S_2CK
                    , mu.int_null(new_line[160:163])  # S_3CK
                    , mu.int_null(new_line[163:166])  # S_GAI
                    , mu.int_null(new_line[166:169])  # M_1CK
                    , mu.int_null(new_line[169:172])  # M_2CK
                    , mu.int_null(new_line[172:175])  # M_3CK
                    , mu.int_null(new_line[175:178])  # M_GAI
                    , mu.int_null(new_line[178:181])  # H_1CK
                    , mu.int_null(new_line[181:184])  # H_2CK
                    , mu.int_null(new_line[184:187])  # H_3CK
                    , mu.int_null(new_line[187:190])  # H_GAI
                    , mu.int_null(new_line[190:193])  # SEASON_1CK
                    , mu.int_null(new_line[193:196])  # SEASON_2CK
                    , mu.int_null(new_line[196:199])  # SEASON_3CK
                    , mu.int_null(new_line[199:202])  # SEASON_GAI
                    , mu.int_null(new_line[202:205])  # WAKU_1CK
                    , mu.int_null(new_line[205:208])  # WAKU_2CK
                    , mu.int_null(new_line[208:211])  # WAKU_3CK
                    , mu.int_null(new_line[211:214])  # WAKU_GAI
                    , mu.int_null(new_line[214:217])  # KISHU_KYORI_1CK
                    , mu.int_null(new_line[217:220])  # KISHU_KYORI_2CK
                    , mu.int_null(new_line[220:223])  # KISHU_KYORI_3CK
                    , mu.int_null(new_line[223:226])  # KISHU_KYORI_GAI
                    , mu.int_null(new_line[226:229])  # KISHU_TRACK_1CK
                    , mu.int_null(new_line[229:232])  # KISHU_TRACK_2CK
                    , mu.int_null(new_line[232:235])  # KISHU_TRACK_3CK
                    , mu.int_null(new_line[235:238])  # KISHU_TRACK_GAI
                    , mu.int_null(new_line[238:241])  # KISHU_CHOKYOSHI_1CK
                    , mu.int_null(new_line[241:244])  # KISHU_CHOKYOSHI_2CK
                    , mu.int_null(new_line[244:247])  # KISHU_CHOKYOSHI_3CK
                    , mu.int_null(new_line[247:250])  # KISHU_CHOKYOSHI_GAI
                    , mu.int_null(new_line[250:253])  # KISHU_BANUSHI_1CK
                    , mu.int_null(new_line[253:256])  # KISHU_BANUSHI_2CK
                    , mu.int_null(new_line[256:259])  # KISHU_BANUSHI_3CK
                    , mu.int_null(new_line[259:262])  # KISHU_BANUSHI_GAI
                    , mu.int_null(new_line[262:265])  # KISHU_BURINKA_1CK
                    , mu.int_null(new_line[265:268])  # KISHU_BURINKA_2CK
                    , mu.int_null(new_line[268:271])  # KISHU_BURINKA_3CK
                    , mu.int_null(new_line[271:274])  # KISHU_BURINKA_GAI
                    , mu.int_null(new_line[274:277])  # CHOKYOSHI_BANUSHI_1CK
                    , mu.int_null(new_line[277:280])  # CHOKYOSHI_BANUSHI_2CK
                    , mu.int_null(new_line[280:283])  # CHOKYOSHI_BANUSHI_3CK
                    , mu.int_null(new_line[283:286])  # CHOKYOSHI_BANUSHI_GAI
                    , mu.int_null(new_line[286:289])  # CHICHI_SHIBA_RENTAI
                    , mu.int_null(new_line[289:292])  # CHICHI_DART_RENTAI
                    , mu.int_null(new_line[292:296])  # CHICHI_RENTAI_KYORI
                    # HAHACHICHI_SHIBA_RENTAI
                    # HAHACHICHI_DART_RENTAI
                    # HAHACHICHI_RENTAI_KYORI
                    , mu.int_null(new_line[296:298]), mu.int_null(new_line[299:302]), mu.int_null(new_line[302:306]), filename, filename[3:9]
                ], index=self.names)
                df = df.append(sr, ignore_index=True)
        return df

    def sql_create(self):
        create_sql = """if object_id('[TALEND].[jrdb].[ORG_KKA]') is null
        CREATE TABLE [TALEND].[jrdb].[ORG_KKA] (
        RACE_KEY               nvarchar(8),
        UMABAN                 nvarchar(2),
        JRA_1CK                 int,
        JRA_2CK                 int,
        JRA_3CK                 int,
        JRA_GAI                 int,
        NRA_1CK                 int,
        NRA_2CK                 int,
        NRA_3CK                 int,
        NRA_GAI                 int,
        OTHER_1CK                 int,
        OTHER_2CK                 int,
        OTHER_3CK                 int,
        OTHER_GAI                 int,
        SHIDA_1CK                 int,
        SHIDA_2CK                 int,
        SHIDA_3CK                 int,
        SHIDA_GAI                 int,
        SHIDA_KYORI_1CK                 int,
        SHIDA_KYORI_2CK                 int,
        SHIDA_KYORI_3CK                 int,
        SHIDA_KYORI_GAI                 int,
        TRACK_1CK                 int,
        TRACK_2CK                 int,
        TRACK_3CK                 int,
        TRACK_GAI                 int,
        ROTE_1CK                 int,
        ROTE_2CK                 int,
        ROTE_3CK                 int,
        ROTE_GAI                 int,
        MAWARI_1CK                 int,
        MAWARI_2CK                 int,
        MAWARI_3CK                 int,
        MAWARI_GAI                 int,
        KISHU_1CK                 int,
        KISHU_2CK                 int,
        KISHU_3CK                 int,
        KISHU_GAI                 int,
        RYO_1CK                 int,
        RYO_2CK                 int,
        RYO_3CK                 int,
        RYO_GAI                 int,
        YAYA_1CK                 int,
        YAYA_2CK                 int,
        YAYA_3CK                 int,
        YAYA_GAI                 int,
        OMO_1CK                 int,
        OMO_2CK                 int,
        OMO_3CK                 int,
        OMO_GAI                 int,
        S_1CK                 int,
        S_2CK                 int,
        S_3CK                 int,
        S_GAI                 int,
        M_1CK                 int,
        M_2CK                 int,
        M_3CK                 int,
        M_GAI                 int,
        H_1CK                 int,
        H_2CK                 int,
        H_3CK                 int,
        H_GAI                 int,
        SEASON_1CK                 int,
        SEASON_2CK                 int,
        SEASON_3CK                 int,
        SEASON_GAI                 int,
        WAKU_1CK                            int,
        WAKU_2CK                            int,
        WAKU_3CK                            int,
        WAKU_GAI                            int,
        KISHU_KYORI_1CK                    int,
        KISHU_KYORI_2CK                    int,
        KISHU_KYORI_3CK                    int,
        KISHU_KYORI_GAI                    int,
        KISHU_TRACK_1CK                    int,
        KISHU_TRACK_2CK                    int,
        KISHU_TRACK_3CK                    int,
        KISHU_TRACK_GAI                    int,
        KISHU_CHOKYOSHI_1CK                int,
        KISHU_CHOKYOSHI_2CK                int,
        KISHU_CHOKYOSHI_3CK                int,
        KISHU_CHOKYOSHI_GAI                int,
        KISHU_BANUSHI_1CK                  int,
        KISHU_BANUSHI_2CK                  int,
        KISHU_BANUSHI_3CK                  int,
        KISHU_BANUSHI_GAI                  int,
        KISHU_BURINKA_1CK                  int,
        KISHU_BURINKA_2CK                  int,
        KISHU_BURINKA_3CK                  int,
        KISHU_BURINKA_GAI                  int,
        CHOKYOSHI_BANUSHI_1CK              int,
        CHOKYOSHI_BANUSHI_2CK              int,
        CHOKYOSHI_BANUSHI_3CK              int,
        CHOKYOSHI_BANUSHI_GAI              int,
        CHICHI_SHIBA_RENTAI                int,
        CHICHI_DART_RENTAI                 int,
        CHICHI_RENTAI_KYORI                int,
        HAHACHICHI_SHIBA_RENTAI            int,
        HAHACHICHI_DART_RENTAI             int,
        HAHACHICHI_RENTAI_KYORI            int,
        filename                 nvarchar(20),
        target_date          nvarchar(6)
        );
        """
        return create_sql


class KYI(JrdbSchemaBase):
    def __init__(self):
        super().__init__()
        self.names = ['RACE_KEY', 'UMABAN', 'NENGAPPI', 'KETTO_TOROKU_BANGO', 'IDM', 'KISHU_RECORD', 'JOHO_RECORD', 'SOGO_RECORD', 'KYAKUSHITSU', 'KYORI_TEKISEI', 'JOSHODO', 'ROTE', 'BASE_ODDS', 'BASE_NINKI_JUNI', 'KIJUN_FUKUSHO_ODDS', 'KIJUN_FUKUSHO_NINKIJUN', 'TOKUTEI_HONMEI', 'TOKUTEI_TAIKO', 'TOKUTEI_TANANA', 'TOKUTEI_OSAE', 'TOKUTEI_HIMO',
                      'SOGO_HONMEI', 'SOGO_TAIKO', 'SOGO_TANANA', 'SOGO_OSAE', 'SOGO_HIMO', 'NINKI_RECORD', 'CHOKYO_RECORD', 'KYUSHA_RECOCRD', 'CHOKYO_YAJIRUSHI', 'KYUSHA_HYOUKA', 'KISHU_KITAI_RENRITSU', 'GEKISO_RECORD', 'HIDUME', 'OMOTEKISEI', 'CLASS_CODE', 'BURINKA', 'KISHU_NAME', 'FUTAN_JURYO', 'MINARAIA_KUBUN', 'CHOKYOSHI_NAME',
                      'CHOKYOSHO_SHOZOKU', 'ZENSO1_KYOSO_RESULT', 'ZENSO2_KYOSO_RESULT', 'ZENSO3_KYOSO_RESULT', 'ZENSO4_KYOSO_RESULT', 'ZENSO5_KYOSO_RESULT', 'ZENSO1_RACE_KEY', 'ZENSO2_RACE_KEY', 'ZENSO3_RACE_KEY', 'ZENSO4_RACE_KEY', 'ZENSO5_RACE_KEY', 'WAKUBAN', 'TOTAL_MARK', 'IDM_MARK', 'JOHO_MARK', 'KISHU_MARK', 'KYUSHA_MARK', 'CHOKYO_MARK', 'GEKISO_MARK', 'SHIBA_TEKISEI',
                      'DART_TEKISEI', 'KISHU_CODE', 'CHOKYKOSHI_CODE', 'KAKUTOKU_SHOKIN', 'SHUTOK_SHOKIN', 'JOKEN_CLASS', 'TEN_RECORD', 'PACE_RECORD', 'AGARI_RECORD', 'ICHI_RECORD', 'PACE_YOSO', 'DOCHU_JUNI', 'DOCHU_SA', 'DOCHU_UCHISOTO', 'ATO3F_JUNI', 'ATO3F_SA', 'ATO3F_UCHISOTO', 'GOAL_JUNI', 'GOAL_SA', 'GOAL_UCHISOTO',
                      'TENAI_MARK', 'KYORI_TEKISEI2', 'WAKU_KAKUTEI_BATAIJU', 'WAKU_KAKUTEI_ZOGEN', 'TORIKESHI', 'SEX', 'BANUSHI_NAME', 'BANUSHI_CODE', 'UMA_KIGO', 'GEKISO_JUNI', 'LS_RECORD_JUNI', 'TEN_RECORD_JUNI', 'PACE_RECORD_JUNI', 'AGARI_RECORD_JUNI', 'ICHI_RECORD_JUNI', 'KISHU_KITAI_1CK', 'KISHU_KITAI_3CK', 'YUSO_KUBUN', 'SOHO',
                      'TAIEKI_01', 'TAIEKI_02', 'TAIEKI_03', 'TAIEKI_04', 'TAIEKI_05', 'TAIEKI_06', 'TAIEKI_07', 'TAIEKI_08', 'TAIEKI_09', 'TAIEKI_10', 'TAIEKI_11', 'TAIEKI_12', 'TAIEKI_13', 'TAIEKI_14', 'TAIEKI_15', 'TAIEKI_16', 'TAIEKI_17', 'TAIEKI_18',
                      'TAIKEI1', 'TAIKEI2', 'TAIKEI3', 'UMA_TOKKI1', 'UMA_TOKKI2', 'UMA_TOKKI3', 'UMA_START_RECORD', 'UMA_DEOKURE_RITSU', 'SANKO_ZENSO', 'SANKO_ZENSO_KISHU_CODE', 'MANKEN_RECORD', 'MANKEN_MARK', 'KOKYU_FLAG', 'GEKISO_TYPE', 'KYUYOU_RIYU_CODE', 'FLAG_1', 'FLAG_2', 'FLAG_3', 'FLAG_4', 'FLAG_5', 'FLAG_6', 'NYUKYO_SOUME', 'NYUKYO_NENGAPPI', 'NYUKYO_NICHIMAE', 'HOUBOKUSAKI',
                      'HOUBOKUSAKI_RANK', 'KYUSHA_RANK', 'file_name', 'target_date']
        self.columns = ['%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s',
                        '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s']
        self.create_sql = self.sql_create()
        self.table_name = '[TALEND].[jrdb].[ORG_KYI]'

    def set_df(self, filename):
        with open('./' + filename, 'r', encoding="ms932") as fh:
            df = pd.DataFrame(index=[], columns=self.names)
            for line in fh:
                new_line = self.replace_line(line)
                sr = pd.Series([
                    new_line[0:8]  # RACE_KEY
                    , new_line[8:10]  # UMABAN
                    , mu.get_kaisai_date(filename)  # NENGAPPI
                    , new_line[10:18]  # KETTO_TOROKU_BANGO
                    , mu.float_null(new_line[54:59])  # IDM
                    , mu.float_null(new_line[59:64])  # KISHU_RECORD
                    , mu.float_null(new_line[64:69])  # JOHO_RECORD
                    , mu.float_null(new_line[84:89])  # SOGO_RECORD
                    , new_line[89:90]  # KYAKUSHITSU
                    , new_line[90:91]  # KYORI_TEKISEI
                    , new_line[91:92]  # JOSHODO
                    , mu.int_null(new_line[92:95])  # ROTE
                    , mu.float_null(new_line[95:100])  # BASE_ODDS
                    , mu.int_null(new_line[100:102])  # BASE_NINKI_JUNI
                    , mu.float_null(new_line[102:107])  # KIJUN_FUKUSHO_ODDS
                    , mu.int_null(new_line[107:109])  # KIJUN_FUKUSHO_NINKIJUN
                    , mu.int_null(new_line[109:112])  # TOKUTEI_HONMEI
                    , mu.int_null(new_line[112:115])  # TOKUTEI_TAIKO
                    , mu.int_null(new_line[115:118])  # TOKUTEI_TANANA
                    , mu.int_null(new_line[118:121])  # TOKUTEI_OSAE
                    , mu.int_null(new_line[121:124])  # TOKUTEI_HIMO
                    , mu.int_null(new_line[124:127])  # SOGO_HONMEI
                    , mu.int_null(new_line[127:130])  # SOGO_TAIKO
                    , mu.int_null(new_line[130:133])  # SOGO_TANANA
                    , mu.int_null(new_line[133:136])  # SOGO_OSAE
                    , mu.int_null(new_line[136:139])  # SOGO_HIMO
                    , mu.float_null(new_line[139:144])  # NINKI_RECORD
                    , mu.float_null(new_line[144:149])  # CHOKYO_RECORD
                    , mu.float_null(new_line[149:154])  # KYUSHA_RECOCRD
                    , new_line[154:155]  # CHOKYO_YAJIRUSHI
                    , new_line[155:156]  # KYUSHA_HYOUKA
                    , mu.float_null(new_line[156:160])  # KISHU_KITAI_RENRITSU
                    , mu.int_null(new_line[160:163])  # GEKISO_RECORD
                    , new_line[163:165]  # HIDUME
                    , new_line[165:166]  # OMOTEKISEI
                    , new_line[166:168]  # CLASS_CODE
                    , new_line[170:171]  # BURINKA
                    , new_line[171:183].replace(" ", "")  # KISHU_NAME
                    , mu.int_null(new_line[183:186])  # FUTAN_JURYO
                    , new_line[186:187]  # MINARAIA_KUBUN
                    , new_line[187:199].replace(" ", "")  # CHOKYOSHI_NAME
                    , new_line[199:203].replace(" ", "")  # CHOKYOSHO_SHOZOKU
                    , new_line[203:219]  # ZENSO1_KYOSO_RESULT
                    , new_line[219:235]  # ZENSO2_KYOSO_RESULT
                    , new_line[235:251]  # ZENSO3_KYOSO_RESULT
                    , new_line[251:267]  # ZENSO4_KYOSO_RESULT
                    , new_line[267:283]  # ZENSO5_KYOSO_RESULT
                    , new_line[283:291]  # ZENSO1_RACE_KEY
                    , new_line[291:299]  # ZENSO2_RACE_KEY
                    , new_line[299:307]  # ZENSO3_RACE_KEY
                    , new_line[307:315]  # ZENSO4_RACE_KEY
                    , new_line[315:323]  # ZENSO5_RACE_KEY
                    , new_line[323:324]  # WAKUBAN
                    , new_line[326:327]  # TOTAL_MARK
                    , new_line[327:328]  # IDM_MARK
                    , new_line[328:329]  # JOHO_MARK
                    , new_line[329:330]  # KISHU_MARK
                    , new_line[330:331]  # KYUSHA_MARK
                    , new_line[331:332]  # CHOKYO_MARK
                    , new_line[332:333]  # GEKISO_MARK
                    , new_line[333:334]  # SHIBA_TEKISEI
                    , new_line[334:335]  # DART_TEKISEI
                    , new_line[335:340]  # KISHU_CODE
                    , new_line[340:345]  # CHOKYKOSHI_CODE
                    , mu.int_null(new_line[346:352])  # KAKUTOKU_SHOKIN
                    , mu.int_null(new_line[352:357])  # SHUTOK_SHOKIN
                    , new_line[357:358]  # JOKEN_CLASS
                    , mu.float_null(new_line[358:363])  # TEN_RECORD
                    , mu.float_null(new_line[363:368])  # PACE_RECORD
                    , mu.float_null(new_line[368:373])  # AGARI_RECORD
                    , mu.float_null(new_line[373:378])  # ICHI_RECORD
                    , new_line[378:379]  # PACE_YOSO
                    , mu.int_null(new_line[379:381])  # DOCHU_JUNI
                    , mu.int_null(new_line[381:383])  # DOCHU_SA
                    , new_line[383:384]  # DOCHU_UCHISOTO
                    , mu.int_null(new_line[384:386])  # ATO3F_JUNI
                    , mu.int_null(new_line[386:388])  # ATO3F_SA
                    , new_line[388:389]  # ATO3F_UCHISOTO
                    , mu.int_null(new_line[389:391])  # GOAL_JUNI
                    , mu.int_null(new_line[391:393])  # GOAL_SA
                    , new_line[393:394]  # GOAL_UCHISOTO
                    , new_line[394:395]  # TENAI_MARK
                    , new_line[395:396]  # KYORI_TEKISEI2
                    , mu.int_null(new_line[396:399])  # WAKU_KAKUTEI_BATAIJU
                    , mu.int_null(new_line[399:402])  # WAKU_KAKUTEI_ZOGEN
                    , new_line[402:403]  # TORIKESHI
                    , new_line[403:404]  # SEX
                    , new_line[404:444].replace(" ", "")  # BANUSHI_NAME
                    , new_line[444:446]  # BANUSHI_CODE
                    , new_line[446:448]  # UMA_KIGO
                    , mu.int_null(new_line[448:450])  # GEKISO_JUNI
                    , mu.int_null(new_line[450:452])  # LS_RECORD_JUNI
                    , mu.int_null(new_line[452:454])  # TEN_RECORD_JUNI
                    , mu.int_null(new_line[454:456])  # PACE_RECORD_JUNI
                    , mu.int_null(new_line[456:458])  # AGARI_RECORD_JUNI
                    , mu.int_null(new_line[458:460])  # ICHI_RECORD_JUNI
                    , mu.float_null(new_line[460:464])  # KISHU_KITAI_1CK
                    , mu.float_null(new_line[464:468])  # KISHU_KITAI_3CK
                    , new_line[468:469]  # YUSO_KUBUN
                    , new_line[469:477]  # SOHO
                    , new_line[477:478]  # TAIEKI_01
                    , new_line[478:479]  # TAIEKI_02
                    , new_line[479:480]  # TAIEKI_03
                    , new_line[480:481]  # TAIEKI_04
                    , new_line[481:482]  # TAIEKI_05
                    , new_line[482:483]  # TAIEKI_06
                    , new_line[483:484]  # TAIEKI_07
                    , new_line[484:485]  # TAIEKI_08
                    , new_line[485:486]  # TAIEKI_09
                    , new_line[486:487]  # TAIEKI_10
                    , new_line[487:488]  # TAIEKI_11
                    , new_line[488:489]  # TAIEKI_12
                    , new_line[489:490]  # TAIEKI_13
                    , new_line[490:491]  # TAIEKI_14
                    , new_line[491:492]  # TAIEKI_15
                    , new_line[492:493]  # TAIEKI_16
                    , new_line[493:494]  # TAIEKI_17
                    , new_line[494:495]  # TAIEKI_18
                    , new_line[501:504]  # TAIKEI1
                    , new_line[504:507]  # TAIKEI2
                    , new_line[507:510]  # TAIKEI3
                    , new_line[510:513]  # UMA_TOKKI1
                    , new_line[513:516]  # UMA_TOKKI2
                    , new_line[516:519]  # UMA_TOKKI3
                    , mu.float_null(new_line[519:523])  # UMA_START_RECORD
                    , mu.float_null(new_line[523:527])  # UMA_DEOKURE_RITSU
                    , new_line[527:529]  # SANKO_ZENSO
                    , new_line[529:534]  # SANKO_ZENSO_KISHU_CODE
                    , mu.int_null(new_line[534:537])  # MANKEN_RECORD
                    , new_line[537:538]  # MANKEN_MARK
                    , new_line[538:539]  # KOKYU_FLAG
                    , new_line[539:541]  # GEKISO_TYPE
                    , new_line[541:543]  # KYUYOU_RIYU_CODE
                    , new_line[543:544]  # FLAG_1
                    , new_line[544:545]  # FLAG_2
                    , new_line[545:546]  # FLAG_3
                    , new_line[546:547]  # FLAG_4
                    , new_line[547:548]  # FLAG_5
                    , new_line[548:549]  # FLAG_6
                    , mu.int_null(new_line[559:561])  # NYUKYO_SOUME
                    , new_line[561:569]  # NYUKYO_NENGAPPI
                    , mu.int_null(new_line[569:572])  # NYUKYO_NICHIMAE
                    , new_line[572:622].replace(" ", "")  # HOUBOKUSAKI
                    , new_line[622:623]  # HOUBOKUSAKI_RANK
                    , new_line[623:624]  # KYUSHA_RANK
                    , filename, filename[3:9]
                ], index=self.names)
                df = df.append(sr, ignore_index=True)
        return df

    def sql_create(self):
        create_sql = """if object_id('[TALEND].[jrdb].[ORG_KYI]') is null
        CREATE TABLE [TALEND].[jrdb].[ORG_KYI] (
        RACE_KEY                nvarchar(8),
        UMABAN                  nvarchar(2),
        NENGAPPI                nvarchar(8),
        KETTO_TOROKU_BANGO      nvarchar(8),
        IDM                     real,
        KISHU_RECORD            real,
        JOHO_RECORD             real,
        SOGO_RECORD             real,
        KYAKUSHITSU             nvarchar(1),
        KYORI_TEKISEI           nvarchar(1),
        JOSHODO                 nvarchar(1),
        ROTE                    int,
        BASE_ODDS               real,
        BASE_NINKI_JUNI         int,
        KIJUN_FUKUSHO_ODDS      real,
        KIJUN_FUKUSHO_NINKIJUN  int,
        TOKUTEI_HONMEI          int,
        TOKUTEI_TAIKO           int,
        TOKUTEI_TANANA          int,
        TOKUTEI_OSAE            int,
        TOKUTEI_HIMO            int,
        SOGO_HONMEI             int,
        SOGO_TAIKO              int,
        SOGO_TANANA             int,
        SOGO_OSAE               int,
        SOGO_HIMO               int,
        NINKI_RECORD            real,
        CHOKYO_RECORD           real,
        KYUSHA_RECOCRD          real,
        CHOKYO_YAJIRUSHI        nvarchar(1),
        KYUSHA_HYOUKA           nvarchar(1),
        KISHU_KITAI_RENRITSU    nvarchar(4),
        GEKISO_RECORD           int,
        HIDUME                  nvarchar(2),
        OMOTEKISEI              nvarchar(1),
        CLASS_CODE              nvarchar(2),
        BURINKA                 nvarchar(1),
        KISHU_NAME              nvarchar(12),
        FUTAN_JURYO             int,
        MINARAIA_KUBUN          nvarchar(1),
        CHOKYOSHI_NAME          nvarchar(12),
        CHOKYOSHO_SHOZOKU       nvarchar(4),
        ZENSO1_KYOSO_RESULT     nvarchar(16),
        ZENSO2_KYOSO_RESULT     nvarchar(16),
        ZENSO3_KYOSO_RESULT     nvarchar(16),
        ZENSO4_KYOSO_RESULT     nvarchar(16),
        ZENSO5_KYOSO_RESULT     nvarchar(16),
        ZENSO1_RACE_KEY         nvarchar(8),
        ZENSO2_RACE_KEY         nvarchar(8),
        ZENSO3_RACE_KEY         nvarchar(8),
        ZENSO4_RACE_KEY         nvarchar(8),
        ZENSO5_RACE_KEY         nvarchar(8),
        WAKUBAN                 nvarchar(1),
        TOTAL_MARK              nvarchar(1),
        IDM_MARK                nvarchar(1),
        JOHO_MARK               nvarchar(1),
        KISHU_MARK              nvarchar(1),
        KYUSHA_MARK             nvarchar(1),
        CHOKYO_MARK             nvarchar(1),
        GEKISO_MARK             nvarchar(1),
        SHIBA_TEKISEI           nvarchar(1),
        DART_TEKISEI            nvarchar(1),
        KISHU_CODE              nvarchar(5),
        CHOKYKOSHI_CODE         nvarchar(5),
        KAKUTOKU_SHOKIN         nvarchar(6),
        SHUTOK_SHOKIN           int,
        JOKEN_CLASS             nvarchar(1),
        TEN_RECORD              real,
        PACE_RECORD             real,
        AGARI_RECORD            real,
        ICHI_RECORD             real,
        PACE_YOSO               nvarchar(1),
        DOCHU_JUNI              nvarchar(2),
        DOCHU_SA                nvarchar(2),
        DOCHU_UCHISOTO          nvarchar(1),
        ATO3F_JUNI              nvarchar(2),
        ATO3F_SA                nvarchar(2),
        ATO3F_UCHISOTO          nvarchar(1),
        GOAL_JUNI               nvarchar(2),
        GOAL_SA                 nvarchar(2),
        GOAL_UCHISOTO           nvarchar(1),
        TENAI_MARK              nvarchar(1),
        KYORI_TEKISEI2          nvarchar(1),
        WAKU_KAKUTEI_BATAIJU    int,
        WAKU_KAKUTEI_ZOGEN      int,
        TORIKESHI               nvarchar(1),
        SEX                     nvarchar(1),
        BANUSHI_NAME            nvarchar(40),
        BANUSHI_CODE            nvarchar(2),
        UMA_KIGO                nvarchar(2),
        GEKISO_JUNI             int,
        LS_RECORD_JUNI          int,
        TEN_RECORD_JUNI         int,
        PACE_RECORD_JUNI        int,
        AGARI_RECORD_JUNI       int,
        ICHI_RECORD_JUNI        int,
        KISHU_KITAI_1CK         real,
        KISHU_KITAI_3CK         real,
        YUSO_KUBUN              nvarchar(1),
        SOHO                    nvarchar(8),
        TAIEKI_01               nvarchar(1),
        TAIEKI_02               nvarchar(1),
        TAIEKI_03               nvarchar(1),
        TAIEKI_04               nvarchar(1),
        TAIEKI_05               nvarchar(1),
        TAIEKI_06               nvarchar(1),
        TAIEKI_07               nvarchar(1),
        TAIEKI_08               nvarchar(1),
        TAIEKI_09               nvarchar(1),
        TAIEKI_10               nvarchar(1),
        TAIEKI_11               nvarchar(1),
        TAIEKI_12               nvarchar(1),
        TAIEKI_13               nvarchar(1),
        TAIEKI_14               nvarchar(1),
        TAIEKI_15               nvarchar(1),
        TAIEKI_16               nvarchar(1),
        TAIEKI_17               nvarchar(1),
        TAIEKI_18               nvarchar(1),
        TAIKEI1                 nvarchar(3),
        TAIKEI2                 nvarchar(3),
        TAIKEI3                 nvarchar(3),
        UMA_TOKKI1              nvarchar(3),
        UMA_TOKIK2              nvarchar(3),
        UMA_TOKKI3              nvarchar(3),
        UMA_START_RECORD        real,
        UMA_DEOKURE_RITSU       real,
        SANKO_ZENSO             nvarchar(2),
        SANKO_ZENSO_KISHU_CODE  nvarchar(5),
        MANKEN_RECORD           int,
        MANKEN_MARK             nvarchar(1),
        KOKYU_FLAG              nvarchar(1),
        GEKISO_TYPE             nvarchar(2),
        KYUYOU_RIYU_CODE        nvarchar(2),
        FLAG_1                  nvarchar(1),
        FLAG_2                  nvarchar(1),
        FLAG_3                  nvarchar(1),
        FLAG_4                  nvarchar(1),
        FLAG_5                  nvarchar(1),
        FLAG_6                  nvarchar(1),
        NYUKYO_SOUME            int,
        NYUKYO_NENGAPPI         nvarchar(8),
        NYUKYO_NICHIMAE         int,
        HOUBOKUSAKI             nvarchar(50),
        HOUBOKUSAKI_RANK        nvarchar(1),
        KYUSHA_RANK             nvarchar(1),
        filename                nvarchar(20),
        target_date          nvarchar(6)
        );
        """
        return create_sql


class UKC(JrdbSchemaBase):
    def __init__(self):
        super().__init__()
        self.names = ['KETTO_TOROKU_BANGO', 'UMA_NAME', 'SEX', 'KEIRO', 'UMA_KIGO', 'CHICHI_NAME', 'HAHA_NAME', 'HAHA_CHICHI_NAME', 'SEINENGAPPI', 'CHICHI_UMARE_YEAR', 'HAHA_UMARE_YEAR',
                      'HAHA_CHICHI_UMARE_YEAR', 'BANUSHI_NAME', 'BANUSHI_CODE', 'SEISANSHA', 'SANCHI', 'MASSHO_FLAG', 'DATA_NENGAPPI', 'CHICHI_KEITO', 'HAHA_CHICHI_KEITO', 'file_name', 'target_date']
        self.columns = ['%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s',
                        '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s']
        self.create_sql = self.sql_create()
        self.table_name = '[TALEND].[jrdb].[ORG_UKC]'

    def set_df(self, filename):
        with open('./' + filename, 'r', encoding="ms932") as fh:
            df = pd.DataFrame(index=[], columns=self.names)
            for line in fh:
                new_line = self.replace_line(line)
                sr = pd.Series([
                    new_line[0:8]  # KETTO_TOROKU_BANGO
                    , new_line[8:44]  # UMA_NAME
                    , new_line[44:45]  # SEX
                    , new_line[45:47]  # KEIRO
                    , new_line[47:49]  # UMA_KIGO
                    , new_line[49:85].replace(" ", "")  # CHICHI_NAME
                    , new_line[85:121].replace(" ", "")  # HAHA_NAME
                    , new_line[121:157].replace(" ", "")  # HAHA_CHICHI_NAME
                    , new_line[157:165]  # SEINENGAPPI
                    , new_line[165:169]  # CHICHI_UMARE_YEAR
                    , new_line[169:173]  # HAHA_UMARE_YEAR
                    , new_line[173:177]  # HAHA_CHICHI_UMARE_YEAR
                    , new_line[177:217].replace(" ", "")  # BANUSHI_NAME
                    , new_line[217:219]  # BANUSHI_CODE
                    , new_line[219:259].replace(" ", "")  # SEISANSHA
                    , new_line[259:267].replace(" ", "")  # SANCHI
                    , new_line[267:268]  # MASSHO_FLAG
                    , new_line[268:276]  # DATA_NENGAPPI
                    , new_line[276:280]  # CHICHI_KEITO
                    , new_line[280:284]  # HAHA_CHICHI_KEITO
                    , filename, filename[3:9]
                ], index=self.names)
                df = df.append(sr, ignore_index=True)
        return df

    def sql_create(self):
        create_sql = """if object_id('[TALEND].[jrdb].[ORG_UKC]') is null
        CREATE TABLE [TALEND].[jrdb].[ORG_UKC] (
        KETTO_TOROKU_BANGO      nvarchar(8),
        UMA_NAME                nvarchar(36),
        SEX                     nvarchar(1),
        KEIRO                   nvarchar(2),
        UMA_KIGO                nvarchar(2),
        CHICHI_NAME             nvarchar(36),
        HAHA_NAME               nvarchar(36),
        HAHA_CHICHI_NAME        nvarchar(36),
        SEINENGAPPI             nvarchar(8),
        CHICHI_UMARE_YEAR       nvarchar(4),
        HAHA_UMARE_YEAR         nvarchar(4),
        HAHA_CHICHI_UMARE_YEAR  nvarchar(4),
        BANUSHI_NAME            nvarchar(40),
        BANUSHI_CODE            nvarchar(2),
        SEISANSHA               nvarchar(40),
        SANCHI                  nvarchar(8),
        MASSHO_FLAG             nvarchar(1),
        DATA_NENGAPPI           nvarchar(8),
        CHICHI_KEITO            nvarchar(4),
        HAHA_CHICHI_KEITO       nvarchar(4),
        filename                nvarchar(20),
        target_date          nvarchar(6)
        );
        """
        return create_sql


class SED(JrdbSchemaBase):
    def __init__(self):
        super().__init__()
        self.names = ['RACE_KEY', 'UMABAN', 'KETTO_TOROKU_BANGO', 'NENGAPPI', 'UMA_NAME', 'KYORI', 'SHIBA_DART', 'MIGIHIDARI', 'UCHISOTO', 'BABA_JOTAI', 'SHUBETSU', 'JOKEN', 'KIGO', 'JURYO', 'GRADE', 'RACE_NAME', 'TOSU', 'RACE_NAME_RYAKUSHO', 'CHAKUJUN', 'IJO_KUBUN', 'TIME', 'KINRYO', 'KISHU_NAME', 'CHOKYOSHI_NAME', 'KAKUTEI_TANSHO_ODDS', 'KAKUTEI_TANSHO_NINKIJUN', 'IDM', 'SOTEN', 'BABA_SA', 'PACE', 'DEOKURE', 'ICHIDORI', 'FURI', 'MAE_FURI', 'NAKA_FURI', 'ATO_FURI', 'RACE', 'COURSE_DORI', 'JOSHODO', 'CLASS_CODE', 'BATAI_CODE', 'KEHAI_CODE',
                      'RACE_PACE', 'UMA_PACE', 'TEN_RECORD', 'AGARI_RECORD', 'PACE_RECORD', 'RACE_PACE_RECORD', 'KACHIUMA_NAME', 'TIME_SA', 'MAE_3F_TIME', 'ATO_3F_TIME', 'KAKUTEI_FUKUSHO_ODDS', 'TANSHO_ODDS_10JI', 'FUKUSHO_ODDS_10JI', 'CORNER_JUNI1', 'CORNER_JUNI2', 'CORNER_JUNI3', 'CORNER_JUNI4', 'MAE_3F_SA', 'ATO_3F_SA', 'KISHU_CODE', 'CHOKYOSHI_CODE', 'BATAIJU', 'BATAIJU_ZOGEN', 'TENKO_CODE', 'COURSE', 'RACE_KYAKUSHITSU', 'TANSHO', 'FUKUSHO', 'HON_SHOKIN', 'SHUTOKU_SHOKIN', 'RACE_PACE_NAGARE', 'UMA_PACE_NAGARE', 'COURSE_4KAKU', 'file_name', 'target_date']
        self.columns = ['%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s',
                        '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s']
        self.create_sql = self.sql_create()
        self.table_name = '[TALEND].[jrdb].[ORG_SED]'

    def set_df(self, filename):
        with open('./' + filename, 'r', encoding="ms932") as fh:
            df = pd.DataFrame(index=[], columns=self.names)
            for line in fh:
                new_line = self.replace_line(line)
                sr = pd.Series([
                    new_line[0:8]  # RACE_KEY
                    , new_line[8:10]  # UMABAN
                    , new_line[10:18]  # KETTO_TOROKU_BANGO
                    , new_line[18:26]  # NENGAPPI
                    , new_line[26:62].replace(" ", "")  # UMA_NAME
                    , mu.int_null(new_line[62:66])  # KYORI
                    , new_line[66:67]  # SHIBA_DART
                    , new_line[67:68]  # MIGIHIDARI
                    , new_line[68:69]  # UCHISOTO
                    , new_line[69:71]  # BABA_JOTAI
                    , new_line[71:73]  # SHUBETSU
                    , new_line[73:75]  # JOKEN
                    , new_line[75:78]  # KIGO
                    , new_line[78:79]  # JURYO
                    , new_line[79:80]  # GRADE
                    , new_line[80:130]  # RACE_NAME
                    , mu.int_null(new_line[130:132])  # TOSU
                    , new_line[132:140]  # RACE_NAME_RYAKUSHO
                    , mu.int_null(new_line[140:142])  # CHAKUJUN
                    , new_line[142:143]  # IJO_KUBUN
                    , mu.convert_time(new_line[143:147])  # TIME
                    , mu.int_null(new_line[147:150])  # KINRYO
                    , new_line[150:162].replace(" ", "")  # KISHU_NAME
                    , new_line[162:174].replace(" ", "")  # CHOKYOSHI_NAME
                    , mu.float_null(new_line[174:180])  # KAKUTEI_TANSHO_ODDS
                    # KAKUTEI_TANSHO_NINKIJUN
                    # IDM
                    # SOTEN
                    # BABA_SA
                    # PACE
                    # DEOKURE
                    # ICHIDORI
                    , mu.int_null(new_line[180:182]), mu.int_null(new_line[182:185]), mu.int_null(new_line[185:188]), mu.int_null(new_line[188:191]), mu.int_null(new_line[191:194]), mu.int_null(new_line[194:197]), mu.int_null(new_line[197:200]), mu.int_null(new_line[200:203])  # FURI
                    , mu.int_null(new_line[203:206])  # MAE_FURI
                    , mu.int_null(new_line[206:209])  # NAKA_FURI
                    , mu.int_null(new_line[209:212])  # ATO_FURI
                    , mu.int_null(new_line[212:215])  # RACE
                    , new_line[215:216]  # COURSE_DORI
                    , new_line[216:217]  # JOSHODO
                    , new_line[217:219]  # CLASS_CODE
                    , new_line[219:220]  # BATAI_CODE
                    , new_line[220:221]  # KEHAI_CODE
                    , new_line[221:222]  # RACE_PACE
                    , new_line[222:223]  # UMA_PACE
                    , mu.float_null(new_line[223:228])  # TEN_RECORD
                    , mu.float_null(new_line[228:233])  # AGARI_RECORD
                    , mu.float_null(new_line[233:238])  # PACE_RECORD
                    , mu.float_null(new_line[238:243])  # RACE_PACE_RECORD
                    , new_line[243:255].replace(" ", "")  # KACHIUMA_NAME
                    , new_line[255:258]  # TIME_SA
                    , mu.int_null(new_line[258:261])  # MAE_3F_TIME
                    , mu.int_null(new_line[261:264])  # ATO_3F_TIME
                    , mu.float_null(new_line[290:296])  # KAKUTEI_FUKUSHO_ODDS
                    , mu.float_null(new_line[296:302])  # TANSHO_ODDS_10JI
                    , mu.float_null(new_line[302:308])  # FUKUSHO_ODDS_10JI
                    , mu.int_null(new_line[308:310])  # CORNER_JUNI1
                    , mu.int_null(new_line[310:312])  # CORNER_JUNI2
                    , mu.int_null(new_line[312:314])  # CORNER_JUNI3
                    , mu.int_null(new_line[314:316])  # CORNER_JUNI4
                    , mu.int_null(new_line[316:319])  # MAE_3F_SA
                    , mu.int_null(new_line[319:322])  # ATO_3F_SA
                    , new_line[322:327]  # KISHU_CODE
                    , new_line[327:332]  # CHOKYOSHI_CODE
                    , mu.int_null(new_line[332:335])  # BATAIJU
                    , mu.int_bataiju_zogen(new_line[335:338])  # BATAIJU_ZOGEN
                    , new_line[338:339]  # TENKO_CODE
                    , new_line[339:340]  # COURSE
                    , new_line[340:341]  # RACE_KYAKUSHITSU
                    , mu.int_haito(new_line[341:348])  # TANSHO
                    , mu.int_haito(new_line[348:355])  # FUKUSHO
                    , mu.float_null(new_line[355:360])  # HON_SHOKIN
                    , mu.float_null(new_line[360:365])  # SHUTOKU_SHOKIN
                    , new_line[365:367]  # RACE_PACE_NAGARE
                    , new_line[367:369]  # UMA_PACE_NAGARE
                    , new_line[369:370]  # COURSE_4KAKU
                    , filename, filename[3:9]
                ], index=self.names)
                df = df.append(sr, ignore_index=True)
        return df

    def sql_create(self):
        create_sql = """if object_id('[TALEND].[jrdb].[ORG_SED]') is null
        CREATE TABLE [TALEND].[jrdb].[ORG_SED] (
        RACE_KEY            nvarchar(8),
        UMABAN              nvarchar(2),
        KETTO_TOROKU_BANGO  nvarchar(8) ,
        NENGAPPI            nvarchar(8),
        UMA_NAME            nvarchar(36),
        KYORI               int,
        SHIBA_DART          nvarchar(1),
        MIGIHIDARI          nvarchar(1),
        UCHISOTO            nvarchar(1),
        BABA_JOTAI          nvarchar(2),
        SHUBETSU            nvarchar(2),
        JOKEN               nvarchar(2),
        KIGO                nvarchar(3),
        JURYO               nvarchar(1),
        GRADE               nvarchar(1),
        RACE_NAME           nvarchar(50),
        TOSU                int,
        RACE_NAME_RYAKUSHO  nvarchar(8),
        CHAKUJUN            int,
        IJO_KUBUN           nvarchar(1),
        TIME                int,
        KINRYO              int,
        KISHU_NAME          nvarchar(12),
        CHOKYOSHI_NAME      nvarchar(12),
        KAKUTEI_TANSHO_ODDS real,
        KAKUTEI_TANSHO_NINKIJUN int,
        IDM                 int,
        SOTEN               int,
        BABA_SA             int,
        PACE                int,
        DEOKURE             int,
        ICHIDORI            int,
        FURI                int,
        MAE_FURI            int,
        NAKA_FURI           int,
        ATO_FURI            int,
        RACE                int,
        COURSE_DORI         nvarchar(1),
        JOSHODO             nvarchar(1) ,
        CLASS_CODE          nvarchar(2),
        BATAI_CODE          nvarchar(1),
        KEHAI_CODE          nvarchar(1),
        RACE_PACE           nvarchar(1),
        UMA_PACE            nvarchar(1),
        TEN_RECORD          real,
        AGARI_RECORD        real,
        PACE_RECORD         real,
        RACE_PACE_RECORD    real,
        KACHIUMA_NAME       nvarchar(12),
        TIME_SA             int,
        MAE_3F_TIME         int,
        ATO_3F_TIME         int,
        AKUTEI_FUKUSHO_ODDS real,
        TANSHO_ODDS_10JI    real,
        FUKUSHO_ODDS_10JI   real,
        CORNER_JUNI1        int,
        CORNER_JUNI2        int,
        CORNER_JUNI3        int,
        CORNER_JUNI4        int,
        MAE_3F_SA           int,
        ATO_3F_SA           int,
        KISHU_CODE          nvarchar(5),
        CHOKYOSHI_CODE      nvarchar(5),
        BATAIJU             int,
        BATAIJU_ZOGEN       int,
        TENKO_CODE          nvarchar(1),
        COURSE              nvarchar(1),
        RACE_KYAKUSHITSU    nvarchar(1),
        TANSHO              int,
        FUKUSHO             int,
        HON_SHOKIN          int,
        SHUTOKU_SHOKIN      int,
        RACE_PACE_NAGARE    nvarchar(2), 
        UMA_PACE_NAGARE     nvarchar(2),
        COURSE_4KAKU        nvarchar(1),
        filename                nvarchar(20),
        target_date          nvarchar(6)
        );
        """
        return create_sql


class SRB(JrdbSchemaBase):
    def __init__(self):
        super().__init__()
        self.names = ['RACE_KEY', 'HARON_01', 'HARON_02', 'HARON_03', 'HARON_04', 'HARON_05', 'HARON_06', 'HARON_07', 'HARON_08', 'HARON_09', 'HARON_10', 'HARON_11', 'HARON_12', 'HARON_13', 'HARON_14', 'HARON_15', 'HARON_16', 'HARON_17', 'HARON_18', 'CORNER_1', 'CORNER_2', 'CORNER_3',
                      'CORNER_4', 'PACE_UP_POINT', 'TB1_1', 'TB1_2', 'TB1_3', 'TB2_1', 'TB2_2', 'TB2_3', 'TBM_1', 'TBM_2', 'TBM_3', 'TB3_1', 'TB3_2', 'TB3_3', 'TB4_0', 'TB4_1', 'TB4_2', 'TB4_3', 'TB4_4', 'TBC_0', 'TBC_1', 'TBC_2', 'TBC_3', 'TBC_4', 'RACE_COMMENT', 'file_name', 'target_date']
        self.columns = ['%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s',
                        '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s']
        self.create_sql = self.sql_create()
        self.table_name = '[TALEND].[jrdb].[ORG_SRB]'

    def set_df(self, filename):
        with open('./' + filename, 'r', encoding="ms932") as fh:
            df = pd.DataFrame(index=[], columns=self.names)
            for line in fh:
                new_line = self.replace_line(line)
                sr = pd.Series([
                    new_line[0:8]  # RACE_KEY
                    , mu.int_null(new_line[8:11])  # HARON_01
                    , mu.int_null(new_line[11:14])  # HARON_02
                    , mu.int_null(new_line[14:17])  # HARON_03
                    , mu.int_null(new_line[17:20])  # HARON_04
                    , mu.int_null(new_line[20:23])  # HARON_05
                    , mu.int_null(new_line[23:26])  # HARON_06
                    , mu.int_null(new_line[26:29])  # HARON_07
                    , mu.int_null(new_line[29:32])  # HARON_08
                    , mu.int_null(new_line[32:35])  # HARON_09
                    , mu.int_null(new_line[35:38])  # HARON_10
                    , mu.int_null(new_line[38:41])  # HARON_11
                    , mu.int_null(new_line[41:44])  # HARON_12
                    , mu.int_null(new_line[44:47])  # HARON_13
                    , mu.int_null(new_line[47:50])  # HARON_14
                    , mu.int_null(new_line[50:53])  # HARON_15
                    , mu.int_null(new_line[53:56])  # HARON_16
                    , mu.int_null(new_line[56:59])  # HARON_17
                    , mu.int_null(new_line[59:62])  # HARON_18
                    , new_line[62:126]  # CORNER_1
                    , new_line[126:190]  # CORNER_2
                    , new_line[190:254]  # CORNER_3
                    , new_line[254:318]  # CORNER_4
                    , mu.int_null(new_line[318:320])  # PACE_UP_POINT
                    , new_line[320:321]  # TRACK_BIAS_1KAKU
                    , new_line[321:322]  # TRACK_BIAS_1KAKU
                    , new_line[322:323]  # TRACK_BIAS_1KAKU
                    , new_line[323:324]  # TRACK_BIAS_2KAKU
                    , new_line[324:325]  # TRACK_BIAS_2KAKU
                    , new_line[325:326]  # TRACK_BIAS_2KAKU
                    , new_line[326:327]  # TRACK_BIAS_MUKAI
                    , new_line[327:328]  # TRACK_BIAS_MUKAI
                    , new_line[328:329]  # TRACK_BIAS_MUKAI
                    , new_line[329:330]  # TRACK_BIAS_3KAKU
                    , new_line[330:331]  # TRACK_BIAS_3KAKU
                    , new_line[331:332]  # TRACK_BIAS_3KAKU
                    , new_line[332:333]  # TRACK_BIAS_4KAKU
                    , new_line[333:334]  # TRACK_BIAS_4KAKU
                    , new_line[334:335]  # TRACK_BIAS_4KAKU
                    , new_line[335:336]  # TRACK_BIAS_4KAKU
                    , new_line[336:337]  # TRACK_BIAS_4KAKU
                    , new_line[337:338]  # TRACK_BIAS_CHOKUSEN
                    , new_line[338:339]  # TRACK_BIAS_CHOKUSEN
                    , new_line[339:340]  # TRACK_BIAS_CHOKUSEN
                    , new_line[340:341]  # TRACK_BIAS_CHOKUSEN
                    , new_line[341:342]  # TRACK_BIAS_CHOKUSEN
                    , new_line[342:842].replace(" ", "")  # RACE_COMMENT
                    , filename, filename[3:9]
                ], index=self.names)
                df = df.append(sr, ignore_index=True)
        return df

    def sql_create(self):
        create_sql = """if object_id('[TALEND].[jrdb].[ORG_SRB]') is null
        CREATE TABLE [TALEND].[jrdb].[ORG_SRB] (
        RACE_KEY    nvarchar(8),
        HARON_01    int,
        HARON_02    int,
        HARON_03    int,
        HARON_04    int,
        HARON_05    int,
        HARON_06    int,
        HARON_07    int,
        HARON_08    int,
        HARON_09    int,
        HARON_10    int,
        HARON_11    int,
        HARON_12    int,
        HARON_13    int,
        HARON_14    int,
        HARON_15    int,
        HARON_16    int,
        HARON_17    int,
        HARON_18    int,
        CORNER_1    nvarchar(64),
        CORNER_2    nvarchar(64),
        CORNER_3    nvarchar(64),
        CORNER_4    nvarchar(64),
        PACE_UP_POINT   int,
        TB1_1       nvarchar(1),
        TB1_2       nvarchar(1),
        TB1_3       nvarchar(1),
        TB2_1       nvarchar(1),
        TB2_2       nvarchar(1),
        TB2_3       nvarchar(1),
        TBM_1       nvarchar(1),
        TBM_2       nvarchar(1),
        TBM_3       nvarchar(1),
        TB3_1       nvarchar(1),
        TB3_2       nvarchar(1),
        TB3_3       nvarchar(1),
        TB4_0       nvarchar(1),
        TB4_1       nvarchar(1),
        TB4_2       nvarchar(1),
        TB4_3       nvarchar(1),
        TB4_4       nvarchar(1),
        TBC_0       nvarchar(1),
        TBC_1       nvarchar(1),
        TBC_2       nvarchar(1),
        TBC_3       nvarchar(1),
        TBC_4       nvarchar(1),
        RACE_COMMENT    nvarchar(500),
        filename                nvarchar(20),
        target_date          nvarchar(6)
        );
        """
        return create_sql


class SKB(JrdbSchemaBase):
    def __init__(self):
        super().__init__()
        self.names = ['RACE_KEY', 'UMABAN', 'KETTO_TOROKU_BANGO', 'NENGAPPI', 'TOKKI_1', 'TOKKI_2', 'TOKKI_3', 'TOKKI_4', 'TOKKI_5', 'TOKKI_6', 'BAGU_1', 'BAGU_2', 'BAGU_3', 'BAGU_4', 'BAGU_5', 'BAGU_6', 'BAGU_7', 'BAGU_8', 'ASHIMOTO_SOGO_1', 'ASHIMOTO_SOGO_2', 'ASHIMOTO_SOGO_3', 'ASHIMOTO_HIDARI_MAE_1', 'ASHIMOTO_HIDARI_MAE_2', 'ASHIMOTO_HIDARI_MAE_3', 'ASHIMOTO_MIGI_MAE_1',
                      'ASHIMOTO_MIGI_MAE_2', 'ASHIMOTO_MIGI_MAE_3', 'ASHIMOTO_HIDARI_USHIRO_1', 'ASHIMOTO_HIDARI_USHIRO_2', 'ASHIMOTO_HIDARI_USHIRO_3', 'ASHIMOTO_MIGI_USHIRO_1', 'ASHIMOTO_MIGI_USHIRO_2', 'ASHIMOTO_MIGI_USHIRO_3', 'PADOC_COMMENNT', 'ASHIMOTO_COMMENT', 'BAGU_COMMENT', 'RACE_COMMENT', 'HAMI', 'BANTEGE', 'TEITETSU', 'HIDUME_JOTAI', 'SOE', 'KOTURYU', 'file_name', 'target_date']
        self.columns = ['%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s',
                        '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s']
        self.create_sql = self.sql_create()
        self.table_name = '[TALEND].[jrdb].[ORG_SKB]'

    def set_df(self, filename):
        with open('./' + filename, 'r', encoding="ms932") as fh:
            df = pd.DataFrame(index=[], columns=self.names)
            for line in fh:
                new_line = self.replace_line(line)
                sr = pd.Series([
                    new_line[0:8]  # RACE_KEY
                    , new_line[8:10]  # UMABAN
                    , new_line[10:18]  # KETTO_TOROKU_BANGO
                    , new_line[18:26]  # NENGAPPI
                    , new_line[26:29]  # TOKKI_1
                    , new_line[29:32]  # TOKKI_2
                    , new_line[32:35]  # TOKKI_3
                    , new_line[35:38]  # TOKKI_4
                    , new_line[38:41]  # TOKKI_5
                    , new_line[41:44]  # TOKKI_6
                    , new_line[44:47]  # BAGU_1
                    , new_line[47:50]  # BAGU_2
                    , new_line[50:53]  # BAGU_3
                    , new_line[53:56]  # BAGU_4
                    , new_line[56:59]  # BAGU_5
                    , new_line[59:62]  # BAGU_6
                    , new_line[62:65]  # BAGU_7
                    , new_line[65:68]  # BAGU_8
                    , new_line[68:71]  # ASHIMOTO_SOGO_1
                    , new_line[71:74]  # ASHIMOTO_SOGO_2
                    , new_line[74:77]  # ASHIMOTO_SOGO_3
                    , new_line[77:80]  # ASHIMOTO_HIDARI_MAE_1
                    , new_line[80:83]  # ASHIMOTO_HIDARI_MAE_2
                    , new_line[83:86]  # ASHIMOTO_HIDARI_MAE_3
                    , new_line[86:89]  # ASHIMOTO_MIGI_MAE_1
                    , new_line[89:92]  # ASHIMOTO_MIGI_MAE_2
                    , new_line[92:95]  # ASHIMOTO_MIGI_MAE_3
                    , new_line[95:98]  # ASHIMOTO_HIDARI_USHIRO_1
                    , new_line[98:101]  # ASHIMOTO_HIDARI_USHIRO_2
                    , new_line[101:104]  # ASHIMOTO_HIDARI_USHIRO_3
                    , new_line[104:107]  # ASHIMOTO_MIGI_USHIRO_1
                    , new_line[107:110]  # ASHIMOTO_MIGI_USHIRO_2
                    , new_line[110:113]  # ASHIMOTO_MIGI_USHIRO_3
                    , new_line[113:153].replace(" ", "")  # PADOC_COMMENNT
                    , new_line[153:193].replace(" ", "")  # ASHIMOTO_COMMENT
                    , new_line[193:233].replace(" ", "")  # BAGU_COMMENT
                    , new_line[233:273].replace(" ", "")  # RACE_COMMENT
                    , new_line[273:276]  # HAMI
                    , new_line[276:279]  # BANTEGE
                    , new_line[279:282]  # TEITETSU
                    , new_line[282:285]  # HIDUME_JOTAI
                    , new_line[285:288]  # SOE
                    , new_line[288:291]  # KOTURYU
                    , filename, filename[3:9]
                ], index=self.names)
                df = df.append(sr, ignore_index=True)
        return df

    def sql_create(self):
        create_sql = """if object_id('[TALEND].[jrdb].[ORG_SKB]') is null
        CREATE TABLE [TALEND].[jrdb].[ORG_SKB] (
        RACE_KEY                nvarchar(8),
        UMABAN                  nvarchar(2),
        KETTO_TOROKU_BANGO      nvarchar(8),
        NENGAPPI                nvarchar(8),
        TOKKI_1                 nvarchar(3),
        TOKKI_2                 nvarchar(3),
        TOKKI_3                 nvarchar(3),
        TOKKI_4                 nvarchar(3),
        TOKKI_5                 nvarchar(3),
        TOKKI_6                 nvarchar(3),
        BAGU_1                  nvarchar(3),
        BAGU_2                  nvarchar(3),
        BAGU_3                  nvarchar(3),
        BAGU_4                  nvarchar(3),
        BAGU_5                  nvarchar(3),
        BAGU_6                  nvarchar(3),
        BAGU_7                  nvarchar(3),
        BAGU_8                  nvarchar(3),
        ASHIMOTO_SOGO_1         nvarchar(3),
        ASHIMOTO_SOGO_2         nvarchar(3),
        ASHIMOTO_SOGO_3         nvarchar(3),
        ASHIMOTO_HIDARI_MAE_1   nvarchar(3),
        ASHIMOTO_HIDARI_MAE_2   nvarchar(3),
        ASHIMOTO_HIDARI_MAE_3   nvarchar(3),
        ASHIMOTO_MIGI_MAE_1     nvarchar(3),
        ASHIMOTO_MIGI_MAE_2     nvarchar(3),
        ASHIMOTO_MIGI_MAE_3     nvarchar(3),
        ASHIMOTO_HIDARI_USHIRO_1    nvarchar(3),
        ASHIMOTO_HIDARI_USHIRO_2    nvarchar(3),
        ASHIMOTO_HIDARI_USHIRO_3    nvarchar(3),
        ASHIMOTO_MIGI_USHIRO_1  nvarchar(3),
        ASHIMOTO_MIGI_USHIRO_2  nvarchar(3),
        ASHIMOTO_MIGI_USHIRO_3  nvarchar(3),
        PADOC_COMMENNT          nvarchar(40),
        ASHIMOTO_COMMENT        nvarchar(40),
        BAGU_COMMENT            nvarchar(40),
        RACE_COMMENT            nvarchar(40),
        HAMI                    nvarchar(3),
        BANTEGE                 nvarchar(3),
        TEITETSU                nvarchar(3),
        HIDUME_JOTAI            nvarchar(3),
        SOE                     nvarchar(3),
        KOTURYU                 nvarchar(3),
        filename                nvarchar(20),
        target_date          nvarchar(6)
        );
        """
        return create_sql


class HJC(JrdbSchemaBase):
    def __init__(self):
        super().__init__()
        self.names = ['RACE_KEY', 'TANSHO1_UMABAN', 'TANSHO1_HARAIMODOSHI', 'TANSHO2_UMABAN', 'TANSHO2_HARAIMODOSHI', 'TANSHO3_UMABAN', 'TANSHO3_HARAIMODOSHI', 'FUKUSHO1_UMABAN', 'FUKUAHO1_HARAIMODOSHI', 'FUKUSHO2_UMABAN', 'FUKUSHO2_HARAIMODOSHI', 'FUKUSHO3_UMABAN', 'FUKUSHO3_HARAIMODOSHI', 'FUKUSHO4_UMABAN', 'FUKUSHO4_HARAIMODOSHI', 'FUKUSHO5_UMABAN', 'FUKUSHO5_HARAIMODOSHI', 'WAKUREN1_WAKUBAN1', 'WAKUBAN1_WAKUBAN2', 'WAKUREN1_HARAIMODOSHI', 'WAKUREN2_WAKUBAN1', 'WAKUREN2_WAKUBAN2', 'WAKUREN2_HARAIMODOSHI', 'WAKUREN3_WAKUBAN1', 'WAKUREN3_WAKUBAN2', 'WAKUREN3_HARAIMODOSHI', 'UMAREN1_UMABAN1', 'UMAREN1_UMABAN2', 'UMAREN1_HARAIMODOSHI', 'UMAREN2_UMABAN1', 'UMAREN2_UMABAN2', 'UMAREN2_HARAIMODOSHI', 'UMAREN3_UMABAN1', 'UMAREN3_UMABAN2', 'UMAREN3_HARAIMODOSHI', 'WIDE1_UMABAN1', 'WIDE1_UMABAN2', 'WIDE1_HARAIMODOSHI', 'WIDE2_UMABAN1', 'WIDE2_UMABAN2', 'WIDE2_HARAIMODOSHI', 'WIDE3_UMABAN1', 'WIDE3_UMABAN2', 'WIDE3_HARAIMODOSHI', 'WIDE4_UMABAN1', 'WIDE4_UMABAN2', 'WIDE4_HARAIMODOSHI', 'WIDE5_UMABAN1', 'WIDE5_UMABAN2', 'WIDE5_HARAIMODOSHI', 'WIDE6_UMABAN1', 'WIDE6_UMABAN2', 'WIDE6_HARAIMODOSHI', 'WIDE7_UMABAN1', 'WIDE7_UMABAN2', 'WIDE7_HARAIMODOSHI', 'UMATAN1_UMABAN1', 'UMATAN1_UMABAN2',
                      'UMATAN1_HARAIMODOSHI', 'UMATAN2_UMABAN1', 'UMATAN2_UMABAN2', 'UMATAN2_HARAIMODOSHI', 'UMATAN3_UMABAN1', 'UMATAN3_UMABAN2', 'UMATAN3_HARAIMODOSHI', 'UMATAN4_UMABAN1', 'UMATAN4_UMABAN2', 'UMATAN4_HARAIMODOSHI', 'UMATAN5_UMABAN1', 'UMATAN5_UMABAN2', 'UMATAN5_HARAIMODOSHI', 'UMATAN6_UMABAN1', 'UMATAN6_UMABAN2', 'UMATAN6_HARAIMODOSHI', 'SANRENPUKU1_UMABAN1', 'SANRENPUKU1_UMABAN2', 'SANRENPUKU1_UMABAN3', 'SANRENPUKU1_HARAIMODOSHI', 'SANRENPUKU2_UMABAN1', 'SANRENPUKU2_UMABAN2', 'SANRENPUKU2_UMABAN3', 'SANRENPUKU2_HARAIMODOSHI', 'SANRENPUKU3_UMABAN1', 'SANRENPUKU3_UMABAN2', 'SANRENPUKU3_UMABAN3', 'SANRENPUKU3_HARAIMODOSHI', 'SANRENTAN1_UMABAN1', 'SANRENTAN1_UMABAN2', 'SANRENTAN1_UMABAN3', 'SANRENTAN1_HARAIMODOSHI', 'SANRENTAN2_UMABAN1', 'SANRENTAN2_UMABAN2', 'SANRENTAN2_UMABAN3', 'SANRENTAN2_HARAIMODOSHI', 'SANRENTAN3_UMABAN1', 'SANRENTAN3_UMABAN2', 'SANRENTAN3_UMABAN3', 'SANRENTAN3_HARAIMODOSHI', 'SANRENTAN4_UMABAN1', 'SANRENTAN4_UMABAN2', 'SANRENTAN4_UMABAN3', 'SANRENTAN4_HARAIMODOSHI', 'SANRENTAN5_UMABAN1', 'SANRENTAN5_UMABAN2', 'SANRENTAN5_UMABAN3', 'SANRENTAN5_HARAIMODOSHI', 'SANRENTAN6_UMABAN1', 'SANRENTAN6_UMABAN2', 'SANRENTAN6_UMABAN3', 'SANRENTAN6_HARAIMODOSHI', 'file_name', 'target_date']
        self.columns = ['%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s',  '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s',  '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s',  '%s', '%s', '%s', '%s', '%s',
                        '%s', '%s', '%s', '%s', '%s',  '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s',  '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s',  '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s']
        self.create_sql = self.sql_create()
        self.table_name = '[TALEND].[jrdb].[ORG_HJC]'

    def set_df(self, filename):
        with open('./' + filename, 'r', encoding="ms932") as fh:
            df = pd.DataFrame(index=[], columns=self.names)
            for line in fh:
                new_line = self.replace_line(line)
                sr = pd.Series([
                    new_line[0:8]  # RACE_KEY
                    # TANSHO1_UMABAN
                    # TANSHO1_HARAIMODOSHI
                    # TANSHO2_UMABAN
                    # TANSHO2_HARAIMODOSHI
                    # TANSHO3_UMABAN
                    # TANSHO3_HARAIMODOSHI
                    # FUKUSHO1_UMABAN
                    # FUKUAHO1_HARAIMODOSHI
                    # FUKUSHO2_UMABAN
                    # FUKUAHO2_HARAIMODOSHI
                    # FUKUSHO3_UMABAN
                    # FUKUAHO3_HARAIMODOSHI
                    # FUKUSHO4_UMABAN
                    # FUKUAHO4_HARAIMODOSHI
                    # FUKUSHO5_UMABAN
                    # FUKUAHO5_HARAIMODOSHI
                    # WAKUREN1_WAKUBAN1
                    # WAKUREN1_WAKUBAN2
                    # WAKUREN1_HARAIMODOSHI
                    # WAKUREN2_WAKUBAN1
                    # WAKUREN2_WAKUBAN2
                    # WAKUREN2_HARAIMODOSHI
                    # WAKUREN3_WAKUBAN1
                    # WAKUREN3_WAKUBAN2
                    # WAKUREN3_HARAIMODOSHI
                    # UMAREN1_UMABAN1
                    # UMAREN1_UMABAN2
                    # UMAREN1_HARAIMODOSHI
                    # UMAREN2_UMABAN1
                    # UMAREN2_UMABAN2
                    # UMAREN2_HARAIMODOSHI
                    # UMAREN3_UMABAN1
                    # UMAREN3_UMABAN2
                    # UMAREN3_HARAIMODOSHI
                    # WIDE1_UMABAN1
                    # WIDE1_UMABAN2
                    # WIDE1_HARAIMODOSHI
                    # WIDE2_UMABAN1
                    # WIDE2_UMABAN2
                    # WIDE2_HARAIMODOSHI
                    # WIDE3_UMABAN1
                    # WIDE3_UMABAN2
                    # WIDE3_HARAIMODOSHI
                    # WIDE4_UMABAN1
                    # WIDE4_UMABAN2
                    # WIDE4_HARAIMODOSHI
                    # WIDE5_UMABAN1
                    # WIDE5_UMABAN2
                    # WIDE5_HARAIMODOSHI
                    # WIDE6_UMABAN1
                    # WIDE6_UMABAN2
                    # WIDE6_HARAIMODOSHI
                    # WIDE7_UMABAN1
                    # WIDE7_UMABAN2
                    # WIDE7_HARAIMODOSHI
                    # UMATAN1_UMABAN1
                    # UMATAN1_UMABAN2
                    # UMATAN1_HARAIMODOSHI
                    # UMATAN2_UMABAN1
                    # UMATAN2_UMABAN2
                    # UMATAN2_HARAIMODOSHI
                    # UMATAN3_UMABAN1
                    # UMATAN3_UMABAN2
                    # UMATAN3_HARAIMODOSHI
                    # UMATAN4_UMABAN1
                    # UMATAN4_UMABAN2
                    # UMATAN4_HARAIMODOSHI
                    # UMATAN5_UMABAN1
                    # UMATAN5_UMABAN2
                    # UMATAN5_HARAIMODOSHI
                    # UMATAN6_UMABAN1
                    # UMATAN6_UMABAN2
                    # UMATAN6_HARAIMODOSHI
                    # SANRENPUKU1_UMABAN1
                    # SANRENPUKU1_UMABAN2
                    # SANRENPUKU1_UMABAN3
                    # SANRENPUKU1_HARAIMODOSHI
                    # SANRENPUKU2_UMABAN1
                    # SANRENPUKU2_UMABAN2
                    # SANRENPUKU2_UMABAN3
                    # SANRENPUKU2_HARAIMODOSHI
                    # SANRENPUKU3_UMABAN1
                    # SANRENPUKU3_UMABAN2
                    # SANRENPUKU3_UMABAN3
                    # SANRENPUKU3_HARAIMODOSHI
                    # SANRENTAN1_UMABAN1
                    # SANRENTAN1_UMABAN2
                    # SANRENTAN1_UMABAN3
                    # SANRENTAN1_HARAIMODOSHI
                    # SANRENTAN2_UMABAN1
                    # SANRENTAN2_UMABAN2
                    # SANRENTAN2_UMABAN3
                    # SANRENTAN2_HARAIMODOSHI
                    # SANRENTAN3_UMABAN1
                    # SANRENTAN3_UMABAN2
                    # SANRENTAN3_UMABAN3
                    # SANRENTAN3_HARAIMODOSHI
                    # SANRENTAN4_UMABAN1
                    # SANRENTAN4_UMABAN2
                    # SANRENTAN4_UMABAN3
                    # SANRENTAN4_HARAIMODOSHI
                    # SANRENTAN5_UMABAN1
                    # SANRENTAN5_UMABAN2
                    # SANRENTAN5_UMABAN3
                    # SANRENTAN5_HARAIMODOSHI
                    # SANRENTAN6_UMABAN1
                    # SANRENTAN6_UMABAN2
                    # SANRENTAN6_UMABAN3
                    # SANRENTAN6_HARAIMODOSHI
                    ,               new_line[8 + 2 * 0 + 7 * 0: 8 + 2 * 1 + 7 * 0], mu.int_null(new_line[8 + 2 * 1 + 7 * 0: 8 + 2 * 1 + 7 * 1]), new_line[8 + 2 * 1 + 7 * 1: 8 + 2 * 2 + 7 * 1], mu.int_null(new_line[8 + 2 * 2 + 7 * 1: 8 + 2 * 2 + 7 * 2]), new_line[8 + 2 * 2 + 7 * 2: 8 + 2 * 3 + 7 * 2], mu.int_null(new_line[8 + 2 * 3 + 7 * 2: 8 + 2 * 3 + 7 * 3]), new_line[35 + 2 * 0 + 7 * 0: 35 + 2 * 1 + 7 * 0], mu.int_null(new_line[35 + 2 * 1 + 7 * 0: 35 + 2 * 1 + 7 * 1]), new_line[35 + 2 * 1 + 7 * 1: 35 + 2 * 2 + 7 * 1], mu.int_null(new_line[35 + 2 * 2 + 7 * 1: 35 + 2 * 2 + 7 * 2]), new_line[35 + 2 * 2 + 7 * 2: 35 + 2 * 3 + 7 * 2], mu.int_null(new_line[35 + 2 * 3 + 7 * 2: 35 + 2 * 3 + 7 * 3]), new_line[35 + 2 * 3 + 7 * 3: 35 + 2 * 4 + 7 * 3], mu.int_null(new_line[35 + 2 * 4 + 7 * 3: 35 + 2 * 4 + 7 * 4]), new_line[35 + 2 * 4 + 7 * 4: 35 + 2 * 5 + 7 * 4], mu.int_null(new_line[35 + 2 * 5 + 7 * 4: 35 + 2 * 5 + 7 * 5]), new_line[80 + 1 * 0 + 7 * 0: 80 + 1 * 1 + 7 * 0], new_line[80 + 1 * 1 + 7 * 0: 80 + 1 * 2 + 7 * 0], mu.int_null(new_line[80 + 1 * 2 + 7 * 0: 80 + 1 * 2 + 7 * 1]), new_line[80 + 1 * 2 + 7 * 1: 80 + 1 * 3 + 7 * 1], new_line[80 + 1 * 3 + 7 * 1: 80 + 1 * 4 + 7 * 1], mu.int_null(new_line[80 + 1 * 4 + 7 * 1: 80 + 1 * 4 + 7 * 2]), new_line[80 + 1 * 4 + 7 * 2: 80 + 1 * 5 + 7 * 2], new_line[80 + 1 * 5 + 7 * 2: 80 + 1 * 6 + 7 * 2], mu.int_null(new_line[80 + 1 * 6 + 7 * 2: 80 + 1 * 6 + 7 * 3]), new_line[107 + 2 * 0 + 8 * 0:107 + 2 * 1 + 8 * 0], new_line[107 + 2 * 1 + 8 * 0:107 + 2 * 2 + 8 * 0], mu.int_null(new_line[107 + 2 * 2 + 8 * 0:107 + 2 * 2 + 8 * 1]), new_line[107 + 2 * 2 + 8 * 1:107 + 2 * 3 + 8 * 1], new_line[107 + 2 * 3 + 8 * 1:107 + 2 * 4 + 8 * 1], mu.int_null(new_line[107 + 2 * 4 + 8 * 1:107 + 2 * 4 + 8 * 2]), new_line[107 + 2 * 4 + 8 * 2:107 + 2 * 5 + 8 * 2], new_line[107 + 2 * 5 + 8 * 2:107 + 2 * 6 + 8 * 2], mu.int_null(new_line[107 + 2 * 6 + 8 * 2:107 + 2 * 6 + 8 * 3]), new_line[143 + 2 * 0 + 8 * 0: 143 + 2 * 1 + 8 * 0], new_line[143 + 2 * 1 + 8 * 0: 143 + 2 * 2 + 8 * 0], mu.int_null(new_line[143 + 2 * 2 + 8 * 0: 143 + 2 * 2 + 8 * 1]), new_line[143 + 2 * 2 + 8 * 1: 143 + 2 * 3 + 8 * 1], new_line[143 + 2 * 3 + 8 * 1: 143 + 2 * 4 + 8 * 1], mu.int_null(new_line[143 + 2 * 4 + 8 * 1: 143 + 2 * 4 + 8 * 2]), new_line[143 + 2 * 4 + 8 * 2: 143 + 2 * 5 + 8 * 2], new_line[143 + 2 * 5 + 8 * 2: 143 + 2 * 6 + 8 * 2], mu.int_null(new_line[143 + 2 * 6 + 8 * 2: 143 + 2 * 6 + 8 * 3]), new_line[143 + 2 * 6 + 8 * 3: 143 + 2 * 7 + 8 * 3], new_line[143 + 2 * 7 + 8 * 3: 143 + 2 * 8 + 8 * 3], mu.int_null(new_line[143 + 2 * 8 + 8 * 3: 143 + 2 * 8 + 8 * 4]), new_line[143 + 2 * 8 + 8 * 4: 143 + 2 * 9 + 8 * 4], new_line[143 + 2 * 9 + 8 * 4: 143 + 2 * 10 + 8 * 4], mu.int_null(new_line[143 + 2 * 10 + 8 * 4: 143 + 2 * 10 + 8 * 5]), new_line[143 + 2 * 10 + 8 * 5: 143 + 2 * 11 + 8 * 5], new_line[143 + 2 * 11 + 8 * 5: 143 + 2 * 12 + 8 * 5], mu.int_null(new_line[143 + 2 * 12 + 8 * 5: 143 + 2 * 12 + 8 * 6]), new_line[143 + 2 * 12 + 8 * 6: 143 + 2 * 13 + 8 * 6], new_line[143 + 2 * 13 + 8 * 6: 143 + 2 * 14 + 8 * 6], mu.int_null(new_line[143 + 2 * 14 + 8 * 6: 143 + 2 * 14 + 8 * 7]), new_line[227 + 2 * 0 + 8 * 0: 227 + 2 * 1 + 8 * 0], new_line[227 + 2 * 1 + 8 * 0: 227 + 2 * 2 + 8 * 0], mu.int_null(new_line[227 + 2 * 2 + 8 * 0: 227 + 2 * 2 + 8 * 1]), new_line[227 + 2 * 2 + 8 * 1: 227 + 2 * 3 + 8 * 1], new_line[227 + 2 * 3 + 8 * 1: 227 + 2 * 4 + 8 * 1], mu.int_null(new_line[227 + 2 * 4 + 8 * 1: 227 + 2 * 4 + 8 * 2]), new_line[227 + 2 * 4 + 8 * 2: 227 + 2 * 5 + 8 * 2], new_line[227 + 2 * 5 + 8 * 2: 227 + 2 * 6 + 8 * 2], mu.int_null(new_line[227 + 2 * 6 + 8 * 2: 227 + 2 * 6 + 8 * 3]), new_line[227 + 2 * 6 + 8 * 3: 227 + 2 * 7 + 8 * 3], new_line[227 + 2 * 7 + 8 * 3: 227 + 2 * 8 + 8 * 3], mu.int_null(new_line[227 + 2 * 8 + 8 * 3: 227 + 2 * 8 + 8 * 4]), new_line[227 + 2 * 8 + 8 * 4: 227 + 2 * 9 + 8 * 4], new_line[227 + 2 * 9 + 8 * 4: 227 + 2 * 10 + 8 * 4], mu.int_null(new_line[227 + 2 * 10 + 8 * 4: 227 + 2 * 10 + 8 * 5]), new_line[227 + 2 * 10 + 8 * 5: 227 + 2 * 11 + 8 * 5], new_line[227 + 2 * 11 + 8 * 5: 227 + 2 * 12 + 8 * 5], mu.int_null(new_line[227 + 2 * 12 + 8 * 5: 227 + 2 * 12 + 8 * 6]), new_line[299 + 2 * 0 + 8 * 0: 299 + 2 * 1 + 8 * 0], new_line[299 + 2 * 1 + 8 * 0: 299 + 2 * 2 + 8 * 0], new_line[299 + 2 * 2 + 8 * 0: 299 + 2 * 3 + 8 * 0], mu.int_null(new_line[299 + 2 * 3 + 8 * 0: 299 + 2 * 3 + 8 * 1]), new_line[299 + 2 * 3 + 8 * 1: 299 + 2 * 4 + 8 * 1], new_line[299 + 2 * 4 + 8 * 1: 299 + 2 * 5 + 8 * 1], new_line[299 + 2 * 5 + 8 * 1: 299 + 2 * 6 + 8 * 1], mu.int_null(new_line[299 + 2 * 6 + 8 * 1: 299 + 2 * 6 + 8 * 2]), new_line[299 + 2 * 6 + 8 * 2: 299 + 2 * 7 + 8 * 2], new_line[299 + 2 * 7 + 8 * 2: 299 + 2 * 8 + 8 * 2], new_line[299 + 2 * 8 + 8 * 2: 299 + 2 * 9 + 8 * 2], mu.int_null(new_line[299 + 2 * 9 + 8 * 2: 299 + 2 * 9 + 8 * 3]), new_line[341 + 2 * 0 + 9 * 0: 341 + 2 * 1 + 9 * 0], new_line[341 + 2 * 1 + 9 * 0: 341 + 2 * 2 + 9 * 0], new_line[341 + 2 * 2 + 9 * 0: 341 + 2 * 3 + 9 * 0], mu.int_null(new_line[341 + 2 * 3 + 9 * 0: 341 + 2 * 3 + 9 * 1]), new_line[341 + 2 * 3 + 9 * 1: 341 + 2 * 4 + 9 * 1], new_line[341 + 2 * 4 + 9 * 1: 341 + 2 * 5 + 9 * 1], new_line[341 + 2 * 5 + 9 * 1: 341 + 2 * 6 + 9 * 1], mu.int_null(new_line[341 + 2 * 6 + 9 * 1: 341 + 2 * 6 + 9 * 2]), new_line[341 + 2 * 6 + 9 * 2: 341 + 2 * 7 + 9 * 2], new_line[341 + 2 * 7 + 9 * 2: 341 + 2 * 8 + 9 * 2], new_line[341 + 2 * 8 + 9 * 2: 341 + 2 * 9 + 9 * 2], mu.int_null(new_line[341 + 2 * 9 + 9 * 2: 341 + 2 * 9 + 9 * 3]), new_line[341 + 2 * 9 + 9 * 3: 341 + 2 * 10 + 9 * 3], new_line[341 + 2 * 10 + 9 * 3: 341 + 2 * 11 + 9 * 3], new_line[341 + 2 * 11 + 9 * 3: 341 + 2 * 12 + 9 * 3], mu.int_null(new_line[341 + 2 * 12 + 9 * 3: 341 + 2 * 12 + 9 * 4]), new_line[341 + 2 * 12 + 9 * 4: 341 + 2 * 13 + 9 * 4], new_line[341 + 2 * 13 + 9 * 4: 341 + 2 * 14 + 9 * 4], new_line[341 + 2 * 14 + 9 * 4: 341 + 2 * 15 + 9 * 4], mu.int_null(new_line[341 + 2 * 15 + 9 * 4: 341 + 2 * 15 + 9 * 5]), new_line[341 + 2 * 15 + 9 * 5: 341 + 2 * 16 + 9 * 5], new_line[341 + 2 * 16 + 9 * 5: 341 + 2 * 17 + 9 * 5], new_line[341 + 2 * 17 + 9 * 5: 341 + 2 * 18 + 9 * 5], mu.int_null(new_line[341 + 2 * 18 + 9 * 5: 341 + 2 * 18 + 9 * 6]), filename, filename[3:9]
                ], index=self.names)
                df = df.append(sr, ignore_index=True)
        return df

    def sql_create(self):
        create_sql = """if object_id('[TALEND].[jrdb].[ORG_HJC]') is null
        CREATE TABLE [TALEND].[jrdb].[ORG_HJC] (
        RACE_KEY                nvarchar(8),
        TANSHO1_UMABAN          nvarchar(2),
        TANSHO1_HARAIMODOSHI     int,
        TANSHO2_UMABAN          nvarchar(2),
        TANSHO2_HARAIMODOSHI     int,
        TANSHO3_UMABAN          nvarchar(2),
        TANSHO3_HARAIMODOSHI     int,
        FUKUSHO1_UMABAN          nvarchar(2),
        FUKUAHO1_HARAIMODOSHI     int,
        FUKUSHO2_UMABAN          nvarchar(2),
        FUKUSHO2_HARAIMODOSHI     int,
        FUKUSHO3_UMABAN          nvarchar(2),
        FUKUSHO3_HARAIMODOSHI     int,
        FUKUSHO4_UMABAN          nvarchar(2),
        FUKUSHO4_HARAIMODOSHI     int,
        FUKUSHO5_UMABAN          nvarchar(2),
        FUKUSHO5_HARAIMODOSHI     int,
        WAKUREN1_WAKUBAN1          nvarchar(1),
        WAKUBAN1_WAKUBAN2          nvarchar(1),
        WAKUREN1_HARAIMODOSHI     int,
        WAKUREN2_WAKUBAN1          nvarchar(1),
        WAKUREN2_WAKUBAN2          nvarchar(1),
        WAKUREN2_HARAIMODOSHI     int,
        WAKUREN3_WAKUBAN1          nvarchar(1),
        WAKUREN3_WAKUBAN2          nvarchar(1),
        WAKUREN3_HARAIMODOSHI     int,
        UMAREN1_UMABAN1          nvarchar(2),
        UMAREN1_UMABAN2          nvarchar(2),
        UMAREN1_HARAIMODOSHI     int,
        UMAREN2_UMABAN1          nvarchar(2),
        UMAREN2_UMABAN2          nvarchar(2),
        UMAREN2_HARAIMODOSHI     int,
        UMAREN3_UMABAN1          nvarchar(2),
        UMAREN3_UMABAN2          nvarchar(2),
        UMAREN3_HARAIMODOSHI     int,
        WIDE1_UMABAN1          nvarchar(2),
        WIDE1_UMABAN2          nvarchar(2),
        WIDE1_HARAIMODOSHI     int,
        WIDE2_UMABAN1          nvarchar(2),
        WIDE2_UMABAN2          nvarchar(2),
        WIDE2_HARAIMODOSHI     int,
        WIDE3_UMABAN1          nvarchar(2),
        WIDE3_UMABAN2          nvarchar(2),
        WIDE3_HARAIMODOSHI     int,
        WIDE4_UMABAN1          nvarchar(2),
        WIDE4_UMABAN2          nvarchar(2),
        WIDE4_HARAIMODOSHI     int,
        WIDE5_UMABAN1          nvarchar(2),
        WIDE5_UMABAN2          nvarchar(2),
        WIDE5_HARAIMODOSHI     int,
        WIDE6_UMABAN1          nvarchar(2),
        WIDE6_UMABAN2          nvarchar(2),
        WIDE6_HARAIMODOSHI     int,
        WIDE7_UMABAN1          nvarchar(2),
        WIDE7_UMABAN2          nvarchar(2),
        WIDE7_HARAIMODOSHI     int,
        UMATAN1_UMABAN1          nvarchar(2),
        UMATAN1_UMABAN2          nvarchar(2),
        UMATAN1_HARAIMODOSHI     int,
        UMATAN2_UMABAN1          nvarchar(2),
        UMATAN2_UMABAN2          nvarchar(2),
        UMATAN2_HARAIMODOSHI     int,
        UMATAN3_UMABAN1          nvarchar(2),
        UMATAN3_UMABAN2          nvarchar(2),
        UMATAN3_HARAIMODOSHI     int,
        UMATAN4_UMABAN1          nvarchar(2),
        UMATAN4_UMABAN2          nvarchar(2),
        UMATAN4_HARAIMODOSHI     int,
        UMATAN5_UMABAN1          nvarchar(2),
        UMATAN5_UMABAN2          nvarchar(2),
        UMATAN5_HARAIMODOSHI     int,
        UMATAN6_UMABAN1          nvarchar(2),
        UMATAN6_UMABAN2          nvarchar(2),
        UMATAN6_HARAIMODOSHI     int,
        SANRENPUKU1_UMABAN1          nvarchar(2),
        SANRENPUKU1_UMABAN2          nvarchar(2),
        SANRENPUKU1_UMABAN3          nvarchar(2),
        SANRENPUKU1_HARAIMODOSHI     int,
        SANRENPUKU2_UMABAN1          nvarchar(2),
        SANRENPUKU2_UMABAN2          nvarchar(2),
        SANRENPUKU2_UMABAN3          nvarchar(2),
        SANRENPUKU2_HARAIMODOSHI     int,
        SANRENPUKU3_UMABAN1          nvarchar(2),
        SANRENPUKU3_UMABAN2          nvarchar(2),
        SANRENPUKU3_UMABAN3          nvarchar(2),
        SANRENPUKU3_HARAIMODOSHI     int,
        SANRENTAN1_UMABAN1          nvarchar(2),
        SANRENTAN1_UMABAN2          nvarchar(2),
        SANRENTAN1_UMABAN3          nvarchar(2),
        SANRENTAN1_HARAIMODOSHI     int,
        SANRENTAN2_UMABAN1          nvarchar(2),
        SANRENTAN2_UMABAN2          nvarchar(2),
        SANRENTAN2_UMABAN3          nvarchar(2),
        SANRENTAN2_HARAIMODOSHI     int,
        SANRENTAN3_UMABAN1          nvarchar(2),
        SANRENTAN3_UMABAN2          nvarchar(2),
        SANRENTAN3_UMABAN3          nvarchar(2),
        SANRENTAN3_HARAIMODOSHI     int,
        SANRENTAN4_UMABAN1          nvarchar(2),
        SANRENTAN4_UMABAN2          nvarchar(2),
        SANRENTAN4_UMABAN3          nvarchar(2),
        SANRENTAN4_HARAIMODOSHI     int,
        SANRENTAN5_UMABAN1          nvarchar(2),
        SANRENTAN5_UMABAN2          nvarchar(2),
        SANRENTAN5_UMABAN3          nvarchar(2),
        SANRENTAN5_HARAIMODOSHI     int,
        SANRENTAN6_UMABAN1          nvarchar(2),
        SANRENTAN6_UMABAN2          nvarchar(2),
        SANRENTAN6_UMABAN3          nvarchar(2),
        SANRENTAN6_HARAIMODOSHI     int,
        filename             nvarchar(20),
        target_date          nvarchar(6)
        );
        """
        return create_sql


class TYB(JrdbSchemaBase):
    def __init__(self):
        super().__init__()
        self.names = ['RACE_KEY', 'UMABAN', 'ODDS_RECORD', 'PADOC_RECORD', 'TOTAL_RECORD', 'BAGU_HENOU', 'ASHIMOTO_JOHO', 'TORIKESHI_FLAG', 'KISHU_CODE', 'BABA_JOTAI_CODE',
                      'TENKO_CODE', 'TANSHO_ODDS', 'FUKUSHO_ODDS', 'ODDS_TIME', 'BATAIJU', 'ZOGEN', 'ODDS_MARK', 'PADOC_MARK', 'CHOKUZEN_MARK', 'file_name', 'target_date']
        self.columns = ['%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s',
                        '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s']
        self.create_sql = self.sql_create()
        self.table_name = '[TALEND].[jrdb].[ORG_TYB]'

    def set_df(self, filename):
        with open('./' + filename, 'r', encoding="ms932") as fh:
            df = pd.DataFrame(index=[], columns=self.names)
            for line in fh:
                new_line = self.replace_line(line)
                sr = pd.Series([
                    new_line[0:8]  # RACE_KEY
                    , new_line[8:10]  # UMABAN
                    , mu.float_null(new_line[25:30])  # ODDS_RECORD
                    , mu.float_null(new_line[30:35])  # PADOC_RECORD
                    , mu.float_null(new_line[40:45])  # TOTAL_RECORD
                    , new_line[45:46]  # BAGU_HENOU
                    , new_line[46:47]  # ASHIMOTO_JOHO
                    , new_line[47:48]  # TORIKESHI_FLAG
                    , new_line[48:53]  # KISHU_CODE
                    , new_line[69:71]  # BABA_JOTAI_CODE
                    , new_line[71:72]  # TENKO_CODE
                    , mu.float_null(new_line[72:78])  # TANSHO_ODDS
                    , mu.float_null(new_line[78:84])  # FUKUSHO_ODDS
                    , new_line[84:88]  # ODDS_TIME
                    , mu.int_null(new_line[88:91])  # BATAIJU
                    , mu.int_bataiju_zogen(new_line[91:94])  # ZOGEN
                    , new_line[94:95]  # ODDS_MARK
                    , new_line[95:96]  # PADOC_MARK
                    , new_line[96:97]  # CHOKUZEN_MARK
                    , filename, filename[3:9]
                ], index=self.names)
                df = df.append(sr, ignore_index=True)
        return df

    def sql_create(self):
        create_sql = """if object_id('[TALEND].[jrdb].[ORG_TYB]') is null
        CREATE TABLE [TALEND].[jrdb].[ORG_TYB] (
        RACE_KEY             nvarchar(8),
        UMABAN               nvarchar(2),
        ODDS_RECORD          real,
        PADOC_RECORD         real,
        TOTAL_RECORD         real,
        BAGU_HENOU           nvarchar(1),
        ASHIMOTO_JOHO        nvarchar(1),
        TORIKESHI_FLAG       nvarchar(1),
        KISHU_CODE           nvarchar(5),
        BABA_JOTAI_CODE      nvarchar(2),
        TENKO_CODE           nvarchar(1),
        TANSHO_ODDS          real,
        FUKUSHO_ODDS         real,
        ODDS_TIME            nvarchar(4),
        BATAIJU              int,
        ZOGEN                int,
        ODDS_MARK            nvarchar(1),
        PADOC_MARK           nvarchar(1),
        CHOKUZEN_MARK        nvarchar(1),
        filename             nvarchar(20),
        target_date          nvarchar(6)
        );
        """
        return create_sql


class OZ(JrdbSchemaBase):
    def __init__(self):
        super().__init__()
        self.names = ['RACE_KEY', 'UMABAN', 'TANSHO_BASE_ODDS', 'FUKUSHO_BASE_ODDS', 'UMAREN_01', 'UMAREN_02', 'UMAREN_03', 'UMAREN_04', 'UMAREN_05', 'UMAREN_06', 'UMAREN_07',
                      'UMAREN_08', 'UMAREN_09', 'UMAREN_10', 'UMAREN_11', 'UMAREN_12', 'UMAREN_13', 'UMAREN_14', 'UMAREN_15', 'UMAREN_16', 'UMAREN_17', 'UMAREN_18', 'file_name', 'target_date']
        self.columns = ['%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s',
                        '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s']
        self.create_sql = self.sql_create()
        self.table_name = '[TALEND].[jrdb].[ORG_OZ]'

    def set_df(self, filename):
        with open('./' + filename, 'r', encoding="ms932") as fh:
            df = pd.DataFrame(index=[], columns=self.names)
            for line in fh:
                new_line = self.replace_line(line)
                race_key = new_line[0:8]
                tosu = int(new_line[8:10])
                odds_tansho_text = new_line[10:100]
                odds_tansho_list = deque(
                    [odds_tansho_text[i: i + 5] for i in range(0, len(odds_tansho_text), 5)])
                odds_fukusho_text = new_line[100:190]
                odds_fukusho_list = deque(
                    [odds_fukusho_text[i: i + 5] for i in range(0, len(odds_fukusho_text), 5)])
                odds_umaren_text = new_line[190:955]
                odds_umaren_list = deque(
                    [odds_umaren_text[i: i + 5] for i in range(0, len(odds_umaren_text), 5)])
                for uma1 in range(1, 19):
                    base_tansho_odds = mu.float_null(
                        odds_tansho_list.popleft())
                    base_fukusho_odds = mu.float_null(
                        odds_fukusho_list.popleft())
                    bul = [0] * 19  # base_umaren_list
                    for uma2 in range(uma1 + 1, 19):
                        bul[uma2] = mu.float_null(odds_umaren_list.popleft())
                    sr = pd.Series([
                        race_key, str(uma1).zfill(2), base_tansho_odds, base_fukusho_odds, bul[1], bul[2], bul[3], bul[4], bul[5], bul[6], bul[7], bul[
                            8], bul[9], bul[10], bul[11], bul[12], bul[13], bul[14], bul[15], bul[16], bul[17], bul[18], filename, filename[2:8]
                    ], index=self.names)
                    df = df.append(sr, ignore_index=True)
        return df

    def sql_create(self):
        create_sql = """if object_id('[TALEND].[jrdb].[ORG_OZ]') is null
        CREATE TABLE [TALEND].[jrdb].[ORG_OZ] (
        RACE_KEY             nvarchar(8),
        UMABAN               nvarchar(2),
        TANSHO_BASE_ODDS     real,
        FUKUSHO_BASE_ODDS    real,
        UMAREN_01            real,
        UMAREN_02            real,
        UMAREN_03            real,
        UMAREN_04            real,
        UMAREN_05            real,
        UMAREN_06            real,
        UMAREN_07            real,
        UMAREN_08            real,
        UMAREN_09            real,
        UMAREN_10            real,
        UMAREN_11            real,
        UMAREN_12            real,
        UMAREN_13            real,
        UMAREN_14            real,
        UMAREN_15            real,
        UMAREN_16            real,
        UMAREN_17            real,
        UMAREN_18            real,
        filename             nvarchar(20),
        target_date          nvarchar(6)
        );
        """
        return create_sql


class OT(JrdbSchemaBase):
    def __init__(self):
        super().__init__()
        self.names = ['RACE_KEY', 'UMABAN_1', 'UMABAN_2', 'UMABAN_3',
                      'SANRENPUKU_BASE_ODDS', 'file_name', 'target_date']
        self.columns = ['%s', '%s', '%s', '%s', '%s', '%s', '%s']
        self.create_sql = self.sql_create()
        self.table_name = '[TALEND].[jrdb].[ORG_OT]'

    def set_df(self, filename):
        with open('./' + filename, 'r', encoding="ms932") as fh:
            df = pd.DataFrame(index=[], columns=self.names)
            print(dt.datetime.now())
            for line in fh:
                new_line = self.replace_line(line)
                race_key = new_line[0:8]
                odds_text = new_line[10:4906]
                base_odds_list = deque([odds_text[i: i + 6]
                                        for i in range(0, len(odds_text), 6)])
                for uma1 in range(1, 17):
                    for uma2 in range(uma1 + 1, 18):
                        for uma3 in range(uma2 + 1, 19):
                            base_odds = mu.float_null(
                                base_odds_list.popleft())
                            if base_odds != None:
                                sr = pd.Series([
                                    race_key, str(uma1).zfill(2), str(uma2).zfill(2), str(
                                        uma3).zfill(2), base_odds, filename, filename[2:8]
                                ], index=self.names)
                                df = df.append(sr, ignore_index=True)
        return df

    def sql_create(self):
        create_sql = """if object_id('[TALEND].[jrdb].[ORG_OT]') is null
        CREATE TABLE [TALEND].[jrdb].[ORG_OT] (
        RACE_KEY             nvarchar(8),
        UMABAN_1             nvarchar(2),
        UMABAN_2             nvarchar(2),
        UMABAN_3             nvarchar(2),
        BASE_SANRENPUKU      real,
        filename             nvarchar(20),
        target_date          nvarchar(6)
        );
        """
        return create_sql
