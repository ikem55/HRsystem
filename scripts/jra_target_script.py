from modules.jra_extract import JRAExtract
from modules.jra_transform import JRATransform
from modules.jra_load import JRALoad
import modules.util as mu
import my_config as mc

from datetime import datetime as dt
from datetime import timedelta
import sys
import pandas as pd
import numpy as np
import math
import scipy
import os
from sklearn import preprocessing

class Ext(JRAExtract):
    pass

class Tf(JRATransform):

    def encode_raceuma_before_df(self, raceuma_df, dict_folder):
        raceuma_df["性別コード"] = raceuma_df["性別コード"].apply(lambda x: mu.convert_sex(x))
        raceuma_df["脚質"] = raceuma_df["脚質"].apply(lambda x: mu.convert_kyakushitsu(x))
        raceuma_df["馬主会コード"] = raceuma_df["馬主会コード"].apply(lambda x: "馬主" + mu.convert_basho(x))
        raceuma_df["馬記号コード"] = raceuma_df["馬記号コード"].apply(lambda x: mu.convert_umakigo(x))
        raceuma_df["輸送区分"] = raceuma_df["輸送区分"].apply(lambda x: mu.convert_yusokubun(x))
        raceuma_df["見習い区分"] = raceuma_df["見習い区分"].apply(lambda x: mu.convert_minaraikubun(x))
        raceuma_df["芝適性コード"] = raceuma_df["芝適性コード"].apply(lambda x: "芝" + mu.convert_tekisei(x))
        raceuma_df["ダ適性コード"] = raceuma_df["ダ適性コード"].apply(lambda x: "ダ" + mu.convert_tekisei(x))
        raceuma_df["距離適性"] = raceuma_df["距離適性"].apply(lambda x: mu.convert_kyori_tekisei(x))
        raceuma_df["上昇度"] = raceuma_df["上昇度"].apply(lambda x: mu.convert_joshodo(x))
        raceuma_df["調教矢印コード"] = raceuma_df["調教矢印コード"].apply(lambda x: mu.convert_chokyo_yajirushi(x))
        raceuma_df["厩舎評価コード"] = raceuma_df["厩舎評価コード"].apply(lambda x: mu.convert_kyusha_hyouka(x))
        raceuma_df["蹄コード"] = raceuma_df["蹄コード"].apply(lambda x: mu.convert_hidume(x))
        raceuma_df["重適正コード"] = raceuma_df["重適正コード"].apply(lambda x: mu.convert_omotekisei(x))
        raceuma_df["クラスコード"] = raceuma_df["クラスコード"].apply(lambda x: mu.convert_class(x))
        raceuma_df["展開記号"] = raceuma_df["展開記号"].apply(lambda x: mu.convert_tenkaimark(x))
        raceuma_df["体型０１"] = raceuma_df["体型０１"].apply(lambda x: "体型" + mu.convert_taikei(x))
        raceuma_df["体型０２"] = raceuma_df["体型０２"].apply(lambda x: "体型背中" + mu.convert_taikei_long(x))
        raceuma_df["体型０３"] = raceuma_df["体型０３"].apply(lambda x: "体型胴" + mu.convert_taikei_long(x))
        raceuma_df["体型０４"] = raceuma_df["体型０４"].apply(lambda x: "体型尻" + mu.convert_taikei_big(x))
        raceuma_df["体型０５"] = raceuma_df["体型０５"].apply(lambda x: "体型トモ" + mu.convert_taikei_kakudo(x))
        raceuma_df["体型０６"] = raceuma_df["体型０６"].apply(lambda x: "体型腹袋" + mu.convert_taikei_big(x))
        raceuma_df["体型０７"] = raceuma_df["体型０７"].apply(lambda x: "体型頭" + mu.convert_taikei_big(x))
        raceuma_df["体型０８"] = raceuma_df["体型０８"].apply(lambda x: "体型首" + mu.convert_taikei_long(x))
        raceuma_df["体型０９"] = raceuma_df["体型０９"].apply(lambda x: "体型胸" + mu.convert_taikei_big(x))
        raceuma_df["体型１０"] = raceuma_df["体型１０"].apply(lambda x: "体型肩" + mu.convert_taikei_kakudo(x))
        raceuma_df["体型１１"] = raceuma_df["体型１１"].apply(lambda x: "体型前長" + mu.convert_taikei_long(x))
        raceuma_df["体型１２"] = raceuma_df["体型１２"].apply(lambda x: "体型後長" + mu.convert_taikei_long(x))
        raceuma_df["体型１３"] = raceuma_df["体型１３"].apply(lambda x: "体型前幅" + mu.convert_taikei_hohaba(x))
        raceuma_df["体型１４"] = raceuma_df["体型１４"].apply(lambda x: "体型後幅" + mu.convert_taikei_hohaba(x))
        raceuma_df["体型１５"] = raceuma_df["体型１５"].apply(lambda x: "体型前繋" + mu.convert_taikei_long(x))
        raceuma_df["体型１６"] = raceuma_df["体型１６"].apply(lambda x: "体型後繋" + mu.convert_taikei_long(x))
        raceuma_df["体型１７"] = raceuma_df["体型１７"].apply(lambda x: "体型尾" + mu.convert_taikei_tsukene(x))
        raceuma_df["体型１８"] = raceuma_df["体型１８"].apply(lambda x: "体型振" + mu.convert_taikei_o(x))
        raceuma_df["体型総合１"] = raceuma_df["体型総合１"].apply(lambda x: mu.convert_tokki(x))
        raceuma_df["体型総合２"] = raceuma_df["体型総合２"].apply(lambda x: mu.convert_tokki(x))
        raceuma_df["体型総合３"] = raceuma_df["体型総合３"].apply(lambda x: mu.convert_tokki(x))
        raceuma_df["馬特記１"] = raceuma_df["馬特記１"].apply(lambda x: mu.convert_tokki(x))
        raceuma_df["馬特記２"] = raceuma_df["馬特記２"].apply(lambda x: mu.convert_tokki(x))
        raceuma_df["馬特記３"] = raceuma_df["馬特記３"].apply(lambda x: mu.convert_tokki(x))
        raceuma_df["調教タイプ"] = raceuma_df["調教タイプ"].apply(lambda x: mu.convert_chokyo_type(x))
        raceuma_df["調教コースコード"] = raceuma_df["調教コースコード"].apply(lambda x: mu.convert_chokyo_course(x))
        raceuma_df["追切種類"] = raceuma_df["追切種類"].apply(lambda x: mu.convert_oikiri_shurui(x))
        raceuma_df["調教コース種別"] = raceuma_df["調教コース種別"].apply(lambda x: mu.convert_chokyo_course_shubetsu(x))
        raceuma_df["調教距離"] = raceuma_df["調教距離"].apply(lambda x: mu.convert_chokyo_kyori(x))
        raceuma_df["調教重点"] = raceuma_df["調教重点"].apply(lambda x: mu.convert_chokyo_juten(x))
        raceuma_df["調教量評価"] = raceuma_df["調教量評価"].apply(lambda x: "調教量評価" + x)
        return raceuma_df

    def normalize_raceuma_df(self, raceuma_df):
        return raceuma_df

    def encode_raceuma_result_df(self, raceuma_df, dict_folder):
        raceuma_df["芝ダ障害コード"] = raceuma_df["芝ダ障害コード"].apply(lambda x: mu.convert_shida(x))
        raceuma_df["芝種類"] = raceuma_df["芝種類"].apply(lambda x: mu.convert_shibatype(x))
        raceuma_df["転圧"] = raceuma_df["転圧"].apply(lambda x: "転圧" if x == '1' else "")
        raceuma_df["凍結防止剤"] = raceuma_df["凍結防止剤"].apply(lambda x: "凍結防止剤散布" if x == '1' else "")
        raceuma_df["馬場状態"] = raceuma_df["馬場状態"].apply(lambda x: mu.convert_babajotai(x))
        raceuma_df["レース脚質"] = raceuma_df["レース脚質"].apply(lambda x: mu.convert_kyakushitsu(x))
        raceuma_df["特記コード１"] = raceuma_df["特記コード１"].apply(lambda x: mu.convert_tokki(x))
        raceuma_df["特記コード２"] = raceuma_df["特記コード２"].apply(lambda x: mu.convert_tokki(x))
        raceuma_df["特記コード３"] = raceuma_df["特記コード３"].apply(lambda x: mu.convert_tokki(x))
        raceuma_df["特記コード４"] = raceuma_df["特記コード４"].apply(lambda x: mu.convert_tokki(x))
        raceuma_df["特記コード５"] = raceuma_df["特記コード５"].apply(lambda x: mu.convert_tokki(x))
        raceuma_df["特記コード６"] = raceuma_df["特記コード６"].apply(lambda x: mu.convert_tokki(x))
        raceuma_df["馬具コード１"] = raceuma_df["馬具コード１"].apply(lambda x: mu.convert_bagu(x))
        raceuma_df["馬具コード２"] = raceuma_df["馬具コード２"].apply(lambda x: mu.convert_bagu(x))
        raceuma_df["馬具コード３"] = raceuma_df["馬具コード３"].apply(lambda x: mu.convert_bagu(x))
        raceuma_df["馬具コード４"] = raceuma_df["馬具コード４"].apply(lambda x: mu.convert_bagu(x))
        raceuma_df["馬具コード５"] = raceuma_df["馬具コード５"].apply(lambda x: mu.convert_bagu(x))
        raceuma_df["馬具コード６"] = raceuma_df["馬具コード６"].apply(lambda x: mu.convert_bagu(x))
        raceuma_df["総合１"] = raceuma_df["総合１"].apply(lambda x: mu.convert_ashimoto(x))
        raceuma_df["総合２"] = raceuma_df["総合２"].apply(lambda x: mu.convert_ashimoto(x))
        raceuma_df["総合３"] = raceuma_df["総合３"].apply(lambda x: mu.convert_ashimoto(x))
        raceuma_df["左前１"] = raceuma_df["左前１"].apply(lambda x: mu.convert_ashimoto(x))
        raceuma_df["左前２"] = raceuma_df["左前２"].apply(lambda x: mu.convert_ashimoto(x))
        raceuma_df["左前３"] = raceuma_df["左前３"].apply(lambda x: mu.convert_ashimoto(x))
        raceuma_df["右前１"] = raceuma_df["右前１"].apply(lambda x: mu.convert_ashimoto(x))
        raceuma_df["右前２"] = raceuma_df["右前２"].apply(lambda x: mu.convert_ashimoto(x))
        raceuma_df["右前３"] = raceuma_df["右前３"].apply(lambda x: mu.convert_ashimoto(x))
        raceuma_df["左後１"] = raceuma_df["左後１"].apply(lambda x: mu.convert_ashimoto(x))
        raceuma_df["左後２"] = raceuma_df["左後２"].apply(lambda x: mu.convert_ashimoto(x))
        raceuma_df["左後３"] = raceuma_df["左後３"].apply(lambda x: mu.convert_ashimoto(x))
        raceuma_df["右後１"] = raceuma_df["右後１"].apply(lambda x: mu.convert_ashimoto(x))
        raceuma_df["右後２"] = raceuma_df["右後２"].apply(lambda x: mu.convert_ashimoto(x))
        raceuma_df["右後３"] = raceuma_df["右後３"].apply(lambda x: mu.convert_ashimoto(x))
        raceuma_df["ハミ"] = raceuma_df["ハミ"].apply(lambda x: "ハミ" + mu.convert_bagu(x))
        raceuma_df["蹄鉄"] = raceuma_df["蹄鉄"].apply(lambda x: "蹄鉄" + mu.convert_bagu(x))
        raceuma_df["バンテージ"] = raceuma_df["バンテージ"].apply(lambda x: "バンテージ" if x == '007' else '')
        raceuma_df["蹄状態"] = raceuma_df["蹄状態"].apply(lambda x: "蹄状態" + mu.convert_bagu(x))
        raceuma_df["ソエ"] = raceuma_df["ソエ"].apply(lambda x: "ソエ" + mu.convert_bagu(x))
        raceuma_df["骨瘤"] = raceuma_df["骨瘤"].apply(lambda x: "骨瘤" + mu.convert_bagu(x))
        raceuma_df["場コード"] = raceuma_df["場コード"].apply(lambda x: mu.convert_basho(x))
        return raceuma_df

    def encode_horse_df(self, horse_df, dict_folder):
        horse_df["毛色コード"] = horse_df["毛色コード"].apply(lambda x: mu.convert_keiro(x))
        horse_df["父系統コード"] = horse_df["父系統コード"].apply(lambda x: mu.convert_keito(x))
        horse_df["母父系統コード"] = horse_df["母父系統コード"].apply(lambda x: mu.convert_keito(x))
        return horse_df

    def normalize_raceuma_result_df(self, raceuma_df):
        return raceuma_df

    def create_feature_raceuma_result_df(self, raceuma_df):
        """  raceuma_dfの特徴量を作成する。馬番→馬番グループを作成して列を追加する。

        :param dataframe raceuma_df:
        :return: dataframe
        """
        temp_raceuma_df = raceuma_df.copy()
        temp_raceuma_df.loc[:, "非根幹"] = temp_raceuma_df["距離"].apply(lambda x: 0 if x % 400 == 0 else 1)
        temp_raceuma_df.loc[:, "距離グループ"] = temp_raceuma_df["距離"] // 400
        temp_raceuma_df.loc[:, "追込率"] = (temp_raceuma_df["コーナー順位４"] - temp_raceuma_df["着順"]) / temp_raceuma_df["頭数"]
        temp_raceuma_df.loc[:, "コーナー順位１"] = (temp_raceuma_df["コーナー順位１"] / temp_raceuma_df["頭数"])
        temp_raceuma_df.loc[:, "コーナー順位２"] = (temp_raceuma_df["コーナー順位２"] / temp_raceuma_df["頭数"])
        temp_raceuma_df.loc[:, "コーナー順位３"] = (temp_raceuma_df["コーナー順位３"] / temp_raceuma_df["頭数"])
        temp_raceuma_df.loc[:, "コーナー順位４"] = (temp_raceuma_df["コーナー順位４"] / temp_raceuma_df["頭数"])
        return temp_raceuma_df

    def drop_columns_raceuma_result_df(self, raceuma_df):
        raceuma_df = raceuma_df.drop(["NENGAPPI", "馬名", "レース名", "レース名略称", "タイム", "斤量", "騎手コード", "調教師コード", "調教師名", "確定単勝オッズ", "確定単勝人気順位",
                                      "ＩＤＭ結果", "素点", "馬場差", "ペース", "出遅", "位置取", "不利", "前不利", "中不利", "中不利", "後不利", "レース", "コース取り",
                                      "上昇度コード", "クラスコード", "馬体コード", "気配コード", "レースペース", "1(2)着馬名", "前３Ｆタイム", "後３Ｆタイム", "確定複勝オッズ下",
                                      "パドックコメント", "脚元コメント", "馬具(その他)コメント", "天候コード", "コース", "本賞金", "収得賞金", "レースペース流れ",
                                      "10時単勝オッズ", "10時複勝オッズ", "馬体重", "馬体重増減", "レースコメント", "異常区分", "血統登録番号", "単勝", "複勝", "KYOSO_RESULT_KEY",
                                      "種別", "条件", "記号", "重量", "グレード", "異常区分", "右左", "内外", "４角コース取り", "レース馬コメント", "KAISAI_KEY",
                                      "１コーナー", "２コーナー", "３コーナー", "４コーナー", "ペースアップ位置", "１角１", "１角２", "１角３", "２角１", "２角２", "２角３",
                                      "向正１", "向正２", "向正３", "３角１", "３角２", "３角３", "４角０", "４角１", "４角２", "４角３", "４角４", "直線０", "直線１", "直線２",
                                      "直線３", "直線４", "COURSE_KEY", "ハロンタイム０１", "ハロンタイム０２", "ハロンタイム０３", "ハロンタイム０４", "ハロンタイム０５", "ハロンタイム０６",
                                      "ハロンタイム０７", "ハロンタイム０８", "ハロンタイム０９", "ハロンタイム１０", "ハロンタイム１１", "ハロンタイム１２", "ハロンタイム１３", "ハロンタイム１４",
                                      "ハロンタイム１５", "ハロンタイム１６", "ハロンタイム１７", "ハロンタイム１８", "ラスト５ハロン", "ラスト４ハロン", "ラスト３ハロン", "ラスト２ハロン",
                                      "ラスト１ハロン", "ラップ差４ハロン", "ラップ差３ハロン", "ラップ差２ハロン", "ラップ差１ハロン", "連続何日目", "草丈", "中間降水量", "１着算入賞金",
                                      "ハロン数", "芝", "外", "重", "軽", "TRACK_BIAS_ZENGO", "TRACK_BIAS_UCHISOTO", "１ハロン平均_mean", "ＩＤＭ結果_mean", "テン指数結果_mean",
                                      "上がり指数結果_mean", "ペース指数結果_mean", "前３Ｆタイム_mean", "後３Ｆタイム_mean", "コーナー順位１", "コーナー順位２", "コーナー順位３",
                                      "コーナー順位４", "前３Ｆ先頭差", "後３Ｆ先頭差", "追走力_mean", "追上力_mean", "後傾指数_mean", "１ハロン平均_std", "上がり指数結果_std",
                                      "ペース指数結果_std", "非根幹", "距離グループ", "追込率", "ペース指数結果", "上がり指数結果", "テン指数結果", "着順", "頭数",
                                      "後３Ｆ先頭差_mean", "前３Ｆ先頭差_mean", "コーナー順位４_mean", "コーナー順位３_mean", "コーナー順位２_mean", "コーナー順位１_mean",
                                      "IDM", "ペース指数結果順位", "ペース指数結果順位", "上がり指数結果順位", "テン指数結果順位", "1(2)着タイム差", "レースＰ指数結果",
                                      "馬場状態", "騎手名", "馬ペース", "馬具コード１", "馬具コード２", "馬具コード３", "馬具コード４", "馬具コード５", "馬具コード６",
                                      "馬具コード７", "馬具コード８", "総合１", "総合２", "総合３", "左前１", "左前２", "左前３", "右前１", "右前２", "右前３",
                                      "左後１", "左後２", "左後３", "右後１", "右後２", "右後３", "ハミ", "蹄鉄", "蹄状態", "ソエ", "バンテージ", "骨瘤", "芝種類", "転圧", "凍結防止剤"], axis=1)
        return raceuma_df

    def drop_columns_cbs_df(self, raceuma_df):
        raceuma_df = raceuma_df.astype({'負担重量': 'int', '枠番': 'int', '激走指数': 'int', '道中内外': 'int', '後３Ｆ内外': 'int', 'ゴール内外': 'int'})
        raceuma_df = raceuma_df.drop(["ZENSO1_KYOSO_RESULT", "ZENSO2_KYOSO_RESULT", "ZENSO3_KYOSO_RESULT", "ZENSO4_KYOSO_RESULT", "ZENSO5_KYOSO_RESULT",
                                      "ZENSO1_RACE_KEY", "ZENSO2_RACE_KEY", "ZENSO3_RACE_KEY", "ZENSO4_RACE_KEY", "ZENSO5_RACE_KEY", "総合印", "ＩＤＭ印",
                                      "情報印", "騎手印", "厩舎印", "調教印", "激走印", "条件クラス", "ペース予想", "距離適性２", "走法", "参考前走", "参考前走騎手コード",
                                      "万券印", "降級フラグ", "激走タイプ", "休養理由分類コード", "芝ダ障害フラグ", "距離フラグ", "クラスフラグ", "転厩フラグ", "去勢フラグ",
                                      "乗替フラグ", "入厩何走目", "入厩年月日", "入厩何日前", "放牧先ランク", "CID", "LS評価", "EM", "厩舎ＢＢ印", "騎手ＢＢ印",
                                      "調教曜日", "調教回数", "追い状態", "乗り役", "併せ結果", "併せ追切種類", "併せクラス", "仕上指数変化", "調教評価", "登録抹消フラグ",
                                      "COURSE_KEY", "芝ダ障害コード",
                                      "距離", "騎手コード", "調教師コード", "ブリンカー",  "条件", "収得賞金", "獲得賞金", "ローテーション", "基準オッズ", "基準人気順位", "基準複勝オッズ",
                                      "基準複勝人気順位", "特定情報◎", "特定情報○", "特定情報▲", "特定情報△", "特定情報×", "総合情報◎", "総合情報○", "総合情報▲",
                                      "総合情報△", "総合情報×", "騎手期待連対率", "IDM", "騎手指数", "情報指数", "総合指数", "人気指数", "調教指数", "厩舎指数",
                                      "激走指数", "激走順位", "ゴール順位", "ゴール差", "LS指数順位", "テン指数順位", "ペース指数順位", "上がり指数順位", "位置指数順位",
                                      "騎手期待単勝率", "騎手期待３着内率", "馬出遅率", "厩舎ランク", "CID調教素点", "CID厩舎素点", "CID素点", "LS指数",
                                      "厩舎ＢＢ◎単勝回収率", "厩舎ＢＢ◎連対率", "騎手ＢＢ◎単勝回収率", "騎手ＢＢ◎連対率", "調教Ｆ", "テンＦ", "中間Ｆ", "終いＦ",
                                      "基準人気グループ", "仕上指数", "追切指数", "血統登録番号", "併せ年齢", "場コード",
                                      "上昇度", "調教矢印コード", "クラスコード", "見習い区分", "展開記号", "馬記号コード", "輸送区分",
                                      "ＪＲＡ成績", "交流成績", "他成績", "芝ダ障害別成績", "芝ダ障害別距離成績", "トラック距離成績", "ローテ成績", "回り成績",
                                      "騎手成績", "枠成績", "騎手距離成績", "騎手トラック距離成績", "騎手調教師別成績", "騎手馬主別成績", "騎手ブリンカ成績", "調教師馬主別成績",
                                      "道中順位", "後３Ｆ順位", "負担重量", "馬スタート指数", "万券指数", "厩舎評価コード",
                                      "距離_1", "fa_1_1", "fa_2_1", "fa_3_1", "fa_4_1", "fa_5_1", "course_cluster_1", "ru_cluster_1"], axis=1)
                                      #"距離_2", "fa_1_2", "fa_2_2", "fa_3_2", "fa_4_2", "fa_5_2", "course_cluster_2", "ru_cluster_2",
                                      #"距離_3", "fa_1_3", "fa_2_3", "fa_3_3", "fa_4_3", "fa_5_3", "course_cluster_3", "ru_cluster_3",
                                      #"距離_4", "fa_1_4", "fa_2_4", "fa_3_4", "fa_4_4", "fa_5_4", "course_cluster_4", "ru_cluster_4",
                                      #"距離_5", "fa_1_5", "fa_2_5", "fa_3_5", "fa_4_5", "fa_5_5", "course_cluster_5", "ru_cluster_5"
        return raceuma_df

class Ld(JRALoad):
    def _get_extract_object(self, start_date, end_date, mock_flag):
        """ 利用するExtクラスを指定する """
        ext = Ext(start_date, end_date, mock_flag)
        return ext

    def _get_transform_object(self, start_date, end_date):
        """ 利用するTransformクラスを指定する """
        tf = Tf(start_date, end_date)
        return tf

    def set_race_file_df(self):
        race_file_df = self.ext.get_race_before_table_base()[["RACE_KEY", "NENGAPPI", "距離", "芝ダ障害コード", "内外", "条件"]]
        race_file_df = race_file_df.groupby("RACE_KEY").first().reset_index()
        race_file_df.loc[:, "RACE_ID"] = race_file_df.apply(lambda x: mu.convert_jrdb_id(x["RACE_KEY"], x["NENGAPPI"]), axis=1)
        race_file_df.loc[:, "file_id"] = race_file_df["RACE_KEY"].apply(lambda x: mu.convert_target_file(x))
        race_file_df.loc[:, "nichiji"] = race_file_df["RACE_KEY"].apply(lambda x: mu.convert_kaiji(x[5:6]))
        race_file_df.loc[:, "race_no"] = race_file_df["RACE_KEY"].str[6:8]
        race_file_df.loc[:, "rc_file_id"] = race_file_df["RACE_KEY"].apply(lambda x: "RC" + x[0:5])
        race_file_df.loc[:, "kc_file_id"] = "KC" + race_file_df["RACE_KEY"].str[0:6]
        self.race_file_df = race_file_df.copy()

    def _proc_prev_df(self, raceuma_5_prev_df):
        """  prev_dfを作成するための処理。prev1_raceuma_df,prev2_raceuma_dfに処理がされたデータをセットする。過去２走のデータと過去走を集計したデータをセットする  """
        raceuma_5_prev_df = self.tf.cluster_course_df(raceuma_5_prev_df, self.dict_path)
        raceuma_5_prev_df = self.tf.cluster_raceuma_result_df(raceuma_5_prev_df, self.dict_path)
        raceuma_5_prev_df = self.tf.factory_analyze_race_result_df(raceuma_5_prev_df, self.dict_path)
#        self.prev5_raceuma_df = self._get_prev_df(5, raceuma_5_prev_df, "")
#        self.prev5_raceuma_df.rename(columns=lambda x: x + "_5", inplace=True)
#        self.prev5_raceuma_df.rename(columns={"RACE_KEY_5": "RACE_KEY", "UMABAN_5": "UMABAN", "target_date_5": "target_date"}, inplace=True)
#        self.prev4_raceuma_df = self._get_prev_df(4, raceuma_5_prev_df, "")
#        self.prev4_raceuma_df.rename(columns=lambda x: x + "_4", inplace=True)
#        self.prev4_raceuma_df.rename(columns={"RACE_KEY_4": "RACE_KEY", "UMABAN_4": "UMABAN", "target_date_4": "target_date"}, inplace=True)
#        self.prev3_raceuma_df = self._get_prev_df(3, raceuma_5_prev_df, "")
#        self.prev3_raceuma_df.rename(columns=lambda x: x + "_3", inplace=True)
#        self.prev3_raceuma_df.rename(columns={"RACE_KEY_3": "RACE_KEY", "UMABAN_3": "UMABAN", "target_date_3": "target_date"}, inplace=True)
#        self.prev2_raceuma_df = self._get_prev_df(2, raceuma_5_prev_df, "")
#        self.prev2_raceuma_df.rename(columns=lambda x: x + "_2", inplace=True)
#        self.prev2_raceuma_df.rename(columns={"RACE_KEY_2": "RACE_KEY", "UMABAN_2": "UMABAN", "target_date_2": "target_date"}, inplace=True)
        self.prev1_raceuma_df = self._get_prev_df(1, raceuma_5_prev_df, "")
        self.prev1_raceuma_df.rename(columns=lambda x: x + "_1", inplace=True)
        self.prev1_raceuma_df.rename(columns={"RACE_KEY_1": "RACE_KEY", "UMABAN_1": "UMABAN", "target_date_1": "target_date"}, inplace=True)
        self.prev_feature_raceuma_df = self._get_prev_feature_df(raceuma_5_prev_df)

    def _get_prev_feature_df(self, raceuma_5_prev_df):
        max_columns = ['血統登録番号', 'target_date', 'テン指数結果', '上がり指数結果', 'ペース指数結果']
        min_columns = ['血統登録番号', 'target_date', 'テン指数結果順位', '上がり指数結果順位', 'ペース指数結果順位']
        max_score_df = raceuma_5_prev_df[max_columns].groupby(['血統登録番号', 'target_date']).max().add_prefix("max_").reset_index()
        min_score_df = raceuma_5_prev_df[min_columns].groupby(['血統登録番号', 'target_date']).min().add_prefix("min_").reset_index()
        feature_df = pd.merge(max_score_df, min_score_df, on=["血統登録番号", "target_date"])
        race_df = self.race_df[["RACE_KEY", "course_cluster"]].copy()
        raceuma_df = self.raceuma_df[["RACE_KEY", "UMABAN", "血統登録番号", "target_date"]].copy()
        raceuma_df = pd.merge(race_df, raceuma_df, on="RACE_KEY")
        filtered_df = pd.merge(raceuma_df, raceuma_5_prev_df.drop(["RACE_KEY", "UMABAN"], axis=1), on=["血統登録番号", "target_date", "course_cluster"])[["RACE_KEY", "UMABAN", "ru_cluster"]]
        filtered_df_c1 = filtered_df.query("ru_cluster == '1'").groupby(["RACE_KEY", "UMABAN"]).count().reset_index()
        filtered_df_c1.columns = ["RACE_KEY", "UMABAN", "c1_cnt"]
        filtered_df_c2 = filtered_df.query("ru_cluster == '2'").groupby(["RACE_KEY", "UMABAN"]).count().reset_index()
        filtered_df_c2.columns = ["RACE_KEY", "UMABAN", "c2_cnt"]
        filtered_df_c3 = filtered_df.query("ru_cluster == '3'").groupby(["RACE_KEY", "UMABAN"]).count().reset_index()
        filtered_df_c3.columns = ["RACE_KEY", "UMABAN", "c3_cnt"]
        filtered_df_c4 = filtered_df.query("ru_cluster == '4'").groupby(["RACE_KEY", "UMABAN"]).count().reset_index()
        filtered_df_c4.columns = ["RACE_KEY", "UMABAN", "c4_cnt"]
        filtered_df_c7 = filtered_df.query("ru_cluster == '7'").groupby(["RACE_KEY", "UMABAN"]).count().reset_index()
        filtered_df_c7.columns = ["RACE_KEY", "UMABAN", "c7_cnt"]
        raceuma_df = pd.merge(raceuma_df, filtered_df_c1, on=["RACE_KEY", "UMABAN"], how="left")
        raceuma_df = pd.merge(raceuma_df, filtered_df_c2, on=["RACE_KEY", "UMABAN"], how="left")
        raceuma_df = pd.merge(raceuma_df, filtered_df_c3, on=["RACE_KEY", "UMABAN"], how="left")
        raceuma_df = pd.merge(raceuma_df, filtered_df_c4, on=["RACE_KEY", "UMABAN"], how="left")
        raceuma_df = pd.merge(raceuma_df, filtered_df_c7, on=["RACE_KEY", "UMABAN"], how="left")
        raceuma_df = raceuma_df.fillna(0)
        raceuma_df = pd.merge(raceuma_df, feature_df, on=["血統登録番号", "target_date"], how="left").drop(["course_cluster", "血統登録番号", "target_date"], axis=1)
        return raceuma_df

    def set_pred_df(self):
        ############ 予想データ作成：レース ###############
        raptype_df = self.ext.get_pred_df("jra_rc_raptype", "RAP_TYPE")[["RACE_KEY", "CLASS", "predict_rank"]].rename(
            columns={"CLASS": "val"})
        raptype_df.loc[:, "val"] = raptype_df["val"].apply(lambda x: mu.decode_rap_type(x))
        raptype_df_1st = raptype_df.query("predict_rank == 1").groupby("RACE_KEY").first().reset_index().drop(
            "predict_rank", axis=1)
        raptype_df_1st = pd.merge(self.race_file_df, raptype_df_1st, on="RACE_KEY", how="left")

        tb_uchisoto_df = self.ext.get_pred_df("jra_rc_raptype", "TRACK_BIAS_UCHISOTO")[
            ["RACE_KEY", "CLASS", "predict_rank"]].rename(columns={"CLASS": "val"})
        tb_uchisoto_df.loc[:, "val"] = tb_uchisoto_df["val"].apply(lambda x: mu._decode_uchisoto_bias(x))
        tb_uchisoto_df_1st = tb_uchisoto_df.query("predict_rank == 1").groupby("RACE_KEY").first().reset_index().drop(
            "predict_rank", axis=1)

        tb_zengo_df = self.ext.get_pred_df("jra_rc_raptype", "TRACK_BIAS_ZENGO")[
            ["RACE_KEY", "CLASS", "predict_rank"]].rename(columns={"CLASS": "val"})
        tb_zengo_df.loc[:, "val"] = tb_zengo_df["val"].apply(lambda x: mu._decode_zengo_bias(x))
        tb_zengo_df_1st = tb_zengo_df.query("predict_rank == 1").groupby("RACE_KEY").first().reset_index().drop(
            "predict_rank", axis=1)

        tb_df = pd.merge(tb_uchisoto_df_1st.rename(columns={"val": "uc"}),
                         tb_zengo_df_1st.rename(columns={"val": "zg"}), on="RACE_KEY")
        tb_df = pd.merge(self.race_file_df, tb_df, on="RACE_KEY", how="left")
        tb_df.loc[:, "val"] = tb_df.apply(lambda x: mu.convert_bias(x["uc"], x["zg"]), axis=1)

        umaren_are_df = self.ext.get_pred_df("jra_rc_haito", "UMAREN_ARE")[["RACE_KEY", "pred"]].rename(
            columns={"pred": "umaren_are"})
        umatan_are_df = self.ext.get_pred_df("jra_rc_haito", "UMATAN_ARE")[["RACE_KEY", "pred"]].rename(
            columns={"pred": "umatan_are"})
        sanrenpuku_are_df = self.ext.get_pred_df("jra_rc_haito", "SANRENPUKU_ARE")[["RACE_KEY", "pred"]].rename(
            columns={"pred": "sanrenpuku_are"})
        are_df = pd.merge(umaren_are_df, umatan_are_df, on="RACE_KEY")
        are_df = pd.merge(are_df, sanrenpuku_are_df, on="RACE_KEY")
        are_df = pd.merge(self.race_file_df, are_df, on="RACE_KEY", how="left")
        are_df.loc[:, "val"] = are_df.apply(
            lambda x: mu.convert_are_flag(x["umaren_are"], x["umatan_are"], x["sanrenpuku_are"]), axis=1)

        ########## 予想データ作成：レース馬指数 ##################
        win_df = self.ext.get_pred_df("jra_ru_mark", "WIN_FLAG")
        win_df.loc[:, "RACEUMA_ID"] = win_df.apply(
            lambda x: mu.convert_jrdb_id(x["RACE_KEY"], x["target_date"]) + x["UMABAN"], axis=1)
        win_df.loc[:, "predict_std"] = round(win_df["predict_std"], 2)
        win_df.loc[:, "predict_rank"] = win_df["predict_rank"].astype(int)

        jiku_df = self.ext.get_pred_df("jra_ru_mark", "JIKU_FLAG")
        jiku_df.loc[:, "RACEUMA_ID"] = jiku_df.apply(
            lambda x: mu.convert_jrdb_id(x["RACE_KEY"], x["target_date"]) + x["UMABAN"], axis=1)
        jiku_df.loc[:, "predict_std"] = round(jiku_df["predict_std"], 2)
        jiku_df.loc[:, "predict_rank"] = jiku_df["predict_rank"].astype(int)

        ana_df = self.ext.get_pred_df("jra_ru_mark", "ANA_FLAG")
        ana_df.loc[:, "RACEUMA_ID"] = ana_df.apply(
            lambda x: mu.convert_jrdb_id(x["RACE_KEY"], x["target_date"]) + x["UMABAN"], axis=1)
        ana_df.loc[:, "predict_std"] = round(ana_df["predict_std"], 2)
        ana_df.loc[:, "predict_rank"] = ana_df["predict_rank"].astype(int)

        nigeuma_df = self.ext.get_pred_df("jra_ru_nigeuma", "NIGEUMA")
        nigeuma_df.loc[:, "RACEUMA_ID"] = nigeuma_df.apply(
            lambda x: mu.convert_jrdb_id(x["RACE_KEY"], x["target_date"]) + x["UMABAN"], axis=1)
        nigeuma_df.loc[:, "predict_std"] = round(nigeuma_df["predict_std"], 2)
        nigeuma_df.loc[:, "predict_rank"] = nigeuma_df["predict_rank"].astype(int)
        agari_df = self.ext.get_pred_df("jra_ru_nigeuma", "AGARI_SAISOKU")
        agari_df.loc[:, "RACEUMA_ID"] = agari_df.apply(
            lambda x: mu.convert_jrdb_id(x["RACE_KEY"], x["target_date"]) + x["UMABAN"], axis=1)
        agari_df.loc[:, "predict_std"] = round(agari_df["predict_std"], 2)
        agari_df.loc[:, "predict_rank"] = agari_df["predict_rank"].astype(int)
        ten_df = self.ext.get_pred_df("jra_ru_nigeuma", "TEN_SAISOKU")
        ten_df.loc[:, "RACEUMA_ID"] = ten_df.apply(
            lambda x: mu.convert_jrdb_id(x["RACE_KEY"], x["target_date"]) + x["UMABAN"], axis=1)
        ten_df.loc[:, "predict_std"] = round(ten_df["predict_std"], 2)
        ten_df.loc[:, "predict_rank"] = ten_df["predict_rank"].astype(int)

        score_df = pd.merge(win_df[["RACE_KEY", "UMABAN", "RACEUMA_ID", "predict_std", "target_date"]].rename(
            columns={"predict_std": "win_std"}),
                            jiku_df[["RACEUMA_ID", "predict_std"]].rename(columns={"predict_std": "jiku_std"}),
                            on="RACEUMA_ID")
        score_df = pd.merge(score_df, ana_df[["RACEUMA_ID", "predict_std"]].rename(columns={"predict_std": "ana_std"}),
                            on="RACEUMA_ID")
        score_df.loc[:, "predict_std"] = score_df["win_std"] * 0.20 + score_df["jiku_std"] * 0.30 + score_df[
            "ana_std"] * 0.50
        grouped_score_df = score_df.groupby("RACE_KEY")
        score_df.loc[:, "predict_rank"] = grouped_score_df["predict_std"].rank("dense", ascending=False)
        score_df.loc[:, "predict_std"] = round(score_df["predict_std"], 2)
        score_df.loc[:, "predict_rank"] = score_df["predict_rank"].astype(int)

        uma_mark_df = pd.merge(
            win_df[["RACE_KEY", "UMABAN", "predict_std", "target_date"]].rename(columns={"predict_std": "win_std"}),
            jiku_df[["RACE_KEY", "UMABAN", "predict_std"]].rename(columns={"predict_std": "jiku_std"}),
            on=["RACE_KEY", "UMABAN"])
        uma_mark_df = pd.merge(uma_mark_df,
                               ana_df[["RACE_KEY", "UMABAN", "predict_std"]].rename(columns={"predict_std": "ana_std"}),
                               on=["RACE_KEY", "UMABAN"])
        uma_mark_df = pd.merge(uma_mark_df, nigeuma_df[["RACE_KEY", "UMABAN", "predict_std"]].rename(
            columns={"predict_std": "nige_std"}), on=["RACE_KEY", "UMABAN"])
        uma_mark_df = pd.merge(uma_mark_df, agari_df[["RACE_KEY", "UMABAN", "predict_std"]].rename(
            columns={"predict_std": "agari_std"}), on=["RACE_KEY", "UMABAN"])
        uma_mark_df = pd.merge(uma_mark_df,
                               ten_df[["RACE_KEY", "UMABAN", "predict_std"]].rename(columns={"predict_std": "ten_std"}),
                               on=["RACE_KEY", "UMABAN"])
        race_course_df = self.tf.cluster_course_df(self.race_file_df, self.dict_path)[["RACE_KEY", "course_cluster", "条件"]].copy()
        uma_mark_df = pd.merge(uma_mark_df, race_course_df, on="RACE_KEY")
        uma_mark_df = pd.merge(uma_mark_df, tb_uchisoto_df_1st.rename(columns={"val": "tb_us"}), on="RACE_KEY")
        uma_mark_df = pd.merge(uma_mark_df, tb_zengo_df_1st.rename(columns={"val": "tb_zg"}), on="RACE_KEY")

        self.score_df = score_df.copy()
        self.are_df = are_df.copy()
        self.uma_mark_df = uma_mark_df.copy()
        self.nigeuma_df = nigeuma_df.copy()
        self.agari_df = agari_df.copy()
        self.win_df = win_df.copy()
        self.jiku_df = jiku_df.copy()
        self.ana_df = ana_df.copy()
        self.ten_df = ten_df.copy()
        self.tb_df = tb_df.copy()
        self.raptype_df_1st = raptype_df_1st.copy()

    def set_result_df(self):
        ######### 結果データ作成 ####################
        race_table_base_df = self.ext.get_race_table_base().drop(["右左", "種別", "記号", "重量", "グレード", "レース名", "頭数", "コース", "天候コード", "馬場状態", "target_date"], axis=1).copy()
        raceuma_table_base_df = self.ext.get_raceuma_table_base().drop(["距離", "芝ダ障害コード", "内外", "条件", "NENGAPPI"], axis=1).copy()
        result_df = pd.merge(race_table_base_df, raceuma_table_base_df, on="RACE_KEY")
        result_df.loc[:, "距離"] = result_df["距離"].astype(int)

        cluster_raceuma_result_df = self.tf.cluster_raceuma_result_df(result_df, self.dict_path)
        factory_analyze_race_result_df = self.tf.factory_analyze_race_result_df(result_df, self.dict_path)

        raceuma_result_df = cluster_raceuma_result_df[["RACE_KEY", "UMABAN", "ru_cluster", "ＩＤＭ結果", "レース馬コメント"]].copy()
        race_result_df = factory_analyze_race_result_df[
            ["RACE_KEY", "target_date", "fa_1", "fa_2", "fa_3", "fa_4", "fa_5", "RAP_TYPE", "TRACK_BIAS_ZENGO",
             "TRACK_BIAS_UCHISOTO", "レースペース流れ", "レースコメント"]].copy()

        race_result_df.loc[:, "val"] = race_result_df["RAP_TYPE"].apply(
            lambda x: mu.decode_rap_type(int(mu.encode_rap_type(x))))
        race_result_df.loc[:, "TB_ZENGO"] = race_result_df["TRACK_BIAS_ZENGO"].apply(
            lambda x: mu._decode_zengo_bias(int(mu._encode_zengo_bias(x))))
        race_result_df.loc[:, "TB_UCHISOTO"] = race_result_df["TRACK_BIAS_UCHISOTO"].apply(
            lambda x: mu._decode_uchisoto_bias(int(mu._calc_uchisoto_bias(x))))
        race_result_df.loc[:, "RACE_PACE"] = race_result_df["レースペース流れ"].apply(
            lambda x: mu._decode_race_pace(int(mu._encode_race_pace(x))))
        race_result_df.loc[:, "TB"] = race_result_df.apply(lambda x: mu.convert_bias(x["TB_UCHISOTO"], x["TB_ZENGO"]),
                                                           axis=1)
        #race_result_df = race_result_df.groupby("RACE_KEY").first().reset_index()

        race_file_df = self.race_file_df.copy()
        race_result_df = pd.merge(race_result_df, race_file_df, on="RACE_KEY")

        result_uchisoto_df = race_result_df[["RACE_KEY", "TB_UCHISOTO", "file_id", "nichiji", "race_no"]].rename(
            columns={"TB_UCHISOTO": "val"})
        result_zengo_df = race_result_df[["RACE_KEY", "TB_ZENGO", "file_id", "nichiji", "race_no"]].rename(
            columns={"TB_ZENGO": "val"})
        result_tb_df = race_result_df[["RACE_KEY", "TB", "file_id", "nichiji", "race_no"]].rename(columns={"TB": "val"})

        raceuma_result_df = pd.merge(raceuma_result_df, race_result_df, on="RACE_KEY")

        raceuma_result_df.loc[:, "RACEUMA_ID"] = raceuma_result_df.apply(
            lambda x: mu.convert_jrdb_id(x["RACE_KEY"], x["target_date"]) + x["UMABAN"], axis=1)
        fa_df = raceuma_result_df[["RACEUMA_ID", "fa_1", "fa_2", "fa_3", "fa_4", "fa_5", "target_date"]]

        self.result_tb_df = result_tb_df.copy()
        self.race_result_df = race_result_df.copy()
        self.fa_df = fa_df.copy()
        self.raceuma_result_df = raceuma_result_df.copy()

    def set_contents_based_filtering_df(self):
        self.set_race_df()
        self.set_raceuma_df()
        self.set_horse_df()
        self.set_prev_df()
        raceuma_df = self.raceuma_df.copy()
        raceuma_df = pd.merge(raceuma_df, self.race_file_df, on=["RACE_KEY", "NENGAPPI"])
        raceuma_df = pd.merge(raceuma_df, self.horse_df, on=["血統登録番号", "target_date"])
        raceuma_df = pd.merge(raceuma_df, self.prev1_raceuma_df, on=["RACE_KEY", "UMABAN", "target_date"], how='left')
#        raceuma_df = pd.merge(raceuma_df, self.prev2_raceuma_df, on=["RACE_KEY", "UMABAN", "target_date"], how='left')
#        raceuma_df = pd.merge(raceuma_df, self.prev3_raceuma_df, on=["RACE_KEY", "UMABAN", "target_date"], how='left')
#        raceuma_df = pd.merge(raceuma_df, self.prev4_raceuma_df, on=["RACE_KEY", "UMABAN", "target_date"], how='left')
#        raceuma_df = pd.merge(raceuma_df, self.prev5_raceuma_df, on=["RACE_KEY", "UMABAN", "target_date"], how='left')
        raceuma_df = pd.merge(raceuma_df, self.prev_feature_raceuma_df, on =["RACE_KEY", "UMABAN"], how='left')

        raceuma_df = pd.merge(raceuma_df, self.nigeuma_df[["RACE_KEY", "UMABAN", "predict_std"]].rename(
            columns={"predict_std": "nige_std"}), on=["RACE_KEY", "UMABAN"], how='left')
        raceuma_df = pd.merge(raceuma_df, self.agari_df[["RACE_KEY", "UMABAN", "predict_std"]].rename(
            columns={"predict_std": "agari_std"}), on=["RACE_KEY", "UMABAN"], how='left')
        raceuma_df = pd.merge(raceuma_df,
                              self.ten_df[["RACE_KEY", "UMABAN", "predict_std"]].rename(columns={"predict_std": "ten_std"}),
                               on=["RACE_KEY", "UMABAN"], how='left')
        raceuma_df = pd.merge(raceuma_df, self.score_df, on=["RACE_KEY", "UMABAN", "target_date"], how='left')
        raceuma_df.loc[:, "距離増減"] = raceuma_df.apply(lambda x: "距離短縮" if x["距離"] - x["距離_1"] <0 else ("距離延長" if x["距離"] - x["距離_1"] > 0 else "同距離"), axis=1)
        raceuma_df = self.tf.drop_columns_cbs_df(raceuma_df)
        self.raceuma_cbf_df = raceuma_df.copy()


class CreateFile(object):
    def __init__(self, start_date, end_date, test_flag):
        self.start_date = start_date
        self.end_date = end_date
        self.test_flag = test_flag
        self.dict_path = mc.return_base_path(test_flag)
        self.target_path = mc.TARGET_PATH
        self.ext_score_path = self.target_path + 'ORIGINAL_DATA/'

    def set_race_file_df(self, race_file_df):
        self.race_file_df = race_file_df.copy()

    def set_update_df(self, update_start_date, update_end_date):
        update_term_df = self.race_file_df.query(f"NENGAPPI >= '{update_start_date}' and NENGAPPI <= '{update_end_date}'")
        print(update_term_df.shape)
        self.file_list = update_term_df["file_id"].drop_duplicates()
        self.date_list = update_term_df["NENGAPPI"].drop_duplicates()
        self.rc_file_list = update_term_df["rc_file_id"].drop_duplicates()
        self.kc_file_list = update_term_df["kc_file_id"].drop_duplicates()

    def _return_mark(self, num):
        if num == 1: return "◎"
        if num == 2: return "○"
        if num == 3: return "▲"
        if num == 4: return "△"
        if num == 5:
            return "×"
        else:
            return "  "

    def _create_rm_file(self, df, pred_df, folder_path):
        """ valの値をレース印としてファイルを作成 """
        for file in self.file_list:
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

    def _create_rc_file(self, df, folder_path):
        """ レースコメントを作成 """
        for file in self.rc_file_list:
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

    def _create_um_mark_file(self, df, folder_path):
        """ ランクを印にして馬印ファイルを作成 """
        df.loc[:, "RACEUMA_ID"] = df.apply(
            lambda x: mu.convert_jrdb_id(x["RACE_KEY"], x["target_date"]) + x["UMABAN"], axis=1)
        df.loc[:, "predict_std"] = df["predict_std"].round(2)
        df.loc[:, "predict_rank"] = df["predict_rank"].astype(int)
        df = pd.merge(self.race_file_df[["RACE_KEY", "file_id", "nichiji", "race_no"]], df, on="RACE_KEY", how="left")
        for file in self.file_list:
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
                        line_text += self._return_mark(val["predict_rank"])
                        i += 1
                    if i != 18:
                        for j in range(i, 18):
                            line_text += "  "
                    file_text += line_text + '\r\n'
            with open(folder_path + "UM" + file + ".DAT", mode='w', encoding="shift-jis") as f:
                f.write(file_text.replace('\r', ''))

    def _create_main_mark_file(self, race_df, raceuma_df, folder_path):
        """ 馬１、レース１用のファイルを作成 """
        for file in self.file_list:
            print(file)
            file_text = ""
            temp_df = self.race_file_df.query(f"file_id == '{file}'")
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
                        line_text += self._return_mark(val["predict_rank"])
                        i += 1
                    if i != 18:
                        for j in range(i, 18):
                            line_text += "  "
                    file_text += line_text + '\r\n'
            with open(folder_path + "UM" + file + ".DAT", mode='w', encoding="shift-jis") as f:
                f.write(file_text.replace('\r', ''))

    def _create_win_flag(self, course_cluster, joken, win, ana, nige, ten, agari):
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
                if joken == '05' and win >= 60 and agari >= 60: return "▲"
                if joken == "16" and (
                        agari >= 50 or (nige >= 50 and ten >= 60) or (nige >= 60 and ten >= 50)): return "☆"
                if joken == "A1":
                    if nige <= 60 and ten <= 60 and agari <= 60: return "消"
                    if ten >= 60 or (nige >= 50 and agari >= 50): return "☆"
                    if win >= 60 or (agari >= 60 and (ten >= 50 or nige >= 50)): return "▲"
                if joken == "A3":
                    if agari >= 70: return "☆"
                    if win >= 60 and (nige >= 50 or ten >= 50 or agari >= 60): return "▲"
            if course_cluster == 6:
                if joken == '16' and (
                        ten >= 60 or nige >= 60 or (nige >= 70 and ten >= 50) or (nige >= 50 and ten >= 70)): return "☆"
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
                    if nige >= 60 or ten >= 70: return "☆"
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
                if joken == 'A3' and (win >= 60 or agari >= 70): return "▲"
        return "  "

    def _create_jiku_flag(self, course_cluster, joken, jiku, ana, nige, ten, agari):
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
                if joken == 'A3' and (
                        jiku >= 60 or agari >= 70 or ((nige >= 50 or ten >= 50) and agari >= 60)): return "▲"
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
                    if jiku >= 60 or agari >= 60 or (agari >= 50 and (nige >= 60 or ten >= 60)): return "▲"
                if joken == 'A3':
                    if (jiku >= 60 or nige >= 50) and agari >= 60: return "○"
                    if jiku >= 60 and agari >= 60 and (ten >= 50 or nige >= 50): return "○"
                    if jiku >= 60 or (nige >= 60 and agari >= 50): return "▲"
            if course_cluster == 6:
                if joken == '05' and jiku >= 60 and (
                        agari >= 70 or (agari >= 50 and (nige >= 50 or ten >= 50))): return "▲"
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
                    if jiku >= 60 or (nige >= 50 and ten >= 50 and agari >= 50): return "○"
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
                    if ((nige >= 60 or ten >= 60) and agari >= 50) or (
                            (nige >= 50 or ten >= 50) and agari >= 60): return "○"
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
                    if jiku >= 50 and (nige >= 50 or ten >= 50 or agari >= 60): return "○"
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
                    if jiku >= 60 or agari >= 70 or (nige >= 50 and agari >= 60): return "▲"
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
                    if jiku >= 60 or agari >= 60 or (ten >= 50 and agari > 50): return "▲"
                if joken == 'OP' and jiku >= 60 and agari >= 50: return "▲"
        return "  "

    def _create_tb_zg_flag(self, course_cluster, tb_zg, jiku, ana, nige, ten, agari):
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

    def _create_tb_us_flag(self, course_cluster, tb_us, jiku, ana, nige, ten, agari):
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
                    if (nige >= 70 and ten >= 70) or (jiku >= 60 and nige >= 60 and ten >= 60) or (
                            ana >= 60 and nige >= 70 and ten >= 60): return "☆"
                    if jiku >= 60 and (ten >= 60 or nige >= 60 or agari >= 60): return "○"
                if tb_us == '03　外':
                    if jiku >= 60 or agari >= 60: return "○"
        return "  "

    def _create_ana_flag(self, course_cluster, joken, ana_std, nige_std, ten_std, agari_std):
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
            if course_cluster == 11 and joken in ('05', 'A3'): return " 2"
            if course_cluster == 12 and joken == '05': return " 2"
            if course_cluster == 13 and joken == '05': return " 2"
            if course_cluster == 14 and joken == '05': return " 2"
            if course_cluster == 14 and joken == 'A3' and (nige_std >= 50 or ten_std >= 50): return " 2"
        return "  "

    def _create_um_mark_file_for_pickup(self, df, folder_path, target):
        """ ランクを印にして馬印ファイルを作成。targetは勝、軸、穴 """
        df.loc[:, "RACEUMA_ID"] = df.apply(
            lambda x: mu.convert_jrdb_id(x["RACE_KEY"], x["target_date"]) + x["UMABAN"], axis=1)
        df = pd.merge(self.race_file_df[["RACE_KEY", "file_id", "nichiji", "race_no"]], df, on="RACE_KEY", how="left")
        for file in self.file_list:
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
                            line_text += self._create_win_flag(val["course_cluster"], val["条件"], val["win_std"],
                                                               val["ana_std"], val["nige_std"], val["ten_std"],
                                                               val["agari_std"])
                        elif target == "軸":
                            line_text += self._create_jiku_flag(val["course_cluster"], val["条件"], val["jiku_std"],
                                                                val["ana_std"], val["nige_std"], val["ten_std"],
                                                                val["agari_std"])
                        elif target == "内外":
                            line_text += self._create_tb_us_flag(val["course_cluster"], val["tb_us"], val["jiku_std"],
                                                                 val["ana_std"], val["nige_std"], val["ten_std"],
                                                                 val["agari_std"])
                        elif target == "前後":
                            line_text += self._create_tb_zg_flag(val["course_cluster"], val["tb_zg"], val["jiku_std"],
                                                                 val["ana_std"], val["nige_std"], val["ten_std"],
                                                                 val["agari_std"])
                        i += 1
                    if i != 18:
                        for j in range(i, 18):
                            line_text += "  "
                    file_text += line_text + '\r\n'
            with open(folder_path + "UM" + file + ".DAT", mode='w', encoding="shift-jis") as f:
                f.write(file_text.replace('\r', ''))


    def export_pred_raceuma_mark(self, uma_mark_df, nigeuma_df, agari_df):
        print("---- 勝マーク --------")
        win_mark_path = self.target_path + "UmaMark2/"
        # create_um_mark_file(win_df, win_mark_path)
        self._create_um_mark_file_for_pickup(uma_mark_df, win_mark_path, "勝")
        print("---- 軸マーク --------")
        jiku_mark_path = self.target_path + "UmaMark3/"
        self._create_um_mark_file_for_pickup(uma_mark_df, jiku_mark_path, "軸")
        # create_um_mark_file(jiku_df, jiku_mark_path)
        print("---- バイアス（内外）マーク --------")
        tb_us_mark_path = self.target_path + "UmaMark4/"
        self._create_um_mark_file_for_pickup(uma_mark_df, tb_us_mark_path, "内外")
        # create_um_mark_file(ana_df, ana_mark_path)
        print("---- バイアス（前後）マーク --------")
        tb_zg_mark_path = self.target_path + "UmaMark5/"
        self._create_um_mark_file_for_pickup(uma_mark_df, tb_zg_mark_path, "前後")
        # create_um_mark_file(ana_df, ana_mark_path)
        print("---- nigeuma_df --------")
        nigeuma_mark_path = self.target_path + "UmaMark6/"
        self._create_um_mark_file(nigeuma_df, nigeuma_mark_path)
        print("---- agari_df --------")
        agari_mark_path = self.target_path + "UmaMark7/"
        self._create_um_mark_file(agari_df, agari_mark_path)
        # print("---- ten_df --------")
        # ten_mark_path = target_path + "UmaMark7/"
        # create_um_mark_file(ten_df, ten_mark_path)

    def export_pred_main(self, score_df, race_file_df, are_df):
        print("---- score_df --------")
        score_mark_path = self.target_path
        main_raceuma_df = pd.merge(score_df, race_file_df, on="RACE_KEY")
        self._create_main_mark_file(are_df, main_raceuma_df, score_mark_path)

    def export_pred_score(self, win_df, jiku_df, ana_df, score_df, nigeuma_df, agari_df, ten_df):
        print("---- 予想外部指数作成 --------")
        for date in self.date_list:
            print(date)
            win_temp_df = win_df.query(f"target_date == '{date}'")[
                ["RACEUMA_ID", "predict_std", "predict_rank"]].sort_values("RACEUMA_ID")
            win_temp_df.loc[:, "predict_std"] = win_temp_df["predict_std"].round(0).astype("int")
            win_temp_df.to_csv(self.ext_score_path + "pred_win/" + date + ".csv", header=False, index=False)
            jiku_temp_df = jiku_df.query(f"target_date == '{date}'")[
                ["RACEUMA_ID", "predict_std", "predict_rank"]].sort_values("RACEUMA_ID")
            jiku_temp_df.loc[:, "predict_std"] = jiku_temp_df["predict_std"].round(0).astype("int")
            jiku_temp_df.to_csv(self.ext_score_path + "pred_jiku/" + date + ".csv", header=False, index=False)
            ana_temp_df = ana_df.query(f"target_date == '{date}'")[
                ["RACEUMA_ID", "predict_std", "predict_rank"]].sort_values("RACEUMA_ID")
            ana_temp_df.loc[:, "predict_std"] = ana_temp_df["predict_std"].round(0).astype("int")
            ana_temp_df.to_csv(self.ext_score_path + "pred_ana/" + date + ".csv", header=False, index=False)
            score_temp_df = score_df.query(f"target_date == '{date}'")[
                ["RACEUMA_ID", "predict_std", "predict_rank"]].sort_values("RACEUMA_ID")
            score_temp_df.loc[:, "predict_std"] = score_temp_df["predict_std"].round(0).astype("int")
            score_temp_df.to_csv(self.ext_score_path + "pred_score/" + date + ".csv", header=False, index=False)
            nigeuma_temp_df = nigeuma_df.query(f"target_date == '{date}'")[
                ["RACEUMA_ID", "predict_std", "predict_rank"]].sort_values("RACEUMA_ID")
            nigeuma_temp_df.loc[:, "predict_std"] = nigeuma_temp_df["predict_std"].round(0).astype("int")
            nigeuma_temp_df.to_csv(self.ext_score_path + "pred_nige/" + date + ".csv", header=False, index=False)
            agari_temp_df = agari_df.query(f"target_date == '{date}'")[
                ["RACEUMA_ID", "predict_std", "predict_rank"]].sort_values("RACEUMA_ID")
            agari_temp_df.loc[:, "predict_std"] = agari_temp_df["predict_std"].round(0).astype("int")
            agari_temp_df.to_csv(self.ext_score_path + "pred_agari/" + date + ".csv", header=False, index=False)
            ten_temp_df = ten_df.query(f"target_date == '{date}'")[
                ["RACEUMA_ID", "predict_std", "predict_rank"]].sort_values("RACEUMA_ID")
            ten_temp_df.loc[:, "predict_std"] = ten_temp_df["predict_std"].round(0).astype("int")
            ten_temp_df.to_csv(self.ext_score_path + "pred_ten/" + date + ".csv", header=False, index=False)

    def export_race_mark(self, result_tb_df, tb_df, race_result_df, raptype_df_1st):
        print("---- tb_df --------")
        tb_mark_folder = self.target_path + "RMark2/"
        self._create_rm_file(result_tb_df, tb_df, tb_mark_folder)
        print("---- result_rap_type --------")
        raptype_mark_folder = self.target_path + "RMark3/"
        self._create_rm_file(race_result_df, raptype_df_1st, raptype_mark_folder)

    def export_result_score(self, fa_df):
        print("---- 結果外部指数作成 --------")
        for date in self.date_list:
            print(date)
            temp_df = fa_df.query(f"target_date == '{date}'")
            fa1_df = temp_df[["RACEUMA_ID", "fa_1"]].copy()
            fa1_df.loc[:, "fa_1"] = round(fa1_df["fa_1"] * 10, 2)
            fa1_df.to_csv(self.ext_score_path + "fa_1/" + date + ".csv", header=False, index=False)
            fa2_df = temp_df[["RACEUMA_ID", "fa_2"]].copy()
            fa2_df.loc[:, "fa_2"] = round(fa2_df["fa_2"] * 10, 2)
            fa2_df.to_csv(self.ext_score_path + "fa_2/" + date + ".csv", header=False, index=False)
            fa3_df = temp_df[["RACEUMA_ID", "fa_3"]].copy()
            fa3_df.loc[:, "fa_3"] = round(fa3_df["fa_3"] * 10, 2)
            fa3_df.to_csv(self.ext_score_path + "fa_3/" + date + ".csv", header=False, index=False)
            fa4_df = temp_df[["RACEUMA_ID", "fa_4"]].copy()
            fa4_df.loc[:, "fa_4"] = round(fa4_df["fa_4"] * 10, 2)
            fa4_df.to_csv(self.ext_score_path + "fa_4/" + date + ".csv", header=False, index=False)
            fa5_df = temp_df[["RACEUMA_ID", "fa_5"]].copy()
            fa5_df.loc[:, "fa_5"] = round(fa5_df["fa_5"] * 10, 2)
            fa5_df.to_csv(self.ext_score_path + "fa_5/" + date + ".csv", header=False, index=False)

    def export_result_raceuma_mark(self, raceuma_result_df):
        print("---- result ru_cluster --------")
        ru_cluster_path = self.target_path + "UmaMark8/"
        for file in self.file_list:
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

    def export_comment(self, race_result_df, raceuma_result_df):
        print("---- コメントファイル作成 --------")
        ######### コメントファイル作成：レース ####################
        for file in self.rc_file_list:
            print(file)
            race_comment_df = race_result_df.query(f"rc_file_id == '{file}'")[["RACE_KEY", "レースコメント"]].sort_values(
                "RACE_KEY")
            race_comment_df.to_csv(self.target_path + "RACE_COM/20" + file[4:6] + "/" + file + ".dat", header=False,
                                   index=False, encoding="cp932")

        ######### コメントファイル作成：レース馬 ####################
        for file in self.kc_file_list:
            print(file)
            race_comment_df = raceuma_result_df.query(f"kc_file_id == '{file}'")[["RACE_KEY", "UMABAN", "レース馬コメント"]]
            race_comment_df.loc[:, "RACE_UMA_KEY"] = race_comment_df["RACE_KEY"] + race_comment_df["UMABAN"]
            race_comment_df = race_comment_df[["RACE_UMA_KEY", "レース馬コメント"]].sort_values("RACE_UMA_KEY")
            race_comment_df.to_csv(self.target_path + "KEK_COM/20" + file[4:6] + "/" + file + ".dat", header=False,
                                   index=False, encoding="cp932")

    def export_sim_score(self, raceuma_cbf_df):
        print("---- export_sim_score --------")
        ru_cluster_path = self.target_path + "UmaMark8/"
        for file in self.file_list:
            print(file)
            file_text = ""
            temp_df = raceuma_cbf_df.query(f"file_id == '{file}'")
            nichiji_list = sorted(temp_df["nichiji"].drop_duplicates())
            for nichiji in nichiji_list:
                temp2_df = temp_df.query(f"nichiji == '{nichiji}'")
                race_list = sorted(temp2_df["RACE_KEY"].drop_duplicates())
                for race in race_list:
                    line_text = "      "
                    temp3_df = temp2_df.query(f"RACE_KEY == '{race}'").sort_values("UMABAN").reset_index(drop=True)
                    sim_df = self.create_sim_score(temp3_df)
                    i = 0
                    for idx, val in sim_df.iterrows():
                        line_text += val["sim_score"]
                        i += 1
                    if i != 18:
                        for j in range(i, 18):
                            line_text += "  "
                    file_text += line_text + '\r\n'
            with open(ru_cluster_path + "UM" + file + ".DAT", mode='w', encoding="shift-jis") as f:
                f.write(file_text.replace('\r', ''))

    def create_sim_score(self, df):
        jiku_df = df.query(f"predict_rank == 1")#.index.values[0]
        if jiku_df.empty:
            sim_df = df[["RACE_KEY", "UMABAN"]].copy()
            sim_df.loc[:, "sim_score"] = '  '
        else:
            jiku_index = jiku_df.index.values[0]
            numerical_feats = df.dtypes[df.dtypes != "object"].index.tolist()
            num_columns_list = numerical_feats + ["RACE_KEY", "UMABAN"]
            categorical_feats = df.dtypes[df.dtypes == "object"].index.tolist()
            num_target_df = df[num_columns_list].fillna(0)
            text_target_df = df[categorical_feats].fillna("")
            text_sim_df = self.create_text_sim_score(text_target_df, jiku_index, categorical_feats)
            text_sim_ketto = ["RACE_KEY", "UMABAN", "NENGAPPI", "RACEUMA_ID", "target_date", "脚質", "蹄コード", "父馬名", "母父馬名", "父系統コード", "母父系統コード", "距離増減"]
            text_sim_ketto_df = self.create_text_sim_score(text_target_df, jiku_index, text_sim_ketto).rename(columns={"sim_score": "ketto_score"})
            text_sim_chokyo = ["RACE_KEY", "UMABAN", "NENGAPPI", "RACEUMA_ID", "target_date", "調教師所属", "放牧先", "調教コースコード", "追切種類", "調教タイプ", "調教コース種別", "調教距離", "調教重点", "調教量評価", "距離増減"]
            text_sim_chokyo_df = self.create_text_sim_score(text_target_df, jiku_index, text_sim_chokyo).rename(columns={"sim_score": "chokyo_score"})
            first_sim_df = self.create_cos_sim_score(num_target_df, jiku_index, True)
            sim_df = pd.merge(first_sim_df, text_sim_df, on =["RACE_KEY", "UMABAN"])
            sim_df = pd.merge(sim_df, text_sim_ketto_df, on =["RACE_KEY", "UMABAN"])
            sim_df = pd.merge(sim_df, text_sim_chokyo_df, on =["RACE_KEY", "UMABAN"])
            sim_df = self.create_cos_sim_score(sim_df, jiku_index, False)
            sim_df.loc[:, "sim_score"] = sim_df["sim_score"].apply(lambda x: self._return_mark(1 + int((10 - x * 10) * 3)) if not np.isnan(x) else '  ')
        return sim_df

    def create_cos_sim_score(self, df, jiku_index,second_flag):
        mm = preprocessing.MinMaxScaler()
        all_np = mm.fit_transform(df.drop(["RACE_KEY", "UMABAN"], axis=1))
        jiku_np = all_np[jiku_index]
        cos_list = []
        for i in range(all_np.shape[0]):
            if (np.linalg.norm(jiku_np) * np.linalg.norm(all_np[i])) == 0:
                cos_sim = 0
            else:
                cos_sim = (np.dot(jiku_np, all_np[i]) / (np.linalg.norm(jiku_np) * np.linalg.norm(all_np[i]))).round(2)
            cos_list.append(cos_sim)
        if second_flag:
            sec_df = df[["枠番", "テン指数", "ペース指数", "上がり指数", "位置指数", "nige_std", "agari_std", "ten_std", "win_std", "jiku_std", "ana_std"]].copy()
            sec_df.loc[:, "類似値"] = cos_list
            all_np = mm.fit_transform(sec_df)
            jiku_np = all_np[jiku_index]
            cos_list = []
            for i in range(all_np.shape[0]):
                if (np.linalg.norm(jiku_np) * np.linalg.norm(all_np[i])) == 0:
                    cos_sim = 0
                else:
                    cos_sim = (np.dot(jiku_np, all_np[i]) / (np.linalg.norm(jiku_np) * np.linalg.norm(all_np[i]))).round(2)
                cos_list.append(cos_sim)
        sim_df = pd.DataFrame({"RACE_KEY": df["RACE_KEY"], "UMABAN": df["UMABAN"], "sim_score": cos_list})
        return sim_df

    def create_text_sim_score(self, df, jiku_index, target_columns):
        df = df[target_columns].copy()
        df.loc[:, "テキスト"] = ""
        for column_name, item in df.iteritems():
            if column_name not in ["RACE_KEY", "UMABAN", "NENGAPPI", "RACEUMA_ID", "target_date"]:
                df.loc[:, "テキスト"] = df["テキスト"].str.cat(item, sep=',')

        df["テキスト"] = df["テキスト"].apply(lambda x: x.split(','))
        docs = df["テキスト"].values.tolist()
        datalist = []
        for j in range(len(docs)):
            data = []
            doc = docs[j]
            tf = self._func_tf(doc)
            idf = self._func_idf(doc, docs)
            tfidf = tf * idf
            for i, word in enumerate(set(doc)):
                data.append([word, tfidf[i]])
            data = sorted(data, key=lambda x: x[1], reverse=True)
            datalist.append(data)

        jiku_text = datalist[jiku_index]
        sim_list = []
        for i in range(len(datalist)):
            tar_text = datalist[i]
            sim = self._similarity(jiku_text, tar_text)
            sim_list.append(sim)
        sim_df = pd.DataFrame({"RACE_KEY": df["RACE_KEY"], "UMABAN": df["UMABAN"], "sim_score": sim_list})
        return sim_df

    def _func_tf(self, docs):
        words = set(docs)
        tf_values = np.array([docs.count(doc) for doc in words])
        tf_list = (tf_values / sum(tf_values)).tolist()
        return tf_list

    def _func_idf(self, words, docs):
        words = set(words)
        idf_list = np.array([math.log10((len(docs) + 1) / (sum([word in doc for doc in docs]) + 1)) for word in words])
        return idf_list

    def _similarity(self, tfidf1, tfidf2):
        """
        Get Cosine Similarity
        cosθ = A・B/|A||B|
        :param tfidf1: list[list[str, float]]
        :param tfidf2: list[list[str, float]]
        :rtype : float
        """
        tfidf2_dict = {key: value for key, value in tfidf2}

        ab = 0  # A・B
        for key, value in tfidf1:
            value2 = tfidf2_dict.get(key)
            if value2:
                ab += float(value * value2)

        # |A| and |B|
        a = math.sqrt(sum([v ** 2 for k, v in tfidf1]))
        b = math.sqrt(sum([v ** 2 for k, v in tfidf2]))

        return float(ab / (a * b))


if __name__ == "__main__":
    args = sys.argv
    print(args)
    print("mode：" + args[1])  # test or init or prod
    mock_flag = False
    test_flag = False
    mode = args[1]
    dict_path = mc.return_base_path(test_flag)
    version_str = "dummy" #dict_folderを取得するのに使用
    pd.set_option('display.max_columns', 3000)
    pd.set_option('display.max_rows', 3000)
    if mode == "test":
        print("Test mode")
        start_date = '2020/04/01'
        end_date = '2020/05/31'
        update_start_date = '20200501'
        update_end_date = '20200531'
    elif mode == "init":
        start_date = '2019/01/01'
        end_date = (dt.now() + timedelta(days=1)).strftime('%Y/%m/%d')
        update_start_date = '20190101'
        update_end_date = (dt.now() + timedelta(days=1)).strftime('%Y%m%d')
    elif mode == "prod":
        start_date = (dt.now() + timedelta(days=-90)).strftime('%Y/%m/%d')
        end_date = (dt.now() + timedelta(days=1)).strftime('%Y/%m/%d')
        update_start_date = (dt.now() + timedelta(days=-9)).strftime('%Y%m%d')
        update_end_date = (dt.now() + timedelta(days=1)).strftime('%Y%m%d')
    print("MODE:" + str(args[1]) + "  update_start_date:" + update_start_date + " update_end_date:" + update_end_date)

    ld = Ld(version_str, start_date, end_date, mock_flag, test_flag)
    ld.set_race_file_df()
    ld.set_pred_df()
    ld.set_result_df()
    ld.set_contents_based_filtering_df()

    cf = CreateFile(start_date, end_date, test_flag)
    cf.set_race_file_df(ld.race_file_df)
    cf.set_update_df(update_start_date, update_end_date)

    cf.export_pred_main(ld.score_df, ld.race_file_df, ld.are_df)
    cf.export_pred_raceuma_mark(ld.uma_mark_df, ld.nigeuma_df, ld.agari_df)
    cf.export_sim_score(ld.raceuma_cbf_df)
    cf.export_pred_score(ld.win_df, ld.jiku_df, ld.ana_df, ld.score_df, ld.nigeuma_df, ld.agari_df, ld.ten_df)
    cf.export_race_mark(ld.result_tb_df, ld.tb_df, ld.race_result_df, ld.raptype_df_1st)
    cf.export_result_score(ld.fa_df)
    cf.export_comment(ld.race_result_df, ld.raceuma_result_df)


