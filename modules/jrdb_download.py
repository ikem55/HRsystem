import datetime as dt
import os
import urllib.request
import pandas as pd
from bs4 import BeautifulSoup

import modules.util as mu


class JrdbDownload(object):
    """ JRDBデータをダウンロードするのに利用

      :param str base_uri: JRDBのURL
      :type str base_uri: str
      :param str jrdb_id: JRDBのID
      :type str jrdb_id: str

      Example::

          from dog import Dog
          dog = Dog()
    """

    def __init__(self):
        self.base_uri = 'http://www.jrdb.com/'
        self.jrdb_id = os.environ["jrdb_id"]
        self.jrdb_pw = os.environ["jrdb_pw"]
        self.download_path = os.environ["PROGRAM_PATH"] + \
            "./import_JRDB/data_org/"
        self.archive_path = os.environ["PROGRAM_PATH"] + \
            "./import_JRDB/data_archive/"
        self.target_folder = 'member/datazip/Paci/2018/'
        self.filename = 'PACI181021.zip'
        self.url = self.base_uri + self.target_folder + self.filename
        mu.setup_basic_auth(self.base_uri)

    def procedure_download_sokuho(self):
        """ 速報データのダウンロードをまとめた手順 """
        print("============== DOWNLOAD JRDB SOKUHO ====================")
        typelist = ["TYB"]
        target_date = dt.date.today().strftime('%Y%m%d')
        print(target_date)
        for type in typelist:
            print("----------------" + type + "---------------")
            filename = type + target_date[2:8] + ".zip"
            url = target_date[0:4] + '/' + filename
            print(url)
            self.download_jrdb_file(type.title(), url, filename)

    def procedure_download(self):
        """ 通常データのダウンロードをまとめた手順  """
        print("============== DOWNLOAD JRDB ====================")
        typelist = ["PACI", "SED", "SKB", "HJC", "TYB"]
        for type in typelist:
            print("----------------" + type + "---------------")
            filelist = mu.get_downloaded_list(type, self.archive_path)
            target_df = self.get_jrdb_page(type)
            for index, row in target_df.iterrows():
                if row['filename'] not in filelist:
                    self.download_jrdb_file(
                        type.title(), row['url'], row['filename'])

    def download_jrdb_file(self, type, url, filename):
        """ 指定したJRDBファイルのダウンロードと解凍を実施

        :param str type: JRDBファイルの種類
        :type str type: str
        :param str url: 該当ファイルのURLのフルパス
        :type str url: str
        :param str filename: ファイル名
        :type str filename: str
        """
        target_file_url = self.base_uri + 'member/datazip/' + type + '/' + url
        print('Downloading ... {0} as {1}'.format(target_file_url, filename))
        urllib.request.urlretrieve(
            target_file_url, self.download_path + filename)
        mu.unzip_file(filename, self.download_path, self.archive_path)

    def get_jrdb_page(self, type):
        """ 指定したファイルタイプに対してページにアクセスして対象ファイルのリストを取得する

        :param str type: ファイルの種類
        :return: データフレーム
        """
        if type == "PACI":
            html = urllib.request.urlopen(
                'http://www.jrdb.com/member/datazip/Paci/index.html').read()
        elif type == "SED":
            html = urllib.request.urlopen(
                'http://www.jrdb.com/member/datazip/Sed/index.html').read()
        elif type == "SKB":
            html = urllib.request.urlopen(
                'http://www.jrdb.com/member/datazip/Skb/index.html').read()
        elif type == "HJC":
            html = urllib.request.urlopen(
                'http://www.jrdb.com/member/datazip/Hjc/index.html').read()
        elif type == "TYB":
            html = urllib.request.urlopen(
                'http://www.jrdb.com/member/datazip/Tyb/index.html').read()

        else:
            html = ""
        soup = BeautifulSoup(html, 'lxml')
        tables = soup.findAll('table')
        target_df = pd.DataFrame(index=[], columns=['filename', 'url'])
        for table in tables:
            for li in table.findAll('li'):
                if type == "PACI":
                    if li.text[0:6] == "PACI16" or li.text[0:6] == "PACI17" or li.text[0:6] == "PACI18" or li.text[0:6] == "PACI19" or li.text[0:5] == "PACI2":
                        sr = pd.Series(
                            [li.text.strip('\n'), li.a.attrs['href']], index=target_df.columns)
                        target_df = target_df.append(sr, ignore_index=True)
                elif type == "SED":
                    if li.text[0:5] == "SED16" or li.text[0:5] == "SED17" or li.text[0:5] == "SED18" or li.text[0:5] == "SED19" or li.text[0:4] == "SED2":
                        sr = pd.Series(
                            [li.text.strip('\n'), li.a.attrs['href']], index=target_df.columns)
                        target_df = target_df.append(sr, ignore_index=True)
                elif type == "SKB":
                    if li.text[0:5] == "SKB16" or li.text[0:5] == "SKB17" or li.text[0:5] == "SKB18" or li.text[0:5] == "SKB19" or li.text[0:4] == "SKB2":
                        sr = pd.Series(
                            [li.text.strip('\n'), li.a.attrs['href']], index=target_df.columns)
                        target_df = target_df.append(sr, ignore_index=True)
                elif type == "HJC":
                    if li.text[0:5] == "HJC16" or li.text[0:5] == "HJC17" or li.text[0:5] == "HJC18" or li.text[0:5] == "HJC19" or li.text[0:4] == "HJC2":
                        sr = pd.Series(
                            [li.text.strip('\n'), li.a.attrs['href']], index=target_df.columns)
                        target_df = target_df.append(sr, ignore_index=True)
                elif type == "TYB":
                    if li.text[0:5] == "TYB16" or li.text[0:5] == "TYB17" or li.text[0:5] == "TYB18" or li.text[0:5] == "TYB19" or li.text[0:4] == "TYB2":
                        sr = pd.Series(
                            [li.text.strip('\n'), li.a.attrs['href']], index=target_df.columns)
                        target_df = target_df.append(sr, ignore_index=True)
                else:
                    print("nothing")
        return target_df
