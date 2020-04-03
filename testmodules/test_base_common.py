import unittest
import os
import shutil

class TestBaseCommon(unittest.TestCase):

    def _proc_check_folder(self):
        """ テストの実施結果を格納するフォルダがあることを確認する。なければ作成する """
        model_folder = self.skmodel.model_folder
        dict_folder = self.skmodel.dict_folder
        intermediate_folder = self.intermediate_folder
        ens_folder = self.skmodel.ens_folder_path
        folder_list = [model_folder, dict_folder, intermediate_folder, ens_folder]
        for folder in folder_list:
            if not os.path.exists(folder):
                os.makedirs(folder)

    def proc_test_contain_columns_check(self, df, contain_columns_set):
        """ データフレームに列が含まれているかのチェック

        :param df:
        :param contain_columns_set:
        :return: bool
        """
        all_columns_set = set(df.columns)
        contain_check = contain_columns_set.issubset(all_columns_set)
        return contain_check

    def proc_test_not_contain_columns_check(self, df, not_contain_columns_set):
        """ データフレームに列が含まれていない（互いに素）かのチェック

        :param df:
        :param not_contain_columns_set:
        :return: bool
        """
        all_columns_set = set(df.columns)
        contain_check = not_contain_columns_set.isdisjoint(all_columns_set)
        return contain_check

    def create_folder(self):
        for folder in ['first/train/', 'first/test/', 'second/train/', 'second/test/', 'third/train/', 'third/test/', 'third/param/']:
            int_folder = self.intermediate_folder + folder
            if not os.path.exists(int_folder):
                os.makedirs(int_folder)
            model_folder = self.skmodel.model_folder + folder
            if not os.path.exists(model_folder):
                os.makedirs(model_folder)
