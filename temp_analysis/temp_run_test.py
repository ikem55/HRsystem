from testmodules.test_lb_task_learning import TestLBv1TaskLearning, TestLBv2TaskLearning, TestLBv3TaskLearning, TestLBv4TaskLearning, TestLBRv1SkModelLearning
from testmodules.test_lb_task_predict import TestLBv1TaskPredict, TestLBv2TaskPredict, TestLBv3TaskPredict, TestLBv4TaskPredict, TestLBRv1TaskPredict
import unittest


class TempTest1(TestLBv4TaskLearning):
  clean_flag = True
  cls_val = "主催者コード"
  val = "2"
  target = "１着"
  start_date = '2017/01/01'
  end_date = '2018/12/31'

def suite():
  suite = unittest.TestSuite()
  suite.addTest(unittest.makeSuite(TempTest1))
#  suite.addTest(TempTest1('test_00_preprocess'))
#  suite.addTest(TempTest1('test_01_create_learning_data'))
#  suite.addTest(TempTest1('test_02_check_dimension'))
#  suite.addTest(TempTest1('test_11_create_feature_select_data'))
#  suite.addTest(TempTest1('test_20_check_learning_df'))
#  suite.addTest(TempTest1('test_21_proc_learning_sk_model'))
#  suite.addTest(TempTest1('test_01_create_predict_data'))
#  suite.addTest(TempTest1('test_11_proc_predict_sk_model'))
  return suite

## 部分的なテストを実施するときに利用
# 上のクラスでテンポラリクラスを作ってclean_flagを設定

if __name__ == '__main__':
  runner = unittest.TextTestRunner()
  test_suite = suite()
  runner.run(test_suite)