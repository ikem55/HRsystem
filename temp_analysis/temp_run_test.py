from testmodules.test_lb_task_learning import TestLBv1TaskLearning, TestLBv2TaskLearning, TestLBv3TaskLearning, TestLBv4TaskLearning, TestLBRv1SkModelLearning
from testmodules.test_lb_task_predict import TestLBv1TaskPredict, TestLBv2TaskPredict, TestLBv3TaskPredict, TestLBv4TaskPredict, TestLBRv1TaskPredict
import unittest


## 部分的なテストを実施するときに利用
# このクラスでテンポラリクラスを作ってclean_flagを設定
class TempTest1(TestLBRv1SkModelLearning):
  clean_flag = True
  cls_val = "主催者コード"
  val = "2"
  target = "UMAREN_ARE"
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


if __name__ == '__main__':
  runner = unittest.TextTestRunner()
  test_suite = suite()
  runner.run(test_suite)