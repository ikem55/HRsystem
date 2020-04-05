from testmodules.test_lb_task_learning import TestLBv1TaskLearning, TestLBv2TaskLearning, TestLBv3TaskLearning
from testmodules.test_lb_task_predict import TestLBv1TaskPredict, TestLBv2TaskPredict, TestLBv3TaskPredict
import unittest


class TempTest1(TestLBv1TaskLearning):
  clean_flag = True

def suite():
  suite = unittest.TestSuite()
  suite.addTest(unittest.makeSuite(TempTest1))
  return suite

## 部分的なテストを実施するときに利用
# 上のクラスでテンポラリクラスを作ってclean_flagを設定

if __name__ == '__main__':
  runner = unittest.TextTestRunner()
  test_suite = suite()
  runner.run(test_suite)