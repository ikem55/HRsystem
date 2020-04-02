from testmodules.test_lb_task_learning import TestLBTaskLearning, TestLBv1TaskLearning, TestLBv2TaskLearning
from testmodules.test_lb_task_predict import TestLBTaskPredict, TestLBv1TaskPredict, TestLBv2TaskPredict
import unittest

def suite():
  suite = unittest.TestSuite()
#  suite.addTest(unittest.makeSuite(TestLBTaskLearning))
  suite.addTest(unittest.makeSuite(TestLBv1TaskLearning))
  suite.addTest(unittest.makeSuite(TestLBv2TaskLearning))
#  suite.addTest(unittest.makeSuite(TestLBTaskPredict))
  suite.addTest(unittest.makeSuite(TestLBv1TaskPredict))
  suite.addTest(unittest.makeSuite(TestLBv2TaskPredict))
  return suite


if __name__ == '__main__':
  runner = unittest.TextTestRunner()
  test_suite = suite()
  runner.run(test_suite)