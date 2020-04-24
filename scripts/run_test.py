from testmodules.test_lb_task_learning import *
from testmodules.test_lb_task_predict import *
import unittest

def suite():
  suite = unittest.TestSuite()
#  suite.addTest(unittest.makeSuite(TestLBv1TaskLearning))
#  suite.addTest(unittest.makeSuite(TestLBv1TaskPredict))
#  suite.addTest(unittest.makeSuite(TestLBv2TaskLearning))
#  suite.addTest(unittest.makeSuite(TestLBv2TaskPredict))
#  suite.addTest(unittest.makeSuite(TestLBv3TaskLearning))
#  suite.addTest(unittest.makeSuite(TestLBv3TaskPredict))
#  suite.addTest(unittest.makeSuite(TestLBv4TaskLearning))
#  suite.addTest(unittest.makeSuite(TestLBv4TaskPredict))
  suite.addTest(unittest.makeSuite(TestLBv5TaskLearning))
  suite.addTest(unittest.makeSuite(TestLBv5TaskPredict))
  suite.addTest(unittest.makeSuite(TestLBRv1TaskLearning))
  suite.addTest(unittest.makeSuite(TestLBRv1TaskPredict))
  return suite


if __name__ == '__main__':
  runner = unittest.TextTestRunner()
  test_suite = suite()
  runner.run(test_suite)