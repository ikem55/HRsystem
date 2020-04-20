from testmodules.test_lb_task_learning import TestLBv1TaskLearning, TestLBv2TaskLearning, TestLBv3TaskLearning, TestLBv4TaskLearning, TestLBRv1SkModelLearning
from testmodules.test_lb_task_predict import TestLBv1TaskPredict, TestLBv2TaskPredict, TestLBv3TaskPredict,TestLBv4TaskPredict, TestLBRv1TaskPredict
import unittest

def suite():
  suite = unittest.TestSuite()
  suite.addTest(unittest.makeSuite(TestLBv1TaskLearning))
  suite.addTest(unittest.makeSuite(TestLBv1TaskPredict))
  suite.addTest(unittest.makeSuite(TestLBv2TaskLearning))
  suite.addTest(unittest.makeSuite(TestLBv2TaskPredict))
  suite.addTest(unittest.makeSuite(TestLBv3TaskLearning))
  suite.addTest(unittest.makeSuite(TestLBv3TaskPredict))
  suite.addTest(unittest.makeSuite(TestLBv4TaskLearning))
  suite.addTest(unittest.makeSuite(TestLBv4TaskPredict))
  suite.addTest(unittest.makeSuite(TestLBRv1SkModelLearning))
  suite.addTest(unittest.makeSuite(TestLBRv1TaskPredict))
  return suite


if __name__ == '__main__':
  runner = unittest.TextTestRunner()
  test_suite = suite()
  runner.run(test_suite)