import unittest
from test_functions import *
import os
import warnings

class Test_Model(unittest.TestCase):

  def test_run(self):
    self.assertTrue(can_run())

  def test_configurate(self):
    warnings.warn("deprecated", DeprecationWarning)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        self.assertTrue(can_configurate())
  def test_frameworks(self):
    for root, dirs, files in os.walk("sample_scripts", topdown=False):
      for name in files:
        self.assertTrue(os.path.join(root,name),name)
#the following is not required if call by pytest instead of python
if __name__ == '__main__':
    unittest.main()
