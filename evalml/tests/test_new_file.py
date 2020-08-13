import unittest

from evalml.new_module.new_file import foo


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(foo(), 2)


if __name__ == '__main__':
    unittest.main()
