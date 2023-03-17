from mockito import mock, verify
import unittest

from testKeras import loadImages

class testKeras(unittest.TestCase):
    def test_should_issue_hello_world_message(self):
        out = mock()

        loadImages(out)

        verify(out).write('60000 train samples, 10000 test samples')