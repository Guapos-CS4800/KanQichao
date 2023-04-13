from mockito import mock, verify
import unittest

from deeplakeupload import loadDeeplakeSet

class Deeplakeupload(unittest.TestCase):
    def test_should_issue_hello_world_message(self):
        out = mock()

        loadDeeplakeSet(out)

        verify(out).write('--------data loaded--------')