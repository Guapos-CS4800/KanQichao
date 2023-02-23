from mockito import mock, verify
import unittest

from helloWorld import unitTest

class HelloWorldTest(unittest.TestCase):
    def test_should_issue_hello_world_message(self):
        out = mock()

        unitTest(out)

        verify(out).write("Hello World")