import os
import sys
import traceback

class BaseException(Exception):
    def __init__(self, value):
        self.value = value
    def __repr__(self):
        return "{} : {}".format(type(self).__name__, self.value)
    def __str__(self):
        return self.value

class EmptyDirectoryException(BaseException):
    def __init__(self):
        self.main = "The directory passed does not contain any .lab files"
        super(EmptyDirectoryException, self).__init__(self.main)