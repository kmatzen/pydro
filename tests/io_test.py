from pydro.io import *

def read_test():
    model = LoadModel('tests/example.dpm')

def write_test():
    model = LoadModel('tests/example.dpm')
    SaveModel('tests/write_test.dpm', model)
