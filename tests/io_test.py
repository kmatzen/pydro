from pydro.io import *

import itertools

def read_test():
    model = LoadModel('tests/example.dpm')

def write_test():
    model = LoadModel('tests/example.dpm')
    SaveModel('tests/write_test.dpm', model)

def compare (obja, objb):
    if type(obja) != type(objb):
        return False

    if type(obja) in (str,int):
        return obja == objb
    elif type(obja) == list:
        if len(obja) != len(objb):
            return False
        return all(compare(a, b) for a, b in itertools.izip(obja, objb)) 
    elif isinstance(obja, object):
        dicta = obja.__dict__
        dictb = objb.__dict__

        if len(dicta) != len(dictb):
            return False

        if set(dicta.keys()).union(dictb.keys()) != set(dicta.keys()):
            return False

        for name in dicta.keys():
            valuea = dicta[name]
            valueb = dictb[name]

            return compare(valuea, valueb)
    else:
        print(type(obja), type(objb))
        return False

def wr_test():
    model = LoadModel('tests/example.dpm')
    model3 = LoadModel('tests/example.dpm')
    SaveModel('tests/wr_test.dpm', model)
    model2 = LoadModel('tests/wr_test.dpm')
    assert compare(model, model2)
    assert compare(model, model3)
