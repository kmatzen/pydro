import subprocess
import os

def voc_dpm2pydro_test():
    print(os.getcwd())
    subprocess.check_call([
        'voc-dpm2pydro', 
        '--input',
        'tests/example.mat', 
        '--output',
        'tests/converted.dpm'
    ])

    with open('tests/example.dpm', 'rb') as f, open('tests/converted.dpm', 'rb') as g:
        assert(f.read() == g.read())
