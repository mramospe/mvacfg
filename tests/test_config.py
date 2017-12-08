'''
Generate a configuration and read it from the output file, checking
that they are the identical.
'''

__author__ = 'Miguel Ramos Pernas'
__email__  = 'miguel.ramos.pernas@cern.ch'


# Python
import os

# mvacfg
import mvacfg


def test_config():
    
    fname = 'test_config.ini'

    testconfig = {
        'string' : 'test',
        'int'    : 1,
        'float'  : 0.1,
        'dict'   : {'sfloat': 1.1, 'sint': 1, 'sstring': 'stest'}
        }
    
    cfg = mvacfg.genconfig('test', testconfig)

    mvacfg.save_config(cfg, fname)

    read = mvacfg.readconfig(fname)

    matches = mvacfg.check_configurations(cfg, [fname])

    os.remove(fname)

    assert len(matches) == 1
    
