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


__fname__ = 'test_config.ini'


def _generate_and_check( func ):
    '''
    Decorator to study the configuration created from an
    input function.

    :param func: input function, with no arguments, that \
    creates a configuration dictionary.
    :type func: function
    '''
    def wrapper():
        '''
        Create the configuration file and read it, checking
        that the two versions match.
        '''
        cfg = mvacfg.ConfigMgr.from_dict(func())
        cfg.save(__fname__)

        read = mvacfg.ConfigMgr.from_config(__fname__)

        matches = mvacfg.check_configurations(cfg, [__fname__])

        os.remove(__fname__)

        assert len(matches) == 1

    return wrapper


@_generate_and_check
def test_basic_config():
    '''
    Test a basic configuration.
    '''
    return {
        'string' : 'this is a test',
        'int'    : 1,
        'float'  : 0.1,
        }


class ttcl:
    '''
    Small class to test the configuration module.
    '''
    def __init__( self, name, first, second = 2. ):
        '''
        Store some attributes.
        '''
        self.name   = name
        self.first  = first
        self.second = second


@_generate_and_check
def test_class_config():
    '''
    Test a configuration holding a class.
    '''
    return {
        'string' : 'this is a test',
        'object' : mvacfg.Configurable(ttcl,
                                       {'name' : 'ttcl',
                                        'first': 1
                                       }),
        'int'    : 1,
        'float'  : 0.1
        }
