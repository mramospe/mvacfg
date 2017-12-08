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
        cfg = mvacfg.genconfig('test', func())

        mvacfg.save_config(cfg, __fname__)

        read = mvacfg.readconfig(__fname__)

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


@_generate_and_check
def test_dict_config():
    '''
    Test a configuration involving a dictionary.
    '''
    return {
        'string' : 'this is a test',
        'int'    : 1,
        'float'  : 0.1,
        'dict'   : {
            'sub-float'  : 1.1,
            'sub-int'    : 1,
            'sub-string' : 'ss'
            }
        }
    

@_generate_and_check
def test_class_config():
    '''
    Test a configuration holding a class.
    '''
    class ttcl:
        
        def __init__( self ):
            '''
            Stores some attributes.
            '''
            self.name   = 'ttcl'
            self.first  = 1
            self.second = 2

    return {
        'string' : 'this is a test',
        'object' : ttcl(),
        'int'    : 1,
        'float'  : 0.1
        }
