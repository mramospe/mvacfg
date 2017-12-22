'''
Auxiliar functions.
'''

__author__ = 'Miguel Ramos Pernas'
__email__  = 'miguel.ramos.pernas@cern.ch'


# Python
import os


__all__ = []


def _makedirs( path ):
    '''
    Check if the path exists and makes the directory if not.

    :param path: input path.
    :type path: str
    '''
    try:
        os.makedirs(path)
        print 'INFO: Creating output directory "{}"'.format(path)
    except:
        pass
