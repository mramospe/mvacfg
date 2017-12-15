'''
Module with some functions to handle configuration files.
'''

__author__ = 'Miguel Ramos Pernas'
__email__  = 'miguel.ramos.pernas@cern.ch'


__all__ = ['available_configuration']


# Name of the section holding the general configuration variables
__main_config_name__ = 'GENERAL'


def available_configuration( flst ):
    '''
    Return the next available configuration index.

    :param flst: list of configuration files.
    :type flst: list of str
    :returns: amount of configuration files.
    :rtype: int
    '''
    numbers = list(sorted(int(f[f.rfind('_') + 1 : f.rfind('.')]) for f in flst))
    
    for i, n in enumerate(numbers):
        if i != n:
            return i
    
    return len(numbers)
