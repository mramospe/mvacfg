'''
Module to manage files with MVA configurations.
'''

__author__ = 'Miguel Ramos Pernas'
__email__  = 'miguel.ramos.pernas@cern.ch'


# Python
import os

# mvacfg
from mvacfg.config import ConfigMgr, __main_config_name__


__all__ = []


def _available_configuration( flst ):
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


def _manage_config_matches( matches, conf_id ):
    '''
    Manage those configurations wich matched one given,
    asking to overwrite the files or create a new one
    with the given configuration ID.

    :param matches: list of files matching the configuration.
    :type matches: list of str
    :param conf_id: proposed configuration ID.
    :type conf_id: str
    :returns: configuration ID.
    :rtype: str
    '''
    if matches:

        print 'WARNING: Found {} file(s) with the same configuration'.format(len(matches))
        
        expath = matches[-1][0]
        
        d = ''
        while d not in ('Y', 'n'):
            d = raw_input('WARNING: Overwrite existing configuration file '\
                              '"{}"? (Y/[n]): '.format(expath)
                          ).strip()
            if not d:
                d = 'n'

        if d == 'Y':

            confmgr = ConfigMgr()
            confmgr.read(expath)
            
            cfg_path = expath
            conf_id  = confmgr[__main_config_name__]['confid']
            
        else:
            d = ''
            while d not in ('Y', 'n'):
                d = raw_input('WARNING: Do you want to create a new configuration file? (Y/[n]): ')
                    
                if not d:
                    d = 'n'
            
            if d == 'n':
                exit(0)

    return conf_id
