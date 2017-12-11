'''
Module to manage MVA configurations.
'''

__author__ = 'Miguel Ramos Pernas'
__email__  = 'miguel.ramos.pernas@cern.ch'


# Python
import configparser, os, re
from collections import OrderedDict as odict


__all__ = ['check_configurations', 'genconfig', 'print_configuration', 'readconfig', 'save_config']


# Name of the section holding the general configuration variables
__main_config_name__ = 'GENERAL'
__manager_name__     = 'manager'


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


def check_configurations( config, flst, skip = None ):
    '''
    Check in the given path if any other files exist with the
    same configuration.

    :param config: configuration to check.
    :type config: dict
    :param flst: list of configuration files.
    :type flst: list of str
    :returns: list of configurations matching the input.
    :rtype: list of str
    '''
    skip = skip or {}
    
    matches = []
    for f in flst:
        
        cfg = configparser.ConfigParser()
        cfg.read(f)

        for s, l in skip.iteritems():
            for e in l:
                cfg.remove_option(s, e)
                
        if cfg == config:
            matches.append((f, cfg))
            
    return matches


def genconfig( name, dic ):
    '''
    Generate a xml file containing the configuration stored
    in the given dictionary.
    
    :param name: name of the configuration.
    :type name: str
    :param dic: dictionary with the configuration.
    :type dic: dict
    '''
    config = configparser.ConfigParser()

    config.add_section(__main_config_name__)
    config.set(__main_config_name__, 'cfgname', name)
    
    for e, v in sorted(dic.iteritems()):
        _proc_config_element(config, __main_config_name__, e, v)

    return config


def _get_configurations( path, name_pattern ):
    '''
    Get the list of current configurations in "path".
    
    :param path: path to get the configurations from.
    :type path: str
    :param name_pattern: regex to filter the configurations.
    :type name_pattern: str
    '''
    comp = re.compile(name_pattern)
    
    matches = [comp.match(f) for f in os.listdir(path)]
    
    full_lst = ['{}/{}'.format(path, f.string) for f in matches if f is not None]
    
    return full_lst


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
            cfg_path = expath
            conf_id  = readconfig(expath)[__main_config_name__]['confid']
        else:
            d = ''
            while d not in ('Y', 'n'):
                d = raw_input('WARNING: Do you want to create a new configuration file? (Y/[n]): ')
                    
                if not d:
                    d = 'n'
            
            if d == 'n':
                exit(0)

    return conf_id


def print_configuration( cfg, indent = 0 ):
    '''
    Display the given configuration.

    :param cfg: configuration.
    :rtype cfg: dict
    :param indent: indentation for the print function.
    :type indent: int
    '''
    maxl = max(map(len, cfg.keys()))

    lsp = maxl + indent
    
    for k, v in cfg.iteritems():
        
        d = '{:>{}}'.format('{:<{}}'.format(k, maxl), lsp)
        
        if hasattr(v, 'iteritems'):
            if len(v.keys()) > 0:
                print '{} = ('.format(d)
                print_configuration(v, indent + 5)
                print '{:>{}}'.format(')', indent + 5)
        else:
            print '{} = {}'.format(d, v)


def _proc_config_element( config, root, name, element ):
    '''
    Process the given element storing a configuration value
    '''
    if hasattr(element, '__dict__'):

        cl = element.__class__

        # So later it can be easily loaded
        full_name = '{}.{}'.format(cl.__module__, cl.__name__)

        config.set(root, name, full_name)

        config.add_section(name)
        
        for e, v in sorted(vars(element).iteritems()):
            _proc_config_element(config, name, e, v)
    else:
        config.set(root, name, str(element))


def readconfig( path ):
    '''
    Read a xml file storing a configuration and return the
    name of the root and a dictionary with the configuration.

    :param path: path to the input file.
    :type path: str
    :returns: tuple with the tag and configuration of the \
    input file.
    :rtype: tuple(str, dict)
    '''
    c = configparser.ConfigParser()

    with open(path, 'rb') as f:
        c.read(path)
    
    return c

def save_config( cfg, path ):
    '''
    :param cfg: configuration.
    :type cfg: ConfigParser
    :param path: path to save the output file.
    :type path: str
    '''
    print 'INFO: Generate new configuration file "{}"'.format(path)

    with open(path, 'wb') as f:
        cfg.write(f)
