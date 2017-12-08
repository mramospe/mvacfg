'''
Module to manage MVA configurations.
'''

__author__ = 'Miguel Ramos Pernas'
__email__  = 'miguel.ramos.pernas@cern.ch'


# Python
import configparser, os, re
from collections import OrderedDict as odict


__all__ = ['check_configurations', 'genconfig', 'print_configuration', 'readconfig', 'save_config']


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


def check_configurations( config, flst ):
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
    matches = []
    for f in flst:
        
        cfg = configparser.ConfigParser()
        cfg.read(f)
        
        if cfg == config:
            matches.append(cfg)
            
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

    config.set('DEFAULT', 'cfgname', name)
    
    for e, v in dic.iteritems():
        _proc_config_element(config, 'DEFAULT', e, v)

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
        
        expath = matches[-1]
    
        d = ''
        while d not in ('Y', 'n'):
            d = raw_input('WARNING: Overwrite existing configuration file '\
                              '"{}"? (Y/[n]): '.format(expath)
                          ).strip()
            if not d:
                d = 'n'

        if d == 'Y':
            cfg_path = expath
            conf_id  = readconfig(expath)[1]['confid']
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
    
    for k, v in sorted(cfg.iteritems()):

        d = '{:>{}}'.format('{:<{}}'.format(k, maxl), lsp)
        
        try:
            dict(v)
            print '{} = ('.format(d)
            print_configuration(v, indent + 5)
            print '{:>{}}'.format(')', maxl)
        except:
            print '{} = {}'.format(d, v)


def _proc_config_element( config, root, name, element ):
    '''
    Process the given element storing a configuration value
    '''
    if hasattr(element, '__dict__'):
        
        config.add_section(name)
        
        d = element.__dict__
        
        for e, v in sorted(d.iteritems()):
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
