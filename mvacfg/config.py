'''
Module to manage configurations.
'''

__author__ = 'Miguel Ramos Pernas'
__email__  = 'miguel.ramos.pernas@cern.ch'


# Python
import importlib, inspect, os, re
from configparser import ConfigParser
from collections import OrderedDict as odict


__all__ = ['check_configurations', 'ConfigMgr', 'Configurable', 'print_configuration']


# Name of the section holding the general configuration variables
__main_config_name__ = 'GENERAL'


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
        
        cfg = ConfigMgr()
        cfg.read(f)
        
        for s, l in skip.iteritems():
            for e in l:
                cfg.remove_option(s, e)
        
        if cfg == config:
            matches.append((f, cfg))
            
    return matches


class ConfigMgr(ConfigParser):
    '''
    Class to manage configurations built using the
    :class:`Configurable` class.
    '''
    def __init__( self, *args, **kwargs ):
        '''
        See :meth:`configparser.ConfigParser.__init__`.
        '''
        ConfigParser.__init__(self, *args, **kwargs)
        
        self._dct = odict()

    def _proc_config( self, name, config ):
        '''
        Process the given configuration dictionary.
        
        :param name: name of the section.
        :type name: str
        :param config: items to process.
        :type config: dict
        :returns: processed dictionary.
        :rtype: dict
        '''
        return odict((k, self._proc_config_element(name, k, v))
                     for k, v in sorted(config.iteritems())
                     )

    def _proc_config_element( self, section, name, element ):
        '''
        Process one configuration element.

        :param section: section attached to the element.
        :type section: str
        :param name: name of the element.
        :type name: str
        :param element: element to process.
        :type element: any class type
        :returns: processed element.
        :rtype: any built class type
        '''
        if isinstance(element, Configurable):

            # So later it can be easily loaded
            self.set(section, name, element.full_name())
            
            self.add_section(name)
            
            dct = self._proc_config(name, element.config())

            return element.build(dct)
        else:
            self.set(section, name, str(element))

            try:
                return eval(element)
            except:
                return element

    @classmethod
    def from_config( cls, path ):
        '''
        Build the class from a configuration file.

        :param path: path to the configuration file.
        :type path: str
        :returns: configuration manager.
        :rtype: this class type
        '''
        cfg = cls()
        cfg.read(path)

        # Process the configuration
        res = odict()
        for name, section in reversed(cfg.items()):
            
            sub = odict()
            
            for k, v in section.iteritems():
                try:
                    sub[k] = eval(v)
                except:
                    sub[k] = v

            res[name] = sub

        # Build all the classes
        its = res.items()
        for i, (name, dct) in enumerate(its):
            for n, d in its[:i]:
                if n in dct:

                    # Access the class constructor
                    path = dct[n]
                    
                    modname = path[:path.rfind('.')]
                    clsname = path[path.rfind('.') + 1:]
                    
                    const  = getattr(importlib.import_module(modname), clsname)
                    
                    # Remove the attributes not present in the constructor
                    args = inspect.getargspec(const.__init__).args
                    args.remove('self')

                    inputs = {k: v for k, v in d.iteritems() if k in args}

                    # Call the constructor
                    dct[n] = const(**inputs)

        cfg._dct = res

        return cfg

    @classmethod
    def from_configurable( cls, name, cfg ):
        '''
        Build the class from a :class:`Configurable` object.

        :param cfg: input configurable.
        :type cfg: Configurable
        :returns: configuration manager.
        :rtype: this class type
        '''
        return cls.from_dict({name: cfg})
        
    @classmethod
    def from_dict( cls, dct ):
        '''
        Build the class from a dictionary.

        :param dct: input dictionary.
        :type dct: dict
        :returns: configuration manager.
        :rtype: this class type
        '''
        cfg = cls()
        
        cfg.add_section(__main_config_name__)
        
        cfg._dct = cfg._proc_config(__main_config_name__, dct)
        
        return cfg
        
    def processed_config( self ):
        '''
        :returns: processed configuration dictionary (where \
        all the built classes are saved.
        :rtype: dict
        '''
        return self._dct

    def save( self, path ):
        '''
        :param cfg: configuration.
        :type cfg: ConfigParser
        :param path: path to save the output file.
        :type path: str
        '''
        print 'INFO: Generate new configuration file "{}"'\
            ''.format(path)

        with open(path, 'wb') as f:
            self.write(f)


class Configurable:
    '''
    Class to store any class constructor plus its
    configuration.
    '''
    def __init__( self, const, dct = None ):
        '''
        :param const: constructor of the configurable class.
        :type const: any class constructor
        :param dct: configuration of the class.
        :type dct: dict
        '''
        dct = dct or {}
        
        self._dct   = dct
        self._const = const

    def build( self, dct ):
        '''
        :param dct: configuration to build the class.
        :type dct: dict
        :returns: built class
        :rtype: built class type
        '''
        return self._const(**dct)

    def config( self ):
        '''
        :returns: configuration for the class in this object.
        :rtype: dict
        '''
        return self._dct

    def full_name( self ):
        '''
        :returns: full name of the class attached to \
        this configurable.
        :rtype: str
        '''
        cl = self._const
        return '{}.{}'.format(cl.__module__, cl.__name__)


def get_configurations( path, name_pattern ):
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
