'''
Module with some functions to handle configuration files.
'''

__author__ = 'Miguel Ramos Pernas'
__email__  = 'miguel.ramos.pernas@cern.ch'


__all__ = ['available_configuration', 'manage_config_matches']


# confmgr
from confmgr import ConfMgr


def available_configuration( cfglst ):
    '''
    Return the next available configuration index.

    :param cfglst: list of configuration files.
    :type cfglst: list(confmgr.ConfMgr)
    :returns: amount of configuration files.
    :rtype: int
    '''
    numbers = list(sorted(int(cfg['confid']) for cfg in cfglst))

    for i, n in enumerate(numbers):
        if i != n:
            return i

    return len(numbers)


def manage_config_matches( matches, conf_id ):
    '''
    Manage the matched configurations, asking
    to overwrite the first matching file or create
    a new one with the given configuration ID.

    :param matches: list of files matching the configuration.
    :type matches: list(ConfMgr)
    :param conf_id: proposed configuration ID.
    :type conf_id: str
    :returns: configuration ID.
    :rtype: str
    '''
    if matches:

        print('WARNING: Found {} file(s) with the same '\
            'configuration'.format(len(matches)))

        match_id = matches[-1]['confid']

        d = ''
        while d not in ('Y', 'n'):
            d = raw_input('WARNING: Overwrite existing '\
                          'configuration file with ID '\
                          '"{}"? (Y/[n]): '.format(match_id)
                          ).strip()
            if not d:
                d = 'n'

        if d == 'Y':
            conf_id = match_id

        else:
            d = ''
            while d not in ('Y', 'n'):
                d = raw_input('WARNING: Do you want to create a '\
                              'new configuration file? (Y/[n]): ')

                if not d:
                    d = 'n'

            if d == 'n':
                exit(0)

    return conf_id
