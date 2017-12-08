'''
To run an example type

>>> from mvacfg import examples as ex
>>> ex.<example-name>.main()

All the examples have a function called "main" which \
executes the example. To see the available examples \
simply access

>>> ex.__all__

what will of course return the modules that will be \
imported when calling

>>> from mvacfg.examples import *

in this case, all the available examples. The examples \
can be executed as a script too.
'''

__author__ = 'Miguel Ramos Pernas'
__email__  = 'miguel.ramos.pernas@cern.ch'


# Python
import os


# All the modules in the current directory.
__all__ = list(set(s[:s.rfind('.py')]
                   for s in os.listdir(os.path.dirname(__file__))
                   if not s.startswith('_')
                   ))
