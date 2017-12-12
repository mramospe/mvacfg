'''
Test some constructors
'''

__author__ = 'Miguel Ramos Pernas'
__email__  = 'miguel.ramos.pernas@cern.ch'


# Python
import os

# Scikit-learn
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# mvacfg
from mvacfg import ConfigMgr, Configurable, StdMVAmgr


__fname__ = 'test_config.ini'


def test_configmgr():
    '''
    Test the configuration manager constructor from a configuration file.
    '''
    # Generate a fake manager and save its configuration
    base = Configurable(DecisionTreeClassifier)
    
    clss = Configurable(AdaBoostClassifier, {'base_estimator': base})

    mgr = Configurable(StdMVAmgr, {'classifier': clss, 'features'  : ['A', 'B', 'C']})

    cfg = ConfigMgr.from_configurable('manager', mgr)
    
    path = './' + __fname__
    
    cfg.save(path)

    # Build the configuration from the file and get the manager
    rcfg = ConfigMgr.from_config(path)
    
    mgr = rcfg.processed_config()['manager']

    os.remove(__fname__)
    
    assert cfg == rcfg
