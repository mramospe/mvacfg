'''
Test some constructors.
'''

__author__ = 'Miguel Ramos Pernas'
__email__  = 'miguel.ramos.pernas@cern.ch'


# Python
import os

# Scikit-learn
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# confmgr
from confmgr import ConfMgr, Config

# mvacfg
from mvacfg import StdMVAmgr, manager_name


__fname__ = 'test_config.xml'


def test_configmgr():
    '''
    Test the configuration manager constructor from a configuration file.
    '''
    # Generate a fake manager and save its configuration
    base = Config(DecisionTreeClassifier)

    clss = Config(AdaBoostClassifier, {'base_estimator': base})

    mgr = Config(StdMVAmgr, {'classifier': clss, 'features'  : ['A', 'B', 'C']})

    cfg = ConfMgr({manager_name(): mgr})

    path = './' + __fname__

    cfg.save(path)

    # Build the configuration from the file and get the manager
    rcfg = ConfMgr.from_file(path)

    mgr = rcfg.proc_conf()[manager_name()]

    os.remove(__fname__)

    assert cfg == rcfg
