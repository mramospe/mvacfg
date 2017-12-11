'''
Test some constructors
'''

__author__ = 'Miguel Ramos Pernas'
__email__  = 'miguel.ramos.pernas@cern.ch'


# Scikit-learn
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# mvacfg
import mvacfg


__fname__ = 'test_config.ini'


def test_mvamgr():
    '''
    Test the MVA manager constructor from a configuration file.
    '''
    # Generate a fake manager and save its configuration
    meth = mvacfg.StdMVAmethod()
    
    fake_mgr = meth.build_mgr(
        AdaBoostClassifier(base_estimator = DecisionTreeClassifier()),
        ['A', 'B', 'C'])

    cfg = {
        'signame' : 'sig',
        'bkgname' : 'bkg',
        'manager' : fake_mgr,
        'outdir'  : '.'
    }

    path = './' + __fname__
    
    cfg = mvacfg.genconfig('dummy', cfg)
    mvacfg.save_config(cfg, path)

    # Build the manager
    mgr = mvacfg.MVAmgr.from_config(path)

test_mvamgr()
