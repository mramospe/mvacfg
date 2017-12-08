'''
Example to generate two MVA methods, using the standard method
and the k-fold method, comparing the results on a common sample.
'''


__author__ = 'Miguel Ramos Pernas'
__email__  = 'miguel.ramos.pernas@cern.ch'


# Python
import numpy as np
import pandas

# Scikit-learn
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# mvacfg
import mvacfg


def main():

    n = 10000

    # Common variables for both signal and background
    def _common( smp ):
        smp['B']   = smp['A']**2
        smp['C']   = np.log(np.abs(smp['A']*smp['B']))
        smp['evt'] = np.arange(len(smp))
    
    sig = pandas.DataFrame()
    sig['A'] = np.random.normal(0., 10., n)
    sig['D'] = np.random.exponential(1e6, n)
    _common(sig)

    bkg = pandas.DataFrame()
    bkg['A'] = np.random.normal(0., 8., n)
    bkg['D'] = np.random.uniform(0., 1, n)
    _common(bkg)
    
    # Use standard method
    classifier = AdaBoostClassifier
    
    cfg = { 'algorithm'     : 'SAMME',
            'base_estimator':
                DecisionTreeClassifier(
            criterion         = 'gini',
            max_depth         = 4,
            max_features      = None,
            max_leaf_nodes    = None,
            min_samples_leaf  = 0.01,
            min_samples_split = 2,
            random_state      = None,
            splitter          = 'best'
            ),
            'learning_rate' : 0.05,
            'n_estimators'  : 100,
            'random_state'  : None,
            }

    std   = mvacfg.StdMVAmethod(0.7, 0.7)
    kfold = mvacfg.KFoldMVAmethod(2, 'evt')

    features = ['A', 'B', 'C', 'D']

    for name, mcfg in [('std', std), ('kfold', kfold)]:
        print '-- Process mode "{}"'.format(name)
        mvacfg.mva_study( name, 'sig', sig, 'bkg', bkg, features, classifier,
                          mvaconfig = cfg,
                          methconfig = mcfg
                          )

    
    # Create a random sample and apply the methods
    print '-- Create independent sample'
    smp = pandas.DataFrame()
    smp['A'] = np.random.normal(0., 9., n)
    smp['D'] = np.concatenate([np.random.exponential(1e6, n - 1000),
                               np.random.uniform(0., 1, 1000)])
    _common(smp)
    
    for name in ('std',  'kfold'):
        print '-- Apply MVA method "{}"'.format(name)

        d = 'mva_configs_{}'.format(name)
        
        f = filter(lambda s: s.endswith('.pkl'), os.listdir(d))[0]
        
        s = joblib.load(os.path.join(d, f))
        
        s.apply(smp, '{}_dec'.format(name), '{}_pred'.format(name))

    common = {'histtype': 'step', 'weights': np.ones(n)/float(n), 'lw': 2}

    plt.hist(smp['std_pred'], color = 'b', ls = 'solid', **common)
    plt.hist(smp['kfold_pred'], color = 'r', ls = 'dashed', **common)
    plt.show()


if __name__ == '__main__':
    main()
