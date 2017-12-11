'''
Example to generate two MVA methods, using the standard method
and the k-fold method, comparing the results on a common sample.
'''


__author__ = 'Miguel Ramos Pernas'
__email__  = 'miguel.ramos.pernas@cern.ch'


# Python
from collections import OrderedDict as odict
import matplotlib.pyplot as plt
import numpy as np
import joblib, os, pandas

# Scikit-learn
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# mvacfg
import mvacfg
from mvacfg.utils import __is_sig__, __mva_dec__


def main():

    n = 1000

    # Common variables for both signal and background
    def _common( smp ):
        smp['D']   = smp['A']**2
        smp['E']   = np.log(np.abs(smp['A']*smp['B']*smp['C']))
        smp['evt'] = np.arange(len(smp))

    # Define signal sample
    sig = pandas.DataFrame()
    sig['A'] = np.random.normal(0., 4., n)
    sig['B'] = np.random.exponential(10, n)
    sig['C'] = np.random.exponential(100, n)
    _common(sig)

    # Define background sample
    bkg = pandas.DataFrame()
    bkg['A'] = np.random.normal(0., 8., n)
    bkg['B'] = np.random.exponential(20, n)
    bkg['C'] = np.random.exponential(200, n)
    _common(bkg)

    # Configuration of the classifier
    classifier = AdaBoostClassifier
    
    cfg = { 'algorithm'     : 'SAMME',
            'base_estimator':
                DecisionTreeClassifier(
            criterion         = 'gini',
            max_depth         = 3,
            max_features      = None,
            max_leaf_nodes    = None,
            min_samples_leaf  = 0.5,
            min_samples_split = 2,
            random_state      = None,
            splitter          = 'best'
            ),
            'learning_rate' : 0.0001,
            'n_estimators'  : 5,
            'random_state'  : None,
            }

    std   = mvacfg.StdMVAmethod(0.7, 0.7)
    kfold = mvacfg.KFoldMVAmethod(2, 'evt')

    features = ['A', 'B', 'C', 'D', 'E']

    # Do the study
    test_smps = odict()
    for name, mcfg in [('std', std), ('kfold', kfold)]:
        print '-- Process mode "{}"'.format(name)
        _, train, test = mvacfg.mva_study( name, 'sig', sig, 'bkg', bkg, features, classifier,
                                           mvaconfig = cfg,
                                           methconfig = mcfg
        )
        mvacfg.plot_overtraining_hists(train, test)
        plt.show()

        test_smps[name] = test
    
    # Create a random sample and apply the methods
    print '-- Create independent sample'

    nsig = 1000
    nbkg = n - nsig
    
    smp = pandas.DataFrame()
    smp['A'] = np.concatenate([np.random.normal(0., 4., nsig),
                               np.random.normal(0., 8., nbkg)])
    smp['B'] = np.concatenate([np.random.exponential(10, nsig),
                               np.random.exponential(20, nbkg)])
    smp['C'] = np.concatenate([np.random.exponential(100, nsig),
                               np.random.exponential(200, nbkg)])
    _common(smp)

    # Apply the two methods to the random sample
    for name in ('std',  'kfold'):
        print '-- Apply MVA method "{}"'.format(name)

        d = 'mva_configs_{}'.format(name)
        
        f = filter(lambda s: s.endswith('.pkl'), os.listdir(d))[0]
        
        s = joblib.load(os.path.join(d, f))
        
        s.apply(smp, '{}_dec'.format(name), '{}_pred'.format(name))

    # Compare the two methods
    com = {'histtype': 'stepfilled', 'range': (-1, 1), 'alpha': 0.5}

    fig, (ax0, ax1) = plt.subplots(1, 2)
    
    ax0.hist(smp['std_dec'], color = 'b', label = 'std', **com)
    ax0.hist(smp['kfold_dec'], color = 'r', label = 'kfold', **com)
    ax0.legend()
    ax0.set_xlabel('MVA decision')
    ax0.set_ylabel('Normalized entries')

    for (s, smp), c in zip(test_smps.iteritems(), ('b', 'r')):
        bkg_rej, sig_eff = mvacfg.ROC(smp[__is_sig__], smp[__mva_dec__])
        ax1.plot(bkg_rej, sig_eff, c = c, label = s)
    ax1.legend()
    ax1.set_xlabel('background rejection')
    ax1.set_ylabel('signal efficiency')
    
    plt.show()
    

if __name__ == '__main__':
    main()
