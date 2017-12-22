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
from sklearn import datasets

# confmgr
from confmgr import Config, ConfMgr

# mvacfg
import mvacfg
from mvacfg.core import __is_sig__, __mva_dec__


def main():

    # Load a sample
    dt = datasets.load_breast_cancer()

    data = pandas.DataFrame(dt.data)
    cols = list(str(c) for c in data.columns)
    data.columns = cols

    data['evt'] = range(len(data))

    sig = data[dt.target == True]
    bkg = data[dt.target == False]

    # Configurable of the base estimator
    bes_cfg = Config(
        DecisionTreeClassifier,
        {
        'criterion'         : 'gini',
        'max_depth'         : 20,
        'max_features'      : None,
        'max_leaf_nodes'    : None,
        'min_samples_leaf'  : 0.01,
        'min_samples_split' : 2,
        'random_state'      : None,
        'splitter'          : 'best'
        })

    # Configurable of the classifier estimator
    class_cfg = Config(
        AdaBoostClassifier,
        { 'algorithm'     : 'SAMME',
          'base_estimator': bes_cfg,
          'learning_rate' : 0.25,
          'n_estimators'  : 27,
          'random_state'  : None,
        })

    # Configurables of the standard and k-folding methods
    std_cfg = ConfMgr(manager = Config(
        mvacfg.StdMVAmgr,
        {'classifier' : class_cfg,
         'features'   : cols
        }))
    kfold_cfg = ConfMgr(manager = Config(
        mvacfg.KFoldMVAmgr,
        {'classifier' : class_cfg,
         'nfolds'     : 2,
         'features'   : cols,
         'splitvar'   : 'evt'
        }))

    # Do the study
    test_smps = odict()
    for name, cfg in [('std', std_cfg), ('kfold', kfold_cfg)]:

        print '-- Process mode "{}"'.format(name)
        _, train, test = mvacfg.mva_study(name, 'sig', sig, 'bkg', bkg, cfg)
        mvacfg.plot_overtraining_hists(train, test)
        plt.show()

        test_smps[name] = test

    # Apply the two methods to the merged sample
    for name in ('std',  'kfold'):
        print '-- Apply MVA method "{}"'.format(name)

        d = 'mva_configs_{}'.format(name)

        f = filter(lambda s: s.endswith('.pkl'), os.listdir(d))[0]

        s = joblib.load(os.path.join(d, f))

        s.apply(data, '{}_dec'.format(name), '{}_pred'.format(name))

    # Compare the two methods
    com = {'histtype': 'stepfilled', 'range': (-1, 1), 'alpha': 0.5}

    fig, (ax0, ax1) = plt.subplots(1, 2)

    ax0.hist(data['std_dec'], color = 'b', label = 'std', **com)
    ax0.hist(data['kfold_dec'], color = 'r', label = 'kfold', **com)
    ax0.legend()
    ax0.set_xlabel('MVA decision')
    ax0.set_ylabel('Normalized entries')

    for (s, data), c in zip(test_smps.iteritems(), ('b', 'r')):
        bkg_rej, sig_eff = mvacfg.ROC(data[__is_sig__], data[__mva_dec__])
        ax1.plot(bkg_rej, sig_eff, c = c, label = s)
    ax1.legend()
    ax1.set_xlabel('background rejection')
    ax1.set_ylabel('signal efficiency')

    plt.show()


if __name__ == '__main__':
    main()
