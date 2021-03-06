'''
Classes and function to plot MVA results using matplotlib.
'''

__author__ = 'Miguel Ramos Pernas'
__email__  = 'miguel.ramos.pernas@cern.ch'


# Python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import warnings

# Scikit learn
from sklearn.metrics import roc_curve


__all__ = ['ks_test', 'overtraining_hists', 'plot_overtraining_hists', 'ROC']


def ks_test( mva_proba_A, mva_proba_B, maxpv ):
    '''
    :param mva_proba_A: MVA output for sample A.
    :type mva_proba_A: array-like
    :param mva_proba_B: MVA output for sample B.
    :type mva_proba_B: array-like
    :param maxpv: maximum value of the p-value of the KS \
    test to get a warning.
    :type maxpv: float
    :returns: KS test statistics and p-value
    :rtype: tuple(float, float)

    .. seealso:: :func:`scipy.stats.ks_2samp`
    '''
    ks_stat, pvalue = st.ks_2samp(np.sort(mva_proba_A),
                                  np.sort(mva_proba_B))
    if pvalue > maxpv:
        warnings.warn('Kolmogorov-Smirnov test rejects the '\
            'null hypothesis (p-value = {:.4f})'.format(pvalue))

    return ks_stat, pvalue


def overtraining_hists( train, test, is_sig = 'is_sig', weights = None, bins = 20 ):
    '''
    Make the overtraining histograms from the training and
    testing samples. To make these histograms and plot the
    results see :func:`plot_overtraining_hists`.

    :param train: training sample.
    :type train: pandas.DataFrame
    :param test: test sample.
    :type test: pandas.DataFrame
    :param is_sig: flag that defines the signal component.
    :type is_sig: bool
    :param weights: name of the column representing possible weights in the \
    samples.
    :type weights: str
    :param bins: bins to consider for the histogram.
    :type bins: see :func:`numpy.histogram`
    :returns: values of the drawn histograms: training \
    background, training signal, testing background, \
    testing signal and the edges of the bins.
    :rtype: tuple(list(4:math:`\times` array-like), array-like)
    '''
    bkg_train = train[train[is_sig] == False]
    sig_train = train[train[is_sig] == True]

    bkg_test = test[test[is_sig] == False]
    sig_test = test[test[is_sig] == True]

    ss = (bkg_train, sig_train, bkg_test, sig_test)

    hists = []
    for s in ss:

        if weights is not None:
            wgts = s[weights]
        else:
            wgts = None

        values, edges = np.histogram(s['mva_proba'], bins=bins, range=(0, 1), weights=wgts)
        hists.append(values)

    return hists, edges


def plot_overtraining_hists( train, test, where = None, **kwargs ):
    '''
    Plot the overtraining histograms using the default
    configuration.

    See :func:`overtraining_hists`.
    '''
    if where is None:
        _, where = plt.subplots(1, 1)

    (bkg_tr, sig_tr, bkg_te, sig_te), edges = overtraining_hists(train, test, **kwargs)

    centers = (edges[1:] + edges[:-1])/2.

    xerr = (edges[1] - edges[0])/2.

    # Plot training histograms
    for s, c in ((bkg_tr, 'r'), (sig_tr, 'b')):

        wgts = s/float(np.sum(s))

        where.hist(centers, edges, weights = wgts, histtype = 'stepfilled', color = c, alpha = 0.5)

    # Plot testing histograms
    for s, c in ((bkg_te, 'r'), (sig_te, 'b')):

        n    = float(np.sum(s))
        yerr = np.sqrt(s)/n
        yv   = s/n

        where.errorbar(centers, yv, yerr, xerr, ls = 'none', c = c)

    where.set_xlabel('MVA decision')
    where.set_ylabel('Normalized entries')


def ROC( sig_sta, mva_proba, **kwargs ):
    '''
    Calculate ROC curve giving the signal status  and the MVA method decision.

    :param sig_sta: array with the signal status of "mva_proba". (True \
    for signal and False for background)
    :type sig_sta: array-like
    :param mva_proba: array with the MVA response.
    :type mva_proba: array-like
    :param kwargs: extra arguments to sklearn.metrics.roc_curve.
    :type kwargs: dict
    :returns: background rejection and signal efficiency.
    :rtype: tuple(array-like, array-like)

    .. seealso:: :func:`sklearn.metrics.roc_curve`
    '''
    bkg_eff, sig_eff, thresholds = roc_curve(sig_sta, mva_proba, **kwargs)
    bkg_rej = 1. - bkg_eff

    return bkg_rej, sig_eff
