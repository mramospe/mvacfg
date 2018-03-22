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


def ks_test( mva_dec_A, mva_dec_B, maxpv ):
    '''
    :param mva_dec_A: MVA output for sample A.
    :type mva_dec_A: array-like
    :param mva_dec_B: MVA output for sample B.
    :type mva_dec_B: array-like
    :param maxpv: maximum value of the p-value of the KS \
    test to get a warning.
    :type maxpv: float
    :returns: KS test statistics and p-value
    :rtype: tuple(float, float)

    .. seealso:: :func:`scipy.stats.ks_2samp`
    '''
    ks_stat, pvalue = st.ks_2samp(np.sort(mva_dec_A),
                                  np.sort(mva_dec_B))
    if pvalue > maxpv:
        warnings.warn('Kolmogorov-Smirnov test rejects the '\
            'null hypothesis (p-value = {:.4f})'.format(pvalue))

    return ks_stat, pvalue


def overtraining_hists( train, test, is_sig = 'is_sig', rg = None, nbins = 20 ):
    '''
    Make the overtraining histograms from the training and
    testing samples. To make these histograms and plot the
    results see :func:`plot_overtraining_hists`.

    :returns: values of the drawn histograms: training \
    background, training signal, testing background, \
    testing signal and the centers of the bins.
    :rtype: tuple(list(4:math:`\times` array-like), array-like)
    '''
    bkg_train = train[train[is_sig] == False]['mva_dec']
    sig_train = train[train[is_sig] == True]['mva_dec']

    bkg_test = test[test[is_sig] == False]['mva_dec']
    sig_test = test[test[is_sig] == True]['mva_dec']

    ss = (bkg_train, sig_train, bkg_test, sig_test)

    if rg == None:

        mn = np.min([a.min() for a in ss])
        mx = np.max([a.max() for a in ss])
        mx = np.nextafter(mx, 2.*mx)
        rg = (mn, mx)

    hists = []

    for s in ss:
        values, edges = np.histogram(s, bins = nbins, range = rg)
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


def ROC( sig_sta, mva_dec, **kwargs ):
    '''
    Calculate ROC curve giving the signal status  and the MVA method decision.

    :param sig_sta: array with the signal status of "mva_dec". (True \
    for signal and False for background)
    :type sig_sta: array-like
    :param mva_dec: array with the MVA response.
    :type mva_dec: array-like
    :param kwargs: extra arguments to sklearn.metrics.roc_curve.
    :type kwargs: dict
    :returns: background rejection and signal efficiency.
    :rtype: tuple(array-like, array-like)

    .. seealso:: :func:`sklearn.metrics.roc_curve`
    '''
    bkg_eff, sig_eff, thresholds = roc_curve(sig_sta, mva_dec, **kwargs)
    bkg_rej = 1. - bkg_eff

    return bkg_rej, sig_eff
