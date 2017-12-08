'''
Classes and function to plot MVA results using matplotlib.
'''

__author__ = 'Miguel Ramos Pernas'
__email__  = 'miguel.ramos.pernas@cern.ch'


# Python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

# Scikit learn
from sklearn.metrics import roc_curve


__all__ = ['ks_test', 'ROC']


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
        print 'WARNING: Kolmogorov-Smirnov test rejects the '\
            'null hypothesis (p-value = {:.4f})'.format(pvalue)

    return ks_stat, pvalue


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
