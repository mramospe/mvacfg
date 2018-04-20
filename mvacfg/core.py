'''
Module to perform trainings of MVA methods and store
their configuration.
'''

__author__ = 'Miguel Ramos Pernas'
__email__  = 'miguel.ramos.pernas@cern.ch'


# Python
import importlib, inspect, joblib, logging, os, pandas
import numpy as np
import warnings
from collections import OrderedDict as odict
from copy import deepcopy

# confmgr
import confmgr

# Scikit-learn
from sklearn.model_selection import train_test_split

# Local
import mvacfg._aux as _aux
import mvacfg.config as config


__all__ = ['MVAmgr', 'KFoldMVAmgr', 'StdMVAmgr', 'manager_name', 'mva_study']


# Global names for the MVA outputs
__mva_dec__  = 'mva_dec'
__mva_pred__ = 'mva_pred'

# Global names for the signal flags
__is_sig__   = 'is_sig'
__sig_flag__ = 1
__bkg_flag__ = 0

# Manager section name in the configuration file
__manager_name__ = 'manager'


def info( msg, indent = 0 ):
    '''
    Display an information message taking into account a indentation level.

    :param msg: message to display.
    :type msg: str
    :param indent: indentation level.
    :type indent: int
    '''
    logging.getLogger(__name__).info('{:>{}}'.format(msg, indent + len(msg)))


def manager_name():
    '''
    Return the name of the manager in the configuration files.

    :returns: name of the manager ("manager" by default).
    :rtype: str
    '''
    return __manager_name__


class MVAmgr:
    '''
    Main class to store and apply a MVA algorithm.
    '''
    def __init__( self, classifier, features ):
        '''
        :param classifier: input MVA classifier.
        :type classifier: MVA classifier
        :param features: variables to use for training.
        :type features: list of str
        '''
        self.classifier = classifier
        self.features   = list(features)

    def _fit( self, train_data, is_sig ):
        '''
        Fit the MVA classifier to the given samples.

        :param train_data: training data.
        :type train_data: pandas.DataFrame
        :param is_sig: flag to determine the signal condition.
        :type is_sig: str
        :returns: trained MVA classifier.
        :rtype: MVA classifier
        '''
        info('Perform MVA training')
        mva = self.classifier.fit(train_data[self.features],
                                  train_data[is_sig])
        info('MVA training finished')
        return mva

    def _process( self, mva, smp ):
        '''
        Apply the given MVA algorithm to the given sample. The
        result is a new DataFrame with the indices being preserved.

        :param mva: processed MVA classifier.
        :type mva: MVA classifier
        :param smp: sample to process.
        :type smp: pandas.DataFrame
        :returns: decision and prediction of the MVA response.
        :rtype: tuple(array-like, array-like)
        '''

        # This returns a DataFrame with two columns, representing
        # the probability to belong to background or signal
        # hypotheses. We keep only the second.
        dec = pandas.DataFrame(mva.predict_proba(smp),
                               index=smp.index)[[__sig_flag__]]

        pred = pandas.DataFrame(mva.predict(smp),
                                index=smp.index)

        return dec, pred

    def apply( self, sample, decname, predname ):
        '''
        Virtual method to apply the stored MVA method to the
        given sample.

        :param sample: input sample to apply the MVA method.
        :type sample: pandas.DataFrame
        :param decname: name of the MVA response decision.
        :type decname: str
        :param predname: name of the MVA response prediction.
        :type predname: str
        '''
        raise NotImplementedError('Attempt to call abstract method')

    def apply_for_overtraining( self, dt, sample, decname = __mva_dec__, predname = __mva_pred__ ):
        '''
        Redefinition of the "apply" method given the sample type
        (train or test) to search for overtraining effects.

        :param dt: type of the input sample.
        :type dt: str ('train' or 'test')

        .. seealso:: :meth:`MVAmgr.apply`.
        '''
        if dt not in ('train', 'test'):
            raise RuntimeError('Unknown data type "{}"'.format(dt))

        return self.apply(sample, decname, predname)

    def extravars( self ):
        '''
        :returns: extra variables needed for the class.
        :rtype: list
        '''
        return []

    def fit( self, sig, bkg, is_sig ):
        '''
        Fit the MVA classifier to the given sample.

        :param sig: signal sample.
        :type sig: pandas.DataFrame
        :param bkg: background sample.
        :type bkg: pandas.DataFrame
        :param is_sig: signal flag.
        :type is_sig: str
        :returns: training and testing data samples.
        :rtype: tuple(pandas.DataFrame, pandas.DataFrame)
        '''
        raise NotImplementedError('Attempt to call abstract method')

    def save( self, path ):
        '''
        Save this MVA manager to a file.

        :param path: path to the output file.
        :type path: str
        '''
        joblib.dump(self, path, compress=True)
        info('Output method saved in {}'.format(path))


class KFoldMVAmgr(MVAmgr):
    '''
    Manager that uses the k-folding technique, splitting the total
    sample in "n" folds, using "n - 1" to train the remaining
    fold, which is used as a test sample.
    '''
    __min_tolerance__ = 0.1

    def __init__( self, classifier, features, splitvar, nfolds = 2 ):
        '''
        Constructor given the classifier, features, number of
        folds and the integer variable used to split the sample.

        See :meth:`MVAmgr.__init__`.

        :param splitvar: variable used to split the sample in folds.
        :type splitvar: str
        :param nfolds: number of folds to generate.
        :type nfolds: int
        '''
        MVAmgr.__init__(self, classifier, features)

        if nfolds <= 1:
            raise RuntimeError('Number of folds must be greater than one')

        self.nfolds   = nfolds
        self.splitvar = splitvar

        info('Prepare {} folds using variable '\
            '"{}"'.format(nfolds, splitvar))

    def _false_cond( self, smp, i ):
        '''
        Condition to be satisfied by the training sample
        (necessary to search for overtraining).

        :param smp: input sample.
        :type smp: pandas.DataFrame
        :param i: fold index.
        :type i: int
        '''
        return smp[self.splitvar] % self.nfolds != i

    def _true_cond( self, smp, i ):
        '''
        Usual apply condition for the sample "smp" with fold
        index "i".

        :param smp: input sample.
        :type smp: pandas.DataFrame
        :param i: fold index.
        :type i: int
        '''
        return smp[self.splitvar] % self.nfolds == i

    def apply( self, sample, decname = __mva_dec__, predname = __mva_pred__ ):
        '''
        Calculate the values for the train and test samples. Apply
        to the full samples, and then split (save computational
        time).

        See :meth:`MVAmgr.apply`.
        '''
        decs  = []
        preds = []

        smp = sample[self.features + self.extravars()]

        for i, mva in enumerate(self.mvas):

            s = smp[self._true_cond(smp, i)][self.features]

            d, p = self._process(mva, s)

            decs.append(d)
            preds.append(p)

        sample[decname]  = pandas.concat(decs)
        sample[predname] = pandas.concat(preds)

    def apply_for_overtraining( self, dt, sample, decname = __mva_dec__, predname = __mva_pred__ ):
        '''
        In this case, the mean of the values of the BDT is taken.

        See :meth:`MVAmgr.apply_for_overtraining`.
        '''
        if dt not in ('train', 'test'):
            raise RuntimeError('Unknown data type "{}"'.format(dt))

        if dt == 'test':
            self.apply(sample, decname, predname)
        else:
            dec_df  = pandas.DataFrame()
            pred_df = pandas.DataFrame()

            info('Processing a training sample, this may '\
                'take a while')

            smp = sample[self.features + self.extravars()]

            for i, mva in enumerate(self.mvas):

                info('Processing MVA number {}'.format(i), indent=1)

                s = smp[self._false_cond(smp, i)][self.features]

                d, p = self._process(mva, s)

                # In the cells corresponding to the test values,
                # it is being filled with NaN. By default the
                # mean is calculated skipping these numbers.
                dec_df  = pandas.concat(
                    [dec_df, d], axis=1, ignore_index=True)
                pred_df = pandas.concat(
                    [pred_df, p], axis=1, ignore_index=True)

            info('Calculating mean of BDT values')

            dm = dec_df.mean(axis=1)
            pm = pred_df.mean(axis=1)

            # Study the deviation of the MVA values
            maximum = dec_df.subtract(dm, axis=0).divide(
                dm, axis = 0).abs().max(axis=1)

            nmax = len(maximum[maximum >
                               KFoldMVAmgr.__min_tolerance__])

            if nmax > 1:
                warnings.warn('Found {} points (out of {}) with '\
                    'a deviation on the MVA value greater than {} '\
                    '%'.format(nmax, len(sample),
                               KFoldMVAmgr.__min_tolerance__*100))

            sample[decname]  = dm
            sample[predname] = pm

    def extravars( self ):
        '''
        See :meth:`MVAmgr.extravars`.
        '''
        return [self.splitvar]

    def fit( self, sig, bkg, is_sig ):
        '''
        See :meth:`MVAmgr.fit`.
        '''
        train_dlst = []
        test_dlst  = []

        mvas = []
        for i in range(self.nfolds):

            info('Processing fold number {}'.format(i + 1), indent=1)

            info('Split signal sample', indent=2)
            train_sig = self.train_sample(sig, i)

            info('Split background sample', indent=2)
            train_bkg = self.train_sample(bkg, i)

            info('Merge training samples', indent=2)
            train_data = pandas.concat([train_sig, train_bkg],
                                       ignore_index=True)

            # The MVA must be a new instance of the classifier type
            mva = deepcopy(self._fit(train_data, is_sig))

            mvas.append(mva)

        train_data = pandas.concat([sig, bkg], ignore_index=True)
        test_data  = train_data.copy()

        self.mvas = mvas

        return train_data, test_data

    def test_sample( self, smp, i ):
        '''
        :param smp: input sample.
        :type smp: pandas.DataFrame
        :param i: fold number.
        :type i: int
        :returns: subsample satisfying the testing condition.
        :rtype: pandas.DataFrame
        '''
        return smp[self._true_cond(smp, i)]

    def train_sample( self, smp, i ):
        '''
        :param smp: input sample.
        :type smp: pandas.DataFrame
        :param i: fold number.
        :type i: int
        :returns: subsample satisfying the training  condition.
        :rtype: pandas.DataFrame
        '''
        return smp[self._false_cond(smp, i)]


class StdMVAmgr(MVAmgr):
    '''
    Manager that uses the standard procedure of splitting the
    samples given the training and testing fractions.
    '''
    def __init__( self, classifier, features, sigtrainfrac = 0.75, bkgtrainfrac = 0.75 ):
        '''
        See :meth:`MVAmgr.__init__`.

        :param sigtrainfrac: fraction of signal events used for training.
        :type sigtrainfrac: float
        :param bkgtrainfrac: fraction of background events used for training.
        :type bkgtrainfrac: float
        '''
        MVAmgr.__init__(self, classifier, features )

        self.sigtrainfrac = sigtrainfrac
        self.bkgtrainfrac = bkgtrainfrac

    def apply( self, sample, decname = __mva_dec__, predname = __mva_pred__ ):
        '''
        Calculate the MVA method values for the given sample.

        See :meth:`MVAmgr.apply`.
        '''
        smp = sample[self.features]

        dec, pred = self._process(self.mva, smp)

        sample[decname]  = dec
        sample[predname] = pred

    def fit( self, sig, bkg, is_sig ):
        '''
        See :meth:`MVAmgr.fit`.
        '''
        info('Divide data in train and test samples')

        info('Signal train fraction: {}'.format(self.sigtrainfrac), indent=1)
        train_sig, test_sig = train_test_split(sig, train_size=self.sigtrainfrac)
        info('Background train fraction: {}'.format(self.bkgtrainfrac), indent=1)
        train_bkg, test_bkg = train_test_split(bkg, train_size=self.bkgtrainfrac)

        info('Merging training and test samples')
        train_data = pandas.concat([train_sig, train_bkg], ignore_index=True)
        test_data  = pandas.concat([test_sig, test_bkg], ignore_index=True)

        self.mva = deepcopy(self._fit(train_data, is_sig))

        return train_data, test_data


def mva_study( name, signame, sigsmp, bkgname, bkgsmp, cfg,
               outdir = '.',
               is_sig = __is_sig__ ):
    '''
    Main function to perform a MVA study. The results are stored
    in three different files: one storing the histograms and the
    ROC curve, another with the configuration used to run this
    function, and the last stores the proper class to store the
    MVA algorithm.

    :param name: name of the study.
    :type name: str
    :param signame: signal sample name.
    :type signame: str
    :param sigsmp: signal sample.
    :type sigsmp: pandas.DataFrame
    :param bkgname: background sample name.
    :type bkgname: str
    :param bkgsmp: background sample.
    :type bkgsmp: pandas.DataFrame
    :param cfg: configurable for the MVA manager.
    :type cfg: ConfMgr or dict
    :param outdir: output directory.
    :type outdir: str
    :param is_sig: name for the additional column holding the \
    signal condition.
    :type is_sig: str
    :returns: MVA manager, training and testing samples and the configuration ID.
    :rtype: tuple(MVAmgr, pandas.DataFrame, pandas.DataFrame, int)
    '''
    cfg = confmgr.ConfMgr(**{__manager_name__ : cfg})

    # Get the raw configuration from the inputs
    for k, v in (
            ('signame', signame),
            ('bkgname', bkgname),
            ('outdir' , outdir)
            ):
        cfg[k] = v

    # Get the manager
    mgr = cfg.proc_conf()[__manager_name__]

    # Create the output directory
    mva_dir = '{}/mva_configs_{}'.format(outdir, name)
    _aux._makedirs(mva_dir)

    # Get the available configuration ID
    cfglst  = confmgr.get_configurations(mva_dir, 'mva_config')
    conf_id = config.available_configuration(cfglst)

    # Check if any other file is storing the same configuration
    matches = confmgr.check_configurations(
        cfg, cfglst, {'funcfile': None, 'confid': None}
    )

    conf_id  = config.manage_config_matches(matches, conf_id)
    cfg_path = '{}/mva_config_{}.xml'.format(mva_dir, conf_id)

    # Save the configuration ID
    cfg['confid'] = str(conf_id)

    # Path to the file storing the MVA function
    func_path = '{}/mva_func_{}.pkl'.format(mva_dir, conf_id)
    cfg['funcfile'] = func_path

    # Generating the XML file must be the last thing to do
    cfg.save(cfg_path)

    # Display the configuration to run
    print('*************************')
    print('*** MVA configuration ***')
    print('*************************')
    print(cfg)
    print('*************************')

    # Add the signal flag
    info('Adding the signal flag')

    sigsmp = sigsmp.copy()
    sigsmp[is_sig] = __sig_flag__

    bkgsmp = bkgsmp.copy()
    bkgsmp[is_sig] = __bkg_flag__

    # Train the MVA method
    info('Initialize training')
    train, test = mgr.fit(sigsmp, bkgsmp, is_sig)

    # Save the output method(s)
    mgr.save(func_path)

    # Apply the MVA method
    info('Apply the trained MVA algorithm')
    for tp, smp in (('train', train), ('test', test)):
        mgr.apply_for_overtraining(tp, smp)

    info('Process finished!')

    return mgr, train, test, conf_id
