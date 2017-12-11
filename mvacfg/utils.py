'''
Module to perform trainings of MVA methods and store
their configuration.
'''

__author__ = 'Miguel Ramos Pernas'
__email__  = 'miguel.ramos.pernas@cern.ch'


# Python
import os, joblib, pandas
import numpy as np
from copy import deepcopy

# Scikit-learn
from sklearn.model_selection import train_test_split

# Local
import mvacfg.config as config


__all__ = ['KFoldMVAmgr', 'StdMVAmgr',
           'KFoldMVAmethod', 'StdMVAmethod',
           'mva_study']

# Global names for the MVA outputs
__mva_dec__  = 'mva_dec'
__mva_pred__ = 'mva_pred'


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

    def _fit( self, train_data, issig ):
        '''
        Fit the MVA classifier to the given samples.
        
        :param train_data: training data.
        :type train_data: pandas.DataFrame
        :param issig: flag to determine the signal condition.
        :type issig: str
        :returns: trained MVA classifier.
        :rtype: MVA classifier
        '''        
        print '---- Perform MVA training'
        mva = self.classifier.fit(train_data[self.features],
                                  train_data[issig])
        print '---- MVA training finished'
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
        dec  = pandas.DataFrame(mva.decision_function(smp),
                                index = smp.index)
        pred = pandas.DataFrame(mva.predict(smp),
                                index = smp.index)
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
            raise 'ERROR: Unknown data type "{}"'.format(dt)
        
        return self.apply(sample, decname, predname)

    def extravars( self ):
        '''
        :returns: extra variables needed for the class.
        :rtype: list
        '''
        return []

    def fit( self, sig, bkg, issig ):
        '''
        Fit the MVA classifier to the given sample.

        :param sig: signal sample.
        :type sig: pandas.DataFrame
        :param bkg: background sample.
        :type bkg: pandas.DataFrame
        :param issig: signal flag.
        :type issig: str
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
        joblib.dump(self, path, compress = True)
        print '-- Output method saved in {}'.format(path)


class KFoldMVAmgr(MVAmgr):
    '''
    Manager that uses the k-folding technique, splitting the total
    sample in "n" folds, using "n - 1" to train the remaining
    fold, which is used as a test sample.
    '''
    __min_tolerance__ = 0.1

    def __init__( self, classifier, features, nfolds, splitvar ):
        '''
        Constructor given the classifier, features, number of
        folds and the integer variable used to split the sample.

        :param classifier: classifier to use.
        :type classifier: MVA classifier.
        :param features: variables for training.
        :type features: list of str
        :param nfolds: number of folds to generate.
        :type nfolds: int
        :param splitvar: variable used to split the sample in folds.
        :type splitvar: str
        '''
        MVAmgr.__init__(self, classifier, features)
        
        if nfolds <= 1:
            raise 'ERROR: Number of folds must be greater than one'
        
        self.nfolds   = nfolds
        self.splitvar = splitvar

        print '-- Prepare {} folds using variable '\
            '"{}"'.format(nfolds, splitvar)

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
            raise 'ERROR: Unknown data type "{}"'.format(dt)
      
        if dt == 'test':
            self.apply(sample, decname, predname)
        else:
            dec_df  = pandas.DataFrame()
            pred_df = pandas.DataFrame()

            print '-- Processing a training sample, this may '\
                'take a while'

            smp = sample[self.features + self.extravars()]

            for i, mva in enumerate(self.mvas):

                print '---- Processing MVA number {}'.format(i)
                
                s = smp[self._false_cond(smp, i)][self.features]
                
                d, p = self._process(mva, s)
                
                # In the cells corresponding to the test values,
                # it is being filled with NaN. By default the
                # mean is calculated skipping these numbers.
                dec_df  = pandas.concat(
                    [dec_df, d], axis = 1, ignore_index = True)
                pred_df = pandas.concat(
                    [pred_df, p], axis = 1, ignore_index = True)
                
            print '---- Calculating mean of BDT values'
            
            dm = dec_df.mean(axis = 1)
            pm = pred_df.mean(axis = 1)
            
            # Study the deviation of the MVA values
            maximum = dec_df.subtract(dm, axis = 0).divide(
                dm, axis = 0).abs().max(axis = 1)
            
            nmax = len(maximum[maximum > 
                               KFoldMVAmgr.__min_tolerance__])

            if nmax > 1:
                print 'WARNING: Found {} points (out of {}) with '\
                    'a deviation on the MVA value greater than {} '\
                    '%'.format(nmax, len(sample),
                               KFoldMVAmgr.__min_tolerance__*100)

            sample[decname]  = dm
            sample[predname] = pm
            
    def extravars( self ):
        '''
        See :meth:`MVAmgr.extravars`.
        '''
        return [self.splitvar]

    def fit( self, sig, bkg, issig ):
        '''
        See :meth:`MVAmgr.fit`.
        '''
        train_dlst = []
        test_dlst  = []
        
        mvas = []
        for i in xrange(self.nfolds):
      
            print '--- Processing fold number {}'.format(i + 1)
            
            print '---- Split signal sample'
            train_sig = self.train_sample(sig, i)
                
            print '---- Split background sample'
            train_bkg = self.train_sample(bkg, i)
            
            print '---- Merge training samples'
            train_data = pandas.concat([train_sig, train_bkg],
                                       ignore_index = True)
            
            # The MVA must be a new instance of the classifier type
            mva = deepcopy(self._fit(train_data, issig))
            
            mvas.append(mva)
            
        train_data = pandas.concat([sig, bkg], ignore_index = True)
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
    def __init__( self, classifier, features, sigtrainfrac, bkgtrainfrac ):
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

    def fit( self, sig, bkg, issig ):
        '''
        See :meth:`MVAmgr.fit`.
        '''
        print '---- Divide data in train and test samples'

        print '---- Signal train fraction: {}'.format(self.sigtrainfrac)
        train_sig, test_sig = train_test_split(sig, random_state = 11, train_size = self.sigtrainfrac)
        print '---- Background train fraction: {}'.format(self.bkgtrainfrac)
        train_bkg, test_bkg = train_test_split(bkg, random_state = 11, train_size = self.bkgtrainfrac)
      
        print '---- Merging training and test samples'
        train_data = pandas.concat([train_sig, train_bkg], ignore_index = True)
        test_data  = pandas.concat([test_sig, test_bkg], ignore_index = True)
        
        self.mva = deepcopy(self._fit(train_data, issig))
        
        return train_data, test_data


class MVAmethod:
    '''
    Base class to define the method to be used for the MVA analysis.
    '''
    def __init__( self, const, cfg ):
        '''
        :param const: MVA constructor from the configuration.
        :type const: scikit-learn MVA constructor
        :param cfg: configuration.
        :type cfg: dict
        '''
        self.config = cfg
        self.const  = const

    def __dict__( self ):
        '''
        :returns: configuration of this class.
        :rtype: dict
        '''
        return self.config
        
    def build_mgr( self, classifier, features ):
        '''
        Build a manager with the given MVA classifier.
        
        :param classifier: MVA classifier.
        :type classifier: scikit-learn classifier
        :param features: features of the classifier.
        :type features: list
        :returns: MVA built object.
        :rtype: MVA classifier.
        '''  
        return self.const(classifier, features, **self.config)


class KFoldMVAmethod(MVAmethod):
    '''
    Define the k-folding method.
    '''
    def __init__( self, nfolds, splitvar ):
        '''
        Number of folds and the variable to do the splitting must
        be provided.

        :param nfolds: number of folds to use.
        :type nfolds: int
        :param splitvar: variable to split in folds.
        :type spitvar: str
        '''
        cfg = {'nfolds': nfolds, 'splitvar': splitvar}
        
        MVAmethod.__init__(self, KFoldMVAmgr, cfg)


class StdMVAmethod(MVAmethod):
    '''
    Define the standard method.
    '''
    def __init__( self, sigtrainfrac = 0.7, bkgtrainfrac = 0.7 ):
        '''
        The training fractions for signal and background must
        be provided.

        :param sigtrainfrac: fraction of signal events used for \
        training.
        :type sigtrainfrac: float
        :param bkgtrainfrac: fraction of background events used \
        for training.
        :type bkgtrainfrac: float
        '''
        cfg = {
            'sigtrainfrac': bkgtrainfrac,
            'bkgtrainfrac': bkgtrainfrac
            }
        
        MVAmethod.__init__(self, StdMVAmgr, cfg)


def mva_study( name, signame, sigsmp, bkgname, bkgsmp, features, mvatype,
               mvaconfig  = None,
               outdir     = '.',
               methconfig = None,
               is_sig     = 'is_sig' ):
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
    :param features: list of variables to use.
    :type features: list of str or None
    :param mvatype: type of MVA to execute.
    :type mvatype: MVA classifier
    :param mvaconfig: configuration of the MVA.
    :type mvaconfig: dict
    :param outdir: output directory.
    :type outdir: str
    :param methconfig: configuration of the MVA method.
    :type methconfig: dict
    :param is_sig: name for the additional column holding the \
    signal condition.
    :type is_sig: str
    :returns: MVA manager, training and testing samples.
    :rtype: tuple(MVAmgr, pandas.DataFrame, pandas.DataFrame)
    '''
    methconfig = methconfig if methconfig else MVAmethod('std')
    
    # Define the manager
    mgr = methconfig.build_mgr(mvatype(**mvaconfig), features)
    
    # Create the output directory
    mva_dir = '{}/mva_configs_{}'.format(outdir, name)
    config._makedirs(mva_dir)
    
    # Get the available configuration ID
    flst    = config._get_configurations(mva_dir, 'mva_config')
    conf_id = config._available_configuration(flst)
    
    # Get the raw configuration from the inputs
    cfg = {
        'signame'    : signame,
        'bkgname'    : bkgname,
        'features'   : str(list(features)),
        'manager'    : mgr,
        'outdir'     : outdir
    }
    
    cfg = config.genconfig(name, cfg)
    
    # Check if any other file is storing the same configuration
    matches  = config.check_configurations(
        cfg, flst, {config.__main_config_name__: ['funcfile', 'confid']}
    )
    conf_id  = config._manage_config_matches(matches, conf_id)
    cfg_path = '{}/mva_config_{}.ini'.format(mva_dir, conf_id)
    
    # Save the configuration ID
    cfg.set(config.__main_config_name__, 'confid', str(conf_id))
    
    # Path to the file storing the MVA function
    func_path = '{}/mva_func_{}.pkl'.format(mva_dir, conf_id)
    cfg.set(config.__main_config_name__, 'funcfile', func_path)
    
    # Display the configuration to run
    print '*************************'
    print '*** MVA configuration ***'
    print '*************************'
    config.print_configuration(cfg)
    print '*************************'
    
    # Generating the INI file must be the last thing to do
    config.save_config(cfg, cfg_path)
    
    # Add the signal flag
    print '-- Adding the signal flag'

    sigsmp = sigsmp.copy()
    sigsmp[is_sig] = True

    bkgsmp = bkgsmp.copy()
    bkgsmp[is_sig] = False
    
    # Train the MVA method
    print '-- Initialize training'
    train, test = mgr.fit(sigsmp, bkgsmp, is_sig)
    
    # Save the output method(s)
    mgr.save(func_path)
    
    # Apply the MVA method
    print '-- Apply the trained MVA algorithm'
    for tp, smp in (('train', train), ('test', test)):
        mgr.apply_for_overtraining(tp, smp)
        
    print '-- Process finished!'
    
    return mgr, train, test
