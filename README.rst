mvacfg
======

Package to handle configurations for Multivariate Analysis
(MVA) using scikit-learn. It provides:

- Functions to correctly read and save configurations in INI format.
- A function to train MVA methods, returning the training and testing samples, together with a class to apply the output algorithm in any sample.

The format of the input samples are pandas.DataFrame objects.

.. toctree::
   :maxdepth: 2

   config
   plot
   utils

Examples
========

.. automodule:: mvacfg.examples
