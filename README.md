# Texture benchmarks


## Dependencies

* numpy
* scipy
* matplotlib
* joblib
* scikit-learn
* scikit-image
* ipcv, http://github.com/andersbll/ipcv


## Setup

You need to specify the following environment variables before executing
benchmark.py.

* CACHE_ROOT
* TEXTURE_DATA_ROOT
* RESULTS_ROOT
* VERBOSITY
* N_THREADS

In Python this can be done with:

    os.environ['CACHE_ROOT'] = '../cache'
    os.environ['TEXTURE_DATA_ROOT'] = '../data'
    os.environ['RESULTS_ROOT'] = '../results'
    os.environ['VERBOSITY'] = '10'
    os.environ['N_THREADS'] = '8'
