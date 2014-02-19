# Texture benchmarks


## Dependencies

* numpy
* scipy
* joblib
* scikit-learn
* scikit-image
* ipcv, http://github.com/andersbll/ipcv


## Setup

You need to specify the environment variables

* JOBLIB_ROOT
* TEXTURE_DATA_ROOT
* VERBOSITY
* N_THREADS

before executing benchmark.py. In Python this can be with e.g.

    os.environ['JOBLIB_ROOT'] = '../working/joblib'
    os.environ['TEXTURE_DATA_ROOT'] = '../data'
    os.environ['VERBOSITY'] = '10'
    os.environ['N_THREADS'] = '8'
