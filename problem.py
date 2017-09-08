from __future__ import division, print_function
import os
import numpy as np
import pandas as pd
import xarray as xr
import rampwf as rw
from glob import glob
from os.path import join
from sklearn.model_selection import LeaveOneGroupOut


problem_title = 'California Winter Extreme Rainfall Prediction'
_prediction_label_names = [0, 1]

Predictions = rw.prediction_types.make_multiclass(label_names=_prediction_label_names)
workflow = rw.workflows.FeatureExtractorClassifier()
score_types = [
    rw.score_types.ROCAUC(name="auc"),
    rw.score_types.BrierScore(name="brier_score")
]


def get_cv(X, y):
    groups = np.zeros(y.shape)
    i = 0
    ens_size = int(y.shape[0] / 4)
    for g in range(4):
        groups[i: i + ens_size] = g
        i += ens_size
    cv = LeaveOneGroupOut()
    return cv.split(X, y, groups)

def _read_data(path, f_prefix):
    nc_files = sorted(glob(join(path, f_prefix + "_*.nc")))      
    X_coll = []
    for nc_file in nc_files:
        x_coll.append(xr.open_dataset(nc_file, decode_times=False))
        x_coll[-1].load()
    X_ds = xr.merge(x_coll)
    y = pd.read_csv(join(path, f_prefix + "_precip_90.csv"))
    y_array = np.concatenate([y[c] for c in y.columns])
    return X_ds, y_array


def get_train_data(path='.'):
    return _read_data(path, 'train')


def get_test_data(path='.'):
    return _read_data(path, 'test')
