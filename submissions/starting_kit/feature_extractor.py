import xarray as xr
import pandas as pd
from sklearn.decomposition import PCA

class FeatureExtractor():
    def __init__(self):
        pass
    
    def fit(self, X_ds, y):
        pass

    def transform(self, X_ds):
        variables = ["TS", "PSL", "TMQ"]
        X = None
        return X
