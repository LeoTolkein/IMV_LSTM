import numpy as np

_EPSILON = 1e-9

class SeriesMinMaxScaler(object):
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.data_min_ = None
        self.data_max_ = None

    def fit(self, X):
        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)

    def transform(self, X):
        # * use _EPSILON to avoid division by zero
        return (X - self.data_min_)/(self.data_max_ - self.data_min_ + _EPSILON)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        return X * (self.data_max_ - self.data_min_) + self.data_min_
