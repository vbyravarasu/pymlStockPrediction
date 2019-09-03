import numpy as np
import pandas as pd

class SampleWeight(object):
    
    def __init__(self, central_feat, stock_cols):
        assert isinstance(central_feat, pd.DataFrame)
        assert central_feat.shape[0] == 1, central_feat.shape
        self.central_feat = central_feat
        self.stock_cols = stock_cols
    
    def weight(self, x_sub, central_feat, cols):
        return 1. / (1 + sum([abs(x_sub[c] - central_feat[c].values[0]) for c in cols]))
    
    def scale_weight(self, x):
        assert np.all(x.columns == self.central_feat.columns)
        x = x.copy()
        weights = {}
        for col in self.stock_cols:
            cols = [c for c in x.columns if col in c] # stock related features containing the keyword col
            x_sub = x[cols]
            weight = self.weight(x_sub, self.central_feat, cols)
            assert len(weight) == x.shape[0]
            weights[col] = np.array(weight)
        return weights
