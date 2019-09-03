import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class SampleWeight(object):
    
    def __init__(self, central_feat, stock_cols):
        assert isinstance(central_feat, pd.DataFrame)
        assert central_feat.shape[0] == 1
        self.central_feat = central_feat
        self.stock_cols = stock_cols
        self.scaler = None
    
    def preprocessor(self):
        self.scaler = StandardScaler()
        return self.scaler
    
    def weight(self, x_sc_sub, central_feat_sc, cols):
        return 1. / (1 + sum([abs(x_sc_sub[c] - central_feat_sc[c].values[0]) for c in cols]))
    
    def scale_weight(self, x, fit=True):
        assert np.all(x.columns == self.central_feat.columns)
        x = x.copy().reset_index(drop=True)
        if fit:
            sc = self.preprocessor()
            x_sc = sc.fit_transform(x)
        else:
            x_sc = sc.transform(x)
        x_sc = pd.DataFrame(x_sc, columns=x.columns)
        
        central_feat_sc = sc.transform(central_feat)
        central_feat_sc = pd.DataFrame(central_feat_sc, columns=central_feat.columns)
        
        weights = {}
        for col in self.stock_cols:
            cols = [c for c in x.columns if col in c] # stock related features containing the keyword col
            x_sc_sub = x_sc[cols]
            display(x_sc_sub.head())
            weight = self.weight(x_sc_sub, central_feat_sc, cols)
            assert len(weight) == x_sc.shape[0]
        return weights
