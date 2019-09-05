import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, QuantileTransformer
import matplotlib.pyplot as plt

class SampleWeight(object):
    
    def __init__(self, central_feat):
        assert isinstance(central_feat, pd.DataFrame)
        assert central_feat.shape[0] == 1, central_feat.shape
        self.central_feat = central_feat
        self.sc = QuantileTransformer()
    
    def weight(self, x_sub, central_feat, cols):
        for c in cols:
            s = x_sub[c] - central_feat[c].values[0]
            s /= 1. + np.mean(x_sub[c])
            x_sub[c].plot()
            plt.axhline(central_feat[c].values[0])
            plt.title(c)
            plt.show()
        
        diff = [abs(x_sub[c] - central_feat[c].values[0]) / (1. + np.mean(x_sub[c])) for c in cols]

        n = x_sub.shape[0]
        weight = np.log(2 + np.array([i for i in range(n)])) / (len(cols) * .1 + sum(diff))

        return weight
    
    def scale_weight(self, x, fit=False, cols=None):
        assert np.all(x.columns == self.central_feat.columns)
        if cols is None:
            cols = x.columns
        x_sub = x[cols].copy()
        
        if fit:
            self.sc.fit(x_sub)
            
        x_sub_sc = self.sc.transform(x_sub)
        x_sub_sc = pd.DataFrame(x_sub_sc, columns=cols)
        
        central_sub = self.central_feat[cols].copy()
        central_feat_sc = self.sc.transform(central_sub)
        central_feat_sc = pd.DataFrame(central_feat_sc, columns=cols)
        
        weight = self.weight(x_sub_sc, central_feat_sc, cols) 
        return weight
