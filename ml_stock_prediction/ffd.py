import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

class FFD(object):
    
    def __init__(self, df, logscale=True, thres=1e-2):
        """All columns of df should be stocks.
        """
        self.df = df.copy().reset_index(drop=True)
        if logscale:
            self.df = np.log(self.df)
        self.thres = thres
    
    def getWeights(self, d):
        w, k = [1.], 1
        while True:
            w_ = - w[-1] / k * (d - k + 1)
            if abs(w_) < self.thres:
                break
            w.append(w_)
            k += 1
        w = np.array(w[::-1]).reshape(-1, 1)
        return w

    def fracDiff(self, d, fpath=None):
        """Times are in index rather than as columns. All columns are fractionally differentiated.
        Returned df's index is offset by len(w) - 1.
        """
        assert isinstance(d, float) or isinstance(d, int)
        df = self.df.copy()
        w = self.getWeights(d)
        width, ret = len(w) - 1, {}
        for col in df.columns:
            subdf, df_ = df[[col]].fillna(method='ffill').dropna(), {}
            for iloc1 in range(width, subdf.shape[0]):
                loc0, loc1 = subdf.index[iloc1 - width], subdf.index[iloc1]
                if not np.isfinite(df.loc[loc1, col]):
                    continue # exclude NAs
                df_[loc1] = np.dot(w.T, subdf.loc[loc0: loc1])[0, 0]
            ret[col] = df_.copy()
        ret = pd.DataFrame(ret)
        if fpath is not None:
            ret.to_csv(fpath, index=True)
        return ret, w, d

    def plotMinFFD(self, col):
        df = self.df[[col]].copy()
        ret = pd.DataFrame(columns=['adfStat', 'pVal', 'lags', 'nObs', '95% conf', 'corr'])
        for d in np.linspace(0, 1, 11):
            df_, _, _ = self.fracDiff(d)
            corr = np.corrcoef(df.loc[df_.index, col], df_[col])[0, 1]
            df_ = adfuller(df_[col], maxlag=1, regression='c', autolag=None)
            ret.loc[d] = list(df_[:4]) + [df_[4]['5%']] + [corr] # with critical value
        ret[['adfStat', 'corr']].plot(secondary_y='adfStat')
        display(ret)
        plt.axhline(ret['95% conf'].mean(), linewidth=1, color='r', linestyle='dotted')
        plt.title(col)
        plt.show()
        plt.close()
        return ret
    
    def plotAllMinFFD(self):
        for col in self.df.columns:
            _ = self.plotMinFFD(col)
        return
    
    def checkFFD(self):
        for col in self.df.columns:
            corr = np.corrcoef(self.df.loc[self.ffd.index, col], self.ffd[col])[0, 1]
            print(corr)
            adf = adfuller(self.ffd[col], maxlag=1, regression='c', autolag=None)
            print(adf)
        return
    