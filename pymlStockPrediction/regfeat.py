import gc
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

class RegressOnePeriod(object):
    
    def __init__(self, df, lo_freq_start, lo_freq_end, lo_freq_col, hi_freq_col):
        df = df.copy()
        df = df[df[lo_freq_col] > lo_freq_start]
        df = df[df[lo_freq_col] <= lo_freq_end]
        if hi_freq_col is not None:
            self.df = df.drop([lo_freq_col, hi_freq_col], axis=1)
        else:
            self.df = df.drop(lo_freq_col, axis=1)
        self.lo_freq_start = lo_freq_start
        self.lo_freq_end = lo_freq_end
        self.lo_freq_col = lo_freq_col
        self.hi_freq_col = hi_freq_col

    def regress(self, x_cols_lst, y_cols, insample=True):
        df = self.df.copy()
        ret = df.copy()
        for x_cols, y_col in zip(x_cols_lst, y_cols):
            if insample:
                y = df[y_col]
            else:
                y = df[y_col].shift(-1).fillna(method='ffill')
            
            x = df[x_cols]
            reg = LinearRegression(fit_intercept=True) # if no intercept R2 can be negative, not true indication
            reg.fit(x, y)
            pred = reg.predict(x)
            ret['%s_regPred' % y_col] = pred
            ret['%s_regR2score' % y_col] = reg.score(x, y)
            ret['%s_mae' % y_col] = mean_absolute_error(y, pred)
            
            err = pred - y
            adf = adfuller(err, maxlag=1, regression='c', autolag=None)
            ret['%s_adfStat' % y_col] = adf[0]
        return ret

class RegressAllPeriod(object):
    
    def __init__(self, df, span, lo_freq_col, hi_freq_col):
        """All columns but lo/hi freq should be stocks.
        """
        self.df = df.copy()
        self.span = span
        self.lo_freq_col = lo_freq_col
        self.hi_freq_col = hi_freq_col
    
    def regress(self, x_cols_lst, y_cols, insample=True):
        df = self.df.copy()
        ret = pd.DataFrame()
        intervals = list(df[self.lo_freq_col].values[0] + range(-1, df[self.lo_freq_col].nunique(), self.span))
        last_day = df[self.lo_freq_col].values[-1]
        if intervals[-1] != last_day:
            intervals.append(last_day)
        print(intervals)
        
        for i in range(len(intervals) - 1):
            rt = RegressOnePeriod(df, intervals[i], intervals[i + 1], self.lo_freq_col, self.hi_freq_col)
            ret = pd.concat([ret, rt.regress(x_cols_lst, y_cols, insample=insample)], axis=0)
        assert ret.shape[0] == df.shape[0]
        assert ret.index[0] == df.index[0]
        assert ret.index[-1] == df.index[-1]
        
        ret[[c for c in ret.columns if 'adfStat' in c]].plot(figsize=(15, 8), alpha=.6)
        plt.show()
        ret[[c for c in ret.columns if 'regPred' in c]].plot(figsize=(15, 8), alpha=.6)
        plt.show()
        ret[[c for c in ret.columns if 'regR2score' in c]].plot(figsize=(15, 8), alpha=.6)
        plt.show()
        ret[[c for c in ret.columns if 'mae' in c]].plot(figsize=(15, 8), alpha=.6)
        plt.show()
        plt.close()
        return ret
    
class RegressAllSpan(object):
    
    def __init__(self, df, spans, lo_freq_col, hi_freq_col, x_cols_lst, y_cols):
        self.df = df.copy()
        self.spans = spans
        self.lo_freq_col = lo_freq_col
        self.hi_freq_col = hi_freq_col
        self.x_cols_lst = x_cols_lst
        self.y_cols = y_cols
        self.reg_dfs = {}
    
    def regress(self, insample=True):
        for span in self.spans:
            self.reg_dfs[span] = RegressAllPeriod(self.df, span, self.lo_freq_col, self.hi_freq_col).regress(self.x_cols_lst, 
                                                                                                             self.y_cols, 
                                                                                                             insample=insample)
            gc.collect()
        return
    
    def regressFeature(self):
        ret = pd.DataFrame(index=self.reg_dfs[self.spans[0]].index)
        for span in self.spans:
            df = self.reg_dfs[span].copy()
            subtitle = str(span)
            cols = [c for c in df.columns if 'regPred' in c]
            subdf = df.loc[:, cols]
            for col in self.y_cols:
                subdf['%s_regScore' % col] = df['%s_adfStat' % col].abs() * df['%s_regR2score' % col] / (1. + df['%s_mae' % col])
                subdf['%s_mae' % col] = df['%s_mae' % col]
            subdf.columns = ['%s_%d' % (c, span) for c in subdf.columns]
            ret = pd.concat([ret, subdf], axis=1)
        gc.collect()

        for s in self.y_cols:
            col = '%s_weightedPred' % s
            ret[col] = sum([ret['%s_regScore_%d' % (s, span)] * ret['%s_regPred_%d' % (s, span)] \
                            for span in self.spans])
            ret[col] /= 1. * sum([ret['%s_regScore_%d' % (s, span)] for span in self.spans])
        ret = ret[['%s_weightedPred' % s for s in self.y_cols]]
        ret.plot(figsize=(15, 8), alpha=.6)
        plt.show()
        return ret
    