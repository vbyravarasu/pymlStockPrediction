import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

class RegressOnePeriod(object):
    
    def __init__(self, df, lo_freq_start, lo_freq_end, lo_freq_col, hi_freq_col):
        """All columns but lo/hi freq should be stocks.
        """
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

    def regress(self, stock_cols_len=6):
        df = self.df.copy()
        ret = df.copy()
        for col in df.columns:
            other_cols = [c for c in df.columns if c != col]
            assert len(other_cols) == stock_cols_len - 1, (other_cols, df.columns)
            y = df[col]
            x = df[other_cols]
            reg = LinearRegression(fit_intercept=True) # if no intercept R2 can be negative, not true indication
            reg.fit(x, y)
            pred = reg.predict(x)
            ret['%s_regPred' % col] = pred
            ret['%s_regR2score' % col] = reg.score(x, y)
            ret['%s_mae' % col] = mean_absolute_error(y, pred)
            
            err = pred - y
            adf = adfuller(err, maxlag=1, regression='c', autolag=None)
            ret['%s_adfStat' % col] = adf[0]

            for coef, other in zip(reg.coef_, other_cols):
                ret['weightOf_%s_for_%s' % (other, col)] = coef
        return ret

class RegressAllPeriod(object):
    
    def __init__(self, df, span, lo_freq_col, hi_freq_col):
        """All columns but lo/hi freq should be stocks.
        """
        self.df = df.copy()
        self.span = span
        self.lo_freq_col = lo_freq_col
        self.hi_freq_col = hi_freq_col
    
    def regress(self):
        df = self.df.copy()
        ret = pd.DataFrame()
        intervals = list(df[self.lo_freq_col].values[0] + range(-1, df[self.lo_freq_col].nunique(), self.span))
        last_day = df[self.lo_freq_col].values[-1]
        if intervals[-1] != last_day:
            intervals.append(last_day)
        print(intervals)
        
        for i in range(len(intervals) - 1):
            rt = RegressOnePeriod(df, intervals[i], intervals[i + 1], self.lo_freq_col, self.hi_freq_col)
            ret = pd.concat([ret, rt.regress()], axis=0)
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
    
    def __init__(self, df, spans, lo_freq_col, hi_freq_col):
        """All columns but lo/hi freq should be stocks.
        """
        self.df = df.copy()
        self.spans = spans
        self.lo_freq_col = lo_freq_col
        self.hi_freq_col = hi_freq_col
        self.stock_cols = [c for c in self.df.columns if c not in [lo_freq_col, hi_freq_col]]
        self.reg_dfs = {}
    
    def regress(self):
        for span in self.spans:
            self.reg_dfs[span] = RegressAllPeriod(self.df, span, self.lo_freq_col, self.hi_freq_col).regress()
        return
    
    def regressFeature(self):
        ret = pd.DataFrame(index=self.reg_dfs[self.spans[0]].index)
        for span in self.spans:
            df = self.reg_dfs[span].copy()
            subtitle = str(span)
            cols = [c for c in df.columns if 'regPred' in c]
            subdf = df.loc[:, cols]
            for col in self.stock_cols:
                subdf['%s_regScore' % col] = df['%s_adfStat' % col].abs() * df['%s_regR2score' % col] * np.exp(- df['%s_mae' % col])
            subdf.columns = ['%s_%d' % (c, span) for c in subdf.columns]
            print(subdf.columns)
            ret = pd.concat([ret, subdf], axis=1)

        for s in self.stock_cols:
            col = '%s_weightedPred' % s
            ret[col] = sum([ret['%s_regScore_%d' % (s, span)] * ret['%s_regPred_%d' % (s, span)] \
                            for span in self.spans])
            ret[col] /= 1. * sum([ret['%s_regScore_%d' % (s, span)] for span in self.spans])
        ret = ret[['%s_weightedPred' % s for s in self.stock_cols]]
        ret.plot(figsize=(15, 8), alpha=.6)
        plt.show()
        return ret
    