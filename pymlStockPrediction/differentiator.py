from itertools import combinations, product
import matplotlib.pyplot as plt

class Differentiator(object):
    
    @staticmethod
    def pairwise(df, cols):
        df = df.copy()
        for col1, col2 in combinations(cols, 2):
            name = '%s_minus_%s' % (col1, col2)
            df[name] = df[col1] - df[col2]
            df[[name, col1, col2]].plot(secondary_y=name)
            plt.show()
            plt.close()
        return df
    
    @staticmethod
    def diff(df, cols, d=1):
        df = df.copy()
        for col in cols:
            name = '%s_diff%d' % (col, d)
            df[name] = df[col].diff(d)
            df[[name, col]].plot(secondary_y=name)
            plt.show()
            plt.close()
        return df
    
    @staticmethod
    def ema(series, periods, fillna=False):
        if fillna:
            return series.ewm(span=periods, min_periods=0).mean()
        return series.ewm(span=periods, min_periods=periods).mean()

    def lessEwm(self, df, cols, periods=20, fillna=False):
        df = df.copy()
        for col in cols:
            name1 = '%s_less_ewm%d' % (col, periods)
            df[name] = df[col] - self.ema(df[col], periods, fillna)
            df[[name, col]].plot(secondary_y=name)
            plt.show()
            plt.close()
        return df
    
    def pairwiseLessEwm(self, df, cols, periods=20, fillna=False):
        df = df.copy()
        for col1, col2 in product(cols, cols):
            name = '%s_less_%s_ewm%d' % (col1, col2, periods)
            df[name] = df[col1] - self.ema(df[col2], periods, fillna)
            df[[name, col1, col2]].plot(secondary_y=name)
            plt.show()
            plt.close()
        return df
            