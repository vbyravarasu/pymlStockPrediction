import numpy as np
import matplotlib.pyplot as plt

class TechnicalAnalysis(object):

    @staticmethod
    def bollingerBandIndicator(df, close_col, subtitle, n_period=20, n_std=2, fillna=False, verbose=False):
        """Returns 1 if close is higher than bollinger high band; -1 if lower than low band; 0 otherwise.
        """
        df = df.copy()
        mavg = df[close_col].rolling(n_period).mean()
        mstd = df[close_col].rolling(n_period).std()
        hband = mavg + n_std * mstd
        lband = mavg - n_std * mstd
        col = '%s_bollinger' % subtitle
        df[col] = 0
        df.loc[df[close_col] > hband, col] = 1
        df.loc[df[close_col] < lband, col] = -1
        if fillna:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)
        if verbose:
            df[col].hist(bins=30)
            plt.title(col)
            plt.show()
            plt.close()
        return df

    @staticmethod
    def bollingerBandScaled(df, close_col, subtitle, n_period=20, n_std=2, fillna=False, verbose=False):
        """Scale price between high and low bands.
        """
        df = df.copy()
        mavg = df[close_col].rolling(n_period).mean()
        mstd = df[close_col].rolling(n_period).std()
        hband = mavg + n_std * mstd
        lband = mavg - n_std * mstd
        col = '%s_bollingerScaled' % subtitle
        df[col] = (df[close_col] - mavg) / (n_std * mstd)
        if fillna:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)
        if verbose:
            df[col].hist(bins=30)
            plt.title(col)
            plt.show()
            plt.close()
        return df

    @staticmethod
    def ema(series, periods, fillna=False):
        if fillna:
            return series.ewm(span=periods, min_periods=0).mean()
        return series.ewm(span=periods, min_periods=periods).mean()

    def rsiScaled(self, df, close_col, subtitle, n_period=14, fillna=False):
        """Relative Strength Index (RSI) compares the magnitude of recent gains and losses over a specified 
        time period to measure speed and change of price movements of a security. It is primarily used to
        attempt to identify overbought or oversold conditions in the trading of an asset.
        """
        df = df.copy()
        diff = df[close_col].diff(1)
        which_dn = diff < 0
        up, dn = diff, diff * 0
        up[which_dn], dn[which_dn] = 0, - up[which_dn]
        emaup = self.ema(up, n_period, fillna)
        emadn = self.ema(dn, n_period, fillna)
        col = '%s_rsiScaled' % subtitle
        df[col] = emaup / (emaup + emadn) - 0.5
        if fillna:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(50)
        df[col].hist(bins=30)
        plt.title(col)
        plt.show()
        plt.close()
        return df

    def macdDiff(self, df, close_col, subtitle, n_period_short=12, n_period_long=26, n_period_signal=9, fillna=False):
        """Moving Average Convergence Divergence (MACD) shows the relationship between MACD and MACD Signal.
        """
        df = df.copy()
        emafast = self.ema(df[close_col], n_period_short, fillna)
        emaslow = self.ema(df[close_col], n_period_long, fillna)
        macd = emafast - emaslow
        macdsign = self.ema(macd, n_period_signal, fillna)
        col = '%s_macd' % subtitle
        df[col] = macd - macdsign
        if fillna:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)
        df[col].hist(bins=50)
        plt.title(col)
        plt.show()
        plt.close()
        return df

    def trix(self, df, close_col, subtitle, n_period=15, fillna=False):
        """Trix (TRIX) shows the percent rate of change of a triple exponentially smoothed moving average.
        """
        df = df.copy()
        ema1 = self.ema(df[close_col], n_period, fillna)
        ema2 = self.ema(ema1, n_period, fillna)
        ema3 = self.ema(ema2, n_period, fillna)
        col = '%s_trix' % subtitle
        df[col] = (ema3 - ema3.shift(1).fillna(ema3.mean())) / ema3.shift(1).fillna(ema3.mean())
        df[col] *= 100
        if fillna:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)
        df[col].hist(bins=50)
        plt.title(col)
        plt.show()
        plt.close()
        return df
    
    def vol(self, df, close_col, subtitle, n_period=30, fillna=False):
        df = df.copy()
        col = '%s_vol' % subtitle
        df[col] = df[close_col].rolling(n_period).std()
        hband = df[col].rolling(n_period).max()
        col2 = '%s_hband' % col
        df[col2] = 0
        df.loc[df[col] >= hband, col2] = 1
        if fillna:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)
        df[col].plot(figsize=(20, 6))
        plt.title(col)
        plt.show()
        df[col2].hist(bins=50)
        plt.title(col2)
        plt.show()
        plt.close()
        return df
    