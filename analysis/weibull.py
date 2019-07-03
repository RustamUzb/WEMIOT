import logging, sys

import analysis.lme as alme

import scipy.stats as ss
import scipy.special as ssp
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from lifelines import WeibullFitter


np.seterr(divide='ignore', invalid='ignore')
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARN)
#logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

logger = logging.getLogger('matplotlib')
# set WARNING for Matplotlib
logger.setLevel(logging.WARNING)

"""
matplotlib ticks: convert lnln  to weibull cdf 
"""
def weibull_ticks(y, pos):
    return "{:.0f}%".format(100 * (1 - np.exp(-np.exp(y))))



class Fit:
    def __init__(self,  dataframe, time: str = 'times', is_failed: str = 'is_failure', lb: float=0.1, fit_type: str = None):

        sns.set(style="darkgrid")
        self._beta = None
        self._eta = None
        #sample population evaluation
        self._spe = None
        self._total_observations = None
        self._total_failures = None
        self._conf_bound = None
        self._r_value = None

        self._conf_bound = lb
        dat = pd.DataFrame({'times': dataframe[time], 'is_failure': dataframe[is_failed]})
        dat['indx'] = dat['times']
        dat.set_index('indx', inplace=True)
        dat.sort_values('times', inplace=True)

        self._total_failures = dat[dat['is_failure'] == 1].count()['is_failure']
        self._total_observations = dat.count()['is_failure']

        self._data = dat

        if self._total_failures < 30:
            self._linear_regression()
        else:
            self._lme()

        # self._plot_weibull_cdf()

    def _linear_regression(self):

        self._data['rank'] = np.arange(1, len(self._data) + 1)
        self._data['reverse_rank'] = self._data['rank'].values[::-1]

        for index, row in self._data.iterrows():
            if row['rank'] == 1:
                prev_aj_rank = 0

            # 1 == failure ; 0 == suspended
            if row['is_failure'] == 1:
                i = ((row['reverse_rank'] * prev_aj_rank) + (len(self._data) + 1)) / (row['reverse_rank'] + 1)
                self._data.loc[index, 'new_increment'] = i
                prev_aj_rank = i

        self._data['cdf'] = (self._data['new_increment'] - 0.3) / (len(self._data) + 0.4)

        self._data['l_bound'] = ss.beta.ppf(self._conf_bound, self._data['new_increment'],
                                           self._total_observations - self._data['new_increment'] + 1)
        self._data['h_bound'] = ss.beta.ppf(1 - self._conf_bound, self._data['new_increment'],
                                           self._total_observations - self._data['new_increment'] + 1)

        x0 = np.log(self._data.dropna()['times'].values)
        y = np.log(-np.log(1.0 - np.asarray(self._data.dropna()['cdf'])))

        slope, intercept, r, p_value, std_err = ss.linregress(y, x0)

        self._beta = 1.0 / slope
        self._r_value = r
        x_intercept = - intercept / self._beta
        self._eta = np.exp(-x_intercept / slope)

    def _lme(self):
        fail = None
        cens = None
        self._beta, self._eta = alme.lme_weibull(fail, cens)

    def plot_weibull_cdf(self):

        x_ideal = self._eta * np.random.weibull(self._beta, size=1000)
        x_ideal.sort()
        f = 1 - np.exp(-(x_ideal / self._eta) ** self._beta)
        x_ideal = x_ideal[f > 0.01]  # take f > 1%
        f = 1 - np.exp(-(x_ideal / self._eta) ** self._beta)
        x_ideal = x_ideal[f < 0.99]  # take f < 99%
        f = f[f < 0.99]
        y_ideal = np.log(-np.log(1 - f))

        plt.semilogx(x_ideal, y_ideal, label="beta: {:.02f}\neta: {:.01f}".format(self._beta, self._eta))

        logging.warning(self._data.loc[self._data['is_failure'] == 1, ['times']].shape)
        logging.warning(self._data.dropna()['cdf'].shape)
        #logging.warning(self._data[:3])

        plt.semilogx(self._data.loc[self._data['is_failure'] == 1, ['times']],
                     np.log(-np.log(1.0 - np.asarray(self._data.loc[self._data['is_failure'] == 1, ['cdf']]))), 'o')

        ax = plt.gca()
        formatter = mpl.ticker.FuncFormatter(weibull_ticks)
        ax.yaxis.set_major_formatter(formatter)
        yt_f = np.array([0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
                         0.6, 0.7, 0.8, 0.9, 0.95, 0.99])

        plt.yticks(np.log(-np.log(1 - np.asarray(yt_f))))
        plt.ylim(np.log(-np.log(1 - np.asarray([.01, .99]))))

        plt.title("Weibull Plot")
        plt.legend(loc=2)

        plt.show(block=True)

    def b(self, x: float=0.1):
        """
        Calculates time when probability of failure reach specific level x. for example if x = 10, the function will return
        time (duration) when 0.1 of the population will fail. Default value = 10%

        :param x: probability of failure
        :return: time or cycles number when probability of failure equals to x
        """
        if not 0 < x < 1.0:
            raise ValueError('level of failure must be between  0.01 and 0.99 (inclusive)')

        # eta * root(beta, -log(1-F(t))
        _log = (np.log(1/(1-x)))
        b = self._eta*_log**(1.0/self._beta)
        return b

    def unreliability_at_time(self, time: float=0.0, plot: bool=False):
        """
        The probability that an item will be in not operational state at a particular point in time.
        Probability of failure is also known as "unreliability" and it is the reciprocal of the reliability.
        The function can also plot CDF graph.

        :param time: time / cycles for which CDF value to be calculated
        :param plot: True if the plot is to be shown, false if otherwise
        :return: CDF value (between 0 and 1)
        """
        # f(t) = 1 - exp((time / eta ) ^- _beta)

        if not plot:
            #cdf
            return 1 - (np.exp(-(time/self._eta)**self._beta))
        else:
           t = np.arange(start=1, stop=self._data['times'].max()*1.5)
           cdf = 1 - (np.exp(-(t/self._eta)**self._beta))

           plt.plot(t, cdf, label="beta: {:.02f}\neta: {:.01f}".format(self._beta, self._eta))
           plt.title("CDF")
           plt.legend(loc=2)

           plt.show()

        return None

    def reliability_at_time(self, time: float=0.1, plot: bool=False):
        """
        Calculates probability that an item will perform its function at a particular point of time.
        :param time:
        :param plot:
        :return:
        """
        # f(t) = 1 - exp((time / eta ) ^- _beta)
        r = (np.exp(-(time / self._eta) ** self._beta))
        return r

    def hazard_rate_at_time(self, time: float=0.1, plot: bool=False):
        """

        :param time:
        :param plot:
        :return:
        """

        if not plot:
            return (self._beta/self._eta)*((time/self._eta)**(self._beta-1))
        else:
            t = np.arange(start=1, stop=self._data['times'].max() * 1.5)
            z = (self._beta / self._eta) * ((t / self._eta) ** (self._beta - 1))
            plt.plot(t, z, label="beta: {:.02f}\neta: {:.01f}".format(self._beta, self._eta))

            plt.title("Hazard rate")
            plt.legend(loc=2)
            plt.show()
        return None

    def to_csv(self, file_name):
        """

        :return:
        """
        _url = file_name + datetime.now().strftime("%m-%d-%Y-%H:%M:%S")+".csv"
        return self._data.to_csv(_url, sep=',', index=False)

    @property
    def beta(self):
        """

        :return:
        """
        return self._beta

    @property
    def eta(self):
        """

        :return:
        """
        return self._eta

    @property
    def average_live(self):
        """

        :return:
        """
        g = 1+(1/self._beta)
        return ssp.gamma(g) * self._eta

    @property
    def totalOperationTime(self):
        """

        :return:
        """
        return self._data['times'].sum()

    @property
    def mtbf(self):
        """
        (total operating time / number of failures)
        :return:
        """
        return (self._data['times'].sum())/self._total_failures

    @property
    def r_value(self):
        """
        (total operating time / number of failures)
        :return:
        """
        return self._r_value

    @property
    def goodnes_of_fit(self):
        """
        to be completed
        :return:
        """
        return None

    @property
    def observetions_numbers(self):
        """

        :return:
        """
        return self._total_observetions

    @property
    def failures_numbers(self):
        """

        :return:
        """
        return  self._total_failures






