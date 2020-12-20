from jax import jacfwd, jacrev
from jax.numpy import linalg
import jax.numpy as jnp

import numpy as np

import scipy.stats as ss
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
import datetime
import configparser
import os

import util.utilities as util
from util.utilities import *


class Weibull:

    def __init__(self, *args, **kwargs):
        self.N = 0 # total sample size
        self.est_data = None # df to save failure time and cdf
        self.r2 = None

        self.failures = None
        self.censored = None

        self.shape = None
        self.scale = None
        self.loc = 0.0

        self.CL = None
        self.variance = None
        self.beta_eta_covar = None
        self.method = ''
        self.converged = False
        f = os.path.join(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'conf'), 'config.ini')
        self.config = configparser.ConfigParser()
        self.config.read(f)


    def __fitComplete2pMLE(self):
        # initial guess:
        shape = 1.2
        scale = self.failures.mean()
        parameters = jnp.array([shape, scale])

        J = jacfwd(self.__logLikelihood2pComp)
        H = jacfwd(jacrev(self.__logLikelihood2pComp))

        epoch = 0
        total = 1
        while not (total < 0.01 or epoch > 200):
            epoch += 1
            grads = J(parameters)
            hess = linalg.inv(H(parameters))
            # Q is a coefficient to reduce gradient ascent step for high delta
            q = 1 / (1 + jnp.sqrt(abs(grads / 800)))
            # Newton-Raphson maximisation
            parameters -= q * hess @ grads
            total = abs(grads[0]) + abs(grads[1])

        if epoch < 200:
            self.converged = True
            self.shape = parameters[0]
            self.scale = parameters[1]
            self.method = '2pComplete'

            # Fisher Matrix confidence bound
            self.variance = [abs(hess[0][0]), abs(hess[1][1])]
            self.beta_eta_covar = [abs(hess[1][0])]

        else:
            # if more than 200 epoch it would be considered that fit is not converged
            self.converged = False
            self.shape = 0.0
            self.scale = 0.0
            self.method = Method.MLEComplete2p

    def __fitTypeICensored2pMLE(self):
        # initial guess:
        shape = 1.2
        scale = (self.failures.mean() + self.censored.mean()) / 2
        parameters = jnp.array([shape, scale])

        J = jacfwd(self.__logLikelihood2pTypeICensored)
        H = jacfwd(jacrev(self.__logLikelihood2pTypeICensored))

        epoch = 0
        total = 1
        while not (total < 0.09 or epoch > 200):
            epoch += 1
            grads = J(parameters)
            hess = linalg.inv(H(parameters))
            q = 1 / (1 + jnp.sqrt(abs(grads / 8)))  # Q is a coefficient to reduce gradient ascent step for high delta
            parameters -= q * hess @ grads  # Newton-Raphson maximisation
            total = abs(grads[0]) + abs(grads[1])

        if epoch < 200:
            self.converged = True
            self.shape = parameters[0]
            self.scale = parameters[1]
            self.method = Method.MLECensored2p
            self.variance = [abs(hess[0][0]), abs(hess[1][1])]
            self.beta_eta_covar = [abs(hess[1][0])]

        else:
            # if more than 200 epoch it would be considered that fit is not converged
            self.converged = False
            self.shape = None
            self.scale = None
            self.method = Method.MLECensored2p
            print('no')

    def __fitCensoredMRR(self):
        '''
        Median rank regression method (50%) to estimate parameters. Better for low number of failures. this method does
        not provides confidence bonds for parameters, however it is possible to estimate confidence bound in time using
        median rank for specified limit (other than 50%).
        :return:
        '''
        iks = jnp.log(self.est_data['time'].to_numpy())
        igrek = jnp.log(jnp.log(1.0 / (1.0 - util.median_rank(self.N, self.est_data['new_order_num'], 0.5))))

        if self.method == Method.MRRCensored2p:
            slope, intercept, r, _, _ = self.__lineregress(iks, igrek)
        elif self.method == Method.MRRCensored3p:
            locs = jnp.linspace(0, jnp.amin(iks), 10)
            arr = np.empty((0, 4), float)
            for l in locs:
                l_iks = iks - l
                l_slope, l_intercept, l_r, _ , _ = self.__lineregress(iks, igrek)
                arr = np.append(arr, np.array([[l, l_slope, l_intercept, l_r]]), axis=0)
            g = arr[np.argmax(arr[:, 3])]
            slope = g[1]
            intercept = g[2]
            r = g[3]
            self.loc = g[0]

        # assigning estimated parameters
        self.shape = slope
        self.scale = jnp.exp(-intercept / slope)

        self.est_data['ub'] = (jnp.log(1.0 / (1.0 - util.median_rank(self.N, self.est_data['new_order_num'], self.CL))) ** (
                1.0 / self.shape) * self.scale)
        self.est_data['lb'] = (jnp.log(1.0 / (1.0 - util.median_rank(self.N, self.est_data['new_order_num'], 1.0 - self.CL))) ** (
                1.0 / self.shape) * self.scale)
        self.r2 = r ** 2
        self.est_data = self.est_data[['time', 'cdf', 'lb', 'ub']]
        self.method = Method.MRRCensored2p
        self.converged = True
        #print(self.est_data)

    def __lineregress(self, x_values, y_values):
        slope, intercept, r, p, std = ss.linregress(x_values, y_values)
        return slope, intercept, r, p, std

    def __logLikelihood2pComp(self, x):
        '''
        log-likelihood function for censored weibull distribution as per https://scholarsarchive.byu.edu/etd/2509 (2.1)
        :param x: array of parameters [shape, scale]
        :return: log-likelihood of 2p complete weibull function with given parameters
        '''

        logl = self.failures.size * jnp.log(x[0]) - x[0] * self.failures.size * jnp.log(x[1]) + (x[0] - 1.0) * \
               jnp.sum(jnp.log(self.failures)) - jnp.sum((self.failures / x[1]) ** x[0])
        return logl

    def __logLikelihood2pTypeICensored(self, x):
        '''
        log-likelihood function for censored weibull distribution as per https://doi.org/10.1016/j.spl.2008.05.019 (3.9)
        :param x:  array of parameters [shape, scale]
        :return: log-likelihood of 2p right censored weibull distribution with given parameters
        '''
        logl = self.failures.size * jnp.log(x[0]) - x[0] * self.failures.size * jnp.log(x[1]) + (x[0] - 1.0) * \
               jnp.sum(jnp.log(self.failures)) - (1 / x[1] ** x[0]) * \
               (jnp.sum(self.failures ** x[0]) + jnp.sum(self.censored ** x[0]))
        return logl

    def fit(self, failures, censored=None, method=Method.AUTO, CL=0.95, mixTest=True, failureCode=None,
            classCode=None, eqId=None):
        '''

        :param failures: list of failure times/cycles. float or int
        :param censored: list of times times/cycles when item was withdrawn or fail with a different failure mode. float or int
        :param method: Method which will be used to estimate parameters of weibull based on provided data. Instance of
         class analysis.util.Method. default: Method.AUTO which automatically decide which method to used based on number
         of failure records.
        TODO :param mixTest:
        :param CL: a number between 0.01 to 0.99 which is represent level of confidence that estimated parameter is in
        the desired range. http://reliawiki.org/index.php/Confidence_Bounds  For One-side confidence interval  the range
        is equal to CL and can be upper or low . For Two-sided confidence interval the range lies between (1.0-CL)/2
        and  1.0 - (1.0-CL)/2. upper and lower bounds for both one-seded and two-sided are calculated if method is MLE,
        for MRR upper and lower bound calculated for one-sided interval only
        :param failureCode: concerned failure code. optional
        :param classCode: concerned class code as per ISO-14224. optional
        :param eqId: optinal description or model number of equipment
        :return: True - if converged , False - if not converged
        '''
        self.CL = CL
        self.failures = jnp.array(failures)
        self.method = method
        if censored is not None:
            self.censored = jnp.array(censored)
        else:
            self.censored = jnp.zeros(1)
        self.__do_rank_regression()
        if method == Method.AUTO:
            # TODO automatically define method (2 parameters only)
            pass
        elif method == Method.MLEComplete2p:
            self.__fitComplete2pMLE()
        elif method == Method.MLECensored2p:
            if censored is not None:
                self.__fitTypeICensored2pMLE()
            else:
                # TODO raise error
                print('Censored data must be provided')
        elif method == Method.MRRCensored2p:
            self.__fitCensoredMRR()
        elif method == Method.MRRCensored3p:
            self.__fitCensoredMRR()
        else:
            # TODO raise error
            print('choose proper method')
        return self.converged

    def __do_rank_regression(self):
        f = jnp.hstack((jnp.atleast_2d(self.failures).T, jnp.zeros((self.failures.shape[0], 1))))
        f = f[f[:, 0].argsort()]
        f = jnp.hstack((f, jnp.reshape(jnp.arange(self.failures.shape[0]), (self.failures.shape[0], -1))))
        # censored items will be having flag '1'
        c = jnp.hstack((jnp.atleast_2d(self.censored).T, jnp.ones((self.censored.shape[0], 1))))
        c = jnp.hstack((c, jnp.reshape(jnp.empty(self.censored.shape[0]), (self.censored.shape[0], -1))))
        d = jnp.concatenate((c, f), axis=0)
        d = d[d[:, 0].argsort()]
        df = pd.DataFrame(data=d, columns=['time', 'is_cens', 'fo'])
        self.N = len(df.index)
        df['new_increment'] = (self.N + 1 - df['fo']) / (self.N + 2 - df.index.values)
        m = 1.0 - df['new_increment'].min()
        df['new_increment'] = df['new_increment'] + m
        df = df.drop(df[df['is_cens'] == 1].index)
        df['new_order_num'] = df['new_increment'].cumsum()
        df['cdf'] = util.median_rank(self.N, df['new_order_num'], 0.5)
        self.est_data = df

    def printResults(self, out=None):
        if self.method in [Method.MLECensored2p, Method.MLEComplete2p]:
            prdf = pd.DataFrame(columns=['a', 'b', 'c', 'd', 'e'])
            prdf.loc[0] = pd.Series(
                {'a': 'REPORT', 'b': 'Date', 'c': datetime.datetime.now().strftime("%c"), 'd': '', 'e': ''})
            prdf.loc[1] = pd.Series(
                {'a': '', 'b': 'Failures:', 'c': str(self.failures.size), 'd': '', 'e': ''})
            prdf.loc[2] = pd.Series(
                {'a': '', 'b': 'Censored:', 'c': str(self.censored.size), 'd': '', 'e': ''})
            prdf.loc[3] = pd.Series({'a': '', 'b': 'Conf. In:', 'c': str(self.CL), 'd': '', 'e': ''})
            prdf.loc[4] = pd.Series({'a': 'OUTPUT:', 'b': '', 'c': 'Est:', 'd': 'LB', 'e': 'UB'})
            #prdf.loc[5] = pd.Series({'a': '', 'b': 'Shape', 'c': str(round(self.shape, 4)),
            #                         'd': str(round(self.OSLBpar[0], 4)), 'e': str(round(self.OSUBpar[0], 4))})
           # prdf.loc[6] = pd.Series({'a': '', 'b': 'Scale', 'c': str(round(self.scale, 4)),
            #                         'd': str(round(self.OSLBpar[1], 4)), 'e': str(round(self.OSUBpar[1], 4))})
            prdf.loc[7] = pd.Series({'a': '', 'b': 'Loc:', 'c': 0.0,
                                     'd': 0.0, 'e': 0.0})
            if out == None:
                # set default output, as per conf.py
                out = self.config.get('DEFAULTS', 'printresults')
            if out == 'file':
                fileName = str(datetime.datetime.now().strftime("%m/%d/%Y%H:%M:%S")) + ".cvs"
                prdf.to_csv(os.path.join(self.config.get('FOLDER_CONFIG', 'files_dir'), fileName.replace('/', '-')))
            elif out == 'json':
                prdf = prdf.drop(['a'], 1)
                prdf = prdf[prdf['b'] != '']
                prdf = prdf.reset_index(drop=True)
                json_split = prdf.to_json(orient='split')
                return json_split
            elif out == 'console':
                print(tabulate(prdf, tablefmt='grid', showindex=False))

        elif self.method == Method.MRRCensored2p:
            prdf = pd.DataFrame(columns=['a', 'b', 'c'])
            prdf.loc[0] = pd.Series({'a': 'REPORT', 'b': '', 'c': ''})
            prdf.loc[1] = pd.Series({'a': '', 'b': 'Fit: ', 'c': 'MRR'})
            prdf.loc[1] = pd.Series({'a': '', 'b': 'Date: ', 'c': datetime.datetime.now().strftime("%c")})
            prdf.loc[2] = pd.Series({'a': 'USR. INFO', 'b': '', 'c': ''})
            prdf.loc[3] = pd.Series({'a': '', 'b': 'Failures:', 'c': str(self.failures.size)})
            prdf.loc[4] = pd.Series({'a': '', 'b': 'Censored:', 'c': str(self.censored.size)})
            prdf.loc[5] = pd.Series({'a': '', 'b': 'Conf. In:', 'c': str(self.CL)})
            prdf.loc[6] = pd.Series({'a': 'OUTPUT:', 'b': '', 'c': '', 'd': '', 'e': ''})
            prdf.loc[7] = pd.Series({'a': '', 'b': 'Shape:', 'c': str(round(self.shape, 6))})
            prdf.loc[8] = pd.Series({'a': '', 'b': 'Scale:', 'c': str(round(self.scale, 2))})
            prdf.loc[9] = pd.Series({'a': '', 'b': 'Loc:', 'c': str(round(self.loc, 4))})
            prdf.loc[10] = pd.Series({'a': '', 'b': 'Rˆ2:', 'c': str(round(self.r2, 4))})

            if out == None:
                # set default output, as per conf.py
                out = self.config.get('DEFAULTS', 'printresults')
            if out == 'file':
                fileName = str(datetime.datetime.now().strftime("%m/%d/%Y%H:%M:%S")) + ".cvs"
                prdf.to_csv(os.path.join(self.config.get('FOLDER_CONFIG', 'files_dir'), fileName.replace('/', '-')))
            elif out == 'json':
                prdf = prdf.drop(['a'], 1)
                prdf = prdf[prdf['b'] != '']
                prdf = prdf.reset_index(drop=True)
                json_split = prdf.to_json(orient='split')
                return json_split
            elif out == 'console':
                print(tabulate(prdf, tablefmt='grid', showindex=False))

    def showPlot(self):
        if self.method == Method.MRRCensored2p:
            xmin = self.est_data[['time', 'lb', 'ub']].min().min() - (
                        self.est_data[['time', 'lb', 'ub']].min().min() * 0.2)
            xmax = self.est_data[['time', 'lb', 'ub']].max().max() * 1.2
            method_text = 'MRR 2P'
            conf_test = 'MR'

            xx = jnp.array(self.est_data['time'] - self.loc) #failures
            yy = jnp.log(1.0 / (1.0 - jnp.array(self.est_data['cdf'])))

            x = jnp.arange (xmin, xmax, (xmax-xmin)/200 )
            est_cdf = jnp.log(1.0 / (1.0 - self.cdf(x)))

            y = yy
            t_lb = self.est_data['lb']
            t_ub = self.est_data['ub']
            xtics = 5 * jnp.around(jnp.arange(xmin, xmax, (xmax - xmin) / 10) / 5)  # round to nearest 5

        elif self.method == Method.MLECensored2p:
            method_text = 'MLE 2P'
            conf_test = 'FM'
            xmin = self.bLive(0.01)
            xmax = self.bLive(0.99)

            xx = jnp.array(self.est_data['time'] - self.loc)
            yy = jnp.log(1.0 / (1.0 - jnp.array(self.est_data['cdf'])))

            x = jnp.arange(xmin, xmax, (xmax - xmin) / 200)
            est_cdf = jnp.log(1.0 / (1.0 - self.cdf(x)))

            y = est_cdf
            t_lb = self.bLiveCL(1.0 - self.cdf(x), Bound.OSLB)
            t_ub = self.bLiveCL(1.0 - self.cdf(x), Bound.OSUB)
            t_tlb = self.bLiveCL(1.0 - self.cdf(x), Bound.TSLB)
            t_tub = self.bLiveCL(1.0 - self.cdf(x), Bound.TSUB)
            xmin = self.bLive(0.01)
            xmax = self.bLive(0.99)
            xtics = 5 * jnp.around(jnp.arange(xmin, xmax, (xmax-xmin)/10) / 5)   # round to nearest 5

        fig = plt.figure(num=None, figsize=(6, 9), dpi=80, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xscale('log')
        ax.set_yscale('log')
        fig.canvas.draw()
        ax.plot(xx, yy, 'rX', label='Failures (MMR)') #failures
        ax.plot(x, est_cdf, label='Reference line') #estimated
        ax.plot(t_lb, y, label='Lower bound - ' + str(round(1.0 - self.CL, 3))) # lower bound
        ax.plot(t_ub, y, label='Upper bound - ' + str(round(1.0 - self.CL, 3))) # upper bound
        if self.method == Method.MLECensored2p:
            # only for MLE
            ax.fill_betweenx(y, t_tlb, t_tub, facecolor='tan', alpha=0.3 ,label='2 side bound 2 x '+
                                                                                str(round((1.0 - self.CL)/2, 3)))
        ax.set_xticks(xtics)
        ax.set_xlabel(xtics)
        ax.set_xticklabels(xtics, rotation=45)

        ax.set_yticks([0.010050336, 0.051293294, 0.105360516, 0.223143551, 0.356674944, 0.510825624, 0.693147181, 0.916290732,
                       1.203972804, 1.609437912, 2.302585093, 4.605170186])
        ax.set_yticklabels(['1%', '5%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '99%'])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.grid(True, which="both")
        ax.set_xlabel('Sample data (time/cycle)')
        ax.set_ylabel('Weibull probability')
        ax.set_title("Weibull probability plot "
                     "\n fit = " + method_text + ", CI =" + conf_test + ", Failures = " + str(self.failures.size) +
                     ", Censored = " + str(self.censored.size) + "\n Est:  Shape = " + str(round(self.shape, 2)) +
                     " ; Scale= " + str(round(self.scale, 2)) + " ; Location = " + str(round(self.loc, 2)))
        plt.legend(loc='upper left')
        plt.show()

    def pdf(self, t):
        return ss.weibull_min.pdf(t, c=self.shape, loc=self.loc, scale=self.scale)

    def cdf(self, t):
        return ss.weibull_min.cdf(t, c=self.shape, loc=self.loc, scale=self.scale)

    def reliability(self, t):
        return ss.weibull_min.sf(t, c=self.shape, loc=self.loc, scale=self.scale)

    def median(self):
        return ss.weibull_min.median(c=self.shape, loc=self.loc, scale=self.scale)

    def mean(self):
        return ss.weibull_min.mean(c=self.shape, loc=self.loc, scale=self.scale)

    def var(self):
        return ss.weibull_min.var(c=self.shape, loc=self.loc, scale=self.scale)

    def std(self):
        return ss.weibull_min.std(c=self.shape, loc=self.loc, scale=self.scale)

    def hf(self, t):
        '''
        https://en.wikipedia.org/wiki/Weibull_distribution
        :param t:
        :return: failure rate at time t
        '''
        return (self.shape / self.scale) * ((t - self.loc) / self.scale) ** (self.shape - 1.0)

    def bLive(self, R, bound=None):
        if bound is None:
            v = ss.weibull_min.ppf(R, self.shape, scale=self.scale, loc=self.loc)
        return v

    def bLiveCL(self, R, bound):
        '''
        https://www.weibull.com/hotwire/issue17/relbasics17.htm
        :param R:
        :param bound:
        :return:
        '''

        if bound == Bound.OSLB:
            ZO = -ss.norm.ppf(1 - self.CL)
            ln_t = 1 / self.shape * jnp.log(- jnp.log(R)) + jnp.log(self.scale)
            var_ln_t = 1.0 / (self.shape ** 4) * (jnp.log(-jnp.log(R))**2) * self.variance[0] \
                    + 1.0 / (self.scale ** 2) * self.variance[1] + 2 * (- 1.0 / (self.shape**2)) \
                    * (jnp.log(- jnp.log(R)) / self.scale) * self.beta_eta_covar[0]
            ans = jnp.exp(ln_t - ZO * jnp.sqrt(var_ln_t)) + self.loc
        elif bound == Bound.OSUB:
            ZO = -ss.norm.ppf(1 - self.CL)
            ln_t = 1 / self.shape * jnp.log(- jnp.log(R)) + jnp.log(self.scale)
            var_ln_t = 1.0 / (self.shape ** 4) * (jnp.log(-jnp.log(R)) ** 2) * self.variance[0] \
                       + 1.0 / (self.scale ** 2) * self.variance[1] + 2 * (- 1.0 / (self.shape ** 2)) \
                       * (jnp.log(- jnp.log(R)) / self.scale) * self.beta_eta_covar[0]
            ans = jnp.exp(ln_t + ZO * jnp.sqrt(var_ln_t)) + self.loc
        elif bound == Bound.TSLB:
            ZT = -ss.norm.ppf((1 - self.CL) / 2)
            ln_t = 1 / self.shape * jnp.log(- jnp.log(R)) + jnp.log(self.scale)
            var_ln_t = 1.0 / (self.shape ** 4) * (jnp.log(-jnp.log(R)) ** 2) * self.variance[0] \
                       + 1.0 / (self.scale ** 2) * self.variance[1] + 2 * (- 1.0 / (self.shape ** 2)) \
                       * (jnp.log(- jnp.log(R)) / self.scale) * self.beta_eta_covar[0]
            ans = jnp.exp(ln_t - ZT * jnp.sqrt(var_ln_t)) + self.loc
        elif bound == Bound.TSUB:
            ZT = -ss.norm.ppf((1 - self.CL) / 2)
            ln_t = 1 / self.shape * jnp.log(- jnp.log(R)) + jnp.log(self.scale)
            var_ln_t = 1.0 / (self.shape ** 4) * (jnp.log(-jnp.log(R)) ** 2) * self.variance[0] \
                       + 1.0 / (self.scale ** 2) * self.variance[1] + 2 * (- 1.0 / (self.shape ** 2)) \
                       * (jnp.log(- jnp.log(R)) / self.scale) * self.beta_eta_covar[0]
            ans = jnp.exp(ln_t + ZT * jnp.sqrt(var_ln_t)) + self.loc

        return ans

    def reliabilityCL(self, t, bound):
        '''
        https://www.weibull.com/hotwire/issue17/relbasics17.htm
        :param t:
        :param bound:
        :return:
        '''
        if bound == Bound.OSLB:
            ZO = -ss.norm.ppf(1 - self.CL)
            r = self.shape * (jnp.log(t + self.loc) - jnp.log(self.scale))
            var_r = (((r**2) / (self.shape **2)) * self.variance[0]) + ((self.shape**2)/ (self.scale**2)) \
                * self.variance[1] - ((2*r)/self.scale) * self.beta_eta_covar[0]
            ans =  jnp.exp(-jnp.exp(r - ZO * jnp.sqrt(var_r)))
        elif bound == Bound.OSUB:
            ZO = -ss.norm.ppf(1 - self.CL)
            r = self.shape * (jnp.log(t + self.loc) - jnp.log(self.scale))
            var_r = (((r ** 2) / (self.shape ** 2)) * self.variance[0]) + ((self.shape ** 2) / (self.scale ** 2)) \
                    * self.variance[1] - ((2 * r) / self.scale) * self.beta_eta_covar[0]
            ans = jnp.exp(-jnp.exp(r + ZO * jnp.sqrt(var_r)))
        elif bound == Bound.TSLB:
            ZT = -ss.norm.ppf((1 - self.CL) / 2)
            r = self.shape * (jnp.log(t + self.loc) - jnp.log(self.scale))
            var_r = (((r ** 2) / (self.shape ** 2)) * self.variance[0]) + ((self.shape ** 2) / (self.scale ** 2)) \
                    * self.variance[1] - ((2 * r) / self.scale) * self.beta_eta_covar[0]
            ans = jnp.exp(-jnp.exp(r - ZT * jnp.sqrt(var_r)))
        elif bound == Bound.TSUB:
            ZT = -ss.norm.ppf((1 - self.CL) / 2)
            r = self.shape * (jnp.log(t + self.loc) - jnp.log(self.scale))
            var_r = (((r ** 2) / (self.shape ** 2)) * self.variance[0]) + ((self.shape ** 2) / (self.scale ** 2)) \
                    * self.variance[1] - ((2 * r) / self.scale) * self.beta_eta_covar[0]
            ans = jnp.exp(-jnp.exp(r + ZT * jnp.sqrt(var_r)))
        return ans

    def chf(self, t):
        '''
        Cumulative hazard function
        :param bound:
        :return:
        '''
        return ((t - self.loc) / self.scale)**self.shape
