import jax.numpy as jnp
from jax import jacfwd, jacrev
from jax.numpy import linalg
import scipy.stats as ss
import matplotlib.pyplot as plt

import logging


class Weibull:

    def __init__(self, *args, **kwargs):

        self.failures = None
        self.censored = None
        self.shape = None
        self.scale = None
        self.loc = 0.0
        self.CF = None
        # two-side bounds
        self.shapeTSUB = None
        self.scaleTSUB = None
        self.locTSUB = 0.0
        self.shapeTSLB = None
        self.scaleTSLB = None
        self.locTSLB = 0.0
        # one-side bounds
        self.shapeOSUB = None
        self.scaleOSUB = None
        self.locOSUB = 0.0
        self.shapeOSLB = None
        self.scaleOSLB = None
        self.locOSLB = 0.0

        self.method = ''
        self.converged = False

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
            q = 1 / (1+jnp.sqrt(abs(grads/800)))
            # Newton-Raphson maximisation
            parameters -= q * hess @ grads
            total = abs(grads[0]) + abs(grads[1])
            #print('epoch: ', epoch, 'parameters: ', ' param: ', parameters, ' grad:', grads, ' Q:',q)
        if epoch < 200:
            self.converged = True
            self.shape = parameters[0]
            self.scale = parameters[1]
            self.method = '2pComplete'
            # Fisher Matrix confidence bound
            ZO = -ss.norm.ppf(1 - self.CF)
            ZT = -ss.norm.ppf((1 - self.CF) / 2)
            self.shapeTSUB = self.shape * jnp.exp((ZT * jnp.sqrt(abs(hess[0][0]))) / self.shape)
            self.shapeTSLB = self.shape / jnp.exp((ZT * jnp.sqrt(abs(hess[0][0]))) / self.shape)
            self.scaleTSUB = self.scale * jnp.exp((ZT * jnp.sqrt(abs(hess[1][1]))) / self.scale)
            self.scaleTSLB = self.scale / jnp.exp((ZT * jnp.sqrt(abs(hess[1][1]))) / self.scale)

            self.shapeOSUB = self.shape * jnp.exp((ZO * jnp.sqrt(abs(hess[0][0]))) / self.shape)
            self.shapeOSLB = self.shape / jnp.exp((ZO * jnp.sqrt(abs(hess[0][0]))) / self.shape)
            self.scaleOSUB = self.scale * jnp.exp((ZO * jnp.sqrt(abs(hess[1][1]))) / self.scale)
            self.scaleOSLB = self.scale / jnp.exp((ZO * jnp.sqrt(abs(hess[1][1]))) / self.scale)

        else:
            # if more than 200 epoch it would be considered that fit is not converged
            self.converged = False
            self.shape = 0.0
            self.scale = 0.0
            self.method = '2pComplete'

        #print(self.shape, self.scale, epoch)


    def __fitTypeICensored2pMLE(self):
        # initial guess:
        shape = 1.2
        scale = (self.failures.mean() + self.censored.mean())/2
        parameters = jnp.array([shape, scale])

        J = jacfwd(self.__logLikelihood2pTypeICensored)
        H = jacfwd(jacrev(self.__logLikelihood2pTypeICensored))

        epoch = 0
        total = 1
        print('intit paramenters: ', parameters)
        while not (total < 0.09 or epoch > 200):
            epoch += 1
            grads = J(parameters)
            hess = linalg.inv(H(parameters))
            # Q is a coefficient to reduce gradient ascent step for high delta
            q = 1 / (1+jnp.sqrt(abs(grads/8)))
            # Newton-Raphson maximisation
            parameters -= q * hess @ grads
            total = abs(grads[0]) + abs(grads[1])
            print('Epoch: ', epoch, 'Param: ', parameters, ' grad:', grads, ' Q:',q, hess[0][0], hess[1][1])
        if epoch < 200:
            self.converged = True
            self.shape = parameters[0]
            self.scale = parameters[1]
            self.method = '2pTypeICensored'

            # Fisher Matrix confidence bound
            ZO = -ss.norm.ppf(1 - self.CF)
            ZT = -ss.norm.ppf((1 - self.CF) / 2)
            self.shapeTSUB = self.shape * jnp.exp((ZT * jnp.sqrt(abs(hess[0][0]))) / self.shape)
            self.shapeTSLB = self.shape / jnp.exp((ZT * jnp.sqrt(abs(hess[0][0]))) / self.shape)
            self.scaleTSUB = self.scale * jnp.exp((ZT * jnp.sqrt(abs(hess[1][1]))) / self.scale)
            self.scaleTSLB = self.scale / jnp.exp((ZT * jnp.sqrt(abs(hess[1][1]))) / self.scale)

            self.shapeOSUB = self.shape * jnp.exp((ZO * jnp.sqrt(abs(hess[0][0]))) / self.shape)
            self.shapeOSLB = self.shape / jnp.exp((ZO * jnp.sqrt(abs(hess[0][0]))) / self.shape)
            self.scaleOSUB = self.scale * jnp.exp((ZO * jnp.sqrt(abs(hess[1][1]))) / self.scale)
            self.scaleOSLB = self.scale / jnp.exp((ZO * jnp.sqrt(abs(hess[1][1]))) / self.scale)

        else:
            # if more than 200 epoch it would be considered that fit is not converged
            self.converged = False
            self.shape = 0.0
            self.scale = 0.0
            self.method = '2pTypeICensored'




    def __logLikelihood2pComp(self, x):
        #x[0] = shape
        #x[1] = scale
        logl = self.failures.size * jnp.log(x[0]) - x[0] * self.failures.size * jnp.log(x[1]) + (x[0] - 1.0) * \
            jnp.sum(jnp.log(self.failures)) - jnp.sum((self.failures/x[1])**x[0])
        return logl

    def __logLikelihood2pTypeICensored(self, x):
        # https://doi.org/10.1016/j.spl.2008.05.019 (3.9)
        #x[0] = shape
        #x[1] = scale
        logl = self.failures.size * jnp.log(x[0]) - x[0] * self.failures.size * jnp.log(x[1]) + (x[0] - 1.0) * \
                        jnp.sum(jnp.log(self.failures)) - (1/x[1]**x[0]) * \
                        (jnp.sum(self.failures**x[0]) + jnp.sum(self.censored ** x[0]))
        return logl

    def fit(self, failures, censored=None, method='all', mixTest = True, CF=0.95):
        self.CF = CF
        self.failures = jnp.array(failures)
        if censored is not None:
            self.censored = jnp.array(censored)
        else:
            self.censored = jnp.zeros(1)

        if method == '2pComplete':
            print('2p Complete')
            self.__fitComplete2pMLE()
        elif method == '2pCensored':
            if censored is not None:
                logging.info('2p Censored')
                self.__fitTypeICensored2pMLE()
            else:
                # TODO raise error
                print('Censored data must be provided')

    def printResults(self):
        print('----------------------------------------------------')
        print('Method: ', self.method)
        print('Shape: ', self.shape, 'Scale: ', self.scale, 'Loc: ', self.loc)
        print('Confidence bounds (One side):')
        print('Shape Lower: ', self.shapeOSLB, 'Shape Upper: ', self.shapeOSUB)
        print('Scale Lower: ', self.scaleOSLB, 'Scale Upper: ', self.scaleOSUB)
        print('Confidence bounds (Two side side):')
        print('Shape Lower: ', self.shapeTSLB, 'Shape Upper: ', self.shapeTSUB)
        print('Scale Lower: ', self.scaleTSLB, 'Scale Upper: ', self.scaleTSUB)

    def pdf(self, x, bound=None):
        if bound is None:
            v = ss.weibull_min.pdf(x, c=self.shape, loc=self.loc, scale=self.scale)
        elif bound == 'OSLB':
            v = ss.weibull_min.pdf(x, c=self.shapeOSLB, loc=self.locOSLB, scale=self.scaleOSLB)
        elif bound == 'TSLB':
            v = ss.weibull_min.pdf(x, c=self.shapeTSLB, loc=self.locTSLB, scale=self.scale)
        elif bound == 'OSUB':
            v = ss.weibull_min.pdf(x, c=self.shapeOSUB, loc=self.locOSUB, scale=self.scaleOSUB)
        elif bound == 'TSUB':
            v = ss.weibull_min.pdf(x, c=self.shapeTSUB, loc=self.locTSUB, scale=self.scale)
        return v

    def cdf(self, x, bound=None):
        if bound is None:
            v = ss.weibull_min.cdf(x, c=self.shape, loc=self.loc, scale=self.scale)
        elif bound == 'OSLB':
            v = ss.weibull_min.cdf(x, c=self.shapeOSLB, loc=self.locOSLB, scale=self.scaleOSLB)
        elif bound == 'TSLB':
            v = ss.weibull_min.cdf(x, c=self.shapeTSLB, loc=self.locTSLB, scale=self.scaleTSLB)
        elif bound == 'OSUB':
            v = ss.weibull_min.cdf(x, c=self.shapeOSUB, loc=self.locOSUB, scale=self.scaleOSUB)
        elif bound == 'TSUB':
            v = ss.weibull_min.cdf(x, c=self.shapeTSUB, loc=self.locTSUB, scale=self.scaleTSUB)
        return v

    def sf(self, x, bound=None):
        if bound is None:
            v = ss.weibull_min.sf(x, c=self.shape, loc=self.loc, scale=self.scale)
        elif bound == 'OSLB':
            v = ss.weibull_min.sf(x, c=self.shapeOSLB, loc=self.locOSLB, scale=self.scaleOSLB)
        elif bound == 'TSLB':
            v = ss.weibull_min.sf(x, c=self.shapeTSLB, loc=self.locTSLB, scale=self.scaleTSLB)
        elif bound == 'OSUB':
            v = ss.weibull_min.sf(x, c=self.shapeOSUB, loc=self.locOSUB, scale=self.scaleOSUB)
        elif bound == 'TSUB':
            v = ss.weibull_min.sf(x, c=self.shapeTSUB, loc=self.locTSUB, scale=self.scaleTSUB)
        return v

    def median(self, bound=None):
        if bound is None:
            v = ss.weibull_min.median(c=self.shape, loc=self.loc, scale=self.scale)
        elif bound == 'OSLB':
            v = ss.weibull_min.median(c=self.shapeOSLB, loc=self.locOSLB, scale=self.scaleOSLB)
        elif bound == 'TSLB':
            v = ss.weibull_min.median(c=self.shapeTSLB, loc=self.locTSLB, scale=self.scaleTSLB)
        elif bound == 'OSUB':
            v = ss.weibull_min.median(c=self.shapeOSUB, loc=self.locOSUB, scale=self.scaleOSUB)
        elif bound == 'TSUB':
            v = ss.weibull_min.median(c=self.shapeTSUB, loc=self.locTSUB, scale=self.scaleTSUB)
        return v

    def mean(self, bound=None):
        if bound is None:
            v = ss.weibull_min.mean(c=self.shape, loc=self.loc, scale=self.scale)
        elif bound == 'OSLB':
            v = ss.weibull_min.mean(c=self.shapeOSLB, loc=self.locOSLB, scale=self.scaleOSLB)
        elif bound == 'TSLB':
            v = ss.weibull_min.mean(c=self.shapeTSLB, loc=self.locTSLB, scale=self.scaleTSLB)
        elif bound == 'OSUB':
            v = ss.weibull_min.mean(c=self.shapeOSUB, loc=self.locOSUB, scale=self.scaleOSUB)
        elif bound == 'TSUB':
            v = ss.weibull_min.mean(c=self.shapeTSUB, loc=self.locTSUB, scale=self.scaleTSUB)
        return v

    def var(self, bound=None):
        if bound is None:
            v = ss.weibull_min.var(c=self.shape, loc=self.loc, scale=self.scale)
        elif bound == 'OSLB':
            v = ss.weibull_min.var(c=self.shapeOSLB, loc=self.locOSLB, scale=self.scaleOSLB)
        elif bound == 'TSLB':
            v = ss.weibull_min.var(c=self.shapeTSLB, loc=self.locTSLB, scale=self.scaleTSLB)
        elif bound == 'OSUB':
            v = ss.weibull_min.var(c=self.shapeOSUB, loc=self.locOSUB, scale=self.scaleOSUB)
        elif bound == 'TSUB':
            v = ss.weibull_min.var(c=self.shapeTSUB, loc=self.locTSUB, scale=self.scaleTSUB)
        return v

    def std(self, bound=None):
        if bound is None:
            v = ss.weibull_min.std(c=self.shape, loc=self.loc, scale=self.scale)
        elif bound == 'OSLB':
            v = ss.weibull_min.std(c=self.shapeOSLB, loc=self.locOSLB, scale=self.scaleOSLB)
        elif bound == 'TSLB':
            v = ss.weibull_min.std(c=self.shapeTSLB, loc=self.locTSLB, scale=self.scaleTSLB)
        elif bound == 'OSUB':
            v = ss.weibull_min.std(c=self.shapeOSUB, loc=self.locOSUB, scale=self.scaleOSUB)
        elif bound == 'TSUB':
            v = ss.weibull_min.std(c=self.shapeTSUB, loc=self.locTSUB, scale=self.scaleTSUB)
        return v

    #hazard function (failure rate)
    def hf(self, x,  bound=None):

        if bound is None:
            v = (self.shape / self.scale) * ((x - self.loc) / self.scale)**(self.shape - 1.0)
        elif bound == 'OSLB':
            v = (self.shapeOSLB / self.scaleOSLB) * (((x - self.locOSLB) / self.scaleOSLB) ** (self.shapeOSLB - 1.0))
        elif bound == 'TSLB':
            v = (self.shapeTSLB / self.scaleTSLB) * (((x - self.locTSLB) / self.scaleTSLB) ** (self.shapeTSLB - 1.0))
        elif bound == 'OSUB':
            v = (self.shapeOSUB / self.scaleOSUB) * (((x - self.locOSUB) / self.scaleOSUB) ** (self.shapeOSUB - 1.0))
        elif bound == 'TSUB':
            v = (self.shapeTSUB / self.scaleTSUB) * (((x - self.locTSUB) / self.scaleTSUB) ** (self.shapeTSUB - 1.0))
        return v

    #cumulative hazard function
    def chf(self, bound=None):
        #TODO (find valid refference for calculation)
        pass
    def plothf(self):

        #ax.fill_between(x, LB, UB, alpha=0.2)
        xadd = self.failures.min() / 3
        x = jnp.arange(self.failures.min() - xadd, self.failures.max() + xadd,
                       ((self.failures.min() - xadd) + (self.failures.max() + xadd)) / 200)
        y = self.hf(x)
        LB = self.hf(x, 'TSLB')
        UB = self.hf(x, 'TSUB')

        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.fill_between(x, LB, UB, alpha=0.2)
        #plt.grid(True, which="both", ls="-", zorder=3)
        plt.show()


    def plotcdf(self):

        #ax.fill_between(x, LB, UB, alpha=0.2)
        xadd = self.failures.min() / 3
        x = jnp.arange(self.failures.min() - xadd, self.failures.max() + xadd,
                       ((self.failures.min() - xadd) + (self.failures.max() + xadd)) / 200)
        y = self.cdf(x)
        LB = self.cdf(x, 'TSLB')
        UB = self.cdf(x, 'TSUB')

        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.fill_between(x, LB, UB, alpha=0.2, interpolate=False)
        #plt.grid(True, which="both", ls="-", zorder=3)
        plt.show()