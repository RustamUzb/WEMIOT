import autograd.numpy as np  # Thinly-wrapped numpy
from autograd import grad
import math


def likehood_weibull(beta, failed, censored):

    a_all = np.sum(failed ** beta * np.log(failed)) + np.sum((censored ** beta * np.log(censored)))
    a_all_d = np.sum(failed**beta) + np.sum(censored**beta)
    b_cens = (1/failed.size) * np.sum(np.log(failed))
    c = 1/beta
    lme = (a_all/a_all_d)-b_cens-c
    #print("lme:", lme)
    loss = 0.5*((0-lme)**2)
    return loss


def eta_estimation (beta, failed, censored):

    eta =  ((np.sum(failed **beta) + np.sum(censored ** beta))/failed.size)**(1/beta)
    return eta

def lme_weibull(failed, censored):

    beta_init = 1.1
    eta_init = 1
    grad_lme = grad(likehood_weibull,0)

    lme = 1
    while lme > 0.000000000001:
        lme = likehood_weibull(beta_init, failed, censored)
        #Newton-raphson
        beta_init = beta_init - (lme / grad_lme(beta_init, failed, censored))
    beta = beta_init
    eta = eta_estimation (beta, failed, censored)

    return beta, eta
