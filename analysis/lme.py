import autograd.numpy as np  # Thinly-wrapped numpy
from autograd import grad
import math


def likehood_weibull(beta, eta, failure, censored):

    a = (np.prod(np.exp(-(censored/eta)**beta))) * np.prod((beta/eta) *
                                                           (failure/eta) ** (beta - 1) * np.exp(-(failure/eta)**beta))

    return np.log(a)


def error_f(beta, eta):
    c = (0-beta) + (0-eta)
    return c



def lme_weibull(failed, censored):
    beta_pr = grad(likehood_weibull, 0)
    eta_pr = grad(likehood_weibull, 1)

    b_cost = grad(error_f, 0)
    e_cost = grad(error_f, 1)

    lr_beta = 0.01
    lr_eta = np.mean(failed)

    beta = 0.1
    eta = np.mean(failed)

    for x in range(10000):
        b_p = beta_pr(beta,eta, failed, censored)
        e_p = eta_pr(beta,eta, failed, censored)
        """
        print(x, beta,eta,
          b_cost(b_p, e_p) * b_p * lr_beta,
          e_cost(b_p, e_p) * e_p * lr_eta
          , abs(error_f(b_p, e_p)))
        """
        beta -= b_cost(b_p, e_p) * b_p * lr_beta
        eta -= e_cost(b_p, e_p) * e_p * lr_eta

        if abs(error_f(b_p, e_p)) < 0.0001:
            break
    return beta, eta
