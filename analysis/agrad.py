import autograd.numpy as np  # Thinly-wrapped numpy
from autograd import grad
import math

#beta =1.22 , eta = 131, loc = 7.8 ,mean =131 ; beta -0.00001, eta -mean/10, loc min/100,
#observ_fail = np.array([16, 34, 53, 75, 93, 120, 150, 191, 240, 339], float)


#observ_fail = np.array([16, 34, 53, 75, 93])
#observ_cens = np.array([120, 120, 120, 120, 120])


#https://www.itl.nist.gov/div898/handbook/apr/section4/apr413.htm
#observ_fail = np.array([54, 187, 216, 240, 244, 335, 361, 373, 375, 386])
#observ_cens = np.array([500, 500,500, 500, 500, 500, 500, 500, 500, 500])

#the weibull book
observ_fail = np.array([1500, 2250, 4000, 4300, 7000])
observ_cens = np.array([1750, 5000])

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
    lr_eta = np.mean(observ_cens)

    beta = 0.1
    eta = np.mean(observ_cens)

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

print(lme_weibull(observ_fail, observ_cens))
