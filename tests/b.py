import analysis.weibull as w
from analysis.datasets import *
from util.utilities import *
import jax.numpy as jnp
import pandas as pd

wd = w.Weibull()
a = Cens2p_02

wd.fit(failures=a.failures, censored=a.censored, method=Method.MLECensored2p)
wd.printResults()
print(a.shape, a.scale)
wd.showPlot()




