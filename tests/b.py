import analysis.weibull as w
from analysis.datasets import *
from util.utilities import *
import jax.numpy as jnp
import pandas as pd

wd = w.Weibull()
a = Cens2p_05

wd.fit(failures=a.failures, censored=a.censored, method=Method.MRRCensored2p)
wd.printResults()
print(a.shape, a.scale)
wd.showPlot()




