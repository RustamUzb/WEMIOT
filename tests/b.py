import analysis.weibull as w
from analysis.datasets import *
from util.utilities import *

wd = w.Weibull()
a = Cens2p_04

wd.fit(failures=a.failures, censored=a.censored, method=Method.MRRCensored2p)
wd.printResults()
print(a.shape, a.scale)
wd.showPlot()



