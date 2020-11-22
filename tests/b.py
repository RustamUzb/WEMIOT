import analysis.weibull as w
from analysis.examples import *
from analysis.util import Method
import numpy as np
from scipy.stats import weibull_min



wd = w.Weibull()
#wd.fit(fail06, method='2pComplete', CL=0.90)
wd.fit(failures=Cens2p_02.failures, censored=Cens2p_02.censored, method=Method.MRRCensored2p)
wd.printResults(out='json')
print(Cens2p_02.scale, Cens2p_02.shape)
#wd.showplot()


