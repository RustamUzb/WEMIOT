import analysis.weibull as w
from analysis.examples import *
from util.utilities import Method

wd = w.Weibull()
#wd.fit(fail06, method='2pComplete', CL=0.90)
wd.fit(failures=Cens2p_02.failures, censored=Cens2p_02.censored, method=Method.MRRCensored2p)
wd.printResults()
print(Cens2p_02.scale, Cens2p_02.shape)
#wd.showplot()


