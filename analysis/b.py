import analysis.weibull as w
import analysis.util as util
import numpy as np
from scipy.stats import weibull_min

size = 100000     # number of samples
k = 0.3     # shape
lam = 105.5   # scale
t = np.array([k, lam])
dist = weibull_min.rvs(t[0], loc=0, scale=t[1], size=size, random_state=1)

# 10.1080/00224065.1997.11979730
# multicensored example
# shape 2.7540 (lb=1.4977 ub=4.0102)
# scale 67.9188 (lb=53.8755 ub=81.9622)
cens01 = [23, 24, 57, 78, 76, 85, 51, 59]
fail01 = [45, 46, 34, 45, 32, 24, 58, 79, 56, 63, 34, 50]

# the new Weibull by Robert B. Abernethy
# C.5
# shape = 2.257
# scale = 4900.1
fail02 = [1500.0, 7000.0, 2250.0, 4000.0, 4300.0]
cens02 = [1750.0, 5000.0]

#https://www.itl.nist.gov/div898/handbook/apr/section4/apr413.htm  beta = 1.7207, eta = 606.52
fail03 = np.array([54, 187, 216, 240, 244, 335, 361, 373, 375, 386])
cens03 = np.array([500, 500, 500, 500, 500, 500, 500, 500, 500, 500])

# from Manchester Univerity M04 course (tab 4.3)
#shape 2.1
#sclae 90
# loc 80
fail04 = [105, 125, 128, 143, 148, 152, 197]
cens04 = [102, 112, 116, 121, 134, 137, 149, 162, 165]


# from Manchester Univerity M04 course (tab 4.2)
# originally used rank regression method
#shape 3.0
#sclae 8200

fail05 = [3300, 4500, 5200, 6300, 8000, 8400, 9100, 11100]
cens05 = [5000, 7200]

# https://doi.org/10.1016/j.cie.2010.02.001
# complete
# scale 0.9295
# shape 811.8514
fail06 = [1.46, 13.75, 137.20, 229.02, 309.39, 373.32, 475.20, 637.80, 767.58, 948.05, 1054.31, 1284.83, 1574.86,
          2000.85, 3184.09, 1.98, 25.79, 138.53, 233.53, 310.53, 383.68, 507.78, 673.82, 799.33, 973.79, 1055.44,
          1341.35, 1680.36, 2135.79, 3304.92, 2.13, 28.51, 147.17, 254.37, 329.05, 390.13, 554.68, 688.02, 841.37,
          994.36, 1097.19, 1354.07, 1689.51, 2252.29, 3.17, 46.56, 147.84, 254.55, 335.20, 439.71, 557.95, 688.54,
          849.86, 999.90, 1130.88, 1380.44, 1726.22, 2441.73, 3.88, 54.62, 166.80, 255.74, 344.72, 441.26, 559.02,
          731.64, 858.26, 999.98, 1139.31, 1387.16, 1765.54, 2571.87, 10.13, 107.75, 183.04, 263.92, 347.81, 446.66,
          581.00, 744.69, 876.76, 1010.75, 1151.90, 1445.69, 1820.57, 2821.35, 13.70, 108.18, 225.62, 273.86, 356.29,
          470.50, 594.39, 760.17, 915.23, 1039.14, 1238.87, 1491.42, 1910.09, 2868.16]

# the new Weibull by Robert B. Abernethy
# Table 2-2
# shape = 2.165
# scale = 79.802
fail07 = [30.0, 49.0, 82.0, 90.0, 96.0]
cens07 = [10.0, 45.0, 100.0]

wd = w.Weibull()
#wd.fit(fail06, method='2pComplete', CF=0.90)
wd.fit(failures=fail07, censored=cens07, method='2pMRRCensored')
wd.printResults()
#print(wd.mean())
wd.plotcdf()


