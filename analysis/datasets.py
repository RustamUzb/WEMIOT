
'''
published datasets
'''
class Cens2p_01:
    failures = [45, 46, 34, 45, 32, 24, 58, 79, 56, 63, 34, 50]
    censored = [23, 24, 57, 78, 76, 85, 51, 59]
    shape = 2.7540
    scale = 67.9188
    source = '10.1080/00224065.1997.11979730'
    # shape 2.7540 (lb=1.4977 ub=4.0102)
    # scale 67.9188 (lb=53.8755 ub=81.9622)

class Cens2p_02(object):
    failures = [1500.0, 7000.0, 2250.0, 4000.0, 4300.0]
    censored = [1750.0, 5000.0]
    shape = 2.257
    scale = 4900.1
    source = 'Book: "The new Weibull" by Robert B. Abernethy, C.5'


class Cens2p_03:
    failures = [54, 187, 216, 240, 244, 335, 361, 373, 375, 386]
    censored = [500, 500, 500, 500, 500, 500, 500, 500, 500, 500]
    shape = 1.7207
    scale = 606.5280
    source = 'https://www.itl.nist.gov/div898/handbook/apr/section4/apr413.htm'

class Cens2p_04:
    failures = [3300, 4500, 5200, 6300, 8000, 8400, 9100, 11100]
    censored = [5000, 7200]
    shape = 3.0
    scale = 8200
    source = 'from Manchester Univerity M04 course (tab 4.2)'

class Cens2p_05:
    failures = [30.0, 49.0, 82.0, 90.0, 96.0]
    censored = [10.0, 45.0, 100.0]
    shape = 2.024
    scale = 94.998
    source = 'Book: "The new Weibull" by Robert B. Abernethy, Fig. 2-5'


'''
2 parameter Complete weibull distributions
'''
class Comp2p_01:
    failures = [1.46, 13.75, 137.20, 229.02, 309.39, 373.32, 475.20, 637.80, 767.58, 948.05, 1054.31, 1284.83, 1574.86,
          2000.85, 3184.09, 1.98, 25.79, 138.53, 233.53, 310.53, 383.68, 507.78, 673.82, 799.33, 973.79, 1055.44,
          1341.35, 1680.36, 2135.79, 3304.92, 2.13, 28.51, 147.17, 254.37, 329.05, 390.13, 554.68, 688.02, 841.37,
          994.36, 1097.19, 1354.07, 1689.51, 2252.29, 3.17, 46.56, 147.84, 254.55, 335.20, 439.71, 557.95, 688.54,
          849.86, 999.90, 1130.88, 1380.44, 1726.22, 2441.73, 3.88, 54.62, 166.80, 255.74, 344.72, 441.26, 559.02,
          731.64, 858.26, 999.98, 1139.31, 1387.16, 1765.54, 2571.87, 10.13, 107.75, 183.04, 263.92, 347.81, 446.66,
          581.00, 744.69, 876.76, 1010.75, 1151.90, 1445.69, 1820.57, 2821.35, 13.70, 108.18, 225.62, 273.86, 356.29,
          470.50, 594.39, 760.17, 915.23, 1039.14, 1238.87, 1491.42, 1910.09, 2868.16]
    shape = 0.9295
    scale = 811.8514
    source = 'https://doi.org/10.1016/j.cie.2010.02.001'



'''
3 parameter weibull distributions
'''
class Cens3p_01:
    failures = [46, 64, 83, 105, 123, 150]
    censored = [200, 200, 200, 200]
    shape = 1.79418
    scale = 144.799390
    loc = 30.92
    source = 'https://www.reliawiki.com/index.php/3P-Weibull_Rank_Regression'

