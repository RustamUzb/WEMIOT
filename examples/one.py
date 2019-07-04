import analysis.weibull as weibull
import analysis.lme as alme
import pandas as pd
import numpy as np

#beta =1.22 , eta = 131, loc = 7.8 ,mean =131 ; beta -0.00001, eta -mean/10, loc min/100,
#observ_fail = np.array([16, 34, 53, 75, 93, 120, 150, 191, 240, 339], float)


#observ_fail = np.array([16, 34, 53, 75, 93])
#observ_cens = np.array([120, 120, 120, 120, 120])


#https://www.itl.nist.gov/div898/handbook/apr/section4/apr413.htm
#observ_fail = np.array([54, 187, 216, 240, 244, 335, 361, 373, 375, 386])
#observ_cens = np.array([500, 500,500, 500, 500, 500, 500, 500, 500, 500])

#the weibull book p-257
observ_fail = np.array([1500, 2250, 4000, 4300, 7000])
observ_cens = np.array([1750, 5000])


# create df
df = pd.DataFrame({'times': observ_fail, 'is_failure': np.full(observ_fail.size, 1)})
df_cens = pd.DataFrame({'times': observ_cens, 'is_failure': np.full(observ_cens.size, 0)})

data = pd.concat ([df, df_cens])


w = weibull.Fit(data, fit_type = 'MLE')
#print(df.loc[df['is_failure'] != 1,['times']].to_numpy().squeeze())
print(w.beta)
print(w.eta)
print(w.method)

''
#print(alme.lme_weibull(observ_fail,observ_cens))