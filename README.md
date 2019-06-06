# WeibullR 
A library for Reliability engineers using Weibull analytical function. Designed for integration with CMMS systems.

# Weibull analysis
Weibull distribution has been found useful in reliability engineering because it can represent all three basic distributions of time to failure of the physical items: 

 1. The Normal Distribution - if items are subject to wear-out failure
 2. The Exponential Distribution - if items are subject to random failure
 3. The Hyper-exponential distribution, if items are subject to running-in failure.

# API 
There are several several standalone commercial software available for  reliability engineers. This library has been designed to integrate with CMMS/ EAM databases. 

# Usage
       import analysis.weibull as wb

as data the tool need pandas data frame with two columns ( `Time` observation duration of items and  `C` whether they censored or not ). If item is not failed or failed with different failure mode the item would be censored . 
print(df.head())

    print(df.head())
    """
         Time  C    
    0   600    1  
    1  1300    1  
    2  1345    1  
    3  1378    0  
    4  1978    1 
    """
After initiating the `Fit` class, access to several properties like `beta, eta` and methods like `hazard_rate_at_time()` will be available:   
# Properties

*`beta`* - 
*`eta`* - 
*`average_live`* - 
*`totalOperationTime`* - 
*`mtbf`* -
*`observetions_numbers`* - 
*`failures_numbers`* - 

# Methods
**`b(x)`** - Calculates time when probability of failure reach specific level x. for example if x = 10, the function will return time (duration) when 0.1 of the population will fail. Default value = 10%

|  |  |
|--|--|
| :param x (float) | : probability of failure |
| :return (float) | : time or cycles number when probability of failure equals to x |

*`unreliability_at_time(time, plot)`* - The probability that an item will be in not operational state at a particular point in time. Probability of failure is also known as "unreliability" and it is the reciprocal of the reliability. The function can also plot CDF graph.

|  |  |
|--|--|
| :param time (float) | time / cycles for which CDF value to be calculated |
| :param plot (Boolean) | True if the plot is to be shown, false if otherwise |
| :return | Unreliability value (between 0 and 1) |

**`reliability_at_time(time)`** - Calculates probability that an item will perform its function at a particular point of time.

|  |  |
|--|--|
| :param time (float) | Time / cycles for which Reliability value to be calculated |
| :return | Reliability value (between 0 and 1) |

**`hazard_rate_at_time(time, plot)`** - hazard rate at time

|  |  |
|--|--|
| :param time (float) | Time / cycles for which hazard rate to be calculated |
| :return | hazard rate value |

**`to_csv(folder_name)`** - exports failure data into cvs file in the specified location

|  |  |
|--|--|
| :param floder_name(String) | path to the folder where file will be saved |
| :return | cvs file saved in specified location |

**`plot_weibull_cdf()`** - shows Weibull plot  with basic information
![enter image description here](https://github.com/RustamUzb/weibullR/blob/master/images/Figure_3.png)
