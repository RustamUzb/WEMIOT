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

As data the tool need pandas data frame with two columns ( `Time` observation duration of items and  `C` whether they censored or not ). If item is not failed or failed with different failure mode the item would be censored . 
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
After initiating the `Fit` class, access to several properties like `beta, eta` and methods like `hazard_rate_at_time()` will be available . 
