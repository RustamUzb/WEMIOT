# Development progress
* optimisation models (TODO)
* weibull estimation
    * 2p complete rank regression (done)
    * 2p complete MLE (done)
    * 2p censored rank regression (done)
    * 2p censored MEL (done)
    * 3p complete rank regression (TODO)
    * 3p censored rank regression (TODO)
    * 3p MLE (TODO)
# Introduction
**_UNDER DEVELOPMENT_** 

Weibull Enabled Maintenance and Inspection Optimisation Tool (WEMIOT) is the ML driven decision analysis tool for selection 
maintenance policy based on data from a CMMS. 



# Description
Utilisation of information collected by CMMS (computerised maintenance management systems) for improving 
a maintenance policy is a long-standing problem. In many organisations failure and cost data from their asset is not 
used at all. This tool will automatically choose most suitable maintenance policy by solving optimisation problems.

## Optimization problems
* Component replacement interval
* Inspection interval
* Capital equipment replacement interval
* Maintenance resource requirements


### Parameters:
* **Cost**. 
    * Maintenance cost
    * Operation cost
    * Replacement cost
* **Failure rate**. WEMIOT has its own module for evaluation parameters of Weibull distribution based on failure information
from CMMS. 

### Constraints
* **Avialability**
* **Safety**
* other

# References
_`MLA format:_ `
1. Labib, Ashraf W. "A decision analysis model for maintenance policy selection using a CMMS." 
   Journal of Quality in Maintenance Engineering (2004).
1. Jardine, Andrew KS, and Albert HC Tsang. Maintenance, replacement, and reliability: theory and applications. 
   CRC press, 2013.
1. Aggarwal, Charu C. Linear Algebra and Optimization for Machine Learning: A Textbook. Springer Nature, 2020.
