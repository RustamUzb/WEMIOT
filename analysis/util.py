from scipy.stats import beta
from enum import Enum

'''
Method returns median rank based on Confidence level (tables in books usually contain 
tables for CL 0.5, 0.05, and 0.95), rank order, and sample size

'''
def median_rank(samp_size, rank_order, CI):
    a = rank_order
    b = samp_size - (rank_order - 1)
    return beta.ppf(CI, a, b) # first

class Method(Enum):
    AUTO = 1
    MLEComplete2p = 2
    MLECensored2p = 3
    MRRCensored2p = 4

