from scipy.stats import beta

samp_size = 3


ro = 3

def median_rank(samp_size, rank_order, CI):
    a = rank_order
    b = samp_size - (rank_order - 1)
    return beta.ppf(CI, a, b)*100 # first


print(median_rank(12, 10, 0.05))
