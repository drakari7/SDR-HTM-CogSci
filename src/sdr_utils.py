from math import comb
from sdr import SDR
import random
import os


# Number of unique SDRs for given n and w
def uniq_sdr(n, w) -> int:
    return comb(n, w)


# Prob that 2 random SDRs are identical
def prob_identical(n, w) -> float:
    return 1/comb(n, w)


# Exact probability of false positive match of sdr with parameters (w, theta)
def exact_prob_fp(sdr:SDR, w, theta) -> float:
    sum = 0
    for b in range(theta, w+1):
        sum += sdr.overlap_set(w, b)
    denom = uniq_sdr(sdr.n, w)
    return sum/denom


# Approximation of above function
def approx_prob_fp(sdr:SDR, w, theta) -> float:
    return sdr.overlap_set(w, theta) / uniq_sdr(sdr.n, w)


# Returns an SDR which is union of given SDRs
def sdr_union(sdrs : list[SDR]) -> SDR:
    ans = sdrs[0]
    for sdr in sdrs[1:]:
        ans |= sdr
    return ans

# Generates a random SDR with parameters n and w
def generate_random_sdr(n, w):
    rand_positions = random.sample(range(n), w)
    res = SDR(n, rand_positions)
    return res

# def check_match(X, theta):
#     for x in X:
#         for y in X:
#             if x!=y and match(x, y, theta):
#                 return False
#     return True

# Returns location to graph directory so everything is constant
def get_graph_dir():
    curr_file = __file__
    root_dir = '/'.join(curr_file.split('/')[:-2]) + '/'
    return root_dir + 'graphs/'


if __name__ == "__main__":
    p = get_graph_dir()
    print(p)
