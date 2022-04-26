import numpy as np
from math import comb

class SDR:
    vec:    np.ndarray          # Vector of n neurons
    n:      int                 # Dimension of vector
    w:      int                 # No of ON bits
    s:      float               # Sparsity measure

    def __init__(self, sdr):
        self.vec = np.asarray(sdr)
        self.n = len(self.vec)
        self.w = np.count_nonzero(self.vec)
        self.s = self.w/self.n

    ## Methods defined here are direct properties of SDRs
    # Returns size of overlap with another SDR
    def overlap(self, other) -> int:
        return np.dot(self.vec, other.vec)

    # Is there a match with threshhold theta?
    def match(self, other, theta) -> bool:
        return self.overlap(other) >= theta

    # Size of overlap set for some value b and w_y
    def overlap_set(self, w_y, b) -> int:
        return comb(self.w, b) * comb(self.n - self.w, w_y - b)

## General functions which are not properties of SDRs

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

# def check_match(X, theta):
#     for x in X:
#         for y in X:
#             if x!=y and match(x, y, theta):
#                 return False
#     return True

def main():
    # For testing without affecting other modules that import this
    pass


if __name__ == "__main__":
    main()

