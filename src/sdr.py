import numpy as np
from math import comb

class SDR:
    """
    Sparse Distributed Representations code implementations.
    Properties and members -
        vec:    n-dimensional vector of n neurons
        index_list: list of ON bits indices
        n:      size of vector
        w:      No. of ON bits
        s:      Sparsity (value between 0 and 1)
    """
    # vec:    np.ndarray          # Vector of n neurons
    n:              int                 # Dimension
    index_list:     set[int]            # Position of ON bits

    # Instantiate with n and list of ON bits
    def __init__(self, n: int, index_list):
        self.n = n
        self.index_list = set(index_list)

    # Instantiate by providing vector instead
    @classmethod
    def from_vector(cls, vector):
        v = np.asarray(vector)
        n = len(vector)
        indices = np.where(v == 1)[0]
        return cls(n, indices)

    # Overloading OR operator to return union of 2 SDRs
    def __or__(self, other):
        assert self.n == other.n
        new_list = self.index_list.union(other.index_list)
        return SDR(self.n, new_list)

    # Returns size of overlap with another SDR
    def overlap(self, other) -> int:
        return len(self.index_list.intersection(other.index_list))

    # Is there a match with threshhold theta?
    def is_match(self, other, theta) -> bool:
        return self.overlap(other) >= theta

    # Size of overlap set for some value b and w_y
    def overlap_set(self, w_y, b) -> int:
        return comb(self.w, b) * comb(self.n - self.w, w_y - b)

    # --------- Properties ---------------
    @property
    def w(self) -> int:
        return len(self.index_list)

    @property
    def s(self) -> float:
        return self.w/self.n

    @property
    def vec(self) -> np.ndarray:
        a = np.zeros(self.n)
        a[list(self.index_list)] = 1
        return a

# ----------- Testing -------------------------

if __name__ == "__main__":
    a = [1, 0, 0, 1, 0, 1, 0, 0, 0, 1]
    b = [0, 1, 0, 0, 1, 1, 0, 0, 0, 1]
    c = [1, 0, 1, 0, 1, 0, 0, 0, 0, 1]

    s1 = SDR.from_vector(a)
    s2 = SDR.from_vector(b)
    s3 = SDR.from_vector(c)

    s4 = s1 | s2 | s3
    print(s4.w)
    print(s1.overlap(s3))

