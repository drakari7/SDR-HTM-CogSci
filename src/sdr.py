import numpy as np
from math import comb

class SDR:
    """
    Sparse Distributed Representations code implementations.
    Properties and members -
        vec:    n-dimensional vector of n neurons
        n:      size of vector
        w:      No. of ON bits
        s:      Sparsity (value between 0 and 1)
    """
    vec:    np.ndarray          # Vector of n neurons

    def __init__(self, sdr):
        self.vec = np.asarray(sdr)

    # Instantiate from list of ON bit indexes
    @classmethod
    def from_index_list(cls, n: int, index_list: list[int]):
        a = np.zeros(n)
        a[index_list] = 1
        return cls(a)

    # Overloading OR operator to return union of 2 SDRs
    def __or__(self, other):
        v = np.logical_or(self.vec, other.vec).astype(int)
        return SDR(v)

    # Returns size of overlap with another SDR
    def overlap(self, other) -> int:
        return np.dot(self.vec, other.vec)

    # Is there a match with threshhold theta?
    def is_match(self, other, theta) -> bool:
        return self.overlap(other) >= theta

    # Size of overlap set for some value b and w_y
    def overlap_set(self, w_y, b) -> int:
        return comb(self.w, b) * comb(self.n - self.w, w_y - b)

    # --------- Properties ---------------
    @property
    def n(self) -> int:
        return len(self.vec)

    @property
    def w(self) -> int:
        return np.count_nonzero(self.vec)

    @property
    def s(self) -> float:
        return self.w/self.n


def main():
    # For testing without affecting other modules that import this
    a = [1, 0, 0, 1, 0, 1, 0, 0, 0, 1]
    b = [0, 1, 0, 0, 1, 1, 0, 0, 0, 1]
    c = [1, 0, 1, 0, 1, 0, 0, 0, 0, 1]

    s1 = SDR(a)
    s2 = SDR(b)

    s3 = s1 | s2
    print(s3.vec)
    pass


if __name__ == "__main__":
    main()

