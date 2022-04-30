import numpy as np

from sdr import SDR
from spatial_pooling import spatial_pooler
import sdr_utils

def generate_random_matrix(n: int, c: int, p: float):
    return np.random.choice([0, 1], size=(n,c), p=[1-p, p])


# Temporal matching phase 1
def phase1(sp_output: SDR, predicted_state: np.ndarray):
    """
    Takes an output from spatial pooling, and a previous predicted state
    and returns the active state of the current time step t.
    """
    l, c = np.shape(predicted_state)
    assert c == sp_output.n

    ans = predicted_state
    for i, _ in enumerate(predicted_state.T):
        if i not in sp_output.index_list:
            ans[:, i] = np.column_stack([0] * l)

    return ans


if __name__ == "__main__":
    n, w = 30, 3
    c = n
    l = 10
    inhibition_factor = int(c * w/n)        # Keeps sparsity equal
    p  = 0.35


    s1 = sdr_utils.generate_random_sdr(n, w)
    s2 = sdr_utils.generate_random_sdr(n, w)

    mat1 = generate_random_matrix(l, c, p)
    mat2 = generate_random_matrix(l, c, p)


    ans = phase1(s1, mat1)
    # print(mat1)



    pass
