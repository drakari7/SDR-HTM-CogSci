import numpy as np
import matplotlib.pyplot as plt

from sdr import SDR
import sdr_utils


def generate_random_matrix(n: int, c: int, p: float):
    return np.random.choice([0, 1], size=(n,c), p=[1-p, p])

def spatial_pooler(
        input_sdr: SDR,
        sp_matrix: np.ndarray,
        inhibition_factor: int,
    ):
    """
    Input params:
        input_sdr: SDR of dimension 1xN
        spatial_pooler: Matrix of shape NxC
        inhibition_factor: top K indices which remain after inhibition
    Returns:
        res: The output SDR of 1xC which has the winning columns
        overlap_count: The sorted overlap counts to make overlap curve
    """
    n = input_sdr.n
    res = np.matmul(input_sdr.vec, sp_matrix)

    top_k_indices = np.argsort(res)[-inhibition_factor:]

    ans_sdr = SDR(n, top_k_indices)
    overlap_counts = np.sort(res)[::-1]
    return ans_sdr, overlap_counts


def main():
    n, w = 1024, 30
    c = n
    inhibition_factor = w       # To maintain sparsity
    matrix_sparsity = 0.35


    s1 = sdr_utils.generate_random_sdr(n, w)
    matrix = generate_random_matrix(n, c, matrix_sparsity)

    _, overlap_counts = spatial_pooler(s1, matrix, inhibition_factor)

    cutoff = 400
    top_overlap = overlap_counts[:cutoff]

    plt.plot(range(cutoff), top_overlap)
    plt.title(f"Overlap Curve for top {cutoff} columns (n={n}, w={w})")
    plt.grid()
    plt.legend(['overlap'])
    # plt.show()

    file = sdr_utils.get_graph_dir() + 'overlap_curve.jpg'
    plt.savefig(file, dpi=250)

if __name__ == "__main__":
    main()
