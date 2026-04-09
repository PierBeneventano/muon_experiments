"""Matrix sensing problem generation and gradient computation."""
import numpy as np
from numpy.linalg import svd, norm


def generate_problem(m, n, p, rank, kappa=1.0, noise_std=0.0, seed=None):
    """
    Generate a matrix sensing problem.

    Parameters:
        m, n: dimensions of the target matrix W_star (m x n)
        p: number of measurement matrices A_i
        rank: rank of W_star
        kappa: condition number of W_star (ratio of largest to smallest nonzero SV)
        noise_std: standard deviation of additive Gaussian noise on measurements
        seed: random seed

    Returns:
        W_star: m x n target matrix of given rank and condition number
        A: p x m x n array of iid Gaussian measurement matrices
        b: p-vector of measurements b_i = <A_i, W_star> + noise
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate W_star with prescribed rank and condition number
    W_star = _generate_target(m, n, rank, kappa)

    # Generate iid Gaussian measurement matrices
    A = np.random.randn(p, m, n) / np.sqrt(p)

    # Compute measurements
    b = np.array([np.sum(A[i] * W_star) for i in range(p)])

    # Add noise
    if noise_std > 0:
        b = b + noise_std * np.random.randn(p)

    return W_star, A, b


def _generate_target(m, n, rank, kappa):
    """
    Generate an m x n matrix with given rank and condition number kappa.
    Singular values are linearly spaced from 1 to kappa, then embedded
    into the full matrix via random orthogonal bases.
    """
    k = min(rank, min(m, n))

    # Singular values: linearly spaced from 1 to kappa
    if k == 1:
        sigmas = np.array([kappa])
    else:
        sigmas = np.linspace(kappa, 1.0, k)

    # Random orthogonal bases
    U_full = np.linalg.qr(np.random.randn(m, m))[0]
    V_full = np.linalg.qr(np.random.randn(n, n))[0]

    U = U_full[:, :k]
    V = V_full[:, :k]

    W_star = U @ np.diag(sigmas) @ V.T
    return W_star


def generate_block_diagonal_target(K, m0, kappa=1.0):
    """
    Generate a K-block diagonal target matrix.
    Each block is m0 x m0 with condition number kappa.
    Total matrix is (K*m0) x (K*m0).

    Returns:
        W_star: (K*m0) x (K*m0) block-diagonal matrix
        block_bounds: list of (row_start, row_end, col_start, col_end) tuples
    """
    m = K * m0
    W_star = np.zeros((m, m))
    block_bounds = []

    for k in range(K):
        r0 = k * m0
        r1 = (k + 1) * m0
        # Each block is a full-rank m0 x m0 matrix with condition number kappa
        block = _generate_target(m0, m0, m0, kappa)
        # Normalize so each block has unit Frobenius norm
        block = block / (norm(block, 'fro') + 1e-15)
        W_star[r0:r1, r0:r1] = block
        block_bounds.append((r0, r1, r0, r1))

    return W_star, block_bounds


def compute_loss(W, A, b):
    """
    Matrix sensing loss: L(W) = (1/2p) sum_i (<A_i, W> - b_i)^2

    Parameters:
        W: m x n current estimate
        A: p x m x n measurement matrices
        b: p-vector of target measurements

    Returns:
        scalar loss value
    """
    p = len(b)
    residuals = np.array([np.sum(A[i] * W) for i in range(p)]) - b
    return 0.5 * np.sum(residuals**2) / p


def compute_gradient(W, A, b):
    """
    Gradient of matrix sensing loss:
    grad L(W) = (1/p) sum_i (<A_i, W> - b_i) A_i

    Parameters:
        W: m x n current estimate
        A: p x m x n measurement matrices
        b: p-vector of target measurements

    Returns:
        m x n gradient matrix
    """
    p = len(b)
    residuals = np.array([np.sum(A[i] * W) for i in range(p)]) - b
    # G = (1/p) sum_i r_i * A_i
    G = np.tensordot(residuals, A, axes=([0], [0])) / p
    return G


def compute_loss_and_gradient(W, A, b):
    """Compute both loss and gradient in one pass (avoids redundant residual computation)."""
    p = len(b)
    residuals = np.array([np.sum(A[i] * W) for i in range(p)]) - b
    loss = 0.5 * np.sum(residuals**2) / p
    G = np.tensordot(residuals, A, axes=([0], [0])) / p
    return loss, G
