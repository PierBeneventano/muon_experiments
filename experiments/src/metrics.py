"""Metric computation for Muon implicit bias experiments."""
import numpy as np
from numpy.linalg import svd, norm


def spectral_entropy(W, eps=1e-12):
    """H(sigma) = -sum_i p_i log p_i where p_i = sigma_i / sum_j sigma_j."""
    s = svd(W, compute_uv=False)
    s = s[s > eps]
    if len(s) == 0:
        return 0.0
    p = s / s.sum()
    return -np.sum(p * np.log(p + eps))


def max_entropy(m, n=None):
    """Maximum spectral entropy = log(min(m,n))."""
    if n is None:
        n = m
    return np.log(min(m, n))


def normalized_entropy(W, eps=1e-12):
    """H(W) / H_max.  Returns value in [0, 1]."""
    m, n = W.shape
    H = spectral_entropy(W, eps=eps)
    H_max = max_entropy(m, n)
    if H_max < eps:
        return 0.0
    return H / H_max


def S_mu(W, eps=1e-12):
    """S(mu) = sum_{i!=j} 1/(sigma_i + sigma_j)^2 (off-diagonal only, our formula)."""
    s = svd(W, compute_uv=False)
    s = s[s > eps]
    m = len(s)
    total = 0.0
    for i in range(m):
        for j in range(m):
            if i != j:
                total += 1.0 / (s[i] + s[j]) ** 2
    return total


def S_mu_fast(W, eps=1e-12):
    """Vectorized S(mu) computation."""
    s = svd(W, compute_uv=False)
    s = s[s > eps]
    if len(s) == 0:
        return 0.0
    # Compute all pairwise (sigma_i + sigma_j)^2
    si, sj = np.meshgrid(s, s)
    denom = (si + sj) ** 2
    mask = ~np.eye(len(s), dtype=bool)  # off-diagonal mask
    return np.sum(1.0 / denom[mask])


def nuclear_norm(W):
    """Sum of singular values."""
    return np.sum(svd(W, compute_uv=False))


def stable_rank(W, eps=1e-12):
    """r = ||W||_F^2 / ||W||_op^2."""
    s = svd(W, compute_uv=False)
    if s[0] < eps:
        return 0.0
    return (s**2).sum() / s[0]**2


def condition_number(W, eps=1e-12):
    """Ratio of largest to smallest non-negligible singular value."""
    s = svd(W, compute_uv=False)
    s = s[s > eps]
    if len(s) < 2:
        return 1.0
    return s[0] / s[-1]


def cos_theta(P_fullbatch, P_minibatch):
    """Cosine similarity between two update directions (vectorized)."""
    P_f = P_fullbatch.ravel()
    P_m = P_minibatch.ravel()
    denom = norm(P_f) * norm(P_m)
    if denom < 1e-12:
        return 0.0
    return np.dot(P_f, P_m) / denom


def nuclear_to_frobenius_ratio(W, eps=1e-12):
    """||W||_* / ||W||_F  -- maximized at 1 when W is rank-1."""
    s = svd(W, compute_uv=False)
    s = s[s > eps]
    if len(s) == 0:
        return 0.0
    nuc = s.sum()
    fro = np.sqrt((s**2).sum())
    if fro < eps:
        return 0.0
    return nuc / fro


def effective_rank(W, eps=1e-12):
    """
    Roy & Vetterli effective rank: exp(H(sigma/||sigma||_1)).
    Returns a continuous measure in [1, min(m,n)].
    """
    H = spectral_entropy(W, eps=eps)
    return np.exp(H)


def gini_coefficient(values):
    """Gini coefficient of an array of non-negative values."""
    vals = np.sort(np.array(values, dtype=float))
    n = len(vals)
    if n == 0 or vals.sum() < 1e-15:
        return 0.0
    idx = np.arange(1, n + 1)
    return (2 * np.sum(idx * vals) / (n * vals.sum())) - (n + 1) / n


def principal_angles(U1, U2):
    """
    Compute principal angles between the column spans of U1 and U2.
    Both should be column-orthonormal matrices (k columns).
    Returns array of angles in radians.
    """
    # Ensure orthonormal via thin QR
    Q1, _ = np.linalg.qr(U1)
    Q2, _ = np.linalg.qr(U2)
    k = min(Q1.shape[1], Q2.shape[1])
    Q1 = Q1[:, :k]
    Q2 = Q2[:, :k]
    M = Q1.T @ Q2
    svals = svd(M, compute_uv=False)
    svals = np.clip(svals, -1.0, 1.0)
    return np.arccos(svals)


def block_singular_value_mass(W, block_sizes):
    """
    For a block-diagonal target, measure how much of W's spectral mass
    lives in each block's subspace.
    block_sizes: list of (row_start, row_end, col_start, col_end) tuples.
    Returns list of nuclear norms per block, normalized by total nuclear norm.
    """
    total_nuc = nuclear_norm(W)
    if total_nuc < 1e-15:
        return [0.0] * len(block_sizes)
    masses = []
    for (r0, r1, c0, c1) in block_sizes:
        block = W[r0:r1, c0:c1]
        masses.append(nuclear_norm(block) / total_nuc)
    return masses


def atsr(acquisition_times):
    """
    Acquisition Time Spread Ratio.
    ATSR = max(t_k) / min(t_k) where t_k is the time block k reaches
    threshold fraction of its final singular value mass.
    Lower ATSR means more simultaneous acquisition.
    """
    t_arr = np.array(acquisition_times, dtype=float)
    t_arr = t_arr[t_arr > 0]  # exclude blocks never acquired
    if len(t_arr) < 2:
        return 1.0
    return t_arr.max() / t_arr.min()
