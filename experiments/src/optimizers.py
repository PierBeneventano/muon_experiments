"""Optimizer implementations for Muon implicit bias experiments."""
import numpy as np
from numpy.linalg import svd, norm, qr
from scipy.stats import ortho_group


def polar(G, full_matrices=False):
    """Polar factor of G: returns UV^T where G = U Sigma V^T."""
    U, s, Vt = svd(G, full_matrices=False)
    return U @ Vt


def muon_step(W, G, eta, beta=0.0, momentum_buffer=None):
    """
    Muon update: W_{t+1} = W_t - eta * polar(G_t + beta * m_t)
    beta=0:    pure polar map (paper's main case)
    beta=0.95: Nesterov momentum (standard Muon)
    Returns: (new_W, new_momentum_buffer)
    """
    if beta > 0 and momentum_buffer is not None:
        G_eff = G + beta * momentum_buffer
    else:
        G_eff = G
    P = polar(G_eff)
    new_W = W - eta * P
    new_buf = G if beta > 0 else None
    return new_W, new_buf


def gd_step(W, G, eta):
    """Vanilla gradient descent."""
    return W - eta * G


def norm_matched_gd_step(W, G, eta, m=None):
    """
    GD with Frobenius norm rescaled to match Muon's update magnitude.
    Muon's update has ||polar(G)||_F = sqrt(min(m,n)) (all singular values = 1).
    Scale GD update to same Frobenius norm.
    """
    if m is None:
        m = min(W.shape)
    target_norm = np.sqrt(m)
    g_norm = norm(G, 'fro')
    if g_norm < 1e-15:
        return W
    return W - eta * G * (target_norm / g_norm)


def polar_unnormalized_step(W, G, eta):
    """
    Directional structure WITHOUT magnitude normalization.
    U_t = ||G||_F * polar(G)
    This preserves the polar map's isometry projection but restores
    the gradient's original Frobenius magnitude.
    """
    g_norm = norm(G, 'fro')
    if g_norm < 1e-15:
        return W
    P = polar(G)
    return W - eta * g_norm * P


def sign_gd_step(W, G, eta):
    """SignSGD: per-element sign normalization."""
    return W - eta * np.sign(G)


def adam_step(W, G, eta, t, m_buf=None, v_buf=None, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    Adam optimizer step (per-element normalization).
    Returns: (new_W, new_m, new_v)
    """
    if m_buf is None:
        m_buf = np.zeros_like(G)
    if v_buf is None:
        v_buf = np.zeros_like(G)
    m_buf = beta1 * m_buf + (1 - beta1) * G
    v_buf = beta2 * v_buf + (1 - beta2) * G**2
    m_hat = m_buf / (1 - beta1**(t+1))
    v_hat = v_buf / (1 - beta2**(t+1))
    W = W - eta * m_hat / (np.sqrt(v_hat) + eps)
    return W, m_buf, v_buf


def adamw_step(W, G, eta, t, m_buf=None, v_buf=None,
               beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
    """
    AdamW optimizer step (Adam with decoupled weight decay, Loshchilov & Hutter 2019).
    Weight decay is applied directly to W, not through the gradient.
    Returns: (new_W, new_m, new_v)
    """
    if m_buf is None:
        m_buf = np.zeros_like(G)
    if v_buf is None:
        v_buf = np.zeros_like(G)
    m_buf = beta1 * m_buf + (1 - beta1) * G
    v_buf = beta2 * v_buf + (1 - beta2) * G**2
    m_hat = m_buf / (1 - beta1**(t+1))
    v_hat = v_buf / (1 - beta2**(t+1))
    # Decoupled weight decay: applied to W directly, not through adaptive term
    W = W * (1 - eta * weight_decay) - eta * m_hat / (np.sqrt(v_hat) + eps)
    return W, m_buf, v_buf


def muon_wd_step(W, G, eta, weight_decay=0.01):
    """
    Muon (polar map) with decoupled weight decay.
    W_{t+1} = (1 - eta * wd) * W_t - eta * polar(G_t)
    Returns: new_W
    """
    P = polar(G)
    W = W * (1 - eta * weight_decay) - eta * P
    return W


def lars_step(W, G, eta, trust_coeff=0.02):
    """
    LARS: layer-wise adaptive rate scaling.
    Scales gradient by ||W||_F / ||G||_F (per-layer normalization).
    """
    w_norm = norm(W, 'fro')
    g_norm = norm(G, 'fro')
    if g_norm < 1e-15 or w_norm < 1e-15:
        return W
    local_lr = trust_coeff * w_norm / g_norm
    return W - eta * local_lr * G


def subspace_preserving_step(W, G, eta, alpha=2.0):
    """
    Control: perturbs W in G's singular subspace but with NON-flat singular values.
    P = U_G diag(sigma_G^alpha / ||sigma_G^alpha||_1 * m) V_G^T
    alpha=1: rescaled gradient direction; alpha=2: emphasizes large singular values
    """
    U_G, s_G, Vt_G = svd(G, full_matrices=False)
    s_alpha = s_G ** alpha
    s_alpha = s_alpha / s_alpha.sum() * len(s_G)  # normalize to sum = m
    P = U_G @ np.diag(s_alpha) @ Vt_G
    return W - eta * P


def random_orthogonal_step(W, G, eta):
    """
    Random isometry baseline: sample a Haar-uniform (partial) orthogonal matrix
    and use it as the update direction instead of polar(G).
    For m x n matrix with m <= n: sample Q from O(n), take first m rows.
    For m > n: sample Q from O(m), take first n columns.
    The update has the same spectral norm as Muon's polar map (all singular values = 1)
    but a random direction unrelated to the gradient.
    """
    m, n = W.shape
    k = min(m, n)
    if k >= 2:
        # Use scipy ortho_group for Haar-uniform sampling
        if m <= n:
            Q = ortho_group.rvs(n)[:m, :]  # m x n with orthonormal rows
        else:
            Q = ortho_group.rvs(m)[:, :n]  # m x n with orthonormal cols
    else:
        # Degenerate case: random unit vector
        Q = np.random.randn(m, n)
        Q = Q / (norm(Q, 'fro') + 1e-15) * np.sqrt(k)
    return W - eta * Q
