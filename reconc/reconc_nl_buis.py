# python
import numpy as np
from typing import Callable
from sklearn.neighbors import KernelDensity
from bayesreconpy.utils import _distr_pmf, _check_weights
import numpy as np

def _logpdf_mvn(X: np.ndarray, mean: np.ndarray, cov: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    X = np.asarray(X, float)
    mean = np.asarray(mean, float).reshape(-1)
    cov = np.asarray(cov, float)

    d = mean.size
    cov = 0.5 * (cov + cov.T) + eps * np.eye(d)

    L = np.linalg.cholesky(cov)
    XC = X - mean
    Y = np.linalg.solve(L, XC.T)               # (d, N)
    maha = np.sum(Y * Y, axis=0)               # (N,)
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    return -0.5 * (d * np.log(2*np.pi) + logdet + maha)


def _compute_weights(
    B: np.ndarray,            # (N, n_bottom)
    U_pred: np.ndarray,       # (N, n_upper)  <-- coherent upper predictions for each bottom particle
    joint_mean: np.ndarray,   # (n_upper+n_bottom,)
    joint_cov: np.ndarray, # (n_upper+n_bottom, n_upper+n_bottom)
    eps_cov: float = 1e-9,
) -> np.ndarray:

    B = np.asarray(B)
    U_pred = np.asarray(U_pred)

    N, n_bottom = B.shape
    if U_pred.ndim != 2 or U_pred.shape[0] != N:
        raise ValueError("U_pred must be shape (N, n_upper)")
    n_upper = U_pred.shape[1]

    d = n_upper + n_bottom
    mu = np.asarray(joint_mean, float).reshape(-1)
    Sig = np.asarray(joint_cov, float)

    if mu.size != d:
        raise ValueError(f"joint_mean must have length {d} (=n_upper+n_bottom), got {mu.size}")
    if Sig.shape != (d, d):
        raise ValueError(f"joint_cov must be shape {(d, d)}, got {Sig.shape}")

    X_joint = np.hstack([U_pred, B])  # (N, d)

    # marginal over B: take bottom block
    mu_b = mu[n_upper:]
    Sig_b = Sig[n_upper:, n_upper:]

    log_joint = _logpdf_mvn(X_joint, mu, Sig, eps=eps_cov)
    log_b = _logpdf_mvn(B, mu_b, Sig_b, eps=eps_cov)

    w = np.exp(log_joint - log_b)
    w[~np.isfinite(w)] = 0.0
    w = np.maximum(w, 0.0)
    if w.sum() <= 0:
        w = np.ones_like(w)
    return w

def reconc_nl_buis(
        #upper_base_forecasts,
        #bottom_base_forecasts,
        #in_type,
        #distr,
        f: Callable[[np.ndarray], np.ndarray],
        num_samples=10000,
        n_bot = None,
        seed=None,
        assume_independent=True,
        joint_mean=None,         # REQUIRED if assume_independent=False (true joint [U,B])
        joint_cov=None,          # REQUIRED if assume_independent=False
        eps_cov: float = 1e-9,
):
    if seed is not None:
        np.random.seed(seed)

    N = num_samples
    n_bottom = n_bot
    n_upper = len(joint_mean) - n_bottom

    # ============================================================
    # TRUE JOINT MODE: one-shot weights using Gaussian on [U,B]
    # ============================================================
    if not assume_independent:
        if joint_mean is None or joint_cov is None:
            raise ValueError("Joint mode requires joint_mean and joint_cov for the full vector [U,B].")

        bot_mean = joint_mean[n_upper:]
        bot_cov = joint_cov[n_upper:, n_upper:]
        B = np.random.multivariate_normal(bot_mean, bot_cov, size=N)
        U_pred = f(B)  # expected shape (n_upper, N)
        if U_pred.shape != (n_upper, N):
            raise ValueError(f"f(B) must return shape (n_upper, N)=({n_upper},{N}), got {U_pred.shape}")

        # compute weights w_j ∝ p(U_pred_j, B_j) / p(B_j)
        w = _compute_weights(
            B=B,
            U_pred=U_pred.T,          # (N, n_upper)
            joint_mean=joint_mean,    # (n_upper+n_bottom,)
            joint_cov=joint_cov,      # (n_upper+n_bottom, n_upper+n_bottom)
            eps_cov=eps_cov,
        )

        # normalize safely
        w = np.maximum(w, 0.0)
        w = w / (w.sum() if w.sum() > 0 else 1.0)

        # resample ONCE
        idx = np.random.choice(N, size=N, replace=True, p=w)
        B = B[idx]

        U_final = f(B)
        return {
            "bottom_reconciled_samples": B.T,
            "upper_reconciled_samples": U_final,
            "reconciled_samples": np.vstack([U_final, B.T]),
        }



    """
    # ============================================================
    # INDEPENDENT MODE: keep your sequential BUIS (unchanged)
    # ============================================================
    for i in range(n_upper):
        U_pred = f(B)  # expected (n_upper, N)
        if U_pred.shape != (n_upper, N):
            raise ValueError(f"f(B) must return shape (n_upper, N)=({n_upper},{N}), got {U_pred.shape}")

        pred_u_i = U_pred[i]
        u_i = upper_base_forecasts[i]

        w = _compute_weights(
            b=pred_u_i,
            u=u_i,
            #in_type=in_type[i],
            #distr=distr[i],
            assume_independent=True,
        )

        w = np.maximum(w, 0.0)
        w = w / (w.sum() if w.sum() > 0 else 1.0)

        idx = np.random.choice(N, size=N, replace=True, p=w)
        B = B[idx]

    U_final = f(B)
    return {
        "bottom_reconciled_samples": B.T,
        "upper_reconciled_samples": U_final,
        "reconciled_samples": np.vstack([U_final, B.T]),
    }
   """