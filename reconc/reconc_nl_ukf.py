import numpy as np
from typing import List, Union, Dict, Callable, Optional, Tuple


def _cov2cor(cov):
    d = np.sqrt(np.diag(cov))
    eps = 1e-6  # Small regularization value
    d = np.where(d == 0, eps, d)  # Replace zero standard deviations with a small value
    corr = cov / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    return corr


def _schafer_strimmer_cov(x):
    n, p = x.shape

    sMat = np.dot(x.T, x) / n
    tMat = np.diag(np.diag(sMat))
    std_devs = np.sqrt(np.diag(sMat))
    eps = 1e-6  # Regularization value to avoid division by zero
    std_devs = np.where(std_devs == 0, eps, std_devs)  # Regularize std_devs to avoid zero
    xscale = x / std_devs

    rSmat = _cov2cor(sMat)
    rTmat = _cov2cor(tMat)

    xscale_sq = xscale ** 2
    crossprod_xscale_sq = np.dot(xscale_sq.T, xscale_sq)
    crossprod_xscale = np.dot(xscale.T, xscale)
    varSij = (1 / (n * (n - 1))) * (crossprod_xscale_sq - (1 / n) * (crossprod_xscale ** 2))
    np.fill_diagonal(varSij, 0)

    sqSij = (rSmat - rTmat) ** 2
    lambda_star = np.sum(varSij) / np.sum(sqSij)
    lambda_star = np.clip(lambda_star, 0, 1)

    shrink_cov = lambda_star * tMat + (1 - lambda_star) * sMat
    return {'shrink_cov': shrink_cov, 'lambda_star': lambda_star}


def _mean_cov_from_params(params: Dict[str, float], distr: str) -> Tuple[np.ndarray, np.ndarray]:
    if distr == "gaussian":
        mu = np.array([params["mean"]])
        cov = np.array([[params["sd"] ** 2]])
        return mu, cov
    else:
        raise NotImplementedError(f"Distribution '{distr}' not implemented yet.")


def _aggregate_mean_cov(means: List[np.ndarray], covs: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.concatenate(means)
    cov = np.block([
        [covs[i] if i == j else np.zeros((covs[i].shape[0], covs[j].shape[0]))
         for j in range(len(covs))]
        for i in range(len(covs))
    ])
    return mu, cov


def sample_multivariate_gaussian(mean: np.ndarray, cov: np.ndarray, n_samples: int, seed: Optional[int] = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mean, cov, size=n_samples)


def unscented_transform(mu_x, Sigma_x, f, R, alpha=1e-3, beta=2, kappa=1):
    n = mu_x.shape[0]
    lam = 1
    gamma = 3

    Wm = np.full(2 * n + 1, 1 / (2 * (n + lam)))
    Wc = np.copy(Wm)
    Wm[0] = lam / (n + lam)
    Wc[0] = Wm[0] + (1 - alpha ** 2 + beta)
    S = np.linalg.cholesky(Sigma_x + 1e-9 * np.eye(n))
    sigma_pts = np.zeros((2 * n + 1, n))
    sigma_pts[0] = mu_x
    for i in range(n):
        sigma_pts[i + 1] = mu_x + gamma * S[:, i]
        sigma_pts[n + i + 1] = mu_x - gamma * S[:, i]

    z_sigma = np.array([f(pt) for pt in sigma_pts])
    m = z_sigma.shape[1] if z_sigma.ndim > 1 else 1
    z_sigma = z_sigma.reshape(2 * n + 1, m)

    u_pred = np.sum(Wm[:, None] * z_sigma, axis=0)

    S_y = R.copy()
    P_xy = np.zeros((n, m))
    for i in range(2 * n + 1):
        dz = z_sigma[i] - u_pred
        dx = sigma_pts[i] - mu_x
        S_y += Wc[i] * np.outer(dz, dz)
        P_xy += Wc[i] * np.outer(dx, dz)

    K = P_xy @ np.linalg.inv(S_y)

    def condition_on(upper_base_forecasts: np.ndarray):
        mu_post = mu_x + K @ (upper_base_forecasts - u_pred)
        Sigma_post = Sigma_x - K @ S_y @ K.T
        Sigma_post = (Sigma_post + Sigma_post.T) / 2  # Symmetrize
        Sigma_post += 1e-6 * np.eye(Sigma_post.shape[0])  # Regularize
        return mu_post, Sigma_post

    return condition_on


def reconc_nl_ukf(
    bottom_base_forecasts: List[Dict[str, Union[float, np.ndarray]]],
    in_type: List[str],
    distr: List[str],
    f: Callable[[np.ndarray], np.ndarray],
    upper_base_forecasts: np.ndarray,
    R: np.ndarray,
    num_samples: int = 10000,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:

    """
    Probabilistic reconciliation using Unscented Kalman conditioning on a user-defined nonlinear manifold.

    # EXAMPLE 1: Using 'params'
    bottom_base_forecasts = [
        {"mean": 1.0, "sd": 0.1},
        {"mean": 2.0, "sd": 0.1},
        {"mean": 3.0, "sd": 0.1},
    ]
    in_type = ["params"] * 3
    distr = ["gaussian"] * 3


    def f(bottom):
        middle1 = bottom[0] + bottom[1]
        middle2 = bottom[2]
        top = middle1 * middle2
        return np.array([top, middle1, middle2])


    upper_base_forecasts = np.array([8.0, 3.0, 3.0])
    R = 0.01 * np.eye(3)

    reconciled = reconc_nl_ukf(bottom_base_forecasts, in_type, distr, f, upper_base_forecasts, R, num_samples=1000, seed=42)
    print("Params input:", reconciled["bottom_reconciled_samples"].shape)

    # EXAMPLE 2: Using 'samples'
    rng = np.random.default_rng(42)
    s0 = rng.normal(1.0, 0.1, 500)
    s1 = rng.normal(2.0, 0.1, 500)
    s2 = rng.normal(3.0, 0.1, 500)
    r0 = s0 - s0.mean()
    r1 = s1 - s1.mean()
    r2 = s2 - s2.mean()

    bottom_base_forecasts = [
        {"samples": s0, "residuals": r0},
        {"samples": s1, "residuals": r1},
        {"samples": s2, "residuals": r2},
    ]
    in_type = ["samples"] * 3

    reconciled = reconc_nl_ukf(bottom_base_forecasts, in_type, distr, f, upper_base_forecasts, R, num_samples=1000, seed=42)
    print("Samples input:", reconciled["reconciled_samples"].shape)
    """
    if all(t == "samples" for t in in_type):
        try:
            sample_mat = np.stack([bf["samples"] for bf in bottom_base_forecasts], axis=1)
            residual_mat = np.stack([bf["residuals"] for bf in bottom_base_forecasts], axis=1)
        except KeyError as e:
            raise ValueError(f"Missing key in sample-based input: {e}")

        mu_b = np.mean(sample_mat, axis=0)
        Sigma_b = _schafer_strimmer_cov(residual_mat)["shrink_cov"]

    elif all(t == "params" for t in in_type):
        bottom_means = []
        bottom_covs = []
        for i, bf in enumerate(bottom_base_forecasts):
            mu, cov = _mean_cov_from_params(bf, distr[i])
            bottom_means.append(mu)
            bottom_covs.append(cov)
        mu_b, Sigma_b = _aggregate_mean_cov(bottom_means, bottom_covs)

    else:
        raise NotImplementedError("Mixed 'params' and 'samples' input types are not supported.")

    ukf = unscented_transform(mu_b, Sigma_b, f, R)
    mu_post, Sigma_post = ukf(upper_base_forecasts)

    B = sample_multivariate_gaussian(mu_post, Sigma_post, n_samples=num_samples, seed=seed).T
    #U = np.stack([f(B[:, i]) for i in range(B.shape[1])], axis=1)
    #Y = np.vstack([U, B])

    return {
        "bottom_reconciled_samples": B,
        #"upper_reconciled_samples": U,
        #"reconciled_samples": Y
    }

