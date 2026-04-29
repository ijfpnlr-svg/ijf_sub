import os
import pickle
import argparse
import time
import numpy as np
import jax.numpy as jnp
from reconc.reconc_nl_buis import reconc_nl_buis
from reconc.reconc_nl_ukf import reconc_nl_ukf, _schafer_strimmer_cov
from reconc.reconc_nl_ols import reconc_nl_ols
from simulation.scripts.score_functions import compute_crps
from scipy.linalg import block_diag
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from scipy.linalg import block_diag
import numpy as np
import jax.numpy as jnp


def sync_any(x):
    if hasattr(x, "block_until_ready"):
        x.block_until_ready()
    elif isinstance(x, dict):
        for v in x.values():
            sync_any(v)
    elif isinstance(x, (list, tuple)):
        for v in x:
            sync_any(v)
    return x


def make_f_ols(U, no_ind_pos):
    no_ind_pos = np.asarray(no_ind_pos)
    expected_dim = 3 * U + 2

    def f_ols(z):
        z = jnp.asarray(z)

        if z.ndim != 1:
            raise ValueError(f"f_ols expects a 1D vector, got shape {z.shape}")
        if z.shape[0] != expected_dim:
            raise ValueError(f"f_ols expects length {expected_dim}, got {z.shape[0]}")

        top = z[0]
        imm_ratio_mid = z[1:U]
        imm_total = z[U]
        pop_total = z[U + 1]
        imm = z[U + 2: 2 * U + 2]
        pop = z[2 * U + 2: 3 * U + 2]

        eps = 1e-8
        c_top = top - imm_total / (pop_total + eps)
        c_mid = imm_ratio_mid - (
                jnp.delete(imm, no_ind_pos) / (jnp.delete(pop, no_ind_pos) + eps)
        )
        c_total_imm = imm_total - jnp.sum(imm)
        c_total_pop = pop_total - jnp.sum(pop)

        return jnp.concatenate([
            jnp.array([c_top]),
            c_mid,
            jnp.array([c_total_imm, c_total_pop]),
        ])

    return f_ols


def _run_single_nl_ols_task(task):
    """
    Worker for one time index i.
    """
    (
        i,
        ratio_top,
        ratio_mid,
        total_target,
        pop_total,
        bot_target,
        pop_bot,
        ratio_top_res,
        ratio_mid_res,
        total_target_res,
        pop_total_res,
        bot_target_res,
        pop_bot_res,
        U,
        no_ind_pos,
        n_iter,
        seed,
    ) = task

    f_ols = make_f_ols(U, no_ind_pos)

    W = _schafer_strimmer_cov(
        np.concatenate([
            ratio_top_res[:, i, :].T,
            ratio_mid_res[:, i, :].T,
            total_target_res[:, i, :].T,
            pop_total_res[:, i, :].T,
            bot_target_res[:, i, :].T,
            pop_bot_res[:, i, :].T,
        ], axis=1)
    )["shrink_cov"]

    P = _to_precision(W)

    Z = np.vstack([
        ratio_top[:, i, :],
        ratio_mid[:, i, :],
        total_target[:, i, :],
        pop_total[:, i, :],
        bot_target[:, i, :],
        pop_bot[:, i, :],
    ]).T

    D = np.diag(np.diag(P))
    B = block_diag(D[:54, :54], P[54:, 54:])

    res_ols = reconc_nl_ols(Z, f_ols, n_iter=n_iter, seed=seed)
    sync_any(res_ols)

    res_wls = reconc_nl_ols(Z, f_ols, W=D, n_iter=n_iter, seed=seed)
    sync_any(res_wls)

    t0_full = time.perf_counter()
    res_full = reconc_nl_ols(Z, f_ols, W=P, n_iter=n_iter, seed=seed)
    sync_any(res_full)
    full_time = time.perf_counter() - t0_full

    res_block = reconc_nl_ols(Z, f_ols, W=B, n_iter=n_iter, seed=seed)
    sync_any(res_block)

    return (
        i,
        res_ols["reconciled_samples"],
        res_wls["reconciled_samples"],
        res_full["reconciled_samples"],
        res_block["reconciled_samples"],
        full_time,
    )


def run_parallel_nl_ols_block(
        ratio_top,
        ratio_mid,
        total_target,
        pop_total,
        bot_target,
        pop_bot,
        ratio_top_res,
        ratio_mid_res,
        total_target_res,
        pop_total_res,
        bot_target_res,
        pop_bot_res,
        U,
        no_ind_pos,
        T,
        n_iter=25,
        seed=42,
        max_workers=None,
):
    tasks = [
        (
            i,
            ratio_top,
            ratio_mid,
            total_target,
            pop_total,
            bot_target,
            pop_bot,
            ratio_top_res,
            ratio_mid_res,
            total_target_res,
            pop_total_res,
            bot_target_res,
            pop_bot_res,
            U,
            no_ind_pos,
            n_iter,
            seed,
        )
        for i in range(T)
    ]

    ols = {}
    wls = {}
    full = {}
    block = {}
    full_time_total = 0.0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_run_single_nl_ols_task, task) for task in tasks]

        for fut in as_completed(futures):
            i, res_ols, res_wls, res_full, res_block, full_time = fut.result()
            ols[i] = res_ols
            wls[i] = res_wls
            full[i] = res_full
            block[i] = res_block
            full_time_total += full_time

    ols_arr = np.stack([ols[t] for t in range(T)], axis=1)
    wls_arr = np.stack([wls[t] for t in range(T)], axis=1)
    full_arr = np.stack([full[t] for t in range(T)], axis=1)
    block_arr = np.stack([block[t] for t in range(T)], axis=1)

    return ols_arr, wls_arr, full_arr, block_arr, full_time_total


# -----------------------------
# Utilities
# -----------------------------
def _row_scale(x_row, method="iqr", eps=1e-12):
    x = np.asarray(x_row, float)
    if method == "iqr":
        q75, q25 = np.nanpercentile(x, [75, 25])
        s = q75 - q25
    elif method == "sd":
        s = np.nanstd(x, ddof=1)
    else:
        raise ValueError("method must be 'iqr' or 'sd'")
    if not np.isfinite(s) or s < eps:
        s = eps
    return float(s)

def _to_precision(cov, eps=1e-6):
    cov = 0.5 * (cov + cov.T)
    lam = eps * np.trace(cov) / cov.shape[0]
    return np.linalg.pinv(cov + lam * np.eye(cov.shape[0]))

# -----------------------------
# Hierarchy functions
# -----------------------------
def f_upper_from_bottom(Bns, ref_id):
    """
    Vectorized map f: bottom -> upper for BUIS.
    Bottom Bns: (N, 2U) with order [imm(U), pop(U)].
    Returns upper: (U+1, N) with order [mid_ratios(U), top_ratio(1)].
    where mid_k = imm_k / pop_k, and top = sum_k w_k * mid_k.
    """
    U = Bns.shape[1] // 2
    imm = Bns[:, :U]
    pop = Bns[:, U:]
    mid_total = np.concatenate([np.sum(imm, axis=1, keepdims=True), np.sum(pop, axis=1, keepdims=True)], axis=1)
    no_ind_pos = np.where(np.array(ref_id) == 'No indication')[0].tolist()
    mid_ratio = np.delete(imm, no_ind_pos, axis=1) / (np.delete(pop, no_ind_pos, axis=1) + 1e-8)
    top = np.sum(imm, axis=1, keepdims=True) / np.sum(pop, axis=1, keepdims=True)
    upper = np.concatenate([top, mid_ratio, mid_total], axis=1)
    return upper.T

def f_upper_from_bottom_single(z, no_ind_pos):
    U = z.shape[0] // 2
    imm = z[:U]
    pop = z[U:]
    mid_total = np.concatenate([np.atleast_1d(np.sum(imm)), np.atleast_1d(np.sum(pop))])
    mid_ratio = np.delete(imm, no_ind_pos) / (np.delete(pop, no_ind_pos) + 1e-8)
    top = np.atleast_1d(np.sum(imm) / np.sum(pop))
    return np.concatenate([top, mid_ratio, mid_total])


# -----------------------------
# CRPS / ES functions
# -----------------------------
import numpy as np

def compute_es(y_true, y_samples):
    """
    Compute the multivariate Energy Score (ES).

    Parameters
    ----------
    y_true : array of shape (n_series, n_splits)
        True multivariate observations.
    y_samples : array of shape (n_series, n_splits, n_samples)
        Predictive samples for each split.

    Returns
    -------
    float
        Mean multivariate energy score across all splits.
    """
    n_series, n_splits, n_samples = y_samples.shape
    es_total = 0.0

    for t in range(n_splits):
        x = y_samples[:, t, :].T  # (n_samples, n_series)
        y = y_true[:, t]          # (n_series,)

        # First term: E||X - y||
        term1 = np.mean(np.linalg.norm(x - y, axis=1))

        # Second term: 0.5 * E||X - X'||
        term2 = 0.5 * np.mean(np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2))

        es_total += term1 - term2

    return es_total / n_splits

def compute_crps_over_level(y_true, y_samples, rows_idx):
    rows_idx = np.asarray(rows_idx)
    T = y_true.shape[1]
    return np.mean([np.mean([compute_crps(y_true[r, t], y_samples[r, t, :]) for r in rows_idx]) for t in range(T)])

def _energy_score_1d(samples, y):
    x = np.asarray(samples, float)
    m = x.shape[0]
    if m < 2:
        return float(np.abs(x.mean() - y))
    term1 = np.mean(np.abs(x - y))
    idx = np.random.permutation(m)
    term2 = np.mean(np.abs(x - x[idx]))
    return float(term1 - 0.5 * term2)


import numpy as np

def compute_es_weighted(y_true, y_samples, w_rows):
    """
    Compute weighted multivariate Energy Score (ES).

    Parameters
    ----------
    y_true : array of shape (R, T)
        True multivariate observations.
    y_samples : array of shape (R, T, m)
        Predictive samples.
    w_rows : array-like of shape (R,)
        Nonnegative weights for each dimension.

    Returns
    -------
    float
        Weighted multivariate energy score averaged over splits.
    """
    R, T, m = y_samples.shape
    w = np.asarray(w_rows, dtype=float)
    sqrt_w = np.sqrt(w)  # used to scale dimensions
    es_total = 0.0

    for t in range(T):
        X = y_samples[:, t, :].T  # (m, R)
        y = y_true[:, t]          # (R,)

        # Weighted coordinates
        Xw = X * sqrt_w
        yw = y * sqrt_w

        d1 = np.mean(np.linalg.norm(Xw - yw, axis=1))
        d2 = 0.5 * np.mean(np.linalg.norm(Xw[:, None, :] - Xw[None, :, :], axis=2))
        es_total += d1 - d2

    return es_total / T


def pbu_block(imm_bot, pop_bot, no_ind_pos):
    U, T, M = pop_bot.shape
    imm_total = np.sum(imm_bot, axis=0).reshape((1,T,M))
    pop_total = np.sum(pop_bot, axis=0).reshape((1,T,M))
    mid_tot = np.concatenate([imm_total,pop_total])
    mid_ratio = np.delete(imm_bot, no_ind_pos, axis=0) / (np.delete(pop_bot, no_ind_pos, axis=0) + 1e-8)
    top = imm_total / pop_total
    return np.concatenate([top, mid_ratio, mid_tot, imm_bot, pop_bot], axis=0)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_pkl",  default="../forecasts/fc_imm_cit_autoarima2_30.pkl")
    parser.add_argument("--base_2_pkl",  default="../forecasts/fc_imm_cit_autoarima_30.pkl")
    parser.add_argument("--train_pkl", default="../forecasts/train_data_new.pkl")
    parser.add_argument("--test_pkl", default="../forecasts/test_autoarima_30.pkl")
    parser.add_argument("--test_2_pkl",  default="../forecasts/test_autoarima2_30.pkl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    # ---- Load data ----
    with open(args.base_2_pkl, "rb") as f: base_2 = pickle.load(f)
    with open(args.base_pkl, "rb") as f: base = pickle.load(f)
    with open(args.train_pkl, "rb") as f: train = pickle.load(f)
    with open(args.test_pkl, "rb") as f: test_data = pickle.load(f)
    with open(args.test_2_pkl, "rb") as f: test_data_2 = pickle.load(f)


    def pick_block(block_key, base=base):
        S = base[block_key]["samples"]
        R = base[block_key]["residuals"]
        return S, R

    pop_bot, pop_bot_res = pick_block("population", base)
    imm_bot, imm_bot_res = pick_block("immigration", base)
    cit_bot, cit_bot_res = pick_block("citizenship", base)
    U, T, M = pop_bot.shape
    ref_uids = base["population"]["uids"]
    no_ind_pos = np.where(np.array(ref_uids) == 'No indication')[0].tolist()
    pop_total, pop_total_res = pick_block("Switzerland_population", base_2)
    imm_total, imm_total_res = pick_block("Switzerland_immigration", base_2)
    cit_total, cit_total_res = pick_block("Switzerland_citizenship", base_2)

    imm_ratio, imm_ratio_res = pick_block("immigration_ratio", base)
    cit_ratio, cit_ratio_res = pick_block("citizenship_ratio", base)
    rat_uids = base['immigration_ratio']['uids']
    ch_uid_pos = np.where(np.array(rat_uids) == 'Switzerland')[0]
    imm_ratio_mid, imm_ratio_mid_res = np.delete(imm_ratio, ch_uid_pos, axis=0), np.delete(imm_ratio_res, ch_uid_pos, axis=0)
    imm_ratio_top, imm_ratio_top_res = imm_ratio[ch_uid_pos,:,:], imm_ratio_res[ch_uid_pos,:,:]
    cit_ratio_mid, cit_ratio_mid_res = np.delete(cit_ratio, ch_uid_pos, axis=0), np.delete(cit_ratio_res, ch_uid_pos, axis=0)
    cit_ratio_top, cit_ratio_top_res = cit_ratio[ch_uid_pos,:,:], cit_ratio_res[ch_uid_pos,:,:]

    timing_summary = {
        "immigration": {},
        "citizenship": {},
    }

    # -----------------------------
    # --- Recon Methods ---
    # -----------------------------
    # 1) PBU

    pbu_imm = pbu_block(imm_bot, pop_bot, no_ind_pos)
    pbu_cit = pbu_block(cit_bot, pop_bot, no_ind_pos)
    print("PBU immigration:", pbu_imm.shape, "PBU citizenship:", pbu_cit.shape)

    # 2) NL-UKF
    print("— Running NL-UKF…")

    t0_ukf_imm = time.perf_counter()
    nlukf_imm = {}
    for t in range(T):
        u_obs = np.mean(np.vstack([imm_ratio_top[:, t, :], imm_ratio_mid[:, t, :],
                                   imm_total[:,t,:], pop_total[:,t,:]]), axis=1)
        R = _schafer_strimmer_cov((np.vstack([imm_ratio_top_res[:, t, :],
                                              imm_ratio_mid_res[:, t, :],
                                              imm_total_res[:,t,:],
                                              pop_total_res[:,t,:]])).T)['shrink_cov']

        bot_list = []
        for k in range(U):  # immigration totals
            bot_list.append({"samples": imm_bot[k, t, :], "residuals": imm_bot_res[k, t, :]})
        for k in range(U):  # population totals
            bot_list.append({"samples": pop_bot[k, t, :], "residuals": pop_bot_res[k, t, :]})

        def f_single(z):
            return f_upper_from_bottom_single(z, no_ind_pos)

        out = reconc_nl_ukf(
            bottom_base_forecasts=bot_list,
            in_type=["samples"] * (2 * U),
            distr=["gaussian"] * (2 * U),
            f=f_single,
            upper_base_forecasts=u_obs,
            R=R,
            num_samples=M,
            seed=args.seed,
        )
        sync_any(out)
        Brec = out["bottom_reconciled_samples"]
        Urec = f_upper_from_bottom(Brec.T, ref_id=ref_uids)
        nlukf_imm[t] = np.vstack([Urec, Brec])

    nlukf_imm = np.stack([nlukf_imm[t] for t in range(T)], axis=1)
    timing_summary["immigration"]["UKF"] = time.perf_counter() - t0_ukf_imm
    print("NL-UKF immigration:", nlukf_imm.shape)

    t0_ukf_cit = time.perf_counter()
    nlukf_cit = {}
    for t in range(T):
        u_obs = np.mean(np.vstack([cit_ratio_top[:, t, :], cit_ratio_mid[:, t, :],
                                   cit_total[:,t,:], pop_total[:,t,:]]), axis=1)
        R = _schafer_strimmer_cov((np.vstack([cit_ratio_top_res[:, t, :],
                                              cit_ratio_mid_res[:, t, :],
                                              cit_total_res[:,t,:],
                                              pop_total_res[:,t,:]])).T)['shrink_cov']

        bot_list = []
        for k in range(U):
            bot_list.append({"samples": cit_bot[k, t, :], "residuals": cit_bot_res[k, t, :]})
        for k in range(U):
            bot_list.append({"samples": pop_bot[k, t, :], "residuals": pop_bot_res[k, t, :]})

        def f_single(z):
            return f_upper_from_bottom_single(z, no_ind_pos)

        out = reconc_nl_ukf(
            bottom_base_forecasts=bot_list,
            in_type=["samples"] * (2 * U),
            distr=["gaussian"] * (2 * U),
            f=f_single,
            upper_base_forecasts=u_obs,
            R=R,
            num_samples=M,
            seed=args.seed,
        )
        sync_any(out)
        Brec = out["bottom_reconciled_samples"]
        Urec = f_upper_from_bottom(Brec.T, ref_id=ref_uids)
        nlukf_cit[t] = np.vstack([Urec, Brec])

    nlukf_cit = np.stack([nlukf_cit[t] for t in range(T)], axis=1)
    timing_summary["citizenship"]["UKF"] = time.perf_counter() - t0_ukf_cit
    print("NL-UKF citizenship:", nlukf_cit.shape)

    # ======================================================
    # 3) NL-OLS
    # ======================================================
    print("— Running NL-OLS…")

    # ======================================================
    # 3) NL-OLS
    # ======================================================
    print("— Running NL-OLS…")

    f_ols = make_f_ols(U, no_ind_pos)  # optional, no longer directly used below

    # Citizenship block
    ols_cit, wls_cit, full_cit, block_cit, full_time_cit = run_parallel_nl_ols_block(
        ratio_top=cit_ratio_top,
        ratio_mid=cit_ratio_mid,
        total_target=cit_total,
        pop_total=pop_total,
        bot_target=cit_bot,
        pop_bot=pop_bot,
        ratio_top_res=cit_ratio_top_res,
        ratio_mid_res=cit_ratio_mid_res,
        total_target_res=cit_total_res,
        pop_total_res=pop_total_res,
        bot_target_res=cit_bot_res,
        pop_bot_res=pop_bot_res,
        U=U,
        no_ind_pos=no_ind_pos,
        T=T,
        n_iter=25,
        seed=42,
        max_workers=None,
    )
    timing_summary["citizenship"]["full"] = full_time_cit

    print("OLS citizenship:", ols_cit.shape)
    print("W-OLS citizenship:", wls_cit.shape)
    print("full citizenship:", full_cit.shape)
    print("block citizenship:", block_cit.shape)

    # Immigration block
    ols_imm, wls_imm, full_imm, block_imm, full_time_imm = run_parallel_nl_ols_block(
        ratio_top=imm_ratio_top,
        ratio_mid=imm_ratio_mid,
        total_target=imm_total,
        pop_total=pop_total,
        bot_target=imm_bot,
        pop_bot=pop_bot,
        ratio_top_res=imm_ratio_top_res,
        ratio_mid_res=imm_ratio_mid_res,
        total_target_res=imm_total_res,
        pop_total_res=pop_total_res,
        bot_target_res=imm_bot_res,
        pop_bot_res=pop_bot_res,
        U=U,
        no_ind_pos=no_ind_pos,
        T=T,
        n_iter=25,
        seed=42,
        max_workers=None,  # or set e.g. 8
    )
    timing_summary["immigration"]["full"] = full_time_imm

    print("OLS immigration:", ols_imm.shape)
    print("W-OLS immigration:", wls_imm.shape)
    print("full immigration:", full_imm.shape)
    print("block immigration:", block_imm.shape)


    # ======================================================
    # 4) P + NL-BUIS
    # ======================================================

    print("— Running IS…")


    def f_buis(B):
        """
        B: (M, 2U) = [imm(U), pop(U)]
        Returns:
            upper: (1 + (U-1) + 2, M)
                 = [top_ratio(1), mid_ratios(U-1), swiss_totals(2)]
        """
        P = B.shape[1] // 2
        imm = B[:, :P]
        pop = B[:, P:]

        mid_total = np.concatenate(
            [
                np.sum(imm, axis=1, keepdims=True),
                np.sum(pop, axis=1, keepdims=True),
            ],
            axis=1,
        )  # (M, 2)

        mid_ratio = np.delete(imm, no_ind_pos, axis=1) / (
                np.delete(pop, no_ind_pos, axis=1) + 1e-8
        )  # (M, U-1)

        top = np.sum(imm, axis=1, keepdims=True) / (
                np.sum(pop, axis=1, keepdims=True) + 1e-8
        )  # (M, 1)

        upper = np.concatenate([top, mid_ratio, mid_total], axis=1)  # (M, 1+(U-1)+2)
        return upper.T  # (n_upper, M)

    nl_buis_imm = {}

    for t in range(T):
        # ----------------------------------------
        # Bottom base samples: [imm_bottoms, pop_bottoms]
        # shape: (2U, M)
        # ----------------------------------------
        fc_bot_arr = np.vstack([
            imm_bot[:, t, :],
            pop_bot[:, t, :]
        ])

        # ----------------------------------------
        # Upper base samples: [top_ratio, mid_ratios, swiss_totals]
        # shape: (1 + (U-1) + 2, M)
        # ----------------------------------------
        fc_upp_arr = np.vstack([
            imm_ratio_top[:, t, :],
            imm_ratio_mid[:, t, :],
            imm_total[:, t, :],
            pop_total[:, t, :]
        ])

        n_upper = fc_upp_arr.shape[0]
        n_bottom = fc_bot_arr.shape[0]

        # Full joint base vector [U, B]
        joint_base = np.vstack([fc_upp_arr, fc_bot_arr])  # (n_upper + n_bottom, M)

        # Joint Gaussian approximation from residuals of the same full structure
        joint_resid = np.concatenate([
            imm_ratio_top_res[:, t, :].T,
            imm_ratio_mid_res[:, t, :].T,
            imm_total_res[:, t, :].T,
            pop_total_res[:, t, :].T,
            imm_bot_res[:, t, :].T,
            pop_bot_res[:, t, :].T,
        ], axis=1)  # (window_or_res_samples, n_upper+n_bottom)

        joint_cov = _schafer_strimmer_cov(joint_resid)["shrink_cov"]
        joint_mean = np.mean(joint_base, axis=1)

        buis_res = reconc_nl_buis(
            assume_independent=False,
            joint_mean=joint_mean,
            joint_cov=joint_cov,
            n_bot=n_bottom,
            f=f_buis,
            num_samples=M,
            seed=args.seed,
        )

        nl_buis_imm[t] = buis_res["reconciled_samples"]

    buis_imm = np.stack([nl_buis_imm[t] for t in range(T)], axis=1)
    print("NL-BUIS immigration:", buis_imm.shape)

    nl_buis_cit = {}

    for t in range(T):
        # ----------------------------------------
        # Bottom base samples: [cit_bottoms, pop_bottoms]
        # shape: (2U, M)
        # ----------------------------------------
        fc_bot_arr = np.vstack([
            cit_bot[:, t, :],
            pop_bot[:, t, :]
        ])

        # ----------------------------------------
        # Upper base samples: [top_ratio, mid_ratios, swiss_totals]
        # shape: (1 + (U-1) + 2, M)
        # ----------------------------------------
        fc_upp_arr = np.vstack([
            cit_ratio_top[:, t, :],
            cit_ratio_mid[:, t, :],
            cit_total[:, t, :],
            pop_total[:, t, :]
        ])

        n_upper = fc_upp_arr.shape[0]
        n_bottom = fc_bot_arr.shape[0]

        # Full joint base vector [U, B]
        joint_base = np.vstack([fc_upp_arr, fc_bot_arr])  # (n_upper + n_bottom, M)

        # Residual matrix for the same stacked structure
        joint_resid = np.concatenate([
            cit_ratio_top_res[:, t, :].T,
            cit_ratio_mid_res[:, t, :].T,
            cit_total_res[:, t, :].T,
            pop_total_res[:, t, :].T,
            cit_bot_res[:, t, :].T,
            pop_bot_res[:, t, :].T,
        ], axis=1)

        joint_cov = _schafer_strimmer_cov(joint_resid)["shrink_cov"]
        joint_mean = np.mean(joint_base, axis=1)

        buis_res = reconc_nl_buis(
            assume_independent=False,
            joint_mean=joint_mean,
            joint_cov=joint_cov,
            n_bot=n_bottom,  # or n_bot if your implementation still expects that exact name
            f=f_buis,
            num_samples=M,
            seed=args.seed,
        )

        nl_buis_cit[t] = buis_res["reconciled_samples"]

    buis_cit = np.stack([nl_buis_cit[t] for t in range(T)], axis=1)
    print("NL-BUIS citizenship:", buis_cit.shape)
    print("----Computing Energy Scores----")

    test_imm_ratio = test_data_2['immigration_ratio']['y_true']
    test_cit_ratio = test_data_2['citizenship_ratio']['y_true']
    test_imm_total = test_data['Switzerland_immigration']['y_true'].reshape(1,30)
    test_pop_total = test_data['Switzerland_population']['y_true'].reshape(1,30)
    test_cit_total = test_data['Switzerland_citizenship']['y_true'].reshape(1,30)
    test_imm = test_data_2['immigration']['y_true']
    test_pop = test_data_2['population']['y_true']
    test_cit = test_data_2['citizenship']['y_true']

    test_imm_data = np.vstack([test_imm_ratio[ch_uid_pos,:],
                                    np.delete(test_imm_ratio, ch_uid_pos, axis=0),
                                    test_imm_total,
                                    test_pop_total,
                                    test_imm,
                                    test_pop])

    test_cit_data = np.vstack([test_cit_ratio[ch_uid_pos,:],
                                    np.delete(test_cit_ratio, ch_uid_pos, axis=0),
                                    test_cit_total,
                                    test_pop_total,
                                    test_cit,
                                    test_pop])

    base_imm = np.vstack([imm_ratio_top,
                              imm_ratio_mid,
                              imm_total,
                              pop_total,
                              imm_bot,
                              pop_bot])

    base_cit = np.vstack([cit_ratio_top,
                              cit_ratio_mid,
                              cit_total,
                              pop_total,
                              cit_bot,
                              pop_bot])

    # forecast_methods_imm = {
    #     "Base": base_imm,
    #     "PBU": pbu_imm,
    #     "UKF": nlukf_imm,
    #     "OLS": ols_imm,
    #     "WLS": wls_imm,
    #     "full": full_imm,
    #     "block": block_imm,
    #     "IS": buis_imm
    # }
    #
    # forecast_methods_cit = {
    #     "Base": base_cit,
    #     "PBU": pbu_cit,
    #     "UKF": nlukf_cit,
    #     "OLS": ols_cit,
    #     "WLS": wls_cit,
    #     "full": full_cit,
    #     "block": block_cit,
    #     "IS": buis_cit
    # }
    #
    # def build_row_weights_from_train(train, target="immigration", method="iqr", eps=1e-12):
    #     """
    #     Construct row weights for (1+3U) rows using last-available train value per (row, test_year).
    #     Matches the new hierarchy structure:
    #         [top_ratio, mid_ratios(U), top_totals(imm,pop), bottoms(imm,pop)]
    #     """
    #     if target == "immigration":
    #         top_pack = train["immigration_ratio"]["train_values"]  # (1,T,W)
    #         mid_pack = train["immigration_ratio_kanton"]["train_values"]  # (U,T,W)
    #         bot_pack = train["immigration"]["train_values"]  # (U,T,W)
    #         top_imm_pack = train["Switzerland_immigration"]["train_values"]  # (1,T,W)
    #     elif target == "citizenship":
    #         top_pack = train["citizenship_ratio"]["train_values"]
    #         mid_pack = train["citizenship_ratio_kanton"]["train_values"]
    #         bot_pack = train["citizenship"]["train_values"]
    #         top_imm_pack = train["Switzerland_citizenship"]["train_values"]
    #     else:
    #         raise ValueError("target must be 'immigration' or 'citizenship'")
    #
    #     pop_pack = train["population"]["train_values"]
    #     top_pop_pack = train["Switzerland_population"]["train_values"]  # (1,T,W)
    #
    #     def last_W(x):  # (R,T,W)->(R,T)
    #         return np.asarray(x, float)[:, :, -1]
    #
    #     # just extract last W; all already aligned
    #     top_ratio = last_W(top_pack)  # (1,T)
    #     mid_ratio = last_W(mid_pack)  # (U,T)
    #     bot_imm = last_W(bot_pack)  # (U,T)
    #     bot_pop = last_W(pop_pack)  # (U,T)
    #     top_imm = last_W(top_imm_pack)  # (1,T)
    #     top_pop = last_W(top_pop_pack)  # (1,T)
    #
    #     # hierarchy stacking order: [top ratio, mid ratios(U), top imm, top pop, bot imm(U), bot pop(U)]
    #     Y = np.vstack([top_ratio, mid_ratio, top_imm, top_pop, bot_imm, bot_pop])
    #     scales = np.array([_row_scale(Y[i, :], method=method, eps=eps) for i in range(Y.shape[0])], dtype=float)
    #     w_rows = 1.0 / scales
    #     w_rows /= np.mean(w_rows)
    #     return w_rows
    #
    # w_rows_imm = build_row_weights_from_train(train, target="immigration", method="iqr")
    # w_rows_cit = build_row_weights_from_train(train, target="citizenship", method="iqr")
    #
    # print(w_rows_imm.shape, w_rows_cit.shape)
    #
    # print("\n🔹 Energy Scores for Immigration")
    # for method, y_hat in forecast_methods_imm.items():
    #     es = compute_es(test_imm_data, y_hat)
    #     wes = compute_es_weighted(test_imm_data, y_hat, w_rows_imm)
    #     print(f"{method:<10s} ES = {es:.4f}  W-ES ={wes:.4f}")
    #
    # print("\n🔹 Energy Scores for Citizenship")
    # for method, y_hat in forecast_methods_cit.items():
    #     es = compute_es(test_cit_data, y_hat)
    #     wes = compute_es_weighted(test_cit_data, y_hat, w_rows_cit)
    #     print(f"{method:<10s} ES = {es:.4f} W-ES ={wes:.4f}")
    #
    def compute_crps_over_level(y_true, y_samples, rows_idx, average_within_level=True):
        """
        Compute CRPS for a subset of rows (level).
        y_true: (R, T)
        y_samples: (R, T, M)
        rows_idx: list or array of row indices for the level
        """
        rows = np.asarray(rows_idx, dtype=int)
        T = y_true.shape[1]
        per_time = []
        for t in range(T):
            crps_rows = compute_crps(y_true[rows, t], y_samples[rows, t, :])
            if average_within_level:
                per_time.append(np.nanmean(crps_rows))
            else:
                per_time.append(crps_rows)
        per_time = np.array(per_time)
        if average_within_level:
            return float(np.nanmean(per_time))
        return per_time.T  # (len(rows), T)

    def crps_table_and_relative(forecast_methods, gt, U, show_levels=None):
        """
        Compute absolute and relative CRPS per level for all methods.
        show_levels: dict[level_name -> list of row indices]
        Hierarchy layout:
            [0]                : top ratio (Switzerland)
            [1:1+U]            : cantonal ratios
            [U+1:U+3]          : Swiss totals (immigration, population)
            [U+3:U+3+U]        : bottom immigration
            [U+3+U:U+3+2U]     : bottom population
        """
        if show_levels is None:
            show_levels = {
                "full": np.arange(gt.shape[0]),
                "top": [0],
                "middle_ratios": list(range(1, 1 + U)),
                "swiss_totals": list(range(1 + U, 1 + U + 2)),
                "bottom": list(range(1 + U + 2, gt.shape[0]))
            }

        abs_crps = {}
        for name, y_hat in forecast_methods.items():
            lvl_scores = {}
            for lvl_name, idxs in show_levels.items():
                lvl_scores[lvl_name] = compute_crps_over_level(gt, y_hat, idxs, average_within_level=True)
            abs_crps[name] = lvl_scores

        # take first entry as baseline if "Base" missing
        base_scores = abs_crps.get("Base", next(iter(abs_crps.values())))
        rel_crps = {}
        for name, scores in abs_crps.items():
            rel = {}
            for lvl_name, val in scores.items():
                base_val = base_scores.get(lvl_name, np.nan)
                rel[lvl_name] = val / base_val if np.isfinite(base_val) and base_val != 0 else np.nan
            rel_crps[name] = rel

        # print table
        print(f"{'Method':<12s} | {'Full':>10s} | {'Top':>10s} | {'Ratios':>10s} | {'Totals':>10s} | {'Bottom':>10s}")
        print("-" * 68)
        for name in abs_crps:
            a, r = abs_crps[name], rel_crps[name]
            print(f"{name:<12s} | "
                  f"{a['full']:.4g} ({r['full']:.3f}x) | "
                  f"{a['top']:.4g} ({r['top']:.3f}x) | "
                  f"{a['middle_ratios']:.4g} ({r['middle_ratios']:.3f}x) | "
                  f"{a['swiss_totals']:.4g} ({r['swiss_totals']:.3f}x) | "
                  f"{a['bottom']:.4g} ({r['bottom']:.3f}x)")

    def crps_relative_geomean_over_series(
            forecast_methods: dict,
            gt: np.ndarray,
            baseline_name: str = "Base",
            eps: float = 1e-12,
    ):
        """
        For each series j:
          1) compute avg CRPS over time: CRPS_method[j] = mean_t CRPS(y_true[j,t], y_samples[j,t,:])
          2) compute ratio per method: r_method[j] = CRPS_method[j] / CRPS_base[j]
          3) aggregate over j with geometric mean: GM = exp(mean_j log(r_method[j]))

        Parameters
        ----------
        forecast_methods : dict[name -> y_samples]
            Each y_samples has shape (R, T, M).
        gt : array (R, T)
            Ground truth.
        baseline_name : str
            Key in forecast_methods to use as baseline; if missing, first entry is used.
        eps : float
            Numerical floor to avoid divide-by-zero / log(0).

        Returns
        -------
        per_series_avg : dict[name -> array (R,)]
            Average CRPS per series (avg over time).
        per_series_ratio : dict[name -> array (R,)]
            Per-series ratio to baseline.
        gm_ratio : dict[name -> float]
            Geometric mean of per-series ratios (ignoring NaNs).
        """
        # pick baseline
        if baseline_name in forecast_methods:
            base_key = baseline_name
        else:
            base_key = next(iter(forecast_methods.keys()))

        R, T = gt.shape
        per_series_avg = {}

        # --- avg CRPS per series for each method
        for name, y_hat in forecast_methods.items():
            if y_hat.shape[:2] != (R, T):
                raise ValueError(f"{name}: expected shape (R,T,M) with (R,T)=({R},{T}), got {y_hat.shape}")

            avg_j = np.full(R, np.nan, dtype=float)
            for j in range(R):
                crps_t = []
                for t in range(T):
                    # compute_crps expects y_section shape (n_series,) and y_hat_section (n_series, M)
                    c = compute_crps(gt[j:j + 1, t], y_hat[j:j + 1, t, :])[0]
                    crps_t.append(c)
                avg_j[j] = np.nanmean(crps_t)
            per_series_avg[name] = avg_j

        # --- ratios to baseline, per series
        base_avg = per_series_avg[base_key]
        base_safe = np.where(np.isfinite(base_avg) & (base_avg > 0), base_avg, np.nan)

        per_series_ratio = {}
        gm_ratio = {}

        for name, avg_j in per_series_avg.items():
            ratio = avg_j / base_safe
            # avoid 0 or negative (shouldn't happen for CRPS, but protect logs)
            ratio = np.where(np.isfinite(ratio), np.maximum(ratio, eps), np.nan)
            per_series_ratio[name] = ratio

            valid = np.isfinite(ratio)
            if np.any(valid):
                gm_ratio[name] = float(np.exp(np.nanmean(np.log(ratio[valid]))))
            else:
                gm_ratio[name] = np.nan

        return per_series_avg, per_series_ratio, gm_ratio

    def crps_gm_table(
            forecast_methods: dict,
            gt: np.ndarray,
            baseline_name: str = "Base",
    ):
        """
        Convenience printer: shows GM over series of (CRPS_method_j / CRPS_base_j),
        plus the baseline GM (=1 by construction if baseline exists).
        """
        per_series_avg, per_series_ratio, gm_ratio = crps_relative_geomean_over_series(
            forecast_methods, gt, baseline_name=baseline_name
        )

        # print
        base_key = baseline_name if baseline_name in forecast_methods else next(iter(forecast_methods.keys()))
        print(f"Baseline for ratios: {base_key}")
        print(f"{'Method':<12s} | {'GM(CRPS/BASE)':>14s}")
        print("-" * 30)
        for name in forecast_methods.keys():
            val = gm_ratio.get(name, np.nan)
            if np.isfinite(val):
                print(f"{name:<12s} | {val:>14.3f}x")
            else:
                print(f"{name:<12s} | {'nan':>14s}")

        return per_series_avg, per_series_ratio, gm_ratio

    def extract_relative_scores_for_middle_level(forecast_methods, ground_truth, U, base_key="Base", pbu_key="PBU"):
        """
        Compute relative CRPS for cantonal ratios only (rows 1:1+U),
        excluding Swiss totals.
        Returns:
            {method_name: {"rel_to_base": x, "rel_to_pbu": y}}
        """
        from simulation.scripts.score_functions import compute_crps

        middle_idx = list(range(1, 1 + U))  # only cantonal ratios
        y_true = ground_truth[middle_idx, :]

        def middle_view(x):
            return x[middle_idx, :, :]

        base_score = np.mean([
            compute_crps(y_true[:, t], middle_view(forecast_methods[base_key])[:, t, :])
            for t in range(y_true.shape[1])
        ])
        pbu_score = np.mean([
            compute_crps(y_true[:, t], middle_view(forecast_methods[pbu_key])[:, t, :])
            for t in range(y_true.shape[1])
        ])

        out = {}
        for name, yhat in forecast_methods.items():
            if name not in [base_key, pbu_key]:
                model_score = np.mean([
                    compute_crps(y_true[:, t], middle_view(yhat)[:, t, :])
                    for t in range(y_true.shape[1])
                ])
                out[name] = {
                    "rel_to_base": model_score / base_score if base_score else np.nan,
                    "rel_to_pbu": model_score / pbu_score if pbu_score else np.nan
                }
        return out

    # ----------------------------------------------------------
    # Evaluate on Immigration & Citizenship
    # ----------------------------------------------------------

    methods_imm = {
        "Base": base_imm,
        "PBU": pbu_imm,
        "UKF": nlukf_imm,
        "OLS": ols_imm,
        "WLS": wls_imm,
        "full": full_imm,
        #"block": block_imm,
        #"IS": buis_imm
    }

    methods_cit = {
        "Base": base_cit,
        "PBU": pbu_cit,
        "UKF": nlukf_cit,
        "OLS": ols_cit,
        "WLS": wls_cit,
        "full": full_cit,
        #"block": block_cit,
        #"IS": buis_cit
    }

    show_levels = {
        "full": np.arange(test_imm_data.shape[0]),
        "top": [0],
        "middle_ratios": list(range(1, U)),
        "swiss_totals": list(range(U, U + 2)),
        "bottom": list(range(U + 2, test_imm_data.shape[0]))
    }

    # If you have the ratio UID list, pass it; otherwise omit `uids=...`
    # For immigration
    #plot_middle_ratios_3sigma("Immigration", base_imm, test_imm_data, U, uids=rat_uids, max_plots=26)

    #plot_middle_ratios_3sigma("Citizenship", base_cit, test_cit_data, U, uids=rat_uids, max_plots=26)

    print("\n🔹 Computation times (seconds)")
    print(f"Immigration  - UKF :  {timing_summary['immigration']['UKF']:.4f}")
    print(f"Immigration  - full:  {timing_summary['immigration']['full']:.4f}")
    print(f"Citizenship  - UKF :  {timing_summary['citizenship']['UKF']:.4f}")
    print(f"Citizenship  - full:  {timing_summary['citizenship']['full']:.4f}")


    print("\n🔹 CRPS — Immigration")
    crps_table_and_relative(methods_imm, test_imm_data, U, show_levels)
    crps_gm_table(methods_imm, test_imm_data, baseline_name="Base")
    crps_relative_geomean_over_series(methods_imm, test_imm_data, baseline_name="Base")

    print("\n🔹 CRPS — Citizenship")
    crps_table_and_relative(methods_cit, test_cit_data, U, show_levels)
    crps_gm_table(methods_cit, test_cit_data, baseline_name="Base")
    crps_relative_geomean_over_series(methods_cit, test_cit_data, baseline_name="Base")

    rel_scores_imm = extract_relative_scores_for_middle_level(methods_imm, test_imm_data, U=U)
    rel_scores_cit = extract_relative_scores_for_middle_level(methods_cit, test_cit_data, U=U)


    with open("../forecasts/relative_scores_imm_2.pkl", "wb") as f:
        pickle.dump(rel_scores_imm, f)

    with open("../forecasts/relative_scores_cit_2.pkl", "wb") as f:
        pickle.dump(rel_scores_cit, f)


    print(0)

if __name__ == "__main__":
    main()