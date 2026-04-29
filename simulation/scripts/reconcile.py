import warnings
warnings.filterwarnings("ignore")

import os
from multiprocessing import get_context
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import jax.numpy as jnp
from bayesreconpy.shrink_cov import _schafer_strimmer_cov as schafer_strimmer_cov
from scipy.linalg import block_diag

from reconc.reconc_nl_buis import reconc_nl_buis
from reconc.reconc_nl_ols import reconc_nl_ols
from reconc.reconc_nl_ukf import reconc_nl_ukf
from score_functions import compute_es
from simulation.scripts.score_functions import compute_crps_new


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


# ============================================================
# PARALLEL PROJECTION
# ============================================================

def _project_at_time_step_worker(task):
    t, z_t, res_t, surface, n_iter, seed = task

    def f_ols(z):
        u = z[0]
        b1 = z[1]
        b2 = z[2]
        return jnp.array([u - f_surface_jax(surface, b1, b2)])

    W = schafer_strimmer_cov(res_t.T)["shrink_cov"]
    P = _to_precision(W)

    out_ols = reconc_nl_ols(z_t, f_ols, n_iter=n_iter, seed=seed)
    sync_any(out_ols)

    out_mint = reconc_nl_ols(z_t, f_ols, n_iter=n_iter, seed=seed, W=P)
    sync_any(out_mint)

    out_wls = reconc_nl_ols(z_t, f_ols, n_iter=n_iter, seed=seed, W=np.diag(np.diag(P)))
    sync_any(out_wls)

    out_block = reconc_nl_ols(
        z_t,
        f_ols,
        n_iter=n_iter,
        seed=seed,
        W=block_diag(np.diag(np.diag(P))[0, 0], P[1:, 1:]),
    )
    sync_any(out_block)

    return {
        "t": t,
        "ols": np.asarray(out_ols["reconciled_samples"]).copy(),
        "mint": np.asarray(out_mint["reconciled_samples"]).copy(),
        "wls": np.asarray(out_wls["reconciled_samples"]).copy(),
        "block": np.asarray(out_block["reconciled_samples"]).copy(),
    }


def run_projection_parallel(indep_base_fc, indep_tr_res, surface, n_iter=20, seed=42, max_workers=None):
    T = indep_base_fc.shape[1]

    tasks = [
        (
            t,
            indep_base_fc[:, t, :].T.copy(),   # (S, 3)
            indep_tr_res[:, t, :].copy(),      # (3, Nres)
            surface,
            n_iter,
            seed,
        )
        for t in range(T)
    ]

    ols_dict = {}
    mint_dict = {}
    wls_dict = {}
    block_dict = {}

    ctx = get_context("spawn")
    if max_workers is None:
        max_workers = min(os.cpu_count() or 1, T)

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        futures = [executor.submit(_project_at_time_step_worker, task) for task in tasks]

        for fut in as_completed(futures):
            result = fut.result()
            t = result["t"]
            ols_dict[t] = result["ols"]
            mint_dict[t] = result["mint"]
            wls_dict[t] = result["wls"]
            block_dict[t] = result["block"]
            print(f"finished projection task t={t}", flush=True)

    ols_fc = np.stack([ols_dict[t] for t in range(T)], axis=1)
    mint_fc = np.stack([mint_dict[t] for t in range(T)], axis=1)
    wls_fc = np.stack([wls_dict[t] for t in range(T)], axis=1)
    block_fc = np.stack([block_dict[t] for t in range(T)], axis=1)

    return ols_fc, mint_fc, wls_fc, block_fc


def f_surface(surface, b1, b2):
    if surface == "paraboloid":
        return b1**2 + b2**2

    if surface == "cone":
        return np.sqrt(np.maximum(b1**2 + b2**2, 0.0))

    if surface == "saddle":
        return b1**2 - b2**2

    if surface == "ripples":
        return np.sin(b1) + np.cos(b2)

    if surface == "linear":
        return b1 + b2

    raise ValueError(f"Unknown surface {surface}")


def f_surface_jax(surface, b1, b2):
    if surface == "paraboloid":
        return b1**2 + b2**2

    if surface == "cone":
        return jnp.sqrt(jnp.maximum(b1**2 + b2**2, 0.0))

    if surface == "saddle":
        return b1**2 - b2**2

    if surface == "ripples":
        return jnp.sin(b1) + jnp.cos(b2)

    if surface == "linear":
        return b1 + b2

    raise ValueError(f"Unknown surface {surface}")


# ============================================================
# BOTTOM-UP CONSTRUCTION (SHAPE-PRESERVING)
# ============================================================

def pbu(B, surface):
    """
    B shape = (2, T, S) = (B1, B2)
    Output = (3, T, S) = (U, B1, B2)
    """
    b1 = B[0, :, :]
    b2 = B[1, :, :]

    U = f_surface(surface, b1, b2)
    U = U.reshape(1, *U.shape)
    return np.concatenate([U, B], axis=0)


# ============================================================
# PRECISION MATRIX UTIL
# ============================================================

def _to_precision(cov, eps=1e-6):
    cov = 0.5 * (cov + cov.T)
    lam = eps * np.trace(cov) / cov.shape[0]
    return np.linalg.pinv(cov + lam * np.eye(cov.shape[0]))


# ============================================================
# MAIN LOOP OVER ALL SURFACES
# ============================================================

def main():
    fc_folder = "../forecasts"
    surfaces = [
        "saddle", "ripples",
        "paraboloid",
        # "linear",
    ]
    n_samples = 2000
    projection_workers = None  # set e.g. 8 if you want to limit CPU usage

    for surface in surfaces:
        print("\n===================================================")
        print(f"Running reconciliation for surface: {surface}")
        print("===================================================\n")

        # ------------- LOAD FORECASTS -------------
        base_path = f"{fc_folder}/base_fc_{surface}_indep_{n_samples}.pkl"
        res_path = f"{fc_folder}/residuals_{surface}_indep_{n_samples}.pkl"
        te_path = f"{fc_folder}/test_data_{surface}_indep_{n_samples}.pkl"

        indep_base_fc = pd.read_pickle(base_path)  # shape (3, T, S)
        T = indep_base_fc.shape[1]
        S = indep_base_fc.shape[2]
        indep_tr_res = pd.read_pickle(res_path)
        indep_tr_res = np.repeat(indep_tr_res[:, None, :], T, axis=1)
        df_te = pd.read_pickle(te_path)

        gt_test = df_te.iloc[:-1].values  # (T, 3)
        indep_base_fc = np.array(indep_base_fc, dtype=np.float64)

        # ============================================================
        # PROBABILISTIC BOTTOM-UP
        # ============================================================
        print("BU started.")

        bot_base = indep_base_fc[1:, :, :]  # (2, T, S)
        bu_fc = pbu(bot_base, surface)

        print("BU completed.")

        # ============================================================
        # OLS/WLS/MINT/BLOCK PROJECTION
        # ============================================================
        print("Projection started.")

        ols_fc, mint_fc, wls_fc, block_fc = run_projection_parallel(
            indep_base_fc=indep_base_fc,
            indep_tr_res=indep_tr_res,
            surface=surface,
            n_iter=20,
            seed=42,
            max_workers=projection_workers,
        )

        print("Projection completed.")

        # ============================================================
        # UKF
        # ============================================================
        def f_ukf_vec(b):
            b1, b2 = b[0], b[1]
            return f_surface(surface, b1, b2)

        def f_ukf_mult(bmat):
            b1 = bmat[:, 0]
            b2 = bmat[:, 1]
            return f_surface(surface, b1, b2)

        bot_res = indep_tr_res[1:, :, :]

        ukf_dict = {}
        for t in range(T):
            u_obs = np.mean(indep_base_fc[0, t, :]).reshape(1,)
            R = schafer_strimmer_cov(indep_tr_res[:, t, :].T)["shrink_cov"][0, 0]

            bot_list = []
            for s in range(2):
                bot_list.append(
                    {
                        "samples": bot_base[s, t, :],
                        "residuals": bot_res[s, t, :],
                    }
                )

            out = reconc_nl_ukf(
                bottom_base_forecasts=bot_list,
                in_type=["samples"],
                distr=["gaussian"],
                f=f_ukf_vec,
                upper_base_forecasts=u_obs,
                R=R,
                num_samples=S,
                seed=42,
            )
            Brec = out["bottom_reconciled_samples"]
            Urec = f_ukf_mult(Brec.T)
            ukf_dict[t] = np.vstack([Urec, Brec])

        ukf_fc = np.stack([ukf_dict[t] for t in range(T)], axis=1)

        # # ============================================================
        # # BUIS
        # # ============================================================
        # print("IS started.")
        #
        # def f_buis(B):
        #     B = B.T
        #     U = f_surface(surface, B[0, :], B[1, :])
        #     return U.reshape(1, -1)
        #
        # buis_dict = {}
        # for t in range(T):
        #     fc_bot = indep_base_fc[1:, t, :]
        #     fc_upp = indep_base_fc[0, t, :].reshape(1, S)
        #
        #     out_buis = reconc_nl_buis(
        #         assume_independent=False,
        #         joint_mean=np.mean(np.vstack([fc_upp, fc_bot]), axis=1),
        #         joint_cov=schafer_strimmer_cov(indep_tr_res[:, t, :].T)["shrink_cov"],
        #         f=f_buis,
        #         n_bot=2,
        #         num_samples=S,
        #     )
        #     buis_dict[t] = out_buis["reconciled_samples"]
        #
        # buis_fc = np.stack([buis_dict[t] for t in range(T)], axis=1)
        #
        # print("IS completed.")

        # ============================================================
        # EVALUATION
        # ============================================================
        forecasts = {
            "base": indep_base_fc,
            "pbu": bu_fc,
            # "buis": buis_fc,
            "ols": ols_fc,
            "wls": wls_fc,
            "full": mint_fc,
            # "block": block_fc,
            "ukf": ukf_fc,
        }

        # ---- Energy Score: absolute + relative to base ----
        es_scores = {}
        for key, arr in forecasts.items():
            es_scores[key] = compute_es(gt_test.T, arr)

        base_es = es_scores["base"]

        print(f"Energy Scores for {surface} (relative to base):")
        for key in forecasts.keys():
            abs_es = es_scores[key]
            rel_es = abs_es / base_es if base_es != 0 else np.nan
            print(f"  {key:<6} :  {rel_es:.2f}")

        # ---- CRPS: absolute + relative to base ----
        crps_scores = {}
        for key, arr in forecasts.items():
            crps_scores[key] = compute_crps_new(gt_test.T, arr)

        base_crps = crps_scores["base"]

        print(f"CRPS for {surface} (relative to base):")
        for key in forecasts.keys():
            abs_crps = crps_scores[key]
            rel_crps = abs_crps / base_crps if base_crps != 0 else np.nan
            print(f"  {key:<6} :  {rel_crps:.2f}")

    print("\nAll surfaces completed.\n")


if __name__ == "__main__":
    main()