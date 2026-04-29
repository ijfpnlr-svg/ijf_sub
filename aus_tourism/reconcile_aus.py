import pickle
import argparse
import numpy as np
import jax.numpy as jnp
from reconc.reconc_nl_buis import reconc_nl_buis
from reconc.reconc_nl_ukf import reconc_nl_ukf, _schafer_strimmer_cov
from reconc.reconc_nl_ols import reconc_nl_ols
from simulation.scripts.score_functions import compute_crps
from scipy.linalg import block_diag
from CH.scripts.reconcile_hybrid import _to_precision
import concurrent.futures


def make_f_ols(B):
    """
    Fixed-shape nonlinear constraint function for projection.

    Layout of z:
      z = [total, ratio_1, ..., ratio_B, bot_1, ..., bot_B]
    so len(z) = 1 + B + B.
    """
    expected_dim = 1 + B + B

    def f_ols(z):
        z = jnp.asarray(z)

        if z.ndim != 1:
            raise ValueError(f"f_ols expects a 1D vector, got shape {z.shape}")
        if z.shape[0] != expected_dim:
            raise ValueError(f"f_ols expects length {expected_dim}, got {z.shape[0]}")

        total = z[0]
        ratio = z[1:1 + B]
        bot = z[1 + B:1 + 2 * B]

        eps = 1e-8
        c_mid = ratio - bot / (total + eps)
        c_total = total - jnp.sum(bot)

        return jnp.concatenate([jnp.array([c_total]), c_mid])

    return f_ols


def process_t(
    t,
    trips_total_res,
    ratio_state_res,
    trips_bottom_res,
    trips_total,
    ratio_state,
    trips_bottom,
    args,
):
    """
    Process one time index t for projection-based reconciliation.
    """
    B = trips_bottom.shape[0]
    f_ols = make_f_ols(B)

    # covariance / precision
    W = _schafer_strimmer_cov(
        np.concatenate(
            [
                trips_total_res[:, t, :].T,
                ratio_state_res[:, t, :].T,
                trips_bottom_res[:, t, :].T,
            ],
            axis=1,
        )
    )["shrink_cov"]
    P = _to_precision(W)

    # stacked sample matrix: (n_samples, 1 + B + B)
    Z = np.vstack(
        [
            trips_total[:, t, :],   # (1, M)
            ratio_state[:, t, :],   # (B, M)
            trips_bottom[:, t, :],  # (B, M)
        ]
    ).T

    # diagonal precision
    D = np.diag(np.diag(P))

    # block precision:
    # first block = [total + ratios] -> size 1+B
    # second block = bottoms -> size B
    block_W = block_diag(D[:1 + B, :1 + B], P[1 + B:, 1 + B:])

    ols_out = reconc_nl_ols(Z, f_ols, n_iter=args.iters, seed=args.seed)
    wls_out = reconc_nl_ols(Z, f_ols, W=D, n_iter=args.iters, seed=args.seed)
    block_out = reconc_nl_ols(Z, f_ols, W=block_W, n_iter=args.iters, seed=args.seed)
    mint_out = reconc_nl_ols(Z, f_ols, W=P, n_iter=args.iters, seed=args.seed)

    return (
        ols_out["reconciled_samples"],
        wls_out["reconciled_samples"],
        block_out["reconciled_samples"],
        mint_out["reconciled_samples"],
    )

def pbu(bot):
    B, T, M = bot.shape
    trips_total = np.sum(bot, axis=0).reshape((1,T,M))
    ratio_state = bot / trips_total
    return np.concatenate([trips_total, ratio_state, bot], axis=0)

def f_upper_from_bottom(bot):
    trip_tot = np.sum(bot, axis=1, keepdims=True)
    ratio_state = bot / trip_tot
    return np.concatenate([trip_tot, ratio_state], axis=1).T

def f_upper_to_bottom_single(bot):
    trip_total = np.atleast_1d(np.sum(bot))
    ratio_state = bot / trip_total
    return np.concatenate([trip_total, ratio_state])

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_pkl", type=str, default="forecasts/fc_tourism_autoarima.pkl")
    parser.add_argument("--test_pkl", type=str, default="forecasts/test_tourism_autoarima.pkl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    with open(args.base_pkl, "rb") as f: base = pickle.load(f)
    with open(args.test_pkl, "rb") as f: test_data = pickle.load(f)

    def pick_block(block_key, base=base):
        S = base[block_key]["samples"]
        R = base[block_key]["residuals"]
        return S, R

    trips, trips_res = pick_block("Trips", base)
    ratio, ratio_res = pick_block("Tourism_Ratio", base)


    uids = list(base["Trips"]["uids"])
    try:
        total_idx = uids.index("Total")
    except ValueError:
        total_idx = None


    mask = np.arange(len(uids)) != total_idx
    trips_bottom = trips[mask]  # bottoms (n_bottoms, n_years, n_samples)
    trips_bottom_res = trips_res[mask]  # residuals for bottoms
    trips_total = np.expand_dims(trips[total_idx], axis=0)  # (1, n_years, n_samples)
    trips_total_res = np.expand_dims(trips_res[total_idx], axis=0)  # (1, n_years, window)
    bottom_uids = [u for i, u in enumerate(uids) if i != total_idx]
    total_uid = uids[total_idx]
    # optional: quick check
    print("bottoms:", len(bottom_uids), "total found:", total_uid is not None)
    if trips_total is not None:
        print("trips_bottom.shape =", trips_bottom.shape, "trips_total.shape =", trips_total.shape)

    ratio_uids = list(base["Tourism_Ratio"]["uids"])
    try:
        ratio_total_idx = ratio_uids.index("Total")
    except ValueError:
        ratio_total_idx = None

    # ratio shape: (n_uids, n_years, n_samples)
    mask = np.arange(len(ratio_uids)) != ratio_total_idx
    ratio_state = ratio[mask]  # bottoms (n_bottoms, n_years, n_samples)
    ratio_state_res = ratio_res[mask]  # residuals for bottoms
    ratio_state_uids = [u for i, u in enumerate(ratio_uids) if i != ratio_total_idx]
    print("ratio_state.shape =", ratio_state.shape)

    B, T, M = trips_bottom.shape

    print("Running reconciliations...")

    # PBU

    print("----Running PBU-----")

    pbu_tourism = pbu(trips_bottom)
    print("PBU complete. Final Shape =", pbu_tourism.shape)

    #  UKF

    print("----Running UKF-----")

    ukf_tourism = {}
    for t in range(T):
        u_obs = np.mean(np.vstack([trips_total[:,t,:],ratio_state[:,t,:]]),axis=1)
        R = _schafer_strimmer_cov((np.vstack([trips_total_res[:,t,:],ratio_state_res[:,t,:]])).T)['shrink_cov']
        bot_list = []
        for k in range(B):
            bot_list.append({
                "samples": trips_bottom[k,t,:],
                "residuals": trips_bottom_res[k,t,:]
            })
        out = reconc_nl_ukf(
            bottom_base_forecasts=bot_list,
            in_type=["samples"]*B,
            distr=["normal"]*B,
            f=f_upper_to_bottom_single,
            upper_base_forecasts=u_obs,
            R=R,
            num_samples=M,
            seed=args.seed
        )
        Brec = out["bottom_reconciled_samples"]
        Urec = f_upper_from_bottom(Brec.T)
        ukf_tourism[t] = np.vstack([Urec, Brec])

    ukf_tourism = np.stack([ukf_tourism[t] for t in range(T)], axis=1)
    print("UKF complete. Final Shape =", ukf_tourism.shape)

    # Projection

    print("---Running projection---")

    ols = {}
    wls = {}
    mint = {}
    block = {}

    # Use ProcessPoolExecutor to parallelize the projection process
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit tasks for each time index t in the range T
        futures = [
            executor.submit(process_t, t, trips_total_res, ratio_state_res, trips_bottom_res, trips_total, ratio_state, trips_bottom, args)
            for t in range(T)
        ]

        # Collect results as they finish
        for t, future in enumerate(futures):
            ols_out, wls_out, block_out, mint_out = future.result()
            ols[t] = ols_out
            wls[t] = wls_out
            block[t] = block_out
            mint[t] = mint_out

    # Stack results for OLS, WLS, Mint, and Block methods
    ols_tourism = np.stack([ols[t] for t in range(T)], axis=1)
    wls_tourism = np.stack([wls[t] for t in range(T)], axis=1)
    block_tourism = np.stack([block[t] for t in range(T)], axis=1)
    mint_tourism = np.stack([mint[t] for t in range(T)], axis=1)

    # Print the results
    print("Projection complete")
    print("OLS Shape =", ols_tourism.shape)
    print("WLS Shape =", wls_tourism.shape)
    #print("Block Shape =", block_tourism.shape)
    print("Mint Shape =", mint_tourism.shape)


    print("---Running BUIS---")

    buis_res = {}
    for t in range(T):
        fc_bot_arr = trips_bottom[:,t,:]
        fc_upp_arr = np.vstack([trips_total[:,t,:], ratio_state[:,t,:]])

        n_upper = fc_upp_arr.shape[0]
        n_bottom = fc_bot_arr.shape[0]
        total_rows = n_upper + n_bottom

        # pass lists of 1-D arrays so reconc_nl_buis will column_stack -> B shape (M, n_bottom)
        upper_list = [fc_upp_arr[i, :] for i in range(n_upper)]
        bottom_list = [fc_bot_arr[i, :] for i in range(n_bottom)]

        # joint_samples must match the final B orientation:
        # since bottom_list -> B has shape (M, n_bottom), set B_joint = fc_bot_arr.T (M, n_bottom)
        # and U_joint = fc_upp_arr.T (M, n_upper)
        joint_samples = (fc_upp_arr.T, fc_bot_arr.T)

        buis_out = reconc_nl_buis(
            f=f_upper_from_bottom,
            n_bot= n_bottom,
            num_samples=M,
            seed=args.seed,
            assume_independent=False,  # use joint_samples for non-independent reconciliation
            joint_mean=np.mean(np.vstack([fc_upp_arr,fc_bot_arr]),axis=1),
            joint_cov=_schafer_strimmer_cov((np.vstack([trips_total_res[:,t,:],ratio_state_res[:,t,:],trips_bottom_res[:,t,:]])).T)['shrink_cov'],
        )

        buis_res[t] = buis_out["reconciled_samples"]

    buis_tourism = np.stack([buis_res[t] for t in range(T)], axis=1)
    print("BUIS complete. Final Shape =", buis_tourism.shape)

    if ratio_total_idx is not None:
        mask_ratio = np.arange(len(ratio_uids)) != ratio_total_idx
    else:
        mask_ratio = np.ones(len(ratio_uids), dtype=bool)

    test_total = test_data["Trips"]["y_true"][total_idx] if total_idx is not None else None
    test_total = test_total.reshape(1,40)
    test_ratio = test_data["Tourism_Ratio"]["y_true"][mask_ratio]
    test_bot = test_data["Trips"]["y_true"][mask]

    tourism_data = np.vstack([test_total, test_ratio, test_bot])

    base_tourism = np.vstack([trips_total, ratio_state, trips_bottom])

    forecast_methods = {
        "Base": base_tourism,
        "PBU": pbu_tourism,
        "UKF": ukf_tourism,
        "OLS": ols_tourism,
        "WLS": wls_tourism,
        #"Block": block_tourism,
        "FULL": mint_tourism,
        #"BUIS": buis_tourism
    }

    print("\n Energy scores for Australian Tourism")
    for method, y_hat in forecast_methods.items():
        score = compute_es(tourism_data, y_hat)
        print(f"{method}: ES = {score:.4f}")

    def compute_crps_over_level(y_true, y_samples, rows_idx, average_within_level=True):
        """
        Compute CRPS for a subset of rows (level).
        y_true: (R, T)
        y_samples: (R, T, M)
        rows_idx: list or array of row indices for the level
        """
        rows = np.asarray(rows_idx, dtype=int)
        if rows.size == 0:
            return np.nan if average_within_level else np.empty((0, y_true.shape[1]))
        T = y_true.shape[1]
        per_time = []
        for t in range(T):
            y_t = np.atleast_1d(y_true[rows, t])
            samples_t = np.atleast_2d(y_samples[rows, t, :])
            crps_rows = np.atleast_1d(compute_crps(y_t, samples_t))
            if average_within_level:
                per_time.append(np.nanmean(crps_rows))
            else:
                per_time.append(crps_rows)
        per_time = np.array(per_time)  # shape (T,) or (T, len(rows))
        if average_within_level:
            return float(np.nanmean(per_time))
        return per_time.T  # (len(rows), T)

    def crps_table_and_relative(forecast_methods, gt, show_levels=None):
        """
        Compute absolute and relative CRPS per level for all methods.
        show_levels: dict[level_name -> list/array of row indices]
        """
        if show_levels is None:
            show_levels = {"full": np.arange(gt.shape[0])}

        abs_crps = {}
        for name, y_hat in forecast_methods.items():
            lvl_scores = {}
            for lvl_name, idxs in show_levels.items():
                lvl_scores[lvl_name] = compute_crps_over_level(gt, y_hat, idxs, average_within_level=True)
            abs_crps[name] = lvl_scores

        # baseline
        base_scores = abs_crps.get("Base", next(iter(abs_crps.values())))
        rel_crps = {}
        for name, scores in abs_crps.items():
            rel = {}
            for lvl_name, val in scores.items():
                base_val = base_scores.get(lvl_name, np.nan)
                rel[lvl_name] = val / base_val if np.isfinite(base_val) and base_val != 0 else np.nan
            rel_crps[name] = rel

        # print table (columns = show_levels keys in given order)
        keys = list(next(iter(show_levels.keys())))
        header_keys = list(show_levels.keys())
        col_widths = [max(12, max(len(k), 6)) for k in header_keys]
        # simple formatted print
        cols = " | ".join([f"{k:^12s}" for k in ["Method"] + header_keys])
        print(cols)
        print("-" * (14 * (1 + len(header_keys))))
        for name in abs_crps:
            row = [f"{name:<12s}"]
            for k in header_keys:
                a = abs_crps[name][k]
                r = rel_crps[name][k]
                row.append(f"{a:.4g} ({r:.3f}x)")
            print(" | ".join(row))

    def crps_relative_geomean_over_series(
            forecast_methods: dict,
            gt: np.ndarray,
            baseline_name: str = "Base",
            eps: float = 1e-12,
    ):
        """
        Returns per-series average CRPS (over time), per-series ratio to baseline,
        and geometric mean of per-series ratios.
        """
        # pick baseline
        if baseline_name in forecast_methods:
            base_key = baseline_name
        else:
            base_key = next(iter(forecast_methods.keys()))

        R, T = gt.shape
        per_series_avg = {}

        for name, y_hat in forecast_methods.items():
            if y_hat.shape[0] != R or y_hat.shape[1] != T:
                raise ValueError(f"{name}: expected shape (R,T,M) with (R,T)=({R},{T}), got {y_hat.shape}")
            avg_j = np.full(R, np.nan, dtype=float)
            for j in range(R):
                crps_t = []
                for t in range(T):
                    c = compute_crps(gt[j:j + 1, t], y_hat[j:j + 1, t, :])[0]
                    crps_t.append(c)
                avg_j[j] = np.nanmean(crps_t)
            per_series_avg[name] = avg_j

        base_avg = per_series_avg[base_key]
        base_safe = np.where(np.isfinite(base_avg) & (base_avg > 0), base_avg, np.nan)

        per_series_ratio = {}
        gm_ratio = {}
        for name, avg_j in per_series_avg.items():
            ratio = avg_j / base_safe
            ratio = np.where(np.isfinite(ratio), np.maximum(ratio, eps), np.nan)
            per_series_ratio[name] = ratio
            valid = np.isfinite(ratio)
            gm_ratio[name] = float(np.exp(np.nanmean(np.log(ratio[valid])))) if np.any(valid) else np.nan

        return per_series_avg, per_series_ratio, gm_ratio

    def crps_gm_table(forecast_methods: dict, gt: np.ndarray, baseline_name: str = "Base"):
        per_series_avg, per_series_ratio, gm_ratio = crps_relative_geomean_over_series(
            forecast_methods, gt, baseline_name=baseline_name
        )
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

    # --- Example usage adapted to your tourism variables (append at end of your script) ---
    # compute index layout for tourism: total at 0, next ratio_count rows are ratios, then B bottoms
    ratio_count = ratio_state.shape[0]
    total_idx = 0
    ratio_start = 1
    ratio_end = 1 + ratio_count
    bottom_start = ratio_end
    bottom_end = bottom_start + B

    show_levels_tourism = {
        "full": list(range(tourism_data.shape[0])),
        "top_total": [total_idx],
        "ratios": list(range(ratio_start, ratio_end)),
        "bottoms": list(range(bottom_start, bottom_end))
    }

    print("\n🔹 CRPS — Tourism")
    crps_table_and_relative(forecast_methods, tourism_data, show_levels=show_levels_tourism)
    crps_gm_table(forecast_methods, tourism_data, baseline_name="Base")
    crps_relative_geomean_over_series(forecast_methods, tourism_data, baseline_name="Base")



if __name__ == "__main__":
    main()