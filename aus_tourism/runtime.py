import argparse
import pickle
import time
import numpy as np
import pandas as pd
import jax.numpy as jnp

from reconc.reconc_nl_ukf import reconc_nl_ukf, _schafer_strimmer_cov
from reconc.reconc_nl_ols import reconc_nl_ols


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


def _to_precision(cov, eps=1e-6):
    cov = 0.5 * (cov + cov.T)
    lam = eps * np.trace(cov) / cov.shape[0]
    return np.linalg.pinv(cov + lam * np.eye(cov.shape[0]))


def make_f_ols(B):
    """
    z = [total, ratio_1, ..., ratio_B, bottom_1, ..., bottom_B]
    constraints:
      - total = sum(bottom)
      - ratio_k = bottom_k / total, k=1,...,B
    """
    expected_dim = 2 * B + 1

    def f_ols(z):
        z = jnp.asarray(z)

        if z.ndim != 1:
            raise ValueError(f"f_ols expects a 1D vector, got shape {z.shape}")
        if z.shape[0] != expected_dim:
            raise ValueError(f"f_ols expects length {expected_dim}, got {z.shape[0]}")

        total = z[0]
        ratio = z[1:B + 1]
        bot = z[B + 1:]

        eps = 1e-8
        c_total = total - jnp.sum(bot)
        c_ratio = ratio - bot / (total + eps)

        return jnp.concatenate([jnp.array([c_total]), c_ratio])

    return f_ols


def f_upper_from_bottom(Bns):
    """
    Bns: (N, B) bottom samples
    returns: (B+1, N) = [total, ratio_1, ..., ratio_B]
    """
    total = np.sum(Bns, axis=1, keepdims=True)
    ratio = Bns / (total + 1e-8)
    upper = np.concatenate([total, ratio], axis=1)
    return upper.T


def f_upper_from_bottom_single(z):
    total = np.atleast_1d(np.sum(z))
    ratio = z / (total + 1e-8)
    return np.concatenate([total, ratio])


def load_case(args):
    with open(args.base_pkl, "rb") as f:
        base = pickle.load(f)

    def pick_block(block_key):
        return base[block_key]["samples"], base[block_key]["residuals"]

    trips, trips_res = pick_block("Trips")
    ratio, ratio_res = pick_block("Tourism_Ratio")

    # Trips: extract bottoms + total
    trip_uids = list(base["Trips"]["uids"])
    try:
        total_idx = trip_uids.index("Total")
    except ValueError:
        raise ValueError("'Total' not found in base['Trips']['uids'].")

    mask_bottom = np.arange(len(trip_uids)) != total_idx
    trips_bottom = trips[mask_bottom]               # (B, T, M)
    trips_bottom_res = trips_res[mask_bottom]       # (B, T, W)
    trips_total = np.expand_dims(trips[total_idx], axis=0)          # (1, T, M)
    trips_total_res = np.expand_dims(trips_res[total_idx], axis=0)  # (1, T, W)

    # Tourism_Ratio: extract bottom ratios, drop total
    ratio_uids = list(base["Tourism_Ratio"]["uids"])
    try:
        ratio_total_idx = ratio_uids.index("Total")
    except ValueError:
        raise ValueError("'Total' not found in base['Tourism_Ratio']['uids'].")

    mask_ratio = np.arange(len(ratio_uids)) != ratio_total_idx
    ratio_state = ratio[mask_ratio]            # (B, T, M)
    ratio_state_res = ratio_res[mask_ratio]    # (B, T, W)

    B, T, M = trips_bottom.shape
    if not (0 <= args.t_idx < T):
        raise ValueError(f"--t_idx must be between 0 and {T-1}, got {args.t_idx}")

    t = args.t_idx

    # UKF inputs
    u_obs = np.mean(
        np.vstack([
            trips_total[:, t, :],
            ratio_state[:, t, :],
        ]),
        axis=1,
    )

    R = _schafer_strimmer_cov(
        np.vstack([
            trips_total_res[:, t, :],
            ratio_state_res[:, t, :],
        ]).T
    )["shrink_cov"]

    bot_list = []
    for k in range(B):
        bot_list.append({
            "samples": trips_bottom[k, t, :],
            "residuals": trips_bottom_res[k, t, :],
        })

    # FULL projection inputs
    W = _schafer_strimmer_cov(
        np.concatenate([
            trips_total_res[:, t, :].T,
            ratio_state_res[:, t, :].T,
            trips_bottom_res[:, t, :].T,
        ], axis=1)
    )["shrink_cov"]
    P = _to_precision(W)

    Z = np.vstack([
        trips_total[:, t, :],
        ratio_state[:, t, :],
        trips_bottom[:, t, :],
    ]).T

    return {
        "B": B,
        "M": M,
        "t": t,
        "u_obs": u_obs,
        "R": R,
        "bot_list": bot_list,
        "Z": Z,
        "P": P,
    }


def run_ukf(case, seed):
    out = reconc_nl_ukf(
        bottom_base_forecasts=case["bot_list"],
        in_type=["samples"] * case["B"],
        distr=["normal"] * case["B"],
        f=f_upper_from_bottom_single,
        upper_base_forecasts=case["u_obs"],
        R=case["R"],
        num_samples=case["M"],
        seed=seed,
    )
    sync_any(out)

    Brec = out["bottom_reconciled_samples"]
    Urec = f_upper_from_bottom(Brec.T)
    rec = np.vstack([Urec, Brec])
    return rec


def run_full(case, seed, n_iter):
    f_ols = make_f_ols(case["B"])
    out = reconc_nl_ols(
        case["Z"],
        f_ols,
        W=case["P"],
        n_iter=n_iter,
        seed=seed,
    )
    sync_any(out)
    return out["reconciled_samples"]


def benchmark(case, repeats, seed, n_iter):
    rows = []
    ukf_last = None
    full_last = None

    for r in range(repeats):
        t0 = time.perf_counter()
        ukf_last = run_ukf(case, seed + r)
        dt = time.perf_counter() - t0
        rows.append({"method": "UKF", "repeat": r + 1, "time_sec": dt})

        t0 = time.perf_counter()
        full_last = run_full(case, seed + r, n_iter)
        dt = time.perf_counter() - t0
        rows.append({"method": "full", "repeat": r + 1, "time_sec": dt})

    df = pd.DataFrame(rows)
    summary = (
        df.groupby("method", as_index=False)
        .agg(
            mean_time_sec=("time_sec", "mean"),
            std_time_sec=("time_sec", "std"),
            min_time_sec=("time_sec", "min"),
            max_time_sec=("time_sec", "max"),
        )
    )

    return df, summary, ukf_last, full_last


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_pkl", default="./forecasts/fc_tourism_autoarima.pkl")
    parser.add_argument("--t_idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out_csv", default=None)
    args = parser.parse_args()

    case = load_case(args)

    df_raw, df_summary, ukf_out, full_out = benchmark(
        case=case,
        repeats=args.repeats,
        seed=args.seed,
        n_iter=args.iters,
    )

    print(f"\nCase        : tourism")
    print(f"Time step   : {args.t_idx}")
    print(f"Bottom dim  : {case['B']}")
    print(f"Sample size : {case['M']}")
    print(f"Repeats     : {args.repeats}")

    print("\nRaw timings:")
    print(df_raw)

    print("\nSummary:")
    print(df_summary)

    print("\nOutput shapes:")
    print("UKF :", ukf_out.shape)
    print("full:", full_out.shape)

    if args.out_csv is not None:
        df_raw.to_csv(args.out_csv, index=False)
        print(f"\nSaved raw timings to: {args.out_csv}")


if __name__ == "__main__":
    main()