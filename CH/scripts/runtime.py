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


def f_upper_from_bottom(Bns, ref_id):
    """
    Bns: (N, 2U) with order [target(U), pop(U)]
    returns: (U+2, N) = [top_ratio(1), mid_ratios(U-1), swiss_totals(2)]
    """
    U = Bns.shape[1] // 2
    target = Bns[:, :U]
    pop = Bns[:, U:]
    no_ind_pos = np.where(np.array(ref_id) == "No indication")[0].tolist()

    mid_total = np.concatenate(
        [np.sum(target, axis=1, keepdims=True), np.sum(pop, axis=1, keepdims=True)],
        axis=1
    )
    mid_ratio = np.delete(target, no_ind_pos, axis=1) / (np.delete(pop, no_ind_pos, axis=1) + 1e-8)
    top = np.sum(target, axis=1, keepdims=True) / (np.sum(pop, axis=1, keepdims=True) + 1e-8)

    upper = np.concatenate([top, mid_ratio, mid_total], axis=1)
    return upper.T


def f_upper_from_bottom_single(z, no_ind_pos):
    U = z.shape[0] // 2
    target = z[:U]
    pop = z[U:]
    mid_total = np.concatenate([
        np.atleast_1d(np.sum(target)),
        np.atleast_1d(np.sum(pop))
    ])
    mid_ratio = np.delete(target, no_ind_pos) / (np.delete(pop, no_ind_pos) + 1e-8)
    top = np.atleast_1d(np.sum(target) / (np.sum(pop) + 1e-8))
    return np.concatenate([top, mid_ratio, mid_total])


def load_case(args):
    with open(args.base_2_pkl, "rb") as f:
        base_2 = pickle.load(f)
    with open(args.base_pkl, "rb") as f:
        base = pickle.load(f)

    def pick_block(block_key, src):
        return src[block_key]["samples"], src[block_key]["residuals"]

    pop_bot, pop_bot_res = pick_block("population", base)
    ref_uids = base["population"]["uids"]
    no_ind_pos = np.where(np.array(ref_uids) == "No indication")[0].tolist()

    U, T, M = pop_bot.shape
    if not (0 <= args.t_idx < T):
        raise ValueError(f"--t_idx must be between 0 and {T-1}, got {args.t_idx}")

    if args.target == "immigration":
        target_bot, target_bot_res = pick_block("immigration", base)
        target_total, target_total_res = pick_block("Switzerland_immigration", base_2)
        ratio_all, ratio_all_res = pick_block("immigration_ratio", base)
    elif args.target == "citizenship":
        target_bot, target_bot_res = pick_block("citizenship", base)
        target_total, target_total_res = pick_block("Switzerland_citizenship", base_2)
        ratio_all, ratio_all_res = pick_block("citizenship_ratio", base)
    else:
        raise ValueError("target must be 'immigration' or 'citizenship'")

    pop_total, pop_total_res = pick_block("Switzerland_population", base_2)

    rat_uids = base["immigration_ratio"]["uids"]
    ch_uid_pos = np.where(np.array(rat_uids) == "Switzerland")[0]

    ratio_mid = np.delete(ratio_all, ch_uid_pos, axis=0)
    ratio_mid_res = np.delete(ratio_all_res, ch_uid_pos, axis=0)
    ratio_top = ratio_all[ch_uid_pos, :, :]
    ratio_top_res = ratio_all_res[ch_uid_pos, :, :]

    t = args.t_idx

    u_obs = np.mean(
        np.vstack([
            ratio_top[:, t, :],
            ratio_mid[:, t, :],
            target_total[:, t, :],
            pop_total[:, t, :],
        ]),
        axis=1,
    )

    R = _schafer_strimmer_cov(
        np.vstack([
            ratio_top_res[:, t, :],
            ratio_mid_res[:, t, :],
            target_total_res[:, t, :],
            pop_total_res[:, t, :],
        ]).T
    )["shrink_cov"]

    bot_list = []
    for k in range(U):
        bot_list.append({
            "samples": target_bot[k, t, :],
            "residuals": target_bot_res[k, t, :],
        })
    for k in range(U):
        bot_list.append({
            "samples": pop_bot[k, t, :],
            "residuals": pop_bot_res[k, t, :],
        })

    W = _schafer_strimmer_cov(
        np.concatenate([
            ratio_top_res[:, t, :].T,
            ratio_mid_res[:, t, :].T,
            target_total_res[:, t, :].T,
            pop_total_res[:, t, :].T,
            target_bot_res[:, t, :].T,
            pop_bot_res[:, t, :].T,
        ], axis=1)
    )["shrink_cov"]
    P = _to_precision(W)

    Z = np.vstack([
        ratio_top[:, t, :],
        ratio_mid[:, t, :],
        target_total[:, t, :],
        pop_total[:, t, :],
        target_bot[:, t, :],
        pop_bot[:, t, :],
    ]).T

    return {
        "U": U,
        "M": M,
        "t": t,
        "ref_uids": ref_uids,
        "no_ind_pos": no_ind_pos,
        "u_obs": u_obs,
        "R": R,
        "bot_list": bot_list,
        "Z": Z,
        "P": P,
    }


def run_ukf(case, seed):
    def f_single(z):
        return f_upper_from_bottom_single(z, case["no_ind_pos"])

    out = reconc_nl_ukf(
        bottom_base_forecasts=case["bot_list"],
        in_type=["samples"] * (2 * case["U"]),
        distr=["gaussian"] * (2 * case["U"]),
        f=f_single,
        upper_base_forecasts=case["u_obs"],
        R=case["R"],
        num_samples=case["M"],
        seed=seed,
    )
    sync_any(out)

    Brec = out["bottom_reconciled_samples"]
    Urec = f_upper_from_bottom(Brec.T, ref_id=case["ref_uids"])
    rec = np.vstack([Urec, Brec])
    return rec


def run_full(case, seed, n_iter):
    f_ols = make_f_ols(case["U"], case["no_ind_pos"])
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
    parser.add_argument("--base_pkl", default="../forecasts/fc_imm_cit_autoarima2_30.pkl")
    parser.add_argument("--base_2_pkl", default="../forecasts/fc_imm_cit_autoarima_30.pkl")
    parser.add_argument("--t_idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out_csv", default=None)
    args = parser.parse_args()

    all_raw = []
    all_summary = []

    for target in ["immigration", "citizenship"]:
        args.target = target
        case = load_case(args)

        df_raw, df_summary, ukf_out, full_out = benchmark(
            case=case,
            repeats=args.repeats,
            seed=args.seed,
            n_iter=args.iters,
        )

        df_raw["target"] = target
        df_summary["target"] = target

        all_raw.append(df_raw)
        all_summary.append(df_summary)

        print(f"\nTarget      : {target}")
        print(f"Time step   : {args.t_idx}")
        print(f"Sample size : {case['M']}")
        print(f"Repeats     : {args.repeats}")


        print("\nSummary:")
        print(df_summary)

    if args.out_csv is not None:
        df_raw.to_csv(args.out_csv, index=False)
        print(f"\nSaved raw timings to: {args.out_csv}")


if __name__ == "__main__":
    main()