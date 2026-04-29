import gc
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import jax.numpy as jnp
from bayesreconpy.shrink_cov import _schafer_strimmer_cov as schafer_strimmer_cov

from reconc.reconc_nl_ols import reconc_nl_ols
from reconc.reconc_nl_ukf import reconc_nl_ukf
from reconcile import f_surface, f_surface_jax, _to_precision


# ============================================================
# CONFIG
# ============================================================

FC_FOLDER = Path("../forecasts")
SURFACES = ["paraboloid", "saddle", "ripples"]
#SAMPLE_SIZES = [1000, 2000, 5000, 10_000]
SAMPLE_SIZES = [2000]
N_REPEATS = 3
N_ITER_OLS = 20
SEED = 42
T_IDX = 0   # single time step to benchmark

RAW_OUT = "timings_raw.csv"
SUMMARY_OUT = "timings_summary.csv"


# ============================================================
# HELPERS
# ============================================================

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


def timed_call(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    sync_any(out)
    elapsed = time.perf_counter() - t0
    return out, elapsed


def load_case(surface: str, n_samples: int, fc_folder: Path):
    base_path = fc_folder / f"base_fc_{surface}_indep_{n_samples}.pkl"
    res_path  = fc_folder / f"residuals_{surface}_indep_{n_samples}.pkl"
    te_path   = fc_folder / f"test_data_{surface}_indep_{n_samples}.pkl"

    indep_base_fc = np.asarray(pd.read_pickle(base_path), dtype=np.float64)
    indep_tr_res = np.asarray(pd.read_pickle(res_path), dtype=np.float64)
    df_te = pd.read_pickle(te_path)

    T = indep_base_fc.shape[1]
    S = indep_base_fc.shape[2]

    indep_tr_res = np.repeat(indep_tr_res[:, None, :], T, axis=1)

    gt_test = df_te.iloc[:-1].values
    return indep_base_fc, indep_tr_res, gt_test, T, S


# ============================================================
# TIMING FUNCTIONS (ONE TIME STEP ONLY)
# ============================================================

def time_full(indep_base_fc, indep_tr_res, surface, t_idx=0, n_iter=20, seed=42):
    def f_ols(z):
        u = z[0]
        b1 = z[1]
        b2 = z[2]
        return jnp.array([u - f_surface_jax(surface, b1, b2)])

    W = schafer_strimmer_cov(indep_tr_res[:, t_idx, :].T)["shrink_cov"]
    P = _to_precision(W)
    Z = indep_base_fc[:, t_idx, :].T

    _, dt = timed_call(
        reconc_nl_ols,
        Z,
        f_ols,
        n_iter=n_iter,
        seed=seed,
        W=P,
    )
    return dt


def time_ukf(indep_base_fc, indep_tr_res, surface, t_idx=0, seed=42):
    S = indep_base_fc.shape[2]

    bot_base = indep_base_fc[1:, :, :]
    bot_res = indep_tr_res[1:, :, :]

    def f_ukf_vec(b):
        b1, b2 = b[0], b[1]
        return f_surface(surface, b1, b2)

    u_obs = np.mean(indep_base_fc[0, t_idx, :]).reshape(1,)
    R = schafer_strimmer_cov(indep_tr_res[:, t_idx, :].T)["shrink_cov"][0, 0]

    bot_list = []
    for s in range(2):
        bot_list.append({
            "samples": bot_base[s, t_idx, :],
            "residuals": bot_res[s, t_idx, :]
        })

    _, dt = timed_call(
        reconc_nl_ukf,
        bottom_base_forecasts=bot_list,
        in_type=["samples"],
        distr=["gaussian"],
        f=f_ukf_vec,
        upper_base_forecasts=u_obs,
        R=R,
        num_samples=S,
        seed=seed,
    )
    return dt


# ============================================================
# MAIN BENCHMARK LOOP
# ============================================================

def main():
    rows = []

    for surface in SURFACES:
        print("\n===================================================")
        print(f"Benchmarking surface: {surface}")
        print("===================================================\n")

        for n_samples in SAMPLE_SIZES:
            print(f"--- n_samples = {n_samples} ---")

            indep_base_fc, indep_tr_res, gt_test, T, S = load_case(
                surface=surface,
                n_samples=n_samples,
                fc_folder=FC_FOLDER,
            )

            if T_IDX >= T:
                raise ValueError(f"T_IDX={T_IDX} is out of range for T={T}")

            print(f"Loaded shape: base_fc = {indep_base_fc.shape}, T = {T}, S = {S}, t_idx = {T_IDX}")

            for rep in range(1, N_REPEATS + 1):
                print(f"  repeat {rep}/{N_REPEATS}")

                dt_full = time_full(
                    indep_base_fc=indep_base_fc,
                    indep_tr_res=indep_tr_res,
                    surface=surface,
                    t_idx=T_IDX,
                    n_iter=N_ITER_OLS,
                    seed=SEED,
                )
                rows.append({
                    "surface": surface,
                    "n_samples": n_samples,
                    "repeat": rep,
                    "method": "full",
                    "time_sec": dt_full,
                    "t_idx": T_IDX,
                    "T": T,
                    "S": S,
                })

                dt_ukf = time_ukf(
                    indep_base_fc=indep_base_fc,
                    indep_tr_res=indep_tr_res,
                    surface=surface,
                    t_idx=T_IDX,
                    seed=SEED,
                )
                rows.append({
                    "surface": surface,
                    "n_samples": n_samples,
                    "repeat": rep,
                    "method": "ukf",
                    "time_sec": dt_ukf,
                    "t_idx": T_IDX,
                    "T": T,
                    "S": S,
                })

                gc.collect()

    df_raw = pd.DataFrame(rows)
    df_raw.to_csv(RAW_OUT, index=False)

    df_summary = (
        df_raw
        .groupby(["surface", "method", "n_samples"], as_index=False)
        .agg(
            mean_time_sec=("time_sec", "mean"),
            std_time_sec=("time_sec", "std"),
        )
        .sort_values(["surface", "method", "n_samples"])
    )
    df_summary.to_csv(SUMMARY_OUT, index=False)

    print("\nRaw timings:")
    print(df_raw)

    print("\nSummary timings:")
    print(df_summary)

    print("\nPivot table: mean seconds")
    print(
        df_summary.pivot_table(
            index=["surface", "method"],
            columns="n_samples",
            values="mean_time_sec"
        )
    )

    # ============================================================
    # PLOT
    # ============================================================
    for surface_name in df_summary["surface"].unique():
        plt.figure(figsize=(8, 5))
        df_surface = df_summary[df_summary["surface"] == surface_name]

        for method in df_surface["method"].unique():
            tmp = df_surface[df_surface["method"] == method].sort_values("n_samples")
            plt.plot(
                tmp["n_samples"],
                tmp["mean_time_sec"],
                marker="o",
                label=method
            )

        plt.xlabel("n_samples")
        plt.ylabel("Mean computation time (s)")
        plt.title(f"One-step computation time vs sample size ({surface_name}, t={T_IDX})")
        plt.xticks(SAMPLE_SIZES, [f"{x:,}" for x in SAMPLE_SIZES])
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"timings_plot_{surface_name}.png", dpi=300)
        plt.show()

    for surface_name in df_summary["surface"].unique():
        plt.figure(figsize=(8, 5))
        df_surface = df_summary[df_summary["surface"] == surface_name]

        for method in df_surface["method"].unique():
            tmp = df_surface[df_surface["method"] == method].sort_values("n_samples")
            plt.plot(
                tmp["n_samples"],
                tmp["mean_time_sec"],
                marker="o",
                label=method
            )

        plt.xlabel("n_samples")
        plt.ylabel("Mean computation time (s)")
        plt.title(f"One-step computation time vs sample size ({surface_name}, t={T_IDX})")
        plt.xticks(SAMPLE_SIZES, [f"{x:,}" for x in SAMPLE_SIZES])
        plt.yscale("log")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"timings_plot_{surface_name}_logy.png", dpi=300)
        plt.show()

    print(f"\nSaved raw timings to: {RAW_OUT}")
    print(f"Saved summary timings to: {SUMMARY_OUT}")


if __name__ == "__main__":
    main()