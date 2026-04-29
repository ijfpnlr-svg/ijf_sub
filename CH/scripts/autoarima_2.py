
import numpy as np
import pandas as pd
import pickle
import warnings
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import argparse
import os
import sys
import hashlib

# R bridge
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

# Prevent collisions with any local Python module named 'forecast'
if "forecast" in sys.modules:
    del sys.modules["forecast"]

warnings.filterwarnings("ignore")

# Gaussian Process (kept because present in your original script)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, DotProduct
R_AUTOARIMA = None
R_FORECAST = None

def init_r_forecast():
    global R_AUTOARIMA, R_FORECAST
    importr("forecast")
    R_AUTOARIMA = ro.r("forecast::auto.arima")
    R_FORECAST = ro.r("forecast::forecast")

# ----------------------------- AutoARIMA helpers -----------------------------

def _py_series_to_r_ts(series: pd.Series, freq: int = 1):
    """Convert pandas Series indexed by year -> R ts."""
    x = ro.FloatVector(series.astype(float).values.tolist())
    start_year = int(pd.Index(series.index).min())
    if freq == 1:
        return ro.r["ts"](x, start=start_year, frequency=1)
    else:
        return ro.r["ts"](x, start=ro.IntVector([start_year, 1]), frequency=freq)


def _fit_single_autoarima(train: pd.Series, r_freq: int = 1):
    global R_AUTOARIMA, R_FORECAST

    if R_AUTOARIMA is None or R_FORECAST is None:
        init_r_forecast()

    r_train = _py_series_to_r_ts(train, freq=r_freq)
    fit = R_AUTOARIMA(r_train)
    fc = R_FORECAST(fit, h=1)

    fc_mean = float(np.array(fc.rx2("mean"))[0])

    resid = np.array(fit.rx2("residuals"), dtype=float)
    if resid.size >= len(train):
        resid = resid[-len(train):]
    else:
        resid = np.pad(resid, (len(train) - resid.size, 0), mode="edge")

    return fc_mean, resid.astype(float)

def _stable_seed_from_uid(uid: str, base_seed: int = 42) -> int:
    """
    Stable deterministic seed from uid + base_seed.
    """
    msg = f"{uid}|{base_seed}".encode("utf-8")
    digest = hashlib.md5(msg).hexdigest()
    return int(digest[:8], 16)


# ----------------------------- Joint rolling AutoARIMA -----------------------------

def fit_arima_forecast_joint(
    group_df: pd.DataFrame,
    uid: str,
    targets: list,
    n_samples: int,
    window: int = 13,
    r_freq: int = 1,
    base_seed: int = 42,
) -> pd.DataFrame:
    """
    Rolling 1-step-ahead AutoARIMA with JOINT residual-row bootstrapping.

    For each rolling window and each uid:
    - fit one AutoARIMA per target on the same training window,
    - collect point forecasts,
    - collect aligned residuals,
    - build residual matrix (window, n_targets),
    - sample WHOLE ROWS to preserve dependence across targets.

    Returns a long DataFrame with one row per (test_year, target).
    """
    warnings.filterwarnings("ignore")

    rng = np.random.default_rng(_stable_seed_from_uid(uid, base_seed))

    group_df = group_df.sort_values("year").drop_duplicates(subset="year")
    ts_df = (
        group_df[["year"] + targets]
        .drop_duplicates(subset="year")
        .set_index("year")
        .sort_index()
        .astype(float)
    )

    results = []
    max_test_year = int(ts_df.index.max())
    max_train_end = max_test_year - 1

    for start_year in range(int(ts_df.index.min()), int(max_train_end - window + 1)):
        train_df = ts_df.loc[start_year:start_year + window - 1].copy()

        if len(train_df) < window or train_df[targets].isnull().any().any():
            continue

        test_year = int(train_df.index.max() + 1)

        try:
            point_forecasts = []
            residual_cols = []

            for target in targets:
                fc_mean, resid = _fit_single_autoarima(train_df[target], r_freq=r_freq)
                point_forecasts.append(fc_mean)
                residual_cols.append(resid)

            # residual matrix: rows = aligned time residuals, cols = targets
            res_mat = np.column_stack(residual_cols).astype(float)

            # remove rows with NaNs jointly
            valid_rows = ~np.isnan(res_mat).any(axis=1)
            res_mat_valid = res_mat[valid_rows]

            n_targets = len(targets)
            fc_vec = np.asarray(point_forecasts, dtype=float)

            if res_mat_valid.shape[0] == 0:
                joint_samples = np.tile(fc_vec[None, :], (n_samples, 1))
                residuals_for_storage = np.full((window, n_targets), np.nan)
            else:
                idx = rng.integers(0, res_mat_valid.shape[0], size=n_samples)
                eps = res_mat_valid[idx, :]                      # (n_samples, n_targets)
                joint_samples = fc_vec[None, :] + eps           # (n_samples, n_targets)

                residuals_for_storage = np.full((window, n_targets), np.nan)
                residuals_for_storage[:res_mat_valid.shape[0], :] = res_mat_valid

            for k, target in enumerate(targets):
                results.append({
                    "uid": uid,
                    "target": target,
                    "train_start": int(train_df.index.min()),
                    "train_end": int(train_df.index.max()),
                    "test_year": test_year,
                    "forecast_samples": joint_samples[:, k].astype(float),
                    "residuals": residuals_for_storage[:, k].astype(float),
                    "point_forecast": float(fc_vec[k]),
                })

        except Exception as e:
            print(f"⚠️ Joint AutoARIMA failed for {uid}, {start_year}: {e}")
            for target in targets:
                results.append({
                    "uid": uid,
                    "target": target,
                    "train_start": int(train_df.index.min()),
                    "train_end": int(train_df.index.max()),
                    "test_year": test_year,
                    "forecast_samples": np.full(n_samples, np.nan),
                    "residuals": np.full(window, np.nan),
                    "point_forecast": np.nan,
                })

    return pd.DataFrame(results)


def process_group_joint(args):
    group_df, uid, targets, n_samples, window, seed = args
    return fit_arima_forecast_joint(
        group_df=group_df,
        uid=uid,
        targets=targets,
        n_samples=n_samples,
        window=window,
        base_seed=seed,
    )


def make_uid(key):
    return "|".join(str(x) for x in key) if isinstance(key, tuple) else str(key)


# ---------------------------------- Main -----------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--output", type=str, default="../forecasts/fc_imm_cit_autoarima2_30.pkl")
    parser.add_argument("--test_output", type=str, default="../forecasts/test_autoarima2_30.pkl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load
    df = pd.read_csv("../data/immigration_citizenship_data.csv")
    df["region"] = df["region"].str.strip()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output = {}
    test_output = {}

    # -------------------- Joint blocks --------------------
    counts_targets = ["immigration", "population", "citizenship"]
    ratio_targets = ["immigration_ratio", "citizenship_ratio"]

    blocks = [
        {
            "name": "counts",
            "targets": counts_targets,
            "df_use": df[df["region"] != "Switzerland"].copy(),
        },
        {
            "name": "ratios",
            "targets": ratio_targets,
            "df_use": df.copy(),
        },
    ]

    for block in blocks:
        block_name = block["name"]
        targets = block["targets"]
        df_use = block["df_use"]

        print(f"🔹 Joint AutoARIMA block: {block_name} -> {targets}")

        grouped = df_use.groupby("region")
        results = []

        with ProcessPoolExecutor(initializer=init_r_forecast) as executor:
            futures = [
                executor.submit(
                    process_group_joint,
                    (group.copy(), make_uid(key), targets, args.n_samples, args.window, args.seed)
                )
                for key, group in grouped
                if group["year"].nunique() >= args.window
            ]

            for future in tqdm(futures, desc=f"Forecasting block {block_name} (AutoARIMA)"):
                res = future.result()
                if not res.empty:
                    results.append(res)

        if not results:
            continue

        combined = pd.concat(results, ignore_index=True)

        for target in targets:
            sub = combined[combined["target"] == target].copy()

            years = sorted(sub["test_year"].unique())
            uids = sorted(sub["uid"].unique())
            uid_to_idx = {uid: i for i, uid in enumerate(uids)}
            year_to_idx = {year: i for i, year in enumerate(years)}

            samples_arr = np.full((len(uids), len(years), args.n_samples), np.nan)
            residuals_arr = np.full((len(uids), len(years), args.window), np.nan)
            point_arr = np.full((len(uids), len(years)), np.nan)
            y_true_arr = np.full((len(uids), len(years)), np.nan)

            for _, row in sub.iterrows():
                i = uid_to_idx[row["uid"]]
                j = year_to_idx[row["test_year"]]

                samples_arr[i, j, :] = row["forecast_samples"]
                residuals_arr[i, j, :] = row["residuals"]
                point_arr[i, j] = row["point_forecast"]

                val = df_use[
                    (df_use["region"] == row["uid"]) &
                    (df_use["year"] == row["test_year"])
                ][target]
                if not val.empty:
                    y_true_arr[i, j] = val.values[0]

            output[target] = {
                "uids": uids,
                "years": years,
                "samples": samples_arr,
                "residuals": residuals_arr,
                "point_forecasts": point_arr,
            }
            test_output[target] = {
                "uids": uids,
                "years": years,
                "y_true": y_true_arr,
            }

    # ------------------------------- Save -------------------------------

    with open(args.output, "wb") as f:
        pickle.dump(output, f)

    with open(args.test_output, "wb") as f:
        pickle.dump(test_output, f)

    print(f"✅ Saved all forecast samples/residuals to: {args.output}")
    print(f"✅ Saved test data to: {args.test_output}")


if __name__ == "__main__":
    main()