
import numpy as np
import pandas as pd
import pickle
import warnings
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import argparse
import os
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import sys

if "forecast" in sys.modules:
    del sys.modules["forecast"]

warnings.filterwarnings("ignore")

R_AUTOARIMA = None
R_FORECAST = None

def init_r_forecast():
    global R_AUTOARIMA, R_FORECAST
    importr("forecast")
    R_AUTOARIMA = ro.r("forecast::auto.arima")
    R_FORECAST = ro.r("forecast::forecast")

# ----------------------------- Helpers -----------------------------

def _py_series_to_r_ts(series: pd.Series, freq: int = 1):
    """
    Convert a pandas Series with integer/period index to an R 'ts' object.
    Assumes annual data by default (freq=1).
    """
    x = ro.FloatVector(series.astype(float).values.tolist())
    start_year = int(pd.Index(series.index).min())
    r_ts = ro.r["ts"](
        x,
        start=ro.IntVector([start_year, 1]) if freq > 1 else start_year,
        frequency=freq
    )
    return r_ts


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

# ----------------------------- Joint ARIMA rolling -----------------------------

def fit_arima_forecast_joint(
    group_df: pd.DataFrame,
    uid: str,
    targets: list,
    n_samples: int,
    window: int = 13,
    r_freq: int = 1,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Rolling AutoARIMA with JOINT residual bootstrapping across multiple targets.

    For each rolling window:
    - fit one AutoARIMA per target,
    - collect point forecasts for all targets,
    - collect aligned in-sample residuals for all targets,
    - sample WHOLE residual rows jointly to preserve dependence across targets.

    Returns a long DataFrame with one row per (test_year, target).
    """
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(seed)

    group_df = group_df.sort_values("year").drop_duplicates(subset="year")

    # Build aligned target table indexed by year
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

        try:
            fc_means = []
            residual_cols = []

            # Fit one model per target for this same training window
            for target in targets:
                fc_mean, resid = _fit_single_autoarima(
                    train=train_df[target],
                    r_freq=r_freq,
                )
                fc_means.append(fc_mean)
                residual_cols.append(resid)

            # residual matrix: (window, n_targets)
            res_mat = np.column_stack(residual_cols).astype(float)

            # Remove any row with NaNs jointly
            valid_rows = ~np.isnan(res_mat).any(axis=1)
            res_mat = res_mat[valid_rows]

            n_targets = len(targets)

            if res_mat.shape[0] == 0:
                # fallback: degenerate samples at point forecast
                joint_samples = np.tile(np.array(fc_means, dtype=float), (n_samples, 1))
                residuals_for_storage = np.full((window, n_targets), np.nan)
            else:
                # sample WHOLE residual rows jointly
                idx = rng.integers(0, res_mat.shape[0], size=n_samples)
                eps = res_mat[idx, :]  # (n_samples, n_targets)

                fc_means_arr = np.array(fc_means, dtype=float)[None, :]  # (1, n_targets)
                joint_samples = fc_means_arr + eps  # (n_samples, n_targets)

                # keep per-target residual history aligned to original window size for storage
                residuals_for_storage = np.full((window, n_targets), np.nan)
                residuals_for_storage[:res_mat.shape[0], :] = res_mat

            test_year = int(train_df.index.max() + 1)

            for k, target in enumerate(targets):
                results.append({
                    "uid": uid,
                    "target": target,
                    "train_start": int(train_df.index.min()),
                    "train_end": int(train_df.index.max()),
                    "test_year": test_year,
                    "forecast_samples": joint_samples[:, k].astype(float),
                    "residuals": residuals_for_storage[:, k].astype(float),
                    "point_forecast": float(fc_means[k]),
                })

        except Exception as e:
            print(f"⚠️ Joint AutoARIMA failed for {uid}, {start_year}: {e}")
            test_year = int(train_df.index.max() + 1)
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
        seed=seed,
    )


# ----------------------------- Main -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--output", type=str, default="../forecasts/fc_imm_cit_autoarima_30.pkl")
    parser.add_argument("--test_output", type=str, default="../forecasts/test_autoarima_30.pkl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    df = pd.read_csv("../data/immigration_citizenship_data.csv")
    df["region"] = df["region"].str.strip()

    output = {}
    test_output = {}

    def make_uid(key):
        return "|".join(str(x) for x in key) if isinstance(key, (tuple, list)) else str(key)

    # -------------------- Block 1: Cantonal and Swiss counts --------------------
    count_targets = ["immigration", "population", "citizenship"]

    for region_filter, uid_prefix in [
        (df["region"] != "Switzerland", "cantons"),
        (df["region"] == "Switzerland", "Switzerland"),
    ]:
        print(f"🔹 Joint AutoARIMA Counts Block: {uid_prefix} -> {count_targets}")

        df_use = df[region_filter].copy()
        grouped = df_use.groupby("region")

        results = []
        with ProcessPoolExecutor(initializer=init_r_forecast) as executor:
            futures = [
                executor.submit(
                    process_group_joint,
                    (group.copy(), make_uid(key), count_targets, args.n_samples, args.window, args.seed)
                )
                for key, group in grouped
                if group["year"].nunique() >= args.window
            ]
            for future in tqdm(futures, desc=f"Forecasting joint counts ({uid_prefix})"):
                res = future.result()
                if not res.empty:
                    results.append(res)

        if results:
            combined = pd.concat(results, ignore_index=True)

            for target in count_targets:
                sub = combined[combined["target"] == target].copy()

                years = sorted(sub["test_year"].unique())
                uids = sorted(sub["uid"].unique())
                uid_to_idx = {uid: i for i, uid in enumerate(uids)}
                year_to_idx = {year: i for i, year in enumerate(years)}

                samples_arr = np.full((len(uids), len(years), args.n_samples), np.nan)
                residuals_arr = np.full((len(uids), len(years), args.window), np.nan)
                y_true_arr = np.full((len(uids), len(years)), np.nan)
                point_arr = np.full((len(uids), len(years)), np.nan)

                for _, row in sub.iterrows():
                    i = uid_to_idx[row["uid"]]
                    j = year_to_idx[row["test_year"]]
                    samples_arr[i, j, :] = row["forecast_samples"]
                    residuals_arr[i, j, :] = row["residuals"]
                    point_arr[i, j] = row["point_forecast"]

                    df_uid = df_use[df_use["region"] == row["uid"]]
                    val = df_uid.loc[df_uid["year"] == row["test_year"], target]
                    if not val.empty:
                        y_true_arr[i, j] = val.values[0]

                save_key = f"{uid_prefix}_{target}"
                output[save_key] = {
                    "uids": uids,
                    "years": years,
                    "samples": samples_arr,
                    "residuals": residuals_arr,
                    "point_forecasts": point_arr,
                }
                test_output[save_key] = {
                    "uids": uids,
                    "years": years,
                    "y_true": y_true_arr,
                }

    # -------------------- Block 2: Switzerland ratios --------------------
    ratio_targets = ["immigration_ratio", "citizenship_ratio"]
    print(f"🔹 Joint AutoARIMA Switzerland Ratio Block: {ratio_targets}")

    df_use = df[df["region"] == "Switzerland"].copy()
    grouped = df_use.groupby("region")

    results = []
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                process_group_joint,
                (group.copy(), make_uid(key), ratio_targets, args.n_samples, args.window, args.seed)
            )
            for key, group in grouped
            if group["year"].nunique() >= args.window
        ]
        for future in tqdm(futures, desc="Forecasting joint Switzerland ratios"):
            res = future.result()
            if not res.empty:
                results.append(res)

    if results:
        combined = pd.concat(results, ignore_index=True)

        for target in ratio_targets:
            sub = combined[combined["target"] == target].copy()

            years = sorted(sub["test_year"].unique())
            uids = sorted(sub["uid"].unique())
            uid_to_idx = {uid: i for i, uid in enumerate(uids)}
            year_to_idx = {year: i for i, year in enumerate(years)}

            samples_arr = np.full((len(uids), len(years), args.n_samples), np.nan)
            residuals_arr = np.full((len(uids), len(years), args.window), np.nan)
            y_true_arr = np.full((len(uids), len(years)), np.nan)
            point_arr = np.full((len(uids), len(years)), np.nan)

            for _, row in sub.iterrows():
                i = uid_to_idx[row["uid"]]
                j = year_to_idx[row["test_year"]]
                samples_arr[i, j, :] = row["forecast_samples"]
                residuals_arr[i, j, :] = row["residuals"]
                point_arr[i, j] = row["point_forecast"]

                df_uid = df_use[df_use["region"] == row["uid"]]
                val = df_uid.loc[df_uid["year"] == row["test_year"], target]
                if not val.empty:
                    y_true_arr[i, j] = val.values[0]

            save_key = f"Switzerland_{target}"
            output[save_key] = {
                "uids": uids,
                "years": years,
                "samples": samples_arr,
                "residuals": residuals_arr,
                "point_forecasts": point_arr,
            }
            test_output[save_key] = {
                "uids": uids,
                "years": years,
                "y_true": y_true_arr,
            }

    # -------------------- Save --------------------
    with open(args.output, "wb") as f:
        pickle.dump(output, f)

    with open(args.test_output, "wb") as f:
        pickle.dump(test_output, f)

    print(f"✅ Saved all forecast samples and residuals to: {args.output}")
    print(f"✅ Saved test data to: {args.test_output}")


if __name__ == "__main__":
    main()