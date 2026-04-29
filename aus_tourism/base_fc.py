import os
import pickle
import argparse
import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import re

# Ensure forecast is loaded once
importr("forecast")
R_AUTOARIMA = ro.r("forecast::auto.arima")
R_FORECAST  = ro.r("forecast::forecast")

def _parse_quarter(val) -> pd.Period:
    """Parse '1998 Q1', '1998Q1', '1998-Q1' -> pandas Period('1998Q1', freq='Q')."""
    if pd.isna(val):
        return None
    s = str(val).strip()
    m = re.search(r"(\d{4})\D*Q?([1-4])", s, flags=re.IGNORECASE)
    if not m:
        return None
    y = int(m.group(1)); q = int(m.group(2))
    return pd.Period(f"{y}Q{q}", freq="Q")

def fit_arima_forecast_quarterly(
    group_df: pd.DataFrame,
    uid: str,
    target: str,
    n_samples: int,
    window: int = 40,
    r_freq: int = 4
) -> pd.DataFrame:
    """
    Rolling 1-step-ahead AutoARIMA via R forecast::auto.arima for quarterly data.

    Returns rows with:
      uid, target, train_start, train_end, test_quarter, forecast_samples (n_samples,), residuals (window,)
    """
    g = group_df.copy()
    g["q"] = g["Quarter"].apply(_parse_quarter)
    g = g.dropna(subset=["q"]).sort_values("q").drop_duplicates(subset="q")

    if g.empty:
        return pd.DataFrame([])

    ts = g.set_index("q")[target].astype(float).sort_index()
    qs = ts.index.to_list()
    y  = ts.values

    # rolling over index positions (robust and simple)
    results = []
    T = len(y)
    if T <= window:
        return pd.DataFrame([])

    for t0 in range(0, T - window):
        train_y = y[t0:t0 + window]
        train_q_start = qs[t0]
        train_q_end   = qs[t0 + window - 1]
        test_q        = qs[t0 + window]  # 1-step ahead

        if np.any(~np.isfinite(train_y)):
            continue

        try:
            # R ts for quarterly needs start=c(year, quarter)
            start = train_q_start
            r_start = ro.IntVector([int(start.year), int(start.quarter)])
            r_train = ro.r["ts"](ro.FloatVector(train_y.tolist()), start=r_start, frequency=r_freq)

            fit = R_AUTOARIMA(r_train)
            fc  = R_FORECAST(fit, h=1)
            fc_mean = float(np.array(fc.rx2("mean"))[0])

            resid = np.array(fit.rx2("residuals"), dtype=float)
            # make resid length == window
            if resid.size >= window:
                resid = resid[-window:]
            else:
                resid = np.pad(resid, (window - resid.size, 0), mode="edge")

            sd = float(np.nanstd(resid, ddof=1))
            if (not np.isfinite(sd)) or sd == 0.0:
                samples = np.full(n_samples, fc_mean, dtype=float)
            else:
                # Apply residual bootstrap: sample residuals with replacement
                bootstrapped_residuals = np.random.choice(resid, size=n_samples, replace=True)
                samples = (fc_mean + bootstrapped_residuals).astype(float)

            results.append({
                "uid": uid,
                "target": target,
                "train_start": str(train_q_start),   # e.g. '1998Q1'
                "train_end": str(train_q_end),
                "test_quarter": str(test_q),
                "forecast_samples": samples,
                "residuals": resid.astype(float),
            })

        except Exception as e:
            print(f"⚠️ R AutoARIMA failed for {uid}, test={test_q}: {e}")
            results.append({
                "uid": uid,
                "target": target,
                "train_start": str(train_q_start),
                "train_end": str(train_q_end),
                "test_quarter": str(test_q),
                "forecast_samples": np.full(n_samples, np.nan),
                "residuals": np.full(window, np.nan),
            })

    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--window", type=int, default=40)
    parser.add_argument("--output", type=str, default="forecasts/fc_tourism_autoarima.pkl")
    parser.add_argument("--test_output", type=str, default="forecasts/test_tourism_autoarima.pkl")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # ----- your preprocessing -----
    data = pd.read_csv("data/tourism_data.csv")

    df_filtered = data[["Quarter", "State", "Trips"]]
    df_agg = df_filtered.groupby(["Quarter", "State"], as_index=False)["Trips"].sum()

    df_total = df_agg.groupby("Quarter", as_index=False)["Trips"].sum()
    df_total["State"] = "Total"

    df_final = pd.concat([df_agg, df_total], ignore_index=True)
    df_final = df_final.sort_values(["Quarter", "State"])
    df_final = pd.merge(df_final, df_total[["Quarter", "Trips"]], on="Quarter", how="left", suffixes=("", "_total"))
    df_final["Tourism_Ratio"] = df_final["Trips"] / df_final["Trips_total"]

    df_stacked = pd.melt(
        df_final,
        id_vars=["Quarter", "State"],
        value_vars=["Trips", "Tourism_Ratio"],
        var_name="variable",
        value_name="Value"
    )

    # ----- forecasting & packaging (like your other script) -----
    output = {}
    test_output = {}

    for var in ["Trips", "Tourism_Ratio"]:
        print(f"🔹 AutoARIMA target: {var}")

        df_use = df_stacked[df_stacked["variable"] == var].copy()

        results = []
        for state, group in df_use.groupby("State"):
            # Need at least window+1 points to produce one forecast
            if group["Quarter"].nunique() < args.window + 1:
                continue
            print(f"   - {state}")
            res = fit_arima_forecast_quarterly(
                group_df=group,
                uid=str(state),
                target="Value",
                n_samples=args.n_samples,
                window=args.window,
                r_freq=4
            )
            if not res.empty:
                results.append(res)

        if not results:
            continue

        combined = pd.concat(results, ignore_index=True)

        # axes
        quarters = sorted(combined["test_quarter"].unique())  # strings like '1998Q1'
        uids     = sorted(combined["uid"].unique())
        uid_to_i = {u: i for i, u in enumerate(uids)}
        q_to_j   = {q: j for j, q in enumerate(quarters)}

        samples_arr   = np.full((len(uids), len(quarters), args.n_samples), np.nan)
        residuals_arr = np.full((len(uids), len(quarters), args.window), np.nan)
        y_true_arr    = np.full((len(uids), len(quarters)), np.nan)

        # map y_true from df_use
        # (quarter parsing once)
        df_use["q"] = df_use["Quarter"].apply(_parse_quarter).astype("period[Q]")
        df_use["q_str"] = df_use["q"].astype(str)

        for _, row in combined.iterrows():
            i = uid_to_i[row["uid"]]
            j = q_to_j[row["test_quarter"]]
            samples_arr[i, j, :]   = row["forecast_samples"]
            residuals_arr[i, j, :] = row["residuals"]

            val = df_use[(df_use["State"] == row["uid"]) & (df_use["q_str"] == row["test_quarter"])]["Value"]
            if not val.empty:
                y_true_arr[i, j] = float(val.values[0])

        output[var] = {
            "uids": uids,
            "years": quarters,          # keep same key name as your other pipeline, but it's quarters
            "samples": samples_arr,
            "residuals": residuals_arr,
        }
        test_output[var] = {
            "uids": uids,
            "years": quarters,
            "y_true": y_true_arr,
        }

    with open(args.output, "wb") as f:
        pickle.dump(output, f)
    with open(args.test_output, "wb") as f:
        pickle.dump(test_output, f)

    print(f"✅ Saved forecast samples/residuals to: {args.output}")
    print(f"✅ Saved test data to: {args.test_output}")

if __name__ == "__main__":
    main()