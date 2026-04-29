
import numpy as np
import pandas as pd
import pickle
import argparse
import os

# --------------------
# Utils
# --------------------
def make_uid(key):
    return "|".join(str(x) for x in key) if isinstance(key, tuple) else str(key)

def _clean(df):
    df = df.copy()
    df["region"] = df["region"].astype(str).str.strip()
    df["year"] = df["year"].astype(int)
    return df

def _safe_ratio(num, den, eps=1e-12):
    return num / np.clip(den, eps, None)

# --------------------
# Window collectors
# --------------------
def collect_windows(df, target, group_cols="region", window=23, exclude_switzerland=False):
    """
    Generic rolling-window collector.
    Returns rows of: uid, test_year, train_years (np.array), train_values (np.array).
    """
    df = df.copy()
    if exclude_switzerland:
        df = df[df["region"] != "Switzerland"]

    grouped = df.groupby(group_cols)
    rows = []

    for key, g in grouped:
        g = g.sort_values("year").drop_duplicates("year")
        ts = g.set_index("year")[target].astype(float).sort_index()

        if ts.index.nunique() < window:
            continue

        for start in range(ts.index.min(), ts.index.max() - window + 1):
            train = ts.loc[start:start + window - 1]
            if train.isna().any():
                continue
            test_year = start + window
            rows.append({
                "uid": make_uid(key),
                "test_year": int(test_year),
                "train_years": train.index.values.astype(int),
                "train_values": train.values.astype(float),
            })

    if not rows:
        return None
    return pd.DataFrame(rows)


def collect_top_windows(df, target, window=23):
    """
    Windows for Swiss-only series (top level).
    Uses region == 'Switzerland'.
    """
    df = df[df["region"] == "Switzerland"].copy()
    return collect_windows(df, target, group_cols="region", window=window, exclude_switzerland=False)


# --------------------
# Packing to arrays
# --------------------
def pack_rows(df_rows, window):
    """
    df_rows with columns: uid, test_year, train_years, train_values
    -> dict(uids, years, train_years [U,T,W], train_values [U,T,W])
    """
    years = sorted(df_rows["test_year"].unique())
    uids  = sorted(df_rows["uid"].unique())
    U, T, W = len(uids), len(years), window

    uid_to_i  = {u:i for i,u in enumerate(uids)}
    year_to_j = {y:j for j,y in enumerate(years)}

    train_years = np.full((U, T, W), np.nan)
    train_vals  = np.full((U, T, W), np.nan)

    for _, r in df_rows.iterrows():
        i = uid_to_i[r["uid"]]
        j = year_to_j[int(r["test_year"])]
        ty = np.asarray(r["train_years"], float)
        tv = np.asarray(r["train_values"], float)
        L = min(W, len(ty))
        train_years[i, j, :L] = ty[-L:]
        train_vals[i, j, :L]  = tv[-L:]

    return {
        "uids": uids,
        "years": years,
        "train_years": train_years,
        "train_values": train_vals,
    }

# --------------------
# Main
# --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="../forecasts/train_data_new.pkl")
    parser.add_argument("--imm_csv", default="../data/immigration_citizenship_data.csv")
    parser.add_argument("--window", type=int, default=23)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Load & clean
    df = pd.read_csv(args.imm_csv)
    df = _clean(df)

    # Compute ratios if missing
    if "immigration_ratio" not in df.columns:
        df["immigration_ratio"] = _safe_ratio(df["immigration"], df["population"])
    if "citizenship_ratio" not in df.columns:
        df["citizenship_ratio"] = _safe_ratio(df["citizenship"], df["population"])

    store = {}

    # -------- Top (Swiss ratios, CH only) -----------------------
    for target in ["immigration_ratio", "citizenship_ratio"]:
        top_rows = collect_top_windows(df, target, window=args.window)
        if top_rows is not None and not top_rows.empty:
            store[target] = pack_rows(top_rows, window=args.window)

    # -------- Middle (cantonal ratios, CH excluded) -------------
    middle_map = {
        "immigration_ratio_kanton": "immigration_ratio",
        "citizenship_ratio_kanton": "citizenship_ratio",
    }
    for new_key, col in middle_map.items():
        rows = collect_windows(df, col, group_cols="region", window=args.window, exclude_switzerland=True)
        if rows is not None and not rows.empty:
            store[new_key] = pack_rows(rows, window=args.window)

    # -------- Swiss totals (CH only): immigration, population ---
    top_totals = {
        "Switzerland_immigration": "immigration",
        "Switzerland_population": "population",
        "Switzerland_citizenship": "citizenship",
    }
    for new_key, col in top_totals.items():
        ch_rows = collect_top_windows(df, col, window=args.window)
        if ch_rows is not None and not ch_rows.empty:
            store[new_key] = pack_rows(ch_rows, window=args.window)

    # -------- Bottom (cantons only, CH excluded) ----------------
    for target in ["immigration", "citizenship", "population"]:
        rows = collect_windows(df, target, group_cols="region", window=args.window, exclude_switzerland=True)
        if rows is not None and not rows.empty:
            store[target] = pack_rows(rows, window=args.window)

    # Save
    with open(args.out, "wb") as f:
        pickle.dump(store, f)

    print(f"✅ Wrote hierarchical training data to {args.out}")
    print("Keys saved:", list(store.keys()))
    for k, v in store.items():
        if isinstance(v, dict) and "train_values" in v:
            arr = v["train_values"]
            print(f"  └─ {k:<30s}: shape {arr.shape}")

if __name__ == "__main__":
    main()
