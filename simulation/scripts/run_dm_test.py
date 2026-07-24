import math
import os
from itertools import combinations

import numpy as np
import pandas as pd

try:
    from scipy import stats as scipy_stats
except Exception:
    scipy_stats = None


# ============================================================
# CONFIG
# ============================================================

RESULTS_FOLDER = "../results"

LOSS_FILE = os.path.join(
    RESULTS_FOLDER,
    "reconciliation_losses_by_time.csv",
)

DM_OUTPUT_FILE = os.path.join(
    RESULTS_FOLDER,
    "diebold_mariano_tests.csv",
)

METHOD_ORDER = [
    "base",
    "pbu",
    "ols",
    "wls",
    "full",
    "ukf",
]

ALPHA = 0.05

HORIZON = 1

HAC_LAG = 0


# ============================================================
# DIEBOLD-MARIANO TEST
# ============================================================

def _normal_two_sided_p_value(
    statistic,
):
    z = abs(
        float(statistic)
    )

    return math.erfc(
        z / math.sqrt(2.0)
    )


def _two_sided_p_value(
    statistic,
    dof,
):
    if not np.isfinite(
        statistic
    ):
        return np.nan

    if scipy_stats is not None and dof > 0:
        return float(
            2.0
            * scipy_stats.t.sf(
                abs(statistic),
                df=dof,
            )
        )

    return float(
        _normal_two_sided_p_value(
            statistic
        )
    )


def diebold_mariano_test(
    loss_a,
    loss_b,
    horizon=1,
    hac_lag=None,
    harvey_correction=True,
):
    """
    Diebold-Mariano test for equal predictive accuracy.

    Loss differential:

        d_t = loss_a_t - loss_b_t

    Therefore:

        DM > 0

    means method A has larger average loss than method B.
    Since lower loss is better, positive DM means method A is worse.
    """
    loss_a = np.asarray(
        loss_a,
        dtype=float,
    )

    loss_b = np.asarray(
        loss_b,
        dtype=float,
    )

    if loss_a.shape != loss_b.shape:
        raise ValueError(
            "loss_a and loss_b must have the same shape, "
            f"got {loss_a.shape} and {loss_b.shape}"
        )

    d = (
        loss_a
        - loss_b
    )

    valid = np.isfinite(
        d
    )

    d = d[
        valid
    ]

    n = len(
        d
    )

    if n < 3:
        return {
            "n_obs": n,
            "mean_loss_a": np.nan,
            "mean_loss_b": np.nan,
            "mean_loss_diff": np.nan,
            "dm_stat": np.nan,
            "p_value": np.nan,
            "hac_lag": np.nan,
        }

    if hac_lag is None:
        hac_lag = max(
            horizon - 1,
            0,
        )

    hac_lag = int(
        min(
            hac_lag,
            n - 1,
        )
    )

    mean_diff = float(
        np.mean(
            d
        )
    )

    centered = (
        d
        - mean_diff
    )

    gamma_0 = float(
        np.sum(
            centered
            * centered
        )
        / n
    )

    long_run_variance = gamma_0

    for lag in range(
        1,
        hac_lag + 1,
    ):
        gamma_lag = float(
            np.sum(
                centered[
                    lag:
                ]
                * centered[
                    :-lag
                ]
            )
            / n
        )

        weight = (
            1.0
            - lag
            / (
                hac_lag
                + 1.0
            )
        )

        long_run_variance += (
            2.0
            * weight
            * gamma_lag
        )

    if long_run_variance <= 0 or not np.isfinite(
        long_run_variance
    ):
        dm_stat = np.nan
        p_value = np.nan

    else:
        dm_stat = (
            mean_diff
            / np.sqrt(
                long_run_variance
                / n
            )
        )

        if harvey_correction:
            correction = np.sqrt(
                (
                    n
                    + 1
                    - 2 * horizon
                    + horizon
                    * (
                        horizon
                        - 1
                    )
                    / n
                )
                / n
            )

            dm_stat = (
                dm_stat
                * correction
            )

        p_value = _two_sided_p_value(
            statistic=dm_stat,
            dof=n - 1,
        )

    return {
        "n_obs": n,

        "mean_loss_a": float(
            np.mean(
                loss_a[
                    valid
                ]
            )
        ),

        "mean_loss_b": float(
            np.mean(
                loss_b[
                    valid
                ]
            )
        ),

        "mean_loss_diff": mean_diff,

        "dm_stat": float(
            dm_stat
        )
        if np.isfinite(
            dm_stat
        )
        else np.nan,

        "p_value": float(
            p_value
        )
        if np.isfinite(
            p_value
        )
        else np.nan,

        "hac_lag": hac_lag,
    }


# ============================================================
# LOAD LOSSES
# ============================================================

def load_losses():
    if not os.path.exists(
        LOSS_FILE
    ):
        raise FileNotFoundError(
            f"Missing loss file:\n"
            f"{LOSS_FILE}\n\n"
            "Run the reconciliation script first to generate "
            "reconciliation_losses_by_time.csv."
        )

    df = pd.read_csv(
        LOSS_FILE
    )

    required_columns = {
        "dimension",
        "surface",
        "method",
        "score",
        "t",
        "loss",
    }

    missing = (
        required_columns
        - set(
            df.columns
        )
    )

    if missing:
        raise ValueError(
            "Loss file is missing columns: "
            f"{sorted(missing)}"
        )

    return df


# ============================================================
# DM TESTS
# ============================================================

def compute_dm_tests(
    df_losses,
):
    rows = []

    grouped = df_losses.groupby(
        [
            "dimension",
            "surface",
            "score",
        ]
    )

    for (
        dimension,
        surface,
        score,
    ), group in grouped:

        available_methods = [
            method
            for method in METHOD_ORDER
            if method
            in set(
                group[
                    "method"
                ].unique()
            )
        ]

        wide = group.pivot_table(
            index="t",
            columns="method",
            values="loss",
            aggfunc="mean",
        ).sort_index()

        for method_a, method_b in combinations(
            available_methods,
            2,
        ):
            pair = wide[
                [
                    method_a,
                    method_b,
                ]
            ].dropna()

            result = diebold_mariano_test(
                loss_a=pair[
                    method_a
                ].values,
                loss_b=pair[
                    method_b
                ].values,
                horizon=HORIZON,
                hac_lag=HAC_LAG,
                harvey_correction=True,
            )

            mean_a = result[
                "mean_loss_a"
            ]

            mean_b = result[
                "mean_loss_b"
            ]

            if np.isfinite(mean_a) and np.isfinite(mean_b):
                if mean_a < mean_b:
                    better_method = method_a
                elif mean_b < mean_a:
                    better_method = method_b
                else:
                    better_method = "tie"
            else:
                better_method = np.nan

            p_value = result[
                "p_value"
            ]

            significant = (
                bool(
                    p_value < ALPHA
                )
                if np.isfinite(
                    p_value
                )
                else False
            )

            rows.append(
                {
                    "dimension": dimension,
                    "surface": surface,
                    "score": score,
                    "method_a": method_a,
                    "method_b": method_b,
                    "n_obs": result[
                        "n_obs"
                    ],
                    "mean_loss_a": mean_a,
                    "mean_loss_b": mean_b,
                    "mean_loss_diff_a_minus_b": result[
                        "mean_loss_diff"
                    ],
                    "dm_stat": result[
                        "dm_stat"
                    ],
                    "p_value": p_value,
                    "significant_5pct": significant,
                    "better_method": better_method,
                    "hac_lag": result[
                        "hac_lag"
                    ],
                }
            )

    return pd.DataFrame(
        rows
    )


def print_summary(
    df_dm,
):
    print()
    print("=" * 100)
    print("DIEBOLD-MARIANO TEST SUMMARY")
    print("=" * 100)
    print()

    if df_dm.empty:
        print(
            "No DM tests were computed."
        )
        return

    print(
        "Positive DM statistic means method_a has larger loss "
        "than method_b."
    )

    print(
        "Since lower loss is better, positive DM means "
        "method_a is worse."
    )

    print()

    significant = df_dm[
        df_dm[
            "significant_5pct"
        ]
        == True
    ].copy()

    if significant.empty:
        print(
            "No pairwise differences are significant at the 5% level."
        )
        return

    print(
        significant.to_string(
            index=False,
            formatters={
                "mean_loss_a": lambda x: f"{x:.6f}",
                "mean_loss_b": lambda x: f"{x:.6f}",
                "mean_loss_diff_a_minus_b": lambda x: f"{x:.6f}",
                "dm_stat": lambda x: f"{x:.4f}",
                "p_value": lambda x: f"{x:.4g}",
            },
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print(
        "Loading per-time-step losses..."
    )

    df_losses = load_losses()

    print(
        "Computing Diebold-Mariano tests..."
    )

    df_dm = compute_dm_tests(
        df_losses
    )

    df_dm = df_dm.sort_values(
        [
            "dimension",
            "surface",
            "score",
            "method_a",
            "method_b",
        ]
    )

    df_dm.to_csv(
        DM_OUTPUT_FILE,
        index=False,
    )

    print_summary(
        df_dm
    )

    print()
    print(
        f"DM results saved to:\n"
        f"  {DM_OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()