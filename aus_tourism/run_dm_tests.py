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

RESULTS_FOLDER = "results"

LOSS_FILE = os.path.join(
    RESULTS_FOLDER,
    "australian_tourism_losses_by_time.csv",
)

DM_OUTPUT_FILE = os.path.join(
    RESULTS_FOLDER,
    "australian_tourism_diebold_mariano_tests.csv",
)

UKF_INTERPRETATION_FILE = os.path.join(
    RESULTS_FOLDER,
    "australian_tourism_diebold_mariano_ukf_interpretation.csv",
)

TEXT_SUMMARY_FILE = os.path.join(
    RESULTS_FOLDER,
    "australian_tourism_diebold_mariano_interpretation.txt",
)

METHOD_ORDER = [
    "Base",
    "PBU",
    "OLS",
    "WLS",
    "FULL",
    "UKF",
]

REFERENCE_METHOD = "UKF"

PROJECTION_METHODS = [
    "OLS",
    "WLS",
    "FULL",
]

ALPHA = 0.05

HORIZON = 1

# Use "auto" for Newey-West/HAC correction of the loss-differential variance.
# Set to 0 if you want the simpler one-step DM test without serial-correlation correction.
HAC_LAG = "auto"


# ============================================================
# P-VALUE UTILITIES
# ============================================================

def _normal_two_sided_p_value(statistic):
    z = abs(float(statistic))
    return math.erfc(z / math.sqrt(2.0))


def _two_sided_p_value(statistic, dof):
    if not np.isfinite(statistic):
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


# ============================================================
# DIEBOLD-MARIANO TEST
# ============================================================

def _resolve_hac_lag(
    n,
    horizon,
    hac_lag,
):
    if hac_lag == "auto":
        return int(
            np.floor(
                4.0
                * (
                    n / 100.0
                )
                ** (
                    2.0 / 9.0
                )
            )
        )

    if hac_lag is None:
        return max(
            horizon - 1,
            0,
        )

    return int(
        hac_lag
    )


def diebold_mariano_test(
    loss_a,
    loss_b,
    horizon=1,
    hac_lag="auto",
    harvey_correction=True,
):
    """
    Paired Diebold-Mariano test for equal predictive accuracy.

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

    loss_a_valid = loss_a[
        valid
    ]

    loss_b_valid = loss_b[
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

    hac_lag = _resolve_hac_lag(
        n=n,
        horizon=horizon,
        hac_lag=hac_lag,
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
                loss_a_valid
            )
        ),

        "mean_loss_b": float(
            np.mean(
                loss_b_valid
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
            "Run the Australian reconciliation script first to generate "
            "australian_tourism_losses_by_time.csv."
        )

    df = pd.read_csv(
        LOSS_FILE
    )

    required_columns = {
        "target",
        "level",
        "score",
        "method",
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
# FULL PAIRWISE DM TABLE
# ============================================================

def compute_dm_tests(
    df_losses,
):
    rows = []

    grouped = df_losses.groupby(
        [
            "target",
            "level",
            "score",
        ]
    )

    for (
        target,
        level,
        score,
    ), group in grouped:

        available_methods = [
            method
            for method in METHOD_ORDER
            if method in set(
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

            p_value = result[
                "p_value"
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
                    "target": target,
                    "level": level,
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


# ============================================================
# UKF-FOCUSED INTERPRETATION
# ============================================================

def build_ukf_interpretation(
    df_dm,
):
    rows = []

    for _, row in df_dm.iterrows():
        method_a = row[
            "method_a"
        ]

        method_b = row[
            "method_b"
        ]

        if REFERENCE_METHOD not in {
            method_a,
            method_b,
        }:
            continue

        other_method = (
            method_b
            if method_a == REFERENCE_METHOD
            else method_a
        )

        if other_method not in PROJECTION_METHODS:
            continue

        if method_a == REFERENCE_METHOD:
            ukf_loss = row[
                "mean_loss_a"
            ]

            other_loss = row[
                "mean_loss_b"
            ]

        else:
            ukf_loss = row[
                "mean_loss_b"
            ]

            other_loss = row[
                "mean_loss_a"
            ]

        if (
            np.isfinite(
                ukf_loss
            )
            and np.isfinite(
                other_loss
            )
            and other_loss != 0
        ):
            loss_ratio = (
                ukf_loss
                / other_loss
            )

            relative_reduction = (
                1.0
                - loss_ratio
            )

        else:
            loss_ratio = np.nan
            relative_reduction = np.nan

        p_value = row[
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

        if not significant:
            conclusion = "not_significant"

        elif ukf_loss < other_loss:
            conclusion = "ukf_significantly_better"

        elif ukf_loss > other_loss:
            conclusion = "ukf_significantly_worse"

        else:
            conclusion = "significant_tie"

        if conclusion == "ukf_significantly_better":
            interpretation = (
                f"UKF has significantly lower {row['score']} loss "
                f"than {other_method} on {row['target']} / {row['level']} "
                f"(p={p_value:.4g})."
            )

        elif conclusion == "ukf_significantly_worse":
            interpretation = (
                f"UKF has significantly higher {row['score']} loss "
                f"than {other_method} on {row['target']} / {row['level']} "
                f"(p={p_value:.4g})."
            )

        else:
            interpretation = (
                f"No statistically significant difference in {row['score']} loss "
                f"between UKF and {other_method} on {row['target']} / {row['level']} "
                f"(p={p_value:.4g})."
            )

        rows.append(
            {
                "target": row[
                    "target"
                ],
                "level": row[
                    "level"
                ],
                "score": row[
                    "score"
                ],
                "comparison": f"UKF_vs_{other_method}",
                "other_method": other_method,
                "ukf_mean_loss": ukf_loss,
                "other_mean_loss": other_loss,
                "ukf_loss_ratio_to_other": loss_ratio,
                "ukf_relative_loss_reduction": relative_reduction,
                "dm_stat_original_orientation": row[
                    "dm_stat"
                ],
                "p_value": p_value,
                "significant_5pct": significant,
                "conclusion": conclusion,
                "interpretation": interpretation,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# TEXT SUMMARY
# ============================================================

def write_text_summary(
    df_ukf,
):
    lines = []

    lines.append(
        "DIEBOLD-MARIANO TEST INTERPRETATION"
    )

    lines.append(
        "=" * 80
    )

    lines.append(
        ""
    )

    lines.append(
        "The Diebold-Mariano test was applied to paired per-time-step loss sequences."
    )

    lines.append(
        "The null hypothesis is equal predictive accuracy between the two methods."
    )

    lines.append(
        "Lower values are better for both CRPS and energy score."
    )

    lines.append(
        "In the full DM table, a positive DM statistic means method_a has larger average loss than method_b."
    )

    lines.append(
        f"Significance level: alpha = {ALPHA}."
    )

    lines.append(
        f"HAC lag used for the loss-differential variance: {HAC_LAG}."
    )

    lines.append(
        ""
    )

    if df_ukf.empty:
        lines.append(
            "No UKF-vs-projection comparisons were found."
        )

    else:
        lines.append(
            "UKF VS PROJECTION METHODS"
        )

        lines.append(
            "-" * 80
        )

        summary = (
            df_ukf.groupby(
                [
                    "target",
                    "level",
                    "score",
                    "other_method",
                    "conclusion",
                ]
            )
            .size()
            .reset_index(
                name="count"
            )
        )

        totals = (
            df_ukf.groupby(
                [
                    "target",
                    "level",
                    "score",
                    "other_method",
                ]
            )
            .size()
            .reset_index(
                name="total"
            )
        )

        for _, total_row in totals.iterrows():
            target = total_row[
                "target"
            ]

            level = total_row[
                "level"
            ]

            score = total_row[
                "score"
            ]

            other_method = total_row[
                "other_method"
            ]

            total = int(
                total_row[
                    "total"
                ]
            )

            sub = summary[
                (
                    summary[
                        "target"
                    ]
                    == target
                )
                &
                (
                    summary[
                        "level"
                    ]
                    == level
                )
                &
                (
                    summary[
                        "score"
                    ]
                    == score
                )
                &
                (
                    summary[
                        "other_method"
                    ]
                    == other_method
                )
            ]

            counts = {
                row[
                    "conclusion"
                ]: int(
                    row[
                        "count"
                    ]
                )
                for _, row in sub.iterrows()
            }

            better = counts.get(
                "ukf_significantly_better",
                0,
            )

            worse = counts.get(
                "ukf_significantly_worse",
                0,
            )

            not_sig = counts.get(
                "not_significant",
                0,
            )

            lines.append(
                (
                    f"{target}, {level}, {score}: UKF vs {other_method}: "
                    f"{better}/{total} significantly better, "
                    f"{worse}/{total} significantly worse, "
                    f"{not_sig}/{total} not significant."
                )
            )

        lines.append(
            ""
        )

        lines.append(
            "DETAILED SIGNIFICANT UKF-VS-PROJECTION COMPARISONS"
        )

        lines.append(
            "-" * 80
        )

        lines.append(
            "Only statistically significant comparisons are listed below; "
            "non-significant comparisons are summarized in the counts above."
        )

        lines.append(
            ""
        )

        significant = df_ukf[
            df_ukf[
                "significant_5pct"
            ]
            == True
        ]

        if significant.empty:
            lines.append(
                "No UKF-vs-projection differences are significant at the 5% level."
            )

        else:
            for _, row in significant.iterrows():
                ratio = row[
                    "ukf_loss_ratio_to_other"
                ]

                if np.isfinite(
                    ratio
                ):
                    ratio_text = (
                        f" UKF/other loss ratio = {ratio:.4f}."
                    )

                else:
                    ratio_text = ""

                lines.append(
                    row[
                        "interpretation"
                    ]
                    + ratio_text
                )

    text = "\n".join(
        lines
    )

    with open(
        TEXT_SUMMARY_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            text
        )

    print()
    print(
        text
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
        "Computing pairwise Diebold-Mariano tests..."
    )

    df_dm = compute_dm_tests(
        df_losses
    )

    df_dm = df_dm.sort_values(
        [
            "target",
            "level",
            "score",
            "method_a",
            "method_b",
        ]
    )

    df_dm.to_csv(
        DM_OUTPUT_FILE,
        index=False,
    )

    print(
        f"Saved full DM table: {DM_OUTPUT_FILE}"
    )

    print(
        "Building UKF interpretation table..."
    )

    df_ukf = build_ukf_interpretation(
        df_dm
    )

    df_ukf = df_ukf.sort_values(
        [
            "target",
            "level",
            "score",
            "other_method",
        ]
    )

    df_ukf.to_csv(
        UKF_INTERPRETATION_FILE,
        index=False,
    )

    print(
        f"Saved UKF interpretation table: {UKF_INTERPRETATION_FILE}"
    )

    write_text_summary(
        df_ukf
    )

    print(
        f"Saved text interpretation: {TEXT_SUMMARY_FILE}"
    )


if __name__ == "__main__":
    main()