import math
import os
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

RESULTS_FOLDER = "../results"

CRPS_SERIES_LOSS_FILE = os.path.join(
    RESULTS_FOLDER,
    "swiss_demo_crps_by_series_time.csv",
)

OUTPUT_PREFIX = "swiss_demo"

MCS_SUMMARY_FILE = os.path.join(
    RESULTS_FOLDER,
    f"{OUTPUT_PREFIX}_sequential_studentized_mcs_crps_summary.csv",
)

MCS_TEXT_SUMMARY_FILE = os.path.join(
    RESULTS_FOLDER,
    f"{OUTPUT_PREFIX}_sequential_studentized_mcs_crps_interpretation.txt",
)

MCS_PLOT_FOLDER = os.path.join(
    RESULTS_FOLDER,
    f"{OUTPUT_PREFIX}_sequential_studentized_mcs_crps_plots",
)

MCS_LATEX_TABLE_FILE = os.path.join(
    RESULTS_FOLDER,
    f"{OUTPUT_PREFIX}_sequential_studentized_mcs_latex_tables.txt",
)

METHOD_ORDER = [
    "base",
    "pbu",
    "ols",
    "wls",
    "full",
    "ukf",
]

METHOD_LABELS = {
    "base": "Base",
    "pbu": "PBU",
    "ols": "OLS",
    "wls": "WLS",
    "full": "FULL",
    "ukf": "UKF",
}

REFERENCE_METHOD = "base"

ALPHA = 0.05

N_BOOTSTRAP = 1000

BLOCK_LENGTH = "auto"

RANDOM_SEED = 42

EPSILON = 1e-12

PLOT_DPI = 300

LEVELS_TO_KEEP = [
    "full",
]
# LEVELS_TO_KEEP = None


# ============================================================
# LOADING
# ============================================================

def load_crps_series_losses():
    if not os.path.exists(
        CRPS_SERIES_LOSS_FILE
    ):
        raise FileNotFoundError(
            f"Missing file:\n{CRPS_SERIES_LOSS_FILE}\n\n"
            "Run the updated Swiss demo reconciliation script first. "
            "It must export per-series CRPS losses."
        )

    df = pd.read_csv(
        CRPS_SERIES_LOSS_FILE
    )

    required_columns = {
        "target",
        "level",
        "score",
        "method",
        "series_index",
        "t",
        "loss",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "CRPS series loss file is missing columns: "
            f"{sorted(missing)}"
        )

    df["target"] = (
        df["target"]
        .astype(str)
        .str.lower()
    )

    df["level"] = (
        df["level"]
        .astype(str)
        .str.lower()
    )

    df["score"] = (
        df["score"]
        .astype(str)
        .str.lower()
    )

    df["method"] = (
        df["method"]
        .astype(str)
        .str.lower()
    )

    df["series_index"] = df[
        "series_index"
    ].astype(
        int
    )

    df["t"] = df[
        "t"
    ].astype(
        int
    )

    df["loss"] = df[
        "loss"
    ].astype(
        float
    )

    df = df[
        df["score"] == "crps"
    ].copy()

    if LEVELS_TO_KEEP is not None:
        levels = [
            str(level).lower()
            for level in LEVELS_TO_KEEP
        ]

        df = df[
            df["level"].isin(
                levels
            )
        ].copy()

    if df.empty:
        raise ValueError(
            "No CRPS rows found after filtering."
        )

    return df


# ============================================================
# UTILITIES
# ============================================================

def sanitize_for_filename(text):
    text = str(
        text
    ).strip()

    text = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        text,
    )

    text = re.sub(
        r"_+",
        "_",
        text,
    )

    return text.strip(
        "_"
    )


def method_display_name(method):
    return METHOD_LABELS.get(
        method,
        str(method),
    )


def order_methods(methods):
    methods = list(
        methods
    )

    ordered = [
        method
        for method in METHOD_ORDER
        if method in methods
    ]

    extra = sorted(
        method
        for method in methods
        if method not in ordered
    )

    return (
        ordered
        + extra
    )


def resolve_block_length(n_obs):
    if BLOCK_LENGTH == "auto":
        return max(
            2,
            int(
                round(
                    math.sqrt(
                        n_obs
                    )
                )
            ),
        )

    block_length = int(
        BLOCK_LENGTH
    )

    if block_length < 1:
        raise ValueError(
            "BLOCK_LENGTH must be positive."
        )

    return min(
        block_length,
        n_obs,
    )


def moving_block_bootstrap_indices(
    n_obs,
    block_length,
    rng,
):
    if n_obs <= 0:
        raise ValueError(
            "n_obs must be positive."
        )

    if block_length <= 0:
        raise ValueError(
            "block_length must be positive."
        )

    if block_length > n_obs:
        block_length = n_obs

    n_blocks = int(
        math.ceil(
            n_obs
            / block_length
        )
    )

    max_start = (
        n_obs
        - block_length
    )

    starts = rng.integers(
        low=0,
        high=max_start + 1,
        size=n_blocks,
    )

    indices = []

    for start in starts:
        indices.extend(
            range(
                start,
                start + block_length,
            )
        )

    return np.asarray(
        indices[:n_obs],
        dtype=int,
    )


def safe_float_or_nan(value):
    try:
        if pd.isna(
            value
        ):
            return np.nan
    except TypeError:
        pass

    return float(
        value
    )


# ============================================================
# CRPS STATISTIC
# ============================================================

def build_crps_tensor(group):
    methods = order_methods(
        group[
            "method"
        ].unique()
    )

    if REFERENCE_METHOD not in methods:
        raise ValueError(
            "Base method missing in CRPS group."
        )

    times = np.array(
        sorted(
            group[
                "t"
            ].unique()
        ),
        dtype=int,
    )

    series_indices = np.array(
        sorted(
            group[
                "series_index"
            ].unique()
        ),
        dtype=int,
    )

    time_to_pos = {
        value: index
        for index, value in enumerate(
            times
        )
    }

    series_to_pos = {
        value: index
        for index, value in enumerate(
            series_indices
        )
    }

    method_to_pos = {
        value: index
        for index, value in enumerate(
            methods
        )
    }

    tensor = np.full(
        (
            len(times),
            len(series_indices),
            len(methods),
        ),
        np.nan,
        dtype=float,
    )

    for row in group.itertuples(
        index=False
    ):
        t_pos = time_to_pos[
            int(
                row.t
            )
        ]

        series_pos = series_to_pos[
            int(
                row.series_index
            )
        ]

        method_pos = method_to_pos[
            row.method
        ]

        tensor[
            t_pos,
            series_pos,
            method_pos,
        ] = float(
            row.loss
        )

    valid_time_mask = np.all(
        np.isfinite(
            tensor
        ),
        axis=(
            1,
            2,
        ),
    )

    tensor = tensor[
        valid_time_mask,
        :,
        :,
    ]

    times = times[
        valid_time_mask
    ]

    if tensor.size == 0:
        raise ValueError(
            "No complete CRPS time steps."
        )

    if (
        tensor < 0.0
    ).any():
        raise ValueError(
            "CRPS losses must be non-negative."
        )

    return (
        tensor,
        methods,
        times,
        series_indices,
    )


def compute_relative_crps_geomean(
    crps_tensor,
    methods,
):
    base_index = methods.index(
        REFERENCE_METHOD
    )

    mean_crps_by_series_method = np.mean(
        crps_tensor,
        axis=0,
    )

    base_mean_by_series = mean_crps_by_series_method[
        :,
        base_index,
    ]

    valid_series = (
        np.isfinite(
            base_mean_by_series
        )
        & (
            base_mean_by_series
            > 0.0
        )
    )

    if not np.any(
        valid_series
    ):
        raise ValueError(
            "No valid base CRPS series."
        )

    mean_crps_by_series_method = mean_crps_by_series_method[
        valid_series,
        :,
    ]

    base_mean_by_series = base_mean_by_series[
        valid_series
    ]

    ratios = (
        mean_crps_by_series_method
        + EPSILON
    ) / (
        base_mean_by_series[
            :,
            None,
        ]
        + EPSILON
    )

    relative_scores = np.exp(
        np.nanmean(
            np.log(
                ratios
            ),
            axis=0,
        )
    )

    diagnostic_arithmetic_mean_crps = np.nanmean(
        mean_crps_by_series_method,
        axis=0,
    )

    return (
        diagnostic_arithmetic_mean_crps,
        relative_scores,
    )


# ============================================================
# STUDENTIZED SEQUENTIAL MCS PROCEDURE
# ============================================================

def compute_relative_score_intervals(
    observed_relative_scores,
    bootstrap_relative_scores,
):
    observed_relative_scores = np.asarray(
        observed_relative_scores,
        dtype=float,
    )

    bootstrap_relative_scores = np.asarray(
        bootstrap_relative_scores,
        dtype=float,
    )

    score_se = np.std(
        bootstrap_relative_scores,
        axis=0,
        ddof=1,
    )

    score_se = np.where(
        score_se <= EPSILON,
        EPSILON,
        score_se,
    )

    centered_score_boot = (
        bootstrap_relative_scores
        - observed_relative_scores[
            None,
            :,
        ]
    )

    studentized_score_boot = (
        centered_score_boot
        / score_se[
            None,
            :,
        ]
    )

    max_abs_studentized_score = np.max(
        np.abs(
            studentized_score_boot
        ),
        axis=1,
    )

    q_score = float(
        np.quantile(
            max_abs_studentized_score,
            1.0 - ALPHA,
        )
    )

    relative_score_lower = np.maximum(
        observed_relative_scores
        - q_score
        * score_se,
        0.0,
    )

    relative_score_upper = (
        observed_relative_scores
        + q_score
        * score_se
    )

    return {
        "score_se": score_se,
        "q_score": q_score,
        "relative_score_lower": relative_score_lower,
        "relative_score_upper": relative_score_upper,
    }


def compute_pairwise_studentized_objects(
    observed_relative_scores,
    bootstrap_relative_scores,
):
    observed_relative_scores = np.asarray(
        observed_relative_scores,
        dtype=float,
    )

    bootstrap_relative_scores = np.asarray(
        bootstrap_relative_scores,
        dtype=float,
    )

    observed_pairwise_difference = (
        observed_relative_scores[
            :,
            None,
        ]
        - observed_relative_scores[
            None,
            :,
        ]
    )

    bootstrap_pairwise_difference = (
        bootstrap_relative_scores[
            :,
            :,
            None,
        ]
        - bootstrap_relative_scores[
            :,
            None,
            :,
        ]
    )

    pairwise_se = np.std(
        bootstrap_pairwise_difference,
        axis=0,
        ddof=1,
    )

    pairwise_se = np.where(
        pairwise_se <= EPSILON,
        EPSILON,
        pairwise_se,
    )

    np.fill_diagonal(
        pairwise_se,
        np.inf,
    )

    observed_t_statistic = (
        observed_pairwise_difference
        / pairwise_se
    )

    np.fill_diagonal(
        observed_t_statistic,
        0.0,
    )

    centered_bootstrap_pairwise_difference = (
        bootstrap_pairwise_difference
        - observed_pairwise_difference[
            None,
            :,
            :,
        ]
    )

    bootstrap_t_statistic = (
        centered_bootstrap_pairwise_difference
        / pairwise_se[
            None,
            :,
            :,
        ]
    )

    for boot_index in range(
        bootstrap_t_statistic.shape[0]
    ):
        np.fill_diagonal(
            bootstrap_t_statistic[
                boot_index,
                :,
                :,
            ],
            0.0,
        )

    return {
        "observed_pairwise_difference": observed_pairwise_difference,
        "bootstrap_pairwise_difference": bootstrap_pairwise_difference,
        "pairwise_se": pairwise_se,
        "observed_t_statistic": observed_t_statistic,
        "bootstrap_t_statistic": bootstrap_t_statistic,
    }


def compute_tr_statistic_for_active_set(
    active_indices,
    observed_t_statistic,
    bootstrap_t_statistic,
):
    active_indices = np.asarray(
        active_indices,
        dtype=int,
    )

    n_active = len(
        active_indices
    )

    if n_active < 2:
        raise ValueError(
            "At least two active methods are required."
        )

    active_observed_t = observed_t_statistic[
        np.ix_(
            active_indices,
            active_indices,
        )
    ]

    active_bootstrap_t = bootstrap_t_statistic[
        :,
        active_indices,
        :,
    ][
        :,
        :,
        active_indices,
    ]

    off_diagonal_mask = ~np.eye(
        n_active,
        dtype=bool,
    )

    observed_values = active_observed_t[
        off_diagonal_mask
    ]

    bootstrap_values = active_bootstrap_t[
        :,
        off_diagonal_mask,
    ]

    observed_tr_statistic = float(
        np.max(
            np.abs(
                observed_values
            )
        )
    )

    bootstrap_tr_statistics = np.max(
        np.abs(
            bootstrap_values
        ),
        axis=1,
    )

    critical_value = float(
        np.quantile(
            bootstrap_tr_statistics,
            1.0 - ALPHA,
        )
    )

    p_value = float(
        np.mean(
            bootstrap_tr_statistics
            >= observed_tr_statistic
        )
    )

    reject_equal_performance = (
        observed_tr_statistic
        > critical_value
    )

    worst_scores = np.max(
        active_observed_t,
        axis=1,
    )

    eliminate_local_index = int(
        np.argmax(
            worst_scores
        )
    )

    eliminate_index = int(
        active_indices[
            eliminate_local_index
        ]
    )

    return {
        "observed_tr_statistic": observed_tr_statistic,
        "critical_value": critical_value,
        "p_value": p_value,
        "reject_equal_performance": reject_equal_performance,
        "eliminate_index": eliminate_index,
    }


def compute_sequential_studentized_mcs_from_bootstrap_statistics(
    observed_relative_scores,
    bootstrap_relative_scores,
    methods,
):
    observed_relative_scores = np.asarray(
        observed_relative_scores,
        dtype=float,
    )

    bootstrap_relative_scores = np.asarray(
        bootstrap_relative_scores,
        dtype=float,
    )

    methods = list(
        methods
    )

    n_methods = len(
        methods
    )

    if observed_relative_scores.shape[0] != n_methods:
        raise ValueError(
            "observed_relative_scores and methods have incompatible lengths."
        )

    if bootstrap_relative_scores.shape[1] != n_methods:
        raise ValueError(
            "bootstrap_relative_scores and methods have incompatible lengths."
        )

    score_intervals = compute_relative_score_intervals(
        observed_relative_scores=observed_relative_scores,
        bootstrap_relative_scores=bootstrap_relative_scores,
    )

    pairwise_objects = compute_pairwise_studentized_objects(
        observed_relative_scores=observed_relative_scores,
        bootstrap_relative_scores=bootstrap_relative_scores,
    )

    observed_t_statistic = pairwise_objects[
        "observed_t_statistic"
    ]

    bootstrap_t_statistic = pairwise_objects[
        "bootstrap_t_statistic"
    ]

    best_index = int(
        np.argmin(
            observed_relative_scores
        )
    )

    best_method = methods[
        best_index
    ]

    best_relative_score = float(
        observed_relative_scores[
            best_index
        ]
    )

    active = np.ones(
        n_methods,
        dtype=bool,
    )

    eliminated_step = np.full(
        n_methods,
        np.nan,
        dtype=float,
    )

    elimination_tr_statistic = np.full(
        n_methods,
        np.nan,
        dtype=float,
    )

    elimination_critical_value = np.full(
        n_methods,
        np.nan,
        dtype=float,
    )

    elimination_p_value = np.full(
        n_methods,
        np.nan,
        dtype=float,
    )

    final_tr_statistic = np.nan
    final_critical_value = np.nan
    final_p_value = np.nan
    stopped_by_non_rejection = False
    ended_with_single_method = False

    step = 1

    while np.sum(
        active
    ) > 1:

        active_indices = np.where(
            active
        )[0]

        active_test = compute_tr_statistic_for_active_set(
            active_indices=active_indices,
            observed_t_statistic=observed_t_statistic,
            bootstrap_t_statistic=bootstrap_t_statistic,
        )

        if not active_test[
            "reject_equal_performance"
        ]:
            final_tr_statistic = active_test[
                "observed_tr_statistic"
            ]

            final_critical_value = active_test[
                "critical_value"
            ]

            final_p_value = active_test[
                "p_value"
            ]

            stopped_by_non_rejection = True

            break

        eliminate_index = active_test[
            "eliminate_index"
        ]

        eliminated_step[
            eliminate_index
        ] = float(
            step
        )

        elimination_tr_statistic[
            eliminate_index
        ] = active_test[
            "observed_tr_statistic"
        ]

        elimination_critical_value[
            eliminate_index
        ] = active_test[
            "critical_value"
        ]

        elimination_p_value[
            eliminate_index
        ] = active_test[
            "p_value"
        ]

        active[
            eliminate_index
        ] = False

        step += 1

    if np.sum(
        active
    ) == 1:
        ended_with_single_method = True

    in_mcs_set = active

    significantly_worse_than_best = (
        ~in_mcs_set
    )

    ratio_to_best = (
        observed_relative_scores
        / (
            best_relative_score
            + EPSILON
        )
    )

    final_active_indices = np.where(
        in_mcs_set
    )[0]

    final_active_best_index = final_active_indices[
        int(
            np.argmin(
                observed_relative_scores[
                    final_active_indices
                ]
            )
        )
    ]

    return {
        "best_index": best_index,
        "best_method": best_method,
        "best_relative_score": best_relative_score,
        "final_active_best_index": int(
            final_active_best_index
        ),
        "final_active_best_method": methods[
            final_active_best_index
        ],
        "score_se": score_intervals[
            "score_se"
        ],
        "q_score": score_intervals[
            "q_score"
        ],
        "relative_score_lower": score_intervals[
            "relative_score_lower"
        ],
        "relative_score_upper": score_intervals[
            "relative_score_upper"
        ],
        "ratio_to_best": ratio_to_best,
        "observed_t_statistic": observed_t_statistic,
        "pairwise_se": pairwise_objects[
            "pairwise_se"
        ],
        "eliminated_step": eliminated_step,
        "elimination_tr_statistic": elimination_tr_statistic,
        "elimination_critical_value": elimination_critical_value,
        "elimination_p_value": elimination_p_value,
        "final_tr_statistic": final_tr_statistic,
        "final_critical_value": final_critical_value,
        "final_p_value": final_p_value,
        "stopped_by_non_rejection": stopped_by_non_rejection,
        "ended_with_single_method": ended_with_single_method,
        "significantly_worse_than_best": significantly_worse_than_best,
        "in_mcs_set": in_mcs_set,
    }


def compute_crps_mcs_group(
    group,
    target,
    level,
):
    crps_tensor, methods, times, series_indices = build_crps_tensor(
        group
    )

    n_obs = crps_tensor.shape[
        0
    ]

    n_series = crps_tensor.shape[
        1
    ]

    if n_obs < 3:
        raise ValueError(
            "At least 3 forecast origins are needed."
        )

    block_length = resolve_block_length(
        n_obs
    )

    mean_raw_scores, relative_scores = compute_relative_crps_geomean(
        crps_tensor=crps_tensor,
        methods=methods,
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    bootstrap_relative_scores = np.zeros(
        (
            N_BOOTSTRAP,
            len(methods),
        ),
        dtype=float,
    )

    for boot_index in range(
        N_BOOTSTRAP
    ):
        indices = moving_block_bootstrap_indices(
            n_obs=n_obs,
            block_length=block_length,
            rng=rng,
        )

        boot_mean_raw_scores, boot_relative_scores = compute_relative_crps_geomean(
            crps_tensor=crps_tensor[
                indices,
                :,
                :,
            ],
            methods=methods,
        )

        bootstrap_relative_scores[
            boot_index,
            :,
        ] = boot_relative_scores

    mcs = compute_sequential_studentized_mcs_from_bootstrap_statistics(
        observed_relative_scores=relative_scores,
        bootstrap_relative_scores=bootstrap_relative_scores,
        methods=methods,
    )

    rows = []

    for method_index, method in enumerate(
        methods
    ):
        rows.append(
            {
                "target": target,
                "level": level,
                "metric": "geometric_mean_relative_crps",
                "method": method,
                "method_label": method_display_name(
                    method
                ),
                "n_obs": n_obs,
                "n_series": n_series,
                "block_length": block_length,
                "n_bootstrap": N_BOOTSTRAP,
                "alpha": ALPHA,
                "mean_raw_crps_diagnostic": float(
                    mean_raw_scores[
                        method_index
                    ]
                ),
                "relative_crps": float(
                    relative_scores[
                        method_index
                    ]
                ),
                "relative_crps_se": float(
                    mcs[
                        "score_se"
                    ][
                        method_index
                    ]
                ),
                "relative_crps_ci_lower": float(
                    mcs[
                        "relative_score_lower"
                    ][
                        method_index
                    ]
                ),
                "relative_crps_ci_upper": float(
                    mcs[
                        "relative_score_upper"
                    ][
                        method_index
                    ]
                ),
                "best_method": mcs[
                    "best_method"
                ],
                "best_method_label": method_display_name(
                    mcs[
                        "best_method"
                    ]
                ),
                "best_relative_crps": float(
                    mcs[
                        "best_relative_score"
                    ]
                ),
                "final_active_best_method": mcs[
                    "final_active_best_method"
                ],
                "final_active_best_method_label": method_display_name(
                    mcs[
                        "final_active_best_method"
                    ]
                ),
                "ratio_to_best": float(
                    mcs[
                        "ratio_to_best"
                    ][
                        method_index
                    ]
                ),
                "q_score": float(
                    mcs[
                        "q_score"
                    ]
                ),
                "eliminated_step": safe_float_or_nan(
                    mcs[
                        "eliminated_step"
                    ][
                        method_index
                    ]
                ),
                "elimination_tr_statistic": safe_float_or_nan(
                    mcs[
                        "elimination_tr_statistic"
                    ][
                        method_index
                    ]
                ),
                "elimination_critical_value": safe_float_or_nan(
                    mcs[
                        "elimination_critical_value"
                    ][
                        method_index
                    ]
                ),
                "elimination_p_value": safe_float_or_nan(
                    mcs[
                        "elimination_p_value"
                    ][
                        method_index
                    ]
                ),
                "final_tr_statistic": safe_float_or_nan(
                    mcs[
                        "final_tr_statistic"
                    ]
                ),
                "final_critical_value": safe_float_or_nan(
                    mcs[
                        "final_critical_value"
                    ]
                ),
                "final_p_value": safe_float_or_nan(
                    mcs[
                        "final_p_value"
                    ]
                ),
                "stopped_by_non_rejection": bool(
                    mcs[
                        "stopped_by_non_rejection"
                    ]
                ),
                "ended_with_single_method": bool(
                    mcs[
                        "ended_with_single_method"
                    ]
                ),
                "significantly_worse_than_best": bool(
                    mcs[
                        "significantly_worse_than_best"
                    ][
                        method_index
                    ]
                ),
                "in_mcs_set": bool(
                    mcs[
                        "in_mcs_set"
                    ][
                        method_index
                    ]
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# PLOTTING
# ============================================================

def plot_mcs_group(
    result,
    target,
    level,
):
    os.makedirs(
        MCS_PLOT_FOLDER,
        exist_ok=True,
    )

    result_plot = result.sort_values(
        "relative_crps",
        ascending=True,
    ).copy()

    fig, ax = plt.subplots(
        figsize=(
            8.5,
            max(
                4.0,
                0.45
                * len(result_plot)
                + 2.0,
            ),
        )
    )

    y_positions = np.arange(
        len(result_plot)
    )

    labels = []

    for row in result_plot.itertuples(
        index=False
    ):
        label = row.method_label

        if row.in_mcs_set:
            label += "  ✓"
        else:
            label += "  ×"

        labels.append(
            label
        )

    x_min_data = float(
        result_plot[
            "relative_crps_ci_lower"
        ].min()
    )

    x_max_data = float(
        result_plot[
            "relative_crps_ci_upper"
        ].max()
    )

    best_relative_crps = float(
        result_plot[
            "best_relative_crps"
        ].iloc[0]
    )

    x_min = min(
        x_min_data,
        best_relative_crps,
    )

    x_max = max(
        x_max_data,
        best_relative_crps,
    )

    margin = 0.04 * (
        x_max
        - x_min
    )

    if margin <= 0:
        margin = 0.01

    x_left = (
        x_min
        - margin
    )

    x_right = (
        x_max
        + margin
    )

    ax.set_xlim(
        x_left,
        x_right,
    )

    ax.axvline(
        best_relative_crps,
        linestyle="--",
        linewidth=1.3,
        alpha=0.8,
        zorder=1,
    )

    for i, row in enumerate(
        result_plot.itertuples(
            index=False
        )
    ):
        x = row.relative_crps

        xerr = np.array(
            [
                [
                    x
                    - row.relative_crps_ci_lower
                ],
                [
                    row.relative_crps_ci_upper
                    - x
                ],
            ]
        )

        marker = (
            "o"
            if row.in_mcs_set
            else "x"
        )

        ax.errorbar(
            x,
            i,
            xerr=xerr,
            fmt=marker,
            capsize=4,
            markersize=6,
            linewidth=1.5,
            zorder=2,
        )

    ax.set_yticks(
        y_positions
    )

    ax.set_yticklabels(
        labels
    )

    ax.invert_yaxis()

    ax.set_xlabel(
        "Relative CRPS"
    )

    ax.set_title(
        r"Sequential studentized MCS"
    )

    ax.grid(
        True,
        axis="x",
        alpha=0.3,
    )

    fig.tight_layout()

    filename = (
        f"{OUTPUT_PREFIX}_sequential_studentized_mcs_crps_"
        f"{sanitize_for_filename(target)}_"
        f"{sanitize_for_filename(level)}.png"
    )

    plot_file = os.path.join(
        MCS_PLOT_FOLDER,
        filename,
    )

    fig.savefig(
        plot_file,
        dpi=PLOT_DPI,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return plot_file


# ============================================================
# TEXT SUMMARY
# ============================================================

def write_text_summary(
    summary_df,
):
    lines = []

    lines.append(
        "SWISS DEMO SEQUENTIAL STUDENTIZED MCS INTERPRETATION"
    )

    lines.append(
        "=" * 80
    )

    lines.append("")

    lines.append(
        "Statistic:"
    )

    lines.append(
        "    RelCRPS_m = exp(mean_j log(mean_t CRPS_{m,j,t} / mean_t CRPS_{base,j,t}))"
    )

    lines.append("")

    lines.append(
        "MCS procedure:"
    )

    lines.append(
        "    Sequential elimination using the studentized pairwise range statistic T_R."
    )

    lines.append(
        "    At each bootstrap replication, the full relative CRPS statistic is recomputed."
    )

    lines.append(
        "    The worst active method is eliminated if equal predictive ability is rejected."
    )

    lines.append("")

    lines.append(
        f"alpha = {ALPHA}"
    )

    lines.append(
        f"n_bootstrap = {N_BOOTSTRAP}"
    )

    lines.append(
        f"block_length = {BLOCK_LENGTH}"
    )

    lines.append("")

    grouped = summary_df.groupby(
        [
            "target",
            "level",
        ],
        dropna=False,
    )

    for (
        target,
        level,
    ), group in grouped:

        group = group.sort_values(
            "relative_crps"
        )

        best_method_label = group[
            "best_method_label"
        ].iloc[0]

        best_relative_crps = group[
            "best_relative_crps"
        ].iloc[0]

        final_active_best_method_label = group[
            "final_active_best_method_label"
        ].iloc[0]

        confidence_set = group[
            group[
                "in_mcs_set"
            ]
        ][
            "method_label"
        ].tolist()

        eliminated = group[
            ~group[
                "in_mcs_set"
            ]
        ].sort_values(
            "eliminated_step"
        )

        stopped_by_non_rejection = bool(
            group[
                "stopped_by_non_rejection"
            ].iloc[0]
        )

        ended_with_single_method = bool(
            group[
                "ended_with_single_method"
            ].iloc[0]
        )

        lines.append(
            "-" * 80
        )

        lines.append(
            f"target={target}, level={level}"
        )

        lines.append(
            f"Best observed method: {best_method_label} "
            f"(relative CRPS = {best_relative_crps:.4f})"
        )

        lines.append(
            f"Best method in final MCS set: {final_active_best_method_label}"
        )

        lines.append(
            "Final MCS set: "
            + ", ".join(
                confidence_set
            )
        )

        if stopped_by_non_rejection:
            final_p_value = group[
                "final_p_value"
            ].iloc[0]

            final_tr_statistic = group[
                "final_tr_statistic"
            ].iloc[0]

            final_critical_value = group[
                "final_critical_value"
            ].iloc[0]

            lines.append(
                f"Final EPA test: T_R = {final_tr_statistic:.4f}, "
                f"critical = {final_critical_value:.4f}, "
                f"p = {final_p_value:.4f}"
            )

        elif ended_with_single_method:
            lines.append(
                "Final EPA test: not applicable because only one method remains."
            )

        if not eliminated.empty:
            lines.append(
                "Eliminated methods:"
            )

            for row in eliminated.itertuples(
                index=False
            ):
                lines.append(
                    f"    step {int(row.eliminated_step)}: "
                    f"{row.method_label} "
                    f"(T_R = {row.elimination_tr_statistic:.4f}, "
                    f"critical = {row.elimination_critical_value:.4f}, "
                    f"p = {row.elimination_p_value:.4f})"
                )

        else:
            lines.append(
                "No method was eliminated."
            )

        lines.append("")

    text = "\n".join(
        lines
    )

    with open(
        MCS_TEXT_SUMMARY_FILE,
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
# LATEX TABLE OUTPUT
# ============================================================

def latex_escape(text):
    text = str(
        text
    )

    replacements = {
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
    }

    for old, new in replacements.items():
        text = text.replace(
            old,
            new,
        )

    return text


def latex_float(value, digits=4):
    if pd.isna(
        value
    ):
        return "--"

    return f"{float(value):.{digits}f}"


def latex_bool(value):
    return "Y" if bool(
        value
    ) else "N"


def write_latex_tables(
    summary_df,
):
    lines = []

    lines.append(
        "% ============================================================"
    )
    lines.append(
        "% Sequential studentized MCS elimination-step tables"
    )
    lines.append(
        "% ============================================================"
    )
    lines.append("")

    grouped = summary_df.groupby(
        [
            "target",
            "level",
        ],
        dropna=False,
    )

    for (
        target,
        level,
    ), group in grouped:

        eliminated = group[
            ~group[
                "in_mcs_set"
            ]
        ].sort_values(
            "eliminated_step"
        )

        stopped_by_non_rejection = bool(
            group[
                "stopped_by_non_rejection"
            ].iloc[0]
        )

        ended_with_single_method = bool(
            group[
                "ended_with_single_method"
            ].iloc[0]
        )

        alpha = group[
            "alpha"
        ].iloc[0]

        target_latex = latex_escape(
            target
        )

        level_latex = latex_escape(
            level
        )

        label_suffix = (
            f"{sanitize_for_filename(target)}_"
            f"{sanitize_for_filename(level)}"
        )

        lines.append(
            r"\begin{table}[h!]"
        )
        lines.append(
            r"    \centering"
        )
        lines.append(
            r"    \begin{tabular}{lcccc}"
        )
        lines.append(
            r"    \toprule"
        )
        lines.append(
            r"    Step & Eliminated method & $T_R$ & Critical value & $p$-value \\"
        )
        lines.append(
            r"    \midrule"
        )

        if not eliminated.empty:
            for row in eliminated.itertuples(
                index=False
            ):
                lines.append(
                    "    "
                    f"{int(row.eliminated_step)} & "
                    f"{latex_escape(row.method_label)} & "
                    f"{latex_float(row.elimination_tr_statistic)} & "
                    f"{latex_float(row.elimination_critical_value)} & "
                    f"{latex_float(row.elimination_p_value)} "
                    r"\\"
                )

            lines.append(
                r"    \midrule"
            )

        else:
            lines.append(
                r"    -- & No method eliminated & -- & -- & -- \\"
            )
            lines.append(
                r"    \midrule"
            )

        if stopped_by_non_rejection:
            final_tr_statistic = group[
                "final_tr_statistic"
            ].iloc[0]

            final_critical_value = group[
                "final_critical_value"
            ].iloc[0]

            final_p_value = group[
                "final_p_value"
            ].iloc[0]

            lines.append(
                "    "
                r"Final EPA test & -- & "
                f"{latex_float(final_tr_statistic)} & "
                f"{latex_float(final_critical_value)} & "
                f"{latex_float(final_p_value)} "
                r"\\"
            )

        elif ended_with_single_method:
            lines.append(
                r"    Final singleton set & -- & -- & -- & -- \\"
            )

        lines.append(
            r"    \bottomrule"
        )
        lines.append(
            r"    \end{tabular}"
        )
        lines.append(
            "    "
            r"\caption{Sequential studentized MCS elimination steps for the Swiss "
            f"{target_latex} dataset, level {level_latex}, with "
            rf"$\alpha={alpha}$."
            r"}"
        )
        lines.append(
            f"    \\label{{tab:swiss_mcs_steps_{label_suffix}}}"
        )
        lines.append(
            r"\end{table}"
        )
        lines.append("")

    lines.append("")
    lines.append(
        "% ============================================================"
    )
    lines.append(
        "% Final MCS membership tables"
    )
    lines.append(
        "% ============================================================"
    )
    lines.append("")

    grouped_membership = summary_df.groupby(
        [
            "target",
            "level",
        ],
        dropna=False,
    )

    for (
        target,
        level,
    ), group in grouped_membership:

        group = group.sort_values(
            "relative_crps"
        )

        alpha = group[
            "alpha"
        ].iloc[0]

        target_latex = latex_escape(
            target
        )

        level_latex = latex_escape(
            level
        )

        label_suffix = (
            f"{sanitize_for_filename(target)}_"
            f"{sanitize_for_filename(level)}"
        )

        lines.append(
            r"\begin{table}[h!]"
        )
        lines.append(
            r"    \centering"
        )
        lines.append(
            r"    \begin{tabular}{lc}"
        )
        lines.append(
            r"    \toprule"
        )
        lines.append(
            rf"    Method & MCS at $\alpha={alpha}$ \\"
        )
        lines.append(
            r"    \midrule"
        )

        for row in group.itertuples(
            index=False
        ):
            lines.append(
                "    "
                f"{latex_escape(row.method_label)} & "
                f"{latex_bool(row.in_mcs_set)} "
                r"\\"
            )

        lines.append(
            r"    \bottomrule"
        )
        lines.append(
            r"    \end{tabular}"
        )
        lines.append(
            "    "
            r"\caption{Final sequential studentized MCS membership for the Swiss "
            f"{target_latex} dataset, level {level_latex}. "
            r"Entries indicate whether each method is retained in the final MCS set.}"
        )
        lines.append(
            f"    \\label{{tab:swiss_mcs_membership_{label_suffix}}}"
        )
        lines.append(
            r"\end{table}"
        )
        lines.append("")

    latex_text = "\n".join(
        lines
    )

    with open(
        MCS_LATEX_TABLE_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            latex_text
        )

    print()
    print(
        f"Saved LaTeX tables to: {MCS_LATEX_TABLE_FILE}"
    )

# ============================================================
# MAIN
# ============================================================

def main():
    print(
        "Loading Swiss demo per-series CRPS losses..."
    )

    df_crps = load_crps_series_losses()

    print(
        f"Loaded: {CRPS_SERIES_LOSS_FILE}"
    )

    all_results = []

    grouped_crps = df_crps.groupby(
        [
            "target",
            "level",
        ],
        dropna=False,
    )

    for (
        target,
        level,
    ), group in grouped_crps:

        print()
        print(
            f"Processing sequential studentized MCS: target={target}, level={level}"
        )

        try:
            result = compute_crps_mcs_group(
                group=group,
                target=target,
                level=level,
            )

            plot_file = plot_mcs_group(
                result=result,
                target=target,
                level=level,
            )

            result[
                "plot_file"
            ] = plot_file

            all_results.append(
                result
            )

            print(
                f"Saved plot: {plot_file}"
            )

        except Exception as exc:
            print(
                "Skipping CRPS group because of error:"
            )

            print(
                exc
            )

    if not all_results:
        raise RuntimeError(
            "No MCS results were computed."
        )

    summary_df = pd.concat(
        all_results,
        axis=0,
        ignore_index=True,
    )

    summary_df = summary_df.sort_values(
        [
            "target",
            "level",
            "relative_crps",
            "method",
        ]
    )

    summary_df.to_csv(
        MCS_SUMMARY_FILE,
        index=False,
    )

    print()
    print(
        f"Saved MCS summary table: {MCS_SUMMARY_FILE}"
    )

    write_text_summary(
        summary_df=summary_df,
    )

    write_latex_tables(
        summary_df=summary_df,
    )

    print()
    print(
        f"Saved MCS text summary: {MCS_TEXT_SUMMARY_FILE}"
    )

    print()
    print(
        f"Saved MCS LaTeX tables: {MCS_LATEX_TABLE_FILE}"
    )

    print()
    print(
        f"Saved MCS plots in: {MCS_PLOT_FOLDER}"
    )


if __name__ == "__main__":
    main()