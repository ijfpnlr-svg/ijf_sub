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

ENERGY_LOSS_FILE = os.path.join(
    RESULTS_FOLDER,
    "reconciliation_losses_by_time.csv",
)

CRPS_SERIES_LOSS_FILE = os.path.join(
    RESULTS_FOLDER,
    "reconciliation_crps_by_series_time.csv",
)

OUTPUT_PREFIX = "reconciliation"

MCS_SUMMARY_FILE = os.path.join(
    RESULTS_FOLDER,
    f"{OUTPUT_PREFIX}_mcs_paper_relative_score_summary.csv",
)

MCS_TEXT_SUMMARY_FILE = os.path.join(
    RESULTS_FOLDER,
    f"{OUTPUT_PREFIX}_mcs_paper_relative_score_interpretation.txt",
)

MCS_PLOT_FOLDER = os.path.join(
    RESULTS_FOLDER,
    f"{OUTPUT_PREFIX}_mcs_paper_relative_score_plots",
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

N_BOOTSTRAP = 5000

BLOCK_LENGTH = "auto"

RANDOM_SEED = 42

EPSILON = 1e-12

PLOT_DPI = 300


# ============================================================
# LOADING
# ============================================================

def load_energy_losses():
    if not os.path.exists(
        ENERGY_LOSS_FILE
    ):
        raise FileNotFoundError(
            f"Missing file:\n{ENERGY_LOSS_FILE}\n\n"
            "Run the updated reconciliation script first."
        )

    df = pd.read_csv(
        ENERGY_LOSS_FILE
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
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Energy loss file is missing columns: "
            f"{sorted(missing)}"
        )

    df["method"] = (
        df["method"]
        .astype(str)
        .str.lower()
    )

    df["surface"] = (
        df["surface"]
        .astype(str)
        .str.lower()
    )

    df["score"] = (
        df["score"]
        .astype(str)
        .str.lower()
    )

    df["dimension"] = df["dimension"].astype(
        int
    )

    df["t"] = df["t"].astype(
        int
    )

    df["loss"] = df["loss"].astype(
        float
    )

    df = df[
        df["score"] == "energy_score"
    ].copy()

    if df.empty:
        raise ValueError(
            "No energy_score rows found in "
            f"{ENERGY_LOSS_FILE}"
        )

    return df


def load_crps_series_losses():
    if not os.path.exists(
        CRPS_SERIES_LOSS_FILE
    ):
        raise FileNotFoundError(
            f"Missing file:\n{CRPS_SERIES_LOSS_FILE}\n\n"
            "Run the updated reconciliation script first. "
            "It must export per-series CRPS losses."
        )

    df = pd.read_csv(
        CRPS_SERIES_LOSS_FILE
    )

    required_columns = {
        "dimension",
        "surface",
        "method",
        "score",
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

    df["method"] = (
        df["method"]
        .astype(str)
        .str.lower()
    )

    df["surface"] = (
        df["surface"]
        .astype(str)
        .str.lower()
    )

    df["score"] = (
        df["score"]
        .astype(str)
        .str.lower()
    )

    df["dimension"] = df["dimension"].astype(
        int
    )

    df["series_index"] = df["series_index"].astype(
        int
    )

    df["t"] = df["t"].astype(
        int
    )

    df["loss"] = df["loss"].astype(
        float
    )

    df = df[
        df["score"] == "crps"
    ].copy()

    if df.empty:
        raise ValueError(
            "No crps rows found in "
            f"{CRPS_SERIES_LOSS_FILE}"
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


# ============================================================
# ENERGY SCORE STATISTIC
# ============================================================

def build_energy_matrix(group):
    wide = group.pivot_table(
        index="t",
        columns="method",
        values="loss",
        aggfunc="mean",
    ).sort_index()

    methods = order_methods(
        wide.columns
    )

    if REFERENCE_METHOD not in methods:
        raise ValueError(
            "Base method missing in Energy Score group."
        )

    wide = wide[
        methods
    ].copy()

    wide = wide.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    wide = wide.dropna(
        axis=0,
        how="any",
    )

    if wide.empty:
        raise ValueError(
            "No complete Energy Score time steps."
        )

    if (
        wide.values < 0.0
    ).any():
        raise ValueError(
            "Energy Score losses must be non-negative."
        )

    return (
        wide.values,
        methods,
        wide.index.to_numpy(),
    )


def compute_relative_energy_score(
    energy_matrix,
    methods,
):
    base_index = methods.index(
        REFERENCE_METHOD
    )

    mean_scores = np.mean(
        energy_matrix,
        axis=0,
    )

    base_score = mean_scores[
        base_index
    ]

    if not np.isfinite(
        base_score
    ) or base_score <= 0.0:
        raise ValueError(
            "Base Energy Score must be positive and finite."
        )

    relative_scores = (
        mean_scores
        / (
            base_score
            + EPSILON
        )
    )

    return (
        mean_scores,
        relative_scores,
    )


# ============================================================
# CRPS STATISTIC
# ============================================================

def build_crps_tensor(group):
    methods = order_methods(
        group["method"].unique()
    )

    if REFERENCE_METHOD not in methods:
        raise ValueError(
            "Base method missing in CRPS group."
        )

    times = np.array(
        sorted(
            group["t"].unique()
        ),
        dtype=int,
    )

    series_indices = np.array(
        sorted(
            group["series_index"].unique()
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
        :
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
# COMMON MCS-STYLE PROCEDURE
# ============================================================

def compute_mcs_from_bootstrap_statistics(
    observed_relative_scores,
    bootstrap_relative_scores,
    methods,
):
    best_index = int(
        np.argmin(
            observed_relative_scores
        )
    )

    best_method = methods[
        best_index
    ]

    best_relative_score = observed_relative_scores[
        best_index
    ]

    difference_to_best = (
        observed_relative_scores
        - best_relative_score
    )

    centered_score_boot = (
        bootstrap_relative_scores
        - observed_relative_scores[
            None,
            :,
        ]
    )

    max_abs_score_deviation = np.max(
        np.abs(
            centered_score_boot
        ),
        axis=1,
    )

    q_score = float(
        np.quantile(
            max_abs_score_deviation,
            1.0 - ALPHA,
        )
    )

    relative_score_lower = np.maximum(
        observed_relative_scores
        - q_score,
        0.0,
    )

    relative_score_upper = (
        observed_relative_scores
        + q_score
    )

    difference_to_best_boot = (
        bootstrap_relative_scores
        - bootstrap_relative_scores[
            :,
            [
                best_index
            ],
        ]
    )

    centered_difference_boot = (
        difference_to_best_boot
        - difference_to_best[
            None,
            :,
        ]
    )

    max_abs_difference_deviation = np.max(
        np.abs(
            centered_difference_boot
        ),
        axis=1,
    )

    q_difference = float(
        np.quantile(
            max_abs_difference_deviation,
            1.0 - ALPHA,
        )
    )

    difference_to_best_lower = (
        difference_to_best
        - q_difference
    )

    difference_to_best_upper = (
        difference_to_best
        + q_difference
    )

    significantly_worse_than_best = (
        difference_to_best_lower
        > 0.0
    )

    in_mcs_set = (
        ~significantly_worse_than_best
    )

    ratio_to_best = (
        observed_relative_scores
        / (
            best_relative_score
            + EPSILON
        )
    )

    mcs_acceptance_cutoff = (
        best_relative_score
        + q_difference
    )

    return {
        "best_index": best_index,
        "best_method": best_method,
        "best_relative_score": best_relative_score,
        "relative_score_lower": relative_score_lower,
        "relative_score_upper": relative_score_upper,
        "difference_to_best": difference_to_best,
        "difference_to_best_lower": difference_to_best_lower,
        "difference_to_best_upper": difference_to_best_upper,
        "ratio_to_best": ratio_to_best,
        "significantly_worse_than_best": significantly_worse_than_best,
        "in_mcs_set": in_mcs_set,
        "q_score": q_score,
        "q_difference": q_difference,
        "mcs_acceptance_cutoff": mcs_acceptance_cutoff,
    }


def compute_energy_mcs_group(
    group,
    dimension,
    surface,
):
    energy_matrix, methods, times = build_energy_matrix(
        group
    )

    n_obs = energy_matrix.shape[
        0
    ]

    if n_obs < 3:
        raise ValueError(
            "At least 3 forecast origins are needed."
        )

    block_length = resolve_block_length(
        n_obs
    )

    mean_raw_scores, relative_scores = compute_relative_energy_score(
        energy_matrix=energy_matrix,
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

        boot_mean_raw_scores, boot_relative_scores = compute_relative_energy_score(
            energy_matrix=energy_matrix[
                indices,
                :,
            ],
            methods=methods,
        )

        bootstrap_relative_scores[
            boot_index,
            :,
        ] = boot_relative_scores

    mcs = compute_mcs_from_bootstrap_statistics(
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
                "dimension": dimension,
                "surface": surface,
                "metric": "relative_energy_score",
                "method": method,
                "method_label": method_display_name(
                    method
                ),
                "n_obs": n_obs,
                "n_series": np.nan,
                "block_length": block_length,
                "n_bootstrap": N_BOOTSTRAP,
                "alpha": ALPHA,
                "mean_raw_score_diagnostic": float(
                    mean_raw_scores[
                        method_index
                    ]
                ),
                "relative_score": float(
                    relative_scores[
                        method_index
                    ]
                ),
                "relative_score_ci_lower": float(
                    mcs[
                        "relative_score_lower"
                    ][
                        method_index
                    ]
                ),
                "relative_score_ci_upper": float(
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
                "best_relative_score": float(
                    mcs[
                        "best_relative_score"
                    ]
                ),
                "difference_to_best": float(
                    mcs[
                        "difference_to_best"
                    ][
                        method_index
                    ]
                ),
                "difference_to_best_ci_lower": float(
                    mcs[
                        "difference_to_best_lower"
                    ][
                        method_index
                    ]
                ),
                "difference_to_best_ci_upper": float(
                    mcs[
                        "difference_to_best_upper"
                    ][
                        method_index
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
                "q_difference": float(
                    mcs[
                        "q_difference"
                    ]
                ),
                "mcs_acceptance_cutoff": float(
                    mcs[
                        "mcs_acceptance_cutoff"
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


def compute_crps_mcs_group(
    group,
    dimension,
    surface,
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

    mcs = compute_mcs_from_bootstrap_statistics(
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
                "dimension": dimension,
                "surface": surface,
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
                "mean_raw_score_diagnostic": float(
                    mean_raw_scores[
                        method_index
                    ]
                ),
                "relative_score": float(
                    relative_scores[
                        method_index
                    ]
                ),
                "relative_score_ci_lower": float(
                    mcs[
                        "relative_score_lower"
                    ][
                        method_index
                    ]
                ),
                "relative_score_ci_upper": float(
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
                "best_relative_score": float(
                    mcs[
                        "best_relative_score"
                    ]
                ),
                "difference_to_best": float(
                    mcs[
                        "difference_to_best"
                    ][
                        method_index
                    ]
                ),
                "difference_to_best_ci_lower": float(
                    mcs[
                        "difference_to_best_lower"
                    ][
                        method_index
                    ]
                ),
                "difference_to_best_ci_upper": float(
                    mcs[
                        "difference_to_best_upper"
                    ][
                        method_index
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
                "q_difference": float(
                    mcs[
                        "q_difference"
                    ]
                ),
                "mcs_acceptance_cutoff": float(
                    mcs[
                        "mcs_acceptance_cutoff"
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
    dimension,
    surface,
    metric,
):
    os.makedirs(
        MCS_PLOT_FOLDER,
        exist_ok=True,
    )

    result_plot = result.sort_values(
        "relative_score",
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

    best_relative_score = float(
        result_plot[
            "best_relative_score"
        ].iloc[0]
    )

    mcs_acceptance_cutoff = float(
        result_plot[
            "mcs_acceptance_cutoff"
        ].iloc[0]
    )

    x_min_data = float(
        result_plot[
            "relative_score_ci_lower"
        ].min()
    )

    x_max_data = float(
        result_plot[
            "relative_score_ci_upper"
        ].max()
    )

    x_min = min(
        x_min_data,
        best_relative_score,
    )

    x_max = max(
        x_max_data,
        mcs_acceptance_cutoff,
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

    ax.axvspan(
        x_left,
        mcs_acceptance_cutoff,
        alpha=0.12,
        zorder=0,
    )

    ax.axvline(
        mcs_acceptance_cutoff,
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
        x = row.relative_score

        x_lower = row.relative_score_ci_lower

        x_upper = row.relative_score_ci_upper

        xerr = np.array(
            [
                [
                    x
                    - x_lower
                ],
                [
                    x_upper
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
        "Relative score"
    )

    ax.set_title(
        "MCS block-bootstrap intervals"
    )

    ax.grid(
        True,
        axis="x",
        alpha=0.3,
    )

    fig.tight_layout()

    filename = (
        f"{OUTPUT_PREFIX}_mcs_"
        f"d{dimension}_"
        f"{sanitize_for_filename(surface)}_"
        f"{sanitize_for_filename(metric)}.png"
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
        "MCS-STYLE PAPER RELATIVE SCORE INTERPRETATION"
    )

    lines.append(
        "=" * 80
    )

    lines.append("")

    lines.append(
        "Energy Score statistic:"
    )

    lines.append(
        "    RelES_m = mean_t ES_{m,t} / mean_t ES_{base,t}"
    )

    lines.append("")

    lines.append(
        "CRPS statistic:"
    )

    lines.append(
        "    RelCRPS_m = exp(mean_j log(mean_t CRPS_{m,j,t} / mean_t CRPS_{base,j,t}))"
    )

    lines.append("")

    lines.append(
        "The bootstrap is applied over forecast origins using moving blocks."
    )

    lines.append(
        "At each bootstrap replication, the full relative statistic is recomputed."
    )

    lines.append("")

    lines.append(
        "The confidence set contains methods that are not significantly worse "
        "than the observed best method on the corresponding relative-score scale."
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
            "dimension",
            "surface",
            "metric",
        ],
        dropna=False,
    )

    for (
        dimension,
        surface,
        metric,
    ), group in grouped:

        group = group.sort_values(
            "relative_score"
        )

        best_method_label = group[
            "best_method_label"
        ].iloc[0]

        best_relative_score = group[
            "best_relative_score"
        ].iloc[0]

        confidence_set = group[
            group[
                "in_mcs_set"
            ]
        ][
            "method_label"
        ].tolist()

        significantly_worse = group[
            ~group[
                "in_mcs_set"
            ]
        ][
            "method_label"
        ].tolist()

        lines.append(
            "-" * 80
        )

        lines.append(
            f"d={dimension}, surface={surface}, metric={metric}"
        )

        lines.append(
            f"Best observed method: {best_method_label} "
            f"(relative score = {best_relative_score:.4f})"
        )

        lines.append(
            "MCS-style confidence set: "
            + ", ".join(
                confidence_set
            )
        )

        if significantly_worse:
            lines.append(
                "Significantly worse than best: "
                + ", ".join(
                    significantly_worse
                )
            )
        else:
            lines.append(
                "No method is significantly worse than the observed best."
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
# MAIN
# ============================================================

def main():
    print(
        "Loading Energy Score losses..."
    )

    df_energy = load_energy_losses()

    print(
        f"Loaded: {ENERGY_LOSS_FILE}"
    )

    print(
        "Loading per-series CRPS losses..."
    )

    df_crps = load_crps_series_losses()

    print(
        f"Loaded: {CRPS_SERIES_LOSS_FILE}"
    )

    all_results = []

    grouped_energy = df_energy.groupby(
        [
            "dimension",
            "surface",
        ],
        dropna=False,
    )

    for (
        dimension,
        surface,
    ), group in grouped_energy:

        print()
        print(
            f"Processing RelES: d={dimension}, surface={surface}"
        )

        try:
            result = compute_energy_mcs_group(
                group=group,
                dimension=dimension,
                surface=surface,
            )

            plot_file = plot_mcs_group(
                result=result,
                dimension=dimension,
                surface=surface,
                metric="relative_energy_score",
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
                "Skipping Energy Score group because of error:"
            )
            print(
                exc
            )

    grouped_crps = df_crps.groupby(
        [
            "dimension",
            "surface",
        ],
        dropna=False,
    )

    for (
        dimension,
        surface,
    ), group in grouped_crps:

        print()
        print(
            f"Processing RelCRPS: d={dimension}, surface={surface}"
        )

        try:
            result = compute_crps_mcs_group(
                group=group,
                dimension=dimension,
                surface=surface,
            )

            plot_file = plot_mcs_group(
                result=result,
                dimension=dimension,
                surface=surface,
                metric="geometric_mean_relative_crps",
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
            "dimension",
            "surface",
            "metric",
            "relative_score",
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

    print()
    print(
        f"Saved MCS text summary: {MCS_TEXT_SUMMARY_FILE}"
    )

    print()
    print(
        f"Saved MCS plots in: {MCS_PLOT_FOLDER}"
    )


if __name__ == "__main__":
    main()