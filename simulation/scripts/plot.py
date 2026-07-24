from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

RESULTS_FOLDER = Path(
    "../results/complexity"
)

INPUT_FILE = (
    RESULTS_FOLDER
    / "timings_summary.csv"
)

OUTPUT_FOLDER = (
    RESULTS_FOLDER
    / "runtime_scaling_plots"
)

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)

UKF_METHOD = "ukf"
PROJECTION_METHOD = "full"

COMBINED_PLOT_FILE = (
    OUTPUT_FOLDER
    / "runtime_scaling_2x2_linear_xaxis.png"
)

SHOW_PLOTS = False


# ============================================================
# LOAD DATA
# ============================================================

def load_results():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing file:\n"
            f"{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    required_columns = {
        "surface",
        "method",
        "dimension",
        "n_samples",
        "mean_time_sec",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Missing columns: "
            f"{sorted(missing)}"
        )

    return df


# ============================================================
# AGGREGATION
# ============================================================

def geometric_mean(values):
    values = np.asarray(
        values,
        dtype=float,
    )

    valid = (
        np.isfinite(values)
        & (values > 0)
    )

    values = values[
        valid
    ]

    if len(values) == 0:
        return np.nan

    return float(
        np.exp(
            np.mean(
                np.log(values)
            )
        )
    )


def aggregate_surfaces(
    df,
    method,
):
    """
    Geometric mean runtime across surfaces for each:

        method
        dimension
        n_samples
    """
    df_method = (
        df[
            df["method"] == method
        ]
        .copy()
    )

    if df_method.empty:
        raise ValueError(
            f"No rows found for method '{method}'."
        )

    return (
        df_method
        .groupby(
            [
                "n_samples",
                "dimension",
            ],
            as_index=False,
        )
        .agg(
            mean_runtime_sec=(
                "mean_time_sec",
                geometric_mean,
            )
        )
        .sort_values(
            [
                "n_samples",
                "dimension",
            ]
        )
    )


# ============================================================
# PLOT HELPERS
# ============================================================

def add_linear_x_margin(
    ax,
    values,
    margin_fraction=0.04,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    x_min = np.nanmin(
        values
    )

    x_max = np.nanmax(
        values
    )

    margin = (
        x_max
        - x_min
    ) * margin_fraction

    if margin == 0:
        margin = 1.0

    ax.set_xlim(
        x_min - margin,
        x_max + margin,
    )


def plot_runtime_vs_samples_by_dimension(
    ax,
    df_runtime,
    method_label,
    panel_label,
):
    """
    x-axis:
        number of forecast samples S

    one line per:
        free-level dimension n_b
    """
    dimensions = sorted(
        df_runtime["dimension"].unique()
    )

    sample_sizes = sorted(
        df_runtime["n_samples"].unique()
    )

    for dimension in dimensions:
        subset = (
            df_runtime[
                df_runtime["dimension"] == dimension
            ]
            .sort_values(
                "n_samples"
            )
        )

        ax.plot(
            subset["n_samples"],
            subset["mean_runtime_sec"],
            marker="o",
            linewidth=2,
            label=f"$n_b = {dimension}$",
        )

    ax.set_xlabel(
        "Number of samples"
    )

    ax.set_ylabel(
        "Runtime (s)"
    )

    ax.set_title(
        f"{panel_label} {method_label}: runtime vs sample size"
    )

    ax.set_xticks(
        sample_sizes
    )

    ax.set_xticklabels(
        [
            f"{value:,}"
            for value in sample_sizes
        ],
        rotation=45,
        ha="right",
    )

    add_linear_x_margin(
        ax,
        sample_sizes,
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend(
        title="Free-level dimension",
        fontsize=8,
        title_fontsize=9,
        ncol=2,
    )


def plot_runtime_vs_dimension_by_samples(
    ax,
    df_runtime,
    method_label,
    panel_label,
):
    """
    x-axis:
        free-level dimension n_b

    one line per:
        number of forecast samples S
    """
    dimensions = sorted(
        df_runtime["dimension"].unique()
    )

    sample_sizes = sorted(
        df_runtime["n_samples"].unique()
    )

    for n_samples in sample_sizes:
        subset = (
            df_runtime[
                df_runtime["n_samples"] == n_samples
            ]
            .sort_values(
                "dimension"
            )
        )

        ax.plot(
            subset["dimension"],
            subset["mean_runtime_sec"],
            marker="o",
            linewidth=2,
            label=f"$N = {n_samples:,}$",
        )

    ax.set_xlabel(
        "Free-level dimension"
    )

    ax.set_ylabel(
        "Runtime (s)"
    )

    ax.set_title(
        f"{panel_label} {method_label}: runtime vs free-level dimension"
    )

    ax.set_xticks(
        dimensions
    )

    ax.set_xticklabels(
        dimensions
    )

    add_linear_x_margin(
        ax,
        dimensions,
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend(
        title="Number of samples",
        fontsize=8,
        title_fontsize=9,
        ncol=2,
    )


# ============================================================
# COMBINED 2x2 FIGURE
# ============================================================

def plot_runtime_scaling_2x2(
    df_ukf,
    df_projection,
):
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13, 9),
    )

    plot_runtime_vs_samples_by_dimension(
        ax=axes[0, 0],
        df_runtime=df_ukf,
        method_label="UKF",
        panel_label="(a)",
    )

    plot_runtime_vs_dimension_by_samples(
        ax=axes[0, 1],
        df_runtime=df_ukf,
        method_label="UKF",
        panel_label="(b)",
    )

    plot_runtime_vs_samples_by_dimension(
        ax=axes[1, 0],
        df_runtime=df_projection,
        method_label="Projection",
        panel_label="(c)",
    )

    plot_runtime_vs_dimension_by_samples(
        ax=axes[1, 1],
        df_runtime=df_projection,
        method_label="Projection",
        panel_label="(d)",
    )

    fig.suptitle(
        "Runtime scaling of UKF-based and projection-based reconciliation",
        fontsize=15,
        y=0.995,
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.97,
        ]
    )

    fig.savefig(
        COMBINED_PLOT_FILE,
        dpi=300,
        bbox_inches="tight",
    )

    print(
        f"Saved combined 2x2 plot:\n"
        f"  {COMBINED_PLOT_FILE}"
    )

    if SHOW_PLOTS:
        plt.show()

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print(
        "Loading timing results..."
    )

    df = load_results()

    print()
    print(
        "Aggregating UKF runtimes across surfaces..."
    )

    df_ukf = aggregate_surfaces(
        df=df,
        method=UKF_METHOD,
    )

    print()
    print(
        "Aggregated UKF runtimes:"
    )

    print(
        df_ukf.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print()
    print(
        "Aggregating projection runtimes across surfaces..."
    )

    df_projection = aggregate_surfaces(
        df=df,
        method=PROJECTION_METHOD,
    )

    print()
    print(
        "Aggregated projection runtimes:"
    )

    print(
        df_projection.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print()
    print(
        "Creating combined 2x2 runtime scaling figure with linear x-axes..."
    )

    plot_runtime_scaling_2x2(
        df_ukf=df_ukf,
        df_projection=df_projection,
    )

    print()
    print(
        "All plots completed."
    )

    print()
    print(
        f"Figure saved in:\n"
        f"  {OUTPUT_FOLDER}"
    )

    print()
    print(
        "Saved file:"
    )

    print(
        f"  {COMBINED_PLOT_FILE}"
    )


if __name__ == "__main__":
    main()