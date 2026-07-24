from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

RESULTS_FOLDER = Path(
    "../results/complexity"
)

SUMMARY_FILE = (
    RESULTS_FOLDER
    / "timings_summary.csv"
)

PLOT_FOLDER = (
    RESULTS_FOLDER
    / "plots_summary"
)

PLOT_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)

SHOW_PLOTS = False


# ============================================================
# HELPERS
# ============================================================

def geometric_mean(values):
    """
    Geometric mean of strictly positive finite values.
    """
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


def format_runtime(value):
    """
    Compact runtime label for heatmap cells.
    """
    if value < 0.01:
        return f"{value:.4f}"

    if value < 1:
        return f"{value:.3f}"

    return f"{value:.2f}"


def truncated_colormap(
    cmap_name,
    min_value=0.05,
    max_value=0.75,
    n_colors=256,
):
    """
    Create a lighter version of a Matplotlib colormap.
    """
    base_cmap = plt.get_cmap(
        cmap_name
    )

    colors = base_cmap(
        np.linspace(
            min_value,
            max_value,
            n_colors,
        )
    )

    return LinearSegmentedColormap.from_list(
        f"{cmap_name}_truncated",
        colors,
    )


RUNTIME_CMAP = truncated_colormap(
    "YlGnBu",
    min_value=0.05,
    max_value=0.65,
)

SPEEDUP_CMAP = truncated_colormap(
    "YlOrBr",
    min_value=0.05,
    max_value=0.70,
)


def readable_text_color(
    image,
    value,
):
    """
    Return black or white depending on the brightness
    of the heatmap cell.
    """
    rgba = image.cmap(
        image.norm(
            value
        )
    )

    r, g, b, _ = rgba

    luminance = (
        0.299 * r
        + 0.587 * g
        + 0.114 * b
    )

    if luminance < 0.45:
        return "white"

    return "black"


def annotate_heatmap(
    ax,
    matrix,
    formatter,
    image=None,
):
    """
    Add text labels to all heatmap cells.
    """
    for row in range(
        matrix.shape[0]
    ):
        for col in range(
            matrix.shape[1]
        ):
            value = matrix[
                row,
                col,
            ]

            if not np.isfinite(
                value
            ):
                continue

            if image is not None:
                text_color = readable_text_color(
                    image=image,
                    value=value,
                )
            else:
                text_color = "black"

            ax.text(
                col,
                row,
                formatter(value),
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
                fontweight="bold",
            )


def maybe_show():
    if SHOW_PLOTS:
        plt.show()

    plt.close()


# ============================================================
# LOAD RESULTS
# ============================================================

def load_results():
    if not SUMMARY_FILE.exists():
        raise FileNotFoundError(
            f"Missing file:\n"
            f"{SUMMARY_FILE}"
        )

    df_summary = pd.read_csv(
        SUMMARY_FILE
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
        - set(
            df_summary.columns
        )
    )

    if missing:
        raise ValueError(
            "Missing columns in timings_summary.csv: "
            f"{sorted(missing)}"
        )

    return df_summary


# ============================================================
# AGGREGATE ACROSS SURFACES
# ============================================================

def aggregate_runtime_across_surfaces(
    df_summary,
):
    """
    Compute geometric mean runtime across surfaces for each:

        method
        dimension
        sample size
    """
    df_runtime = (
        df_summary
        .groupby(
            [
                "method",
                "dimension",
                "n_samples",
            ],
            as_index=False,
        )
        .agg(
            geom_mean_time_sec=(
                "mean_time_sec",
                geometric_mean,
            )
        )
    )

    return df_runtime


def compute_speedup(
    df_summary,
):
    """
    Compute projection runtime / UKF runtime.

    Speedup is first computed separately for each surface,
    dimension, and sample size, then aggregated across surfaces
    using the geometric mean.
    """
    df_wide = (
        df_summary
        .pivot_table(
            index=[
                "surface",
                "dimension",
                "n_samples",
            ],
            columns="method",
            values="mean_time_sec",
        )
        .reset_index()
    )

    required_methods = {
        "full",
        "ukf",
    }

    if not required_methods.issubset(
        df_wide.columns
    ):
        raise ValueError(
            "timings_summary.csv must contain "
            "both 'full' and 'ukf' methods."
        )

    df_wide[
        "speedup"
    ] = (
        df_wide[
            "full"
        ]
        / df_wide[
            "ukf"
        ]
    )

    df_speedup = (
        df_wide
        .groupby(
            [
                "dimension",
                "n_samples",
            ],
            as_index=False,
        )
        .agg(
            geom_mean_speedup=(
                "speedup",
                geometric_mean,
            )
        )
    )

    return df_speedup


# ============================================================
# HEATMAP FIGURE
# ============================================================

def plot_runtime_landscape(
    df_runtime,
    df_speedup,
):
    """
    Three-panel heatmap:

        (a) projection runtime
        (b) UKF runtime
        (c) projection / UKF speedup

    Values are geometric means across surfaces.
    """
    dimensions = sorted(
        df_runtime[
            "dimension"
        ].unique()
    )

    sample_sizes = sorted(
        df_runtime[
            "n_samples"
        ].unique()
    )

    def runtime_matrix(method):
        subset = df_runtime[
            df_runtime[
                "method"
            ]
            == method
        ]

        matrix = (
            subset
            .pivot(
                index="dimension",
                columns="n_samples",
                values="geom_mean_time_sec",
            )
            .reindex(
                index=dimensions,
                columns=sample_sizes,
            )
            .values
        )

        return matrix

    full_matrix = runtime_matrix(
        "full"
    )

    ukf_matrix = runtime_matrix(
        "ukf"
    )

    speedup_matrix = (
        df_speedup
        .pivot(
            index="dimension",
            columns="n_samples",
            values="geom_mean_speedup",
        )
        .reindex(
            index=dimensions,
            columns=sample_sizes,
        )
        .values
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5.5),
    )

    # --------------------------------------------------------
    # Projection runtime
    # --------------------------------------------------------

    image_full = axes[0].imshow(
        full_matrix,
        aspect="auto",
        origin="lower",
        cmap=RUNTIME_CMAP,
        norm=LogNorm(
            vmin=np.nanmin(
                full_matrix
            ),
            vmax=np.nanmax(
                full_matrix
            ),
        ),
    )

    axes[0].set_title(
        "Full projection"
    )

    annotate_heatmap(
        axes[0],
        full_matrix,
        format_runtime,
        image=image_full,
    )

    colorbar_full = fig.colorbar(
        image_full,
        ax=axes[0],
    )

    colorbar_full.set_label(
        "Runtime (s)"
    )

    # --------------------------------------------------------
    # UKF runtime
    # --------------------------------------------------------

    image_ukf = axes[1].imshow(
        ukf_matrix,
        aspect="auto",
        origin="lower",
        cmap=RUNTIME_CMAP,
        norm=LogNorm(
            vmin=np.nanmin(
                ukf_matrix
            ),
            vmax=np.nanmax(
                ukf_matrix
            ),
        ),
    )

    axes[1].set_title(
        "UKF"
    )

    annotate_heatmap(
        axes[1],
        ukf_matrix,
        format_runtime,
        image=image_ukf,
    )

    colorbar_ukf = fig.colorbar(
        image_ukf,
        ax=axes[1],
    )

    colorbar_ukf.set_label(
        "Runtime (s)"
    )

    # --------------------------------------------------------
    # Speedup
    # --------------------------------------------------------

    image_speedup = axes[2].imshow(
        speedup_matrix,
        aspect="auto",
        origin="lower",
        cmap=SPEEDUP_CMAP,
        norm=LogNorm(
            vmin=np.nanmin(
                speedup_matrix
            ),
            vmax=np.nanmax(
                speedup_matrix
            ),
        ),
    )

    axes[2].set_title(
        "Projection / UKF"
    )

    annotate_heatmap(
        axes[2],
        speedup_matrix,
        lambda value: (
            f"{value:.1f}×"
        ),
        image=image_speedup,
    )

    colorbar_speedup = fig.colorbar(
        image_speedup,
        ax=axes[2],
    )

    colorbar_speedup.set_label(
        "Speedup factor"
    )

    # --------------------------------------------------------
    # Common axis formatting
    # --------------------------------------------------------

    for ax in axes:
        ax.set_xticks(
            range(
                len(sample_sizes)
            )
        )

        ax.set_xticklabels(
            [
                f"{value:,}"
                for value in sample_sizes
            ],
            rotation=45,
            ha="right",
        )

        ax.set_yticks(
            range(
                len(dimensions)
            )
        )

        ax.set_yticklabels(
            dimensions
        )

        ax.set_xlabel(
            "Number of forecast samples"
        )

        ax.set_ylabel(
            "Free-level dimension"
        )

    fig.tight_layout()

    output_path = (
        PLOT_FOLDER
        / "complexity_runtime_landscape.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    print(
        f"Saved: {output_path}"
    )

    maybe_show()


# ============================================================
# MAIN
# ============================================================

def main():
    print(
        "Loading completed benchmark results..."
    )

    df_summary = load_results()

    print(
        "Computing aggregated runtime and speedup..."
    )

    df_runtime = aggregate_runtime_across_surfaces(
        df_summary
    )

    df_speedup = compute_speedup(
        df_summary
    )

    print()
    print(
        "Creating runtime landscape heatmap..."
    )

    plot_runtime_landscape(
        df_runtime=df_runtime,
        df_speedup=df_speedup,
    )

    print()
    print(
        "Heatmap completed."
    )

    print(
        f"Figure saved in:\n"
        f"  {PLOT_FOLDER}"
    )


if __name__ == "__main__":
    main()