import gc
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bayesreconpy.shrink_cov import (
    _schafer_strimmer_cov as schafer_strimmer_cov,
)

from reconc.reconc_nl_ols import reconc_nl_ols
from reconc.reconc_nl_ukf import reconc_nl_ukf
from reconcile import (
    f_surface,
    f_surface_jax,
    _to_precision,
)


# ============================================================
# CONFIG
# ============================================================

FC_FOLDER = Path("../forecasts")

OUT_FOLDER = Path("../results/complexity")
OUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)

SURFACES = [
    "paraboloid",
    "saddle",
    "ripples",
]

DIMENSIONS = [
    2,
    10,
    20,
    50,
    100,
    200
]

SAMPLE_SIZES = [
    500,
    1000,
    2000,
    5000,
    10000,
    20000,
]

# One forecast time step only.
T_IDX = 0

# Number of timed repetitions per case.
N_REPEATS = 3

# Warm-up repetitions are excluded from timing.
N_WARMUPS = 1

N_ITER_OLS = 20
SEED = 42


RAW_OUT = OUT_FOLDER / "timings_raw.csv"

SUMMARY_OUT = (
    OUT_FOLDER
    / "timings_summary.csv"
)

SAMPLE_SCALING_OUT = (
    OUT_FOLDER
    / "timings_scaling_samples.csv"
)

DIMENSION_SCALING_OUT = (
    OUT_FOLDER
    / "timings_scaling_dimensions.csv"
)


# ============================================================
# HELPERS
# ============================================================

def sync_any(x):
    """
    Block until all JAX computations contained in x
    are complete.
    """
    if hasattr(
        x,
        "block_until_ready",
    ):
        x.block_until_ready()

    elif isinstance(
        x,
        dict,
    ):
        for value in x.values():
            sync_any(value)

    elif isinstance(
        x,
        (list, tuple),
    ):
        for value in x:
            sync_any(value)

    return x


def timed_call(
    fn,
    *args,
    **kwargs,
):
    """
    Time one complete function call.

    Garbage collection is performed before timing,
    not during the timed interval.
    """
    gc.collect()

    t0 = time.perf_counter()

    out = fn(
        *args,
        **kwargs,
    )

    sync_any(
        out
    )

    elapsed = (
        time.perf_counter()
        - t0
    )

    return (
        out,
        elapsed,
    )


# ============================================================
# DATA LOADING
# ============================================================

def load_case(
    surface: str,
    dimension: int,
    n_samples: int,
    fc_folder: Path,
):
    """
    Load one benchmark case.

    Expected forecast shape:

        (d + 1, T, S)

    Expected residual shape:

        (d + 1, Nres)
    """
    base_path = (
        fc_folder
        / (
            f"base_fc_{surface}_"
            f"d{dimension}_"
            f"{n_samples}.pkl"
        )
    )

    res_path = (
        fc_folder
        / (
            f"residuals_{surface}_"
            f"d{dimension}_"
            f"{n_samples}.pkl"
        )
    )

    if not base_path.exists():
        raise FileNotFoundError(
            f"Missing forecast file:\n"
            f"{base_path}"
        )

    if not res_path.exists():
        raise FileNotFoundError(
            f"Missing residual file:\n"
            f"{res_path}"
        )

    base_fc = np.asarray(
        pd.read_pickle(
            base_path
        ),
        dtype=np.float64,
    )

    tr_res = np.asarray(
        pd.read_pickle(
            res_path
        ),
        dtype=np.float64,
    )

    expected_variables = (
        dimension
        + 1
    )

    if base_fc.ndim != 3:
        raise ValueError(
            "base_fc must have shape "
            "(d + 1, T, S), "
            f"got {base_fc.shape}"
        )

    if tr_res.ndim != 2:
        raise ValueError(
            "tr_res must have shape "
            "(d + 1, Nres), "
            f"got {tr_res.shape}"
        )

    if (
        base_fc.shape[0]
        != expected_variables
    ):
        raise ValueError(
            f"Expected {expected_variables} "
            f"forecast variables for d={dimension}, "
            f"got {base_fc.shape[0]}"
        )

    if (
        tr_res.shape[0]
        != expected_variables
    ):
        raise ValueError(
            f"Expected {expected_variables} "
            f"residual variables for d={dimension}, "
            f"got {tr_res.shape[0]}"
        )

    T = base_fc.shape[1]
    S = base_fc.shape[2]

    if S != n_samples:
        raise ValueError(
            f"Requested n_samples={n_samples}, "
            f"but loaded forecast array has S={S}"
        )

    if T_IDX >= T:
        raise ValueError(
            f"T_IDX={T_IDX} is out of range "
            f"for T={T}"
        )

    return (
        base_fc,
        tr_res,
    )


# ============================================================
# PROJECTION CONSTRAINT
# ============================================================

def make_projection_constraint(
    surface: str,
    dimension: int,
):
    """
    Create one fixed-dimensional nonlinear constraint.

    The constraint is:

        U - f(B1, ..., Bd) = 0

    The explicit access to z[dimension] is required so
    that JNLR can infer the correct input dimension.
    """
    def f_ols(z):
        u = z[0]

        # Force JNLR to detect input size d + 1.
        _ = z[dimension]

        B = z[
            1:dimension + 1
        ]

        return jnp.array([
            u
            - f_surface_jax(
                surface=surface,
                B=B,
                axis=0,
            )
        ])

    return f_ols


# ============================================================
# FULL-PRECISION PROJECTION
# ============================================================

def run_full_once(
    base_fc,
    tr_res,
    f_ols,
    t_idx=0,
    n_iter=20,
    seed=42,
):
    """
    Run full-precision nonlinear projection for one
    forecast time step.

    The complete one-step operation is timed, including:

        residual covariance estimation
        precision matrix construction
        reconciliation
    """
    W = schafer_strimmer_cov(
        tr_res.T
    )["shrink_cov"]

    P = _to_precision(
        W
    )

    # Shape:
    #
    #     (S, d + 1)
    #
    Z = base_fc[
        :,
        t_idx,
        :,
    ].T

    return reconc_nl_ols(
        Z,
        f_ols,
        n_iter=n_iter,
        seed=seed,
        W=P,
    )


# ============================================================
# UKF
# ============================================================

def make_ukf_surface_function(
    surface: str,
):
    """
    Create the nonlinear mapping used by the UKF.

    Input:

        (d,)

    Output:

        scalar
    """
    def f_ukf_vec(b):
        return f_surface(
            surface=surface,
            B=b,
            axis=0,
        )

    return f_ukf_vec


def run_ukf_once(
    base_fc,
    tr_res,
    surface,
    dimension,
    f_ukf_vec,
    t_idx=0,
    seed=42,
):
    """
    Run UKF reconciliation for one forecast time step.

    The complete one-step operation is timed, including:

        construction of UKF inputs
        covariance estimation inside reconc_nl_ukf
        unscented transformation
        posterior sampling
    """
    S = base_fc.shape[2]

    # Shape:
    #
    #     (d, S)
    #
    bot_base = base_fc[
        1:,
        t_idx,
        :,
    ]

    # Shape:
    #
    #     (d, Nres)
    #
    bot_res = tr_res[
        1:,
        :,
    ]

    # Upper point forecast.
    u_obs = np.mean(
        base_fc[
            0,
            t_idx,
            :,
        ]
    ).reshape(
        1,
    )

    # Upper residual covariance.
    #
    # Shape:
    #
    #     (1, 1)
    #
    R = np.atleast_2d(
        schafer_strimmer_cov(
            tr_res.T
        )["shrink_cov"][0, 0]
    )

    bot_list = []

    for bottom_index in range(
        dimension
    ):
        bot_list.append(
            {
                "samples": bot_base[
                    bottom_index,
                    :
                ],

                "residuals": bot_res[
                    bottom_index,
                    :
                ],
            }
        )

    return reconc_nl_ukf(
        bottom_base_forecasts=bot_list,

        in_type=[
            "samples"
        ] * dimension,

        distr=[
            "gaussian"
        ] * dimension,

        f=f_ukf_vec,

        upper_base_forecasts=u_obs,

        R=R,

        num_samples=S,

        seed=seed,
    )


# ============================================================
# EMPIRICAL SCALING EXPONENT
# ============================================================

def estimate_scaling_exponent(
    x,
    y,
):
    """
    Estimate the empirical exponent in:

        time ~ x^exponent

    by fitting:

        log(time) = intercept + exponent * log(x)

    Returns
    -------
    exponent : float

    r_squared : float
    """
    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x > 0)
        & (y > 0)
    )

    x = x[valid]
    y = y[valid]

    if len(x) < 2:
        return (
            np.nan,
            np.nan,
        )

    log_x = np.log(
        x
    )

    log_y = np.log(
        y
    )

    slope, intercept = np.polyfit(
        log_x,
        log_y,
        deg=1,
    )

    fitted = (
        intercept
        + slope * log_x
    )

    ss_res = np.sum(
        (
            log_y
            - fitted
        ) ** 2
    )

    ss_tot = np.sum(
        (
            log_y
            - np.mean(log_y)
        ) ** 2
    )

    if ss_tot > 0:
        r_squared = (
            1.0
            - ss_res / ss_tot
        )
    else:
        r_squared = np.nan

    return (
        float(slope),
        float(r_squared),
    )


# ============================================================
# SAMPLE-SIZE SCALING ANALYSIS
# ============================================================

def compute_sample_scaling(
    df_summary,
):
    """
    Estimate runtime scaling with sample size:

        time ~ S^alpha

    separately for every:

        surface
        method
        dimension
    """
    rows = []

    grouped = df_summary.groupby(
        [
            "surface",
            "method",
            "dimension",
        ]
    )

    for (
        surface,
        method,
        dimension,
    ), group in grouped:

        group = group.sort_values(
            "n_samples"
        )

        exponent, r_squared = (
            estimate_scaling_exponent(
                group[
                    "n_samples"
                ].values,

                group[
                    "mean_time_sec"
                ].values,
            )
        )

        rows.append(
            {
                "surface": surface,
                "method": method,
                "dimension": dimension,
                "sample_size_exponent": exponent,
                "r_squared": r_squared,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# DIMENSION SCALING ANALYSIS
# ============================================================

def compute_dimension_scaling(
    df_summary,
):
    """
    Estimate runtime scaling with bottom dimension:

        time ~ d^beta

    separately for every:

        surface
        method
        sample size
    """
    rows = []

    grouped = df_summary.groupby(
        [
            "surface",
            "method",
            "n_samples",
        ]
    )

    for (
        surface,
        method,
        n_samples,
    ), group in grouped:

        group = group.sort_values(
            "dimension"
        )

        exponent, r_squared = (
            estimate_scaling_exponent(
                group[
                    "dimension"
                ].values,

                group[
                    "mean_time_sec"
                ].values,
            )
        )

        rows.append(
            {
                "surface": surface,
                "method": method,
                "n_samples": n_samples,
                "dimension_exponent": exponent,
                "r_squared": r_squared,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# PLOTTING
# ============================================================

def plot_time_vs_samples(
    df_summary,
):
    """
    Plot runtime versus sample size for each:

        surface
        dimension
    """
    for surface in SURFACES:
        for dimension in DIMENSIONS:

            subset = df_summary[
                (
                    df_summary[
                        "surface"
                    ]
                    == surface
                )
                &
                (
                    df_summary[
                        "dimension"
                    ]
                    == dimension
                )
            ]

            if subset.empty:
                continue

            plt.figure(
                figsize=(8, 5)
            )

            for method in [
                "full",
                "ukf",
            ]:
                tmp = subset[
                    subset[
                        "method"
                    ]
                    == method
                ].sort_values(
                    "n_samples"
                )

                plt.plot(
                    tmp[
                        "n_samples"
                    ],
                    tmp[
                        "mean_time_sec"
                    ],
                    marker="o",
                    label=method,
                )

            plt.xlabel(
                "Number of forecast samples"
            )

            plt.ylabel(
                "Mean computation time (s)"
            )

            plt.title(
                f"Runtime vs sample size\n"
                f"{surface}, d={dimension}, "
                f"one time step"
            )

            plt.xticks(
                SAMPLE_SIZES,
                [
                    f"{x:,}"
                    for x in SAMPLE_SIZES
                ],
            )

            plt.xscale(
                "log"
            )

            plt.yscale(
                "log"
            )

            plt.grid(
                True,
                which="both",
                alpha=0.3,
            )

            plt.legend()

            plt.tight_layout()

            output_path = (
                OUT_FOLDER
                / (
                    f"timings_vs_samples_"
                    f"{surface}_"
                    f"d{dimension}.png"
                )
            )

            plt.savefig(
                output_path,
                dpi=300,
            )

            plt.close()


def plot_time_vs_dimension(
    df_summary,
):
    """
    Plot runtime versus state dimension for each:

        surface
        sample size
    """
    for surface in SURFACES:
        for n_samples in SAMPLE_SIZES:

            subset = df_summary[
                (
                    df_summary[
                        "surface"
                    ]
                    == surface
                )
                &
                (
                    df_summary[
                        "n_samples"
                    ]
                    == n_samples
                )
            ]

            if subset.empty:
                continue

            plt.figure(
                figsize=(8, 5)
            )

            for method in [
                "full",
                "ukf",
            ]:
                tmp = subset[
                    subset[
                        "method"
                    ]
                    == method
                ].sort_values(
                    "dimension"
                )

                plt.plot(
                    tmp[
                        "dimension"
                    ],
                    tmp[
                        "mean_time_sec"
                    ],
                    marker="o",
                    label=method,
                )

            plt.xlabel(
                "Number of bottom-level variables"
            )

            plt.ylabel(
                "Mean computation time (s)"
            )

            plt.title(
                f"Runtime vs dimension\n"
                f"{surface}, S={n_samples:,}, "
                f"one time step"
            )

            plt.xticks(
                DIMENSIONS
            )

            plt.xscale(
                "log"
            )

            plt.yscale(
                "log"
            )

            plt.grid(
                True,
                which="both",
                alpha=0.3,
            )

            plt.legend()

            plt.tight_layout()

            output_path = (
                OUT_FOLDER
                / (
                    f"timings_vs_dimension_"
                    f"{surface}_"
                    f"S{n_samples}.png"
                )
            )

            plt.savefig(
                output_path,
                dpi=300,
            )

            plt.close()


# ============================================================
# MAIN BENCHMARK LOOP
# ============================================================

def main():
    rows = []

    total_cases = (
        len(SURFACES)
        * len(DIMENSIONS)
        * len(SAMPLE_SIZES)
    )

    case_counter = 0

    for surface in SURFACES:
        print()
        print(
            "=" * 70
        )
        print(
            f"Benchmarking surface: {surface}"
        )
        print(
            "=" * 70
        )

        for dimension in DIMENSIONS:
            for n_samples in SAMPLE_SIZES:

                case_counter += 1

                print()
                print(
                    "-" * 70
                )

                print(
                    f"Case {case_counter}/{total_cases}: "
                    f"surface={surface}, "
                    f"d={dimension}, "
                    f"S={n_samples}"
                )

                print(
                    "-" * 70
                )

                # ====================================================
                # LOAD ONE CASE
                # ====================================================

                (
                    base_fc,
                    tr_res,
                ) = load_case(
                    surface=surface,
                    dimension=dimension,
                    n_samples=n_samples,
                    fc_folder=FC_FOLDER,
                )

                T = base_fc.shape[1]
                S = base_fc.shape[2]

                print(
                    f"Loaded base_fc shape: "
                    f"{base_fc.shape}"
                )

                print(
                    f"Loaded residual shape: "
                    f"{tr_res.shape}"
                )

                print(
                    f"Benchmark time step: "
                    f"t={T_IDX}"
                )

                # ====================================================
                # CREATE FIXED FUNCTIONS ONCE FOR THIS CASE
                # ====================================================

                f_ols = make_projection_constraint(
                    surface=surface,
                    dimension=dimension,
                )

                f_ukf_vec = (
                    make_ukf_surface_function(
                        surface=surface,
                    )
                )

                # ====================================================
                # WARM-UP
                #
                # Especially important for JAX/JNLR.
                #
                # Warm-up timings are NOT recorded.
                # ====================================================

                print(
                    f"Warm-up runs: {N_WARMUPS}"
                )

                for warmup in range(
                    N_WARMUPS
                ):
                    print(
                        f"  warm-up "
                        f"{warmup + 1}/{N_WARMUPS}: "
                        f"full"
                    )

                    out_full = run_full_once(
                        base_fc=base_fc,
                        tr_res=tr_res,
                        f_ols=f_ols,
                        t_idx=T_IDX,
                        n_iter=N_ITER_OLS,
                        seed=SEED,
                    )

                    sync_any(
                        out_full
                    )

                    del out_full

                    print(
                        f"  warm-up "
                        f"{warmup + 1}/{N_WARMUPS}: "
                        f"ukf"
                    )

                    out_ukf = run_ukf_once(
                        base_fc=base_fc,
                        tr_res=tr_res,
                        surface=surface,
                        dimension=dimension,
                        f_ukf_vec=f_ukf_vec,
                        t_idx=T_IDX,
                        seed=SEED,
                    )

                    sync_any(
                        out_ukf
                    )

                    del out_ukf

                    gc.collect()

                # ====================================================
                # TIMED REPEATS
                # ====================================================

                for rep in range(
                    1,
                    N_REPEATS + 1,
                ):
                    print(
                        f"Timed repeat "
                        f"{rep}/{N_REPEATS}"
                    )

                    # --------------------------------------------
                    # FULL PROJECTION
                    # --------------------------------------------

                    out_full, dt_full = timed_call(
                        run_full_once,

                        base_fc=base_fc,

                        tr_res=tr_res,

                        f_ols=f_ols,

                        t_idx=T_IDX,

                        n_iter=N_ITER_OLS,

                        seed=SEED,
                    )

                    rows.append(
                        {
                            "surface": surface,
                            "dimension": dimension,
                            "n_samples": n_samples,
                            "repeat": rep,
                            "method": "full",
                            "time_sec": dt_full,
                            "t_idx": T_IDX,
                            "T": T,
                            "S": S,
                        }
                    )

                    print(
                        f"  full: "
                        f"{dt_full:.6f} s"
                    )

                    del out_full

                    # --------------------------------------------
                    # UKF
                    # --------------------------------------------

                    out_ukf, dt_ukf = timed_call(
                        run_ukf_once,

                        base_fc=base_fc,

                        tr_res=tr_res,

                        surface=surface,

                        dimension=dimension,

                        f_ukf_vec=f_ukf_vec,

                        t_idx=T_IDX,

                        seed=SEED,
                    )

                    rows.append(
                        {
                            "surface": surface,
                            "dimension": dimension,
                            "n_samples": n_samples,
                            "repeat": rep,
                            "method": "ukf",
                            "time_sec": dt_ukf,
                            "t_idx": T_IDX,
                            "T": T,
                            "S": S,
                        }
                    )

                    print(
                        f"  ukf:  "
                        f"{dt_ukf:.6f} s"
                    )

                    del out_ukf

                    gc.collect()

                # ====================================================
                # SAVE RAW CHECKPOINT
                #
                # Results are preserved after every case.
                # ====================================================

                df_checkpoint = pd.DataFrame(
                    rows
                )

                df_checkpoint.to_csv(
                    RAW_OUT,
                    index=False,
                )

                print(
                    f"Raw timing checkpoint saved to: "
                    f"{RAW_OUT}"
                )

                # ====================================================
                # RELEASE CASE MEMORY
                # ====================================================

                del base_fc
                del tr_res
                del f_ols
                del f_ukf_vec

                # Clear accumulated JAX compilation caches before
                # moving to a new shape/dimension case.
                jax.clear_caches()

                gc.collect()

    # ============================================================
    # RAW RESULTS
    # ============================================================

    df_raw = pd.DataFrame(
        rows
    )

    df_raw.to_csv(
        RAW_OUT,
        index=False,
    )

    # ============================================================
    # SUMMARY RESULTS
    # ============================================================

    df_summary = (
        df_raw
        .groupby(
            [
                "surface",
                "method",
                "dimension",
                "n_samples",
            ],
            as_index=False,
        )
        .agg(
            mean_time_sec=(
                "time_sec",
                "mean",
            ),

            std_time_sec=(
                "time_sec",
                "std",
            ),

            median_time_sec=(
                "time_sec",
                "median",
            ),

            min_time_sec=(
                "time_sec",
                "min",
            ),

            max_time_sec=(
                "time_sec",
                "max",
            ),
        )
        .sort_values(
            [
                "surface",
                "method",
                "dimension",
                "n_samples",
            ]
        )
    )

    df_summary.to_csv(
        SUMMARY_OUT,
        index=False,
    )

    # ============================================================
    # EMPIRICAL SAMPLE-SIZE COMPLEXITY
    # ============================================================

    df_sample_scaling = (
        compute_sample_scaling(
            df_summary
        )
    )

    df_sample_scaling.to_csv(
        SAMPLE_SCALING_OUT,
        index=False,
    )

    # ============================================================
    # EMPIRICAL DIMENSION COMPLEXITY
    # ============================================================

    df_dimension_scaling = (
        compute_dimension_scaling(
            df_summary
        )
    )

    df_dimension_scaling.to_csv(
        DIMENSION_SCALING_OUT,
        index=False,
    )

    # ============================================================
    # PRINT RESULTS
    # ============================================================

    print()
    print(
        "=" * 100
    )
    print(
        "TIMING SUMMARY"
    )
    print(
        "=" * 100
    )
    print()

    print(
        df_summary.to_string(
            index=False
        )
    )

    # ------------------------------------------------------------
    # Mean runtime by sample size
    # ------------------------------------------------------------

    print()
    print(
        "=" * 100
    )
    print(
        "MEAN RUNTIME BY SAMPLE SIZE"
    )
    print(
        "=" * 100
    )
    print()

    print(
        df_summary.pivot_table(
            index=[
                "surface",
                "method",
                "dimension",
            ],

            columns="n_samples",

            values="mean_time_sec",
        ).to_string(
            float_format=lambda x: (
                f"{x:.6f}"
            )
        )
    )

    # ------------------------------------------------------------
    # Mean runtime by dimension
    # ------------------------------------------------------------

    print()
    print(
        "=" * 100
    )
    print(
        "MEAN RUNTIME BY DIMENSION"
    )
    print(
        "=" * 100
    )
    print()

    print(
        df_summary.pivot_table(
            index=[
                "surface",
                "method",
                "n_samples",
            ],

            columns="dimension",

            values="mean_time_sec",
        ).to_string(
            float_format=lambda x: (
                f"{x:.6f}"
            )
        )
    )

    # ------------------------------------------------------------
    # Sample-size scaling exponents
    # ------------------------------------------------------------

    print()
    print(
        "=" * 100
    )
    print(
        "EMPIRICAL SAMPLE-SIZE SCALING"
    )
    print(
        "time ~ n_samples ^ exponent"
    )
    print(
        "=" * 100
    )
    print()

    print(
        df_sample_scaling.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.4f}"
            ),
        )
    )

    # ------------------------------------------------------------
    # Dimension scaling exponents
    # ------------------------------------------------------------

    print()
    print(
        "=" * 100
    )
    print(
        "EMPIRICAL DIMENSION SCALING"
    )
    print(
        "time ~ dimension ^ exponent"
    )
    print(
        "=" * 100
    )
    print()

    print(
        df_dimension_scaling.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.4f}"
            ),
        )
    )

    # ============================================================
    # PLOTS
    # ============================================================

    plot_time_vs_samples(
        df_summary
    )

    plot_time_vs_dimension(
        df_summary
    )

    # ============================================================
    # FINAL OUTPUT LOCATIONS
    # ============================================================

    print()
    print(
        "=" * 100
    )
    print(
        "OUTPUT FILES"
    )
    print(
        "=" * 100
    )

    print(
        f"Raw timings:\n"
        f"  {RAW_OUT}"
    )

    print(
        f"Summary timings:\n"
        f"  {SUMMARY_OUT}"
    )

    print(
        f"Sample-size scaling:\n"
        f"  {SAMPLE_SCALING_OUT}"
    )

    print(
        f"Dimension scaling:\n"
        f"  {DIMENSION_SCALING_OUT}"
    )



if __name__ == "__main__":
    main()