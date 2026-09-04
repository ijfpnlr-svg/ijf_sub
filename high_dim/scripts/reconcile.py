import warnings
warnings.filterwarnings("ignore")

import os
from multiprocessing import get_context
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import jax.numpy as jnp
from bayesreconpy.shrink_cov import (
    _schafer_strimmer_cov as schafer_strimmer_cov,
)

from reconc.reconc_nl_ols import reconc_nl_ols
from reconc.reconc_nl_ukf import reconc_nl_ukf
from simulation.scripts.score_functions import compute_es
from simulation.scripts.score_functions import compute_crps_new


# ============================================================
# CONFIG
# ============================================================

FORECASTS_FOLDER = "../forecasts"
RESULTS_FOLDER = "../results"

SCORE_FILE = os.path.join(
    RESULTS_FOLDER,
    "reconciliation_scores.csv",
)

LOSS_FILE = os.path.join(
    RESULTS_FOLDER,
    "reconciliation_losses_by_time.csv",
)

CRPS_SERIES_LOSS_FILE = os.path.join(
    RESULTS_FOLDER,
    "reconciliation_crps_by_series_time.csv",
)

SIMULATION_SCORE_SUMMARY_FILE = os.path.join(
    RESULTS_FOLDER,
    "simulation_relative_scores_summary.csv",
)

SIMULATION_SCORE_LATEX_TABLE_FILE = os.path.join(
    RESULTS_FOLDER,
    "simulation_relative_scores_table.tex",
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

SURFACE_ORDER = [
    "paraboloid",
    "saddle",
    "ripples",
]

SURFACE_LABELS = {
    "paraboloid": "Paraboloid",
    "saddle": "Saddle",
    "ripples": "Ripples",
}

EPSILON = 1e-12


# ============================================================
# JAX synchronization utility
# ============================================================

def sync_any(x):
    if hasattr(x, "block_until_ready"):
        x.block_until_ready()

    elif isinstance(x, dict):
        for value in x.values():
            sync_any(value)

    elif isinstance(x, (list, tuple)):
        for value in x:
            sync_any(value)

    return x


# ============================================================
# SURFACE DEFINITIONS
# ============================================================

def f_surface(
    surface,
    B,
    axis=0,
):
    """
    High-dimensional nonlinear surface.

    B contains only the bottom-level variables.

    If axis=0:
        B has shape (d, ...)

    If axis=1:
        B has shape (..., d, ...)
    """

    B = np.asarray(
        B,
        dtype=float,
    )

    B = np.moveaxis(
        B,
        axis,
        0,
    )

    d = B.shape[0]

    if surface == "paraboloid":
        return np.sum(
            B ** 2,
            axis=0,
        )

    if surface == "saddle":
        if d % 2 != 0:
            raise ValueError(
                "Saddle surface requires an even number "
                f"of bottom-level dimensions, got d={d}"
            )

        half = d // 2

        return (
            np.sum(
                B[:half] ** 2,
                axis=0,
            )
            - np.sum(
                B[half:] ** 2,
                axis=0,
            )
        )

    if surface == "ripples":
        return (
            np.sum(
                np.sin(B[0::2]),
                axis=0,
            )
            + np.sum(
                np.cos(B[1::2]),
                axis=0,
            )
        )

    if surface == "linear":
        return np.sum(
            B,
            axis=0,
        )

    raise ValueError(
        f"Unknown surface {surface}"
    )


def f_surface_jax(
    surface,
    B,
    axis=0,
):
    B = jnp.asarray(
        B
    )

    B = jnp.moveaxis(
        B,
        axis,
        0,
    )

    d = B.shape[0]

    if surface == "paraboloid":
        return jnp.sum(
            B ** 2,
            axis=0,
        )

    if surface == "saddle":
        if d % 2 != 0:
            raise ValueError(
                "Saddle surface requires an even number "
                f"of bottom-level dimensions, got d={d}"
            )

        half = d // 2

        return (
            jnp.sum(
                B[:half] ** 2,
                axis=0,
            )
            - jnp.sum(
                B[half:] ** 2,
                axis=0,
            )
        )

    if surface == "ripples":
        return (
            jnp.sum(
                jnp.sin(B[0::2]),
                axis=0,
            )
            + jnp.sum(
                jnp.cos(B[1::2]),
                axis=0,
            )
        )

    if surface == "linear":
        return jnp.sum(
            B,
            axis=0,
        )

    raise ValueError(
        f"Unknown surface {surface}"
    )


# ============================================================
# PRECISION MATRIX UTILITY
# ============================================================

def _to_precision(
    cov,
    eps=1e-6,
):
    cov = 0.5 * (
        cov
        + cov.T
    )

    lam = (
        eps
        * np.trace(
            cov
        )
        / cov.shape[0]
    )

    return np.linalg.pinv(
        cov
        + lam
        * np.eye(
            cov.shape[0]
        )
    )


# ============================================================
# PARALLEL PROJECTION
# ============================================================

def _project_at_time_step_worker(task):
    (
        t,
        z_t,
        res_t,
        surface,
        dimension,
        n_iter,
        seed,
    ) = task

    def f_ols(z):
        u = z[0]

        B = z[
            1:dimension + 1
        ]

        return jnp.array(
            [
                u
                - f_surface_jax(
                    surface=surface,
                    B=B,
                    axis=0,
                )
            ]
        )

    W = schafer_strimmer_cov(
        res_t.T
    )["shrink_cov"]

    P = _to_precision(
        W
    )

    out_ols = reconc_nl_ols(
        z_t,
        f_ols,
        n_iter=n_iter,
        seed=seed,
    )

    sync_any(
        out_ols
    )

    out_full = reconc_nl_ols(
        z_t,
        f_ols,
        n_iter=n_iter,
        seed=seed,
        W=P,
    )

    sync_any(
        out_full
    )

    out_wls = reconc_nl_ols(
        z_t,
        f_ols,
        n_iter=n_iter,
        seed=seed,
        W=np.diag(
            np.diag(
                P
            )
        ),
    )

    sync_any(
        out_wls
    )

    return {
        "t": t,
        "ols": np.asarray(
            out_ols[
                "reconciled_samples"
            ]
        ).copy(),
        "full": np.asarray(
            out_full[
                "reconciled_samples"
            ]
        ).copy(),
        "wls": np.asarray(
            out_wls[
                "reconciled_samples"
            ]
        ).copy(),
    }


def run_projection_parallel(
    base_fc,
    tr_res,
    surface,
    dimension,
    n_iter=20,
    seed=42,
    max_workers=None,
):
    T = base_fc.shape[1]

    tasks = [
        (
            t,
            base_fc[
                :,
                t,
                :,
            ].T.copy(),
            tr_res.copy(),
            surface,
            dimension,
            n_iter,
            seed,
        )
        for t in range(T)
    ]

    ols_dict = {}
    full_dict = {}
    wls_dict = {}

    ctx = get_context(
        "spawn"
    )

    if max_workers is None:
        max_workers = min(
            os.cpu_count()
            or 1,
            T,
        )

    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=ctx,
    ) as executor:

        futures = [
            executor.submit(
                _project_at_time_step_worker,
                task,
            )
            for task in tasks
        ]

        for future in as_completed(
            futures
        ):
            result = future.result()

            t = result[
                "t"
            ]

            ols_dict[
                t
            ] = result[
                "ols"
            ]

            full_dict[
                t
            ] = result[
                "full"
            ]

            wls_dict[
                t
            ] = result[
                "wls"
            ]

            print(
                f"finished projection task t={t}",
                flush=True,
            )

    ols_fc = np.stack(
        [
            ols_dict[
                t
            ]
            for t in range(T)
        ],
        axis=1,
    )

    full_fc = np.stack(
        [
            full_dict[
                t
            ]
            for t in range(T)
        ],
        axis=1,
    )

    wls_fc = np.stack(
        [
            wls_dict[
                t
            ]
            for t in range(T)
        ],
        axis=1,
    )

    return (
        ols_fc,
        full_fc,
        wls_fc,
    )


# ============================================================
# PROBABILISTIC BOTTOM-UP
# ============================================================

def pbu(
    B,
    surface,
):
    """
    B shape = (d, T, S)

    Output shape = (d + 1, T, S), with variable order:

        U, B1, ..., Bd
    """

    B = np.asarray(
        B,
        dtype=float,
    )

    if B.ndim != 3:
        raise ValueError(
            "B must have shape (d, T, S), "
            f"got {B.shape}"
        )

    U = f_surface(
        surface=surface,
        B=B,
        axis=0,
    )

    U = U.reshape(
        1,
        *U.shape,
    )

    return np.concatenate(
        [
            U,
            B,
        ],
        axis=0,
    )


# ============================================================
# PER-TIME-STEP LOSS FUNCTIONS
# ============================================================

def _mean_pairwise_euclidean_distance(
    samples,
    chunk_size=256,
):
    samples = np.asarray(
        samples,
        dtype=float,
    )

    n_samples = samples.shape[0]

    total = 0.0
    count = 0

    for start in range(
        0,
        n_samples,
        chunk_size,
    ):
        end = min(
            start + chunk_size,
            n_samples,
        )

        block = samples[
            start:end,
            :,
        ]

        distances = np.linalg.norm(
            block[
                :,
                None,
                :,
            ]
            - samples[
                None,
                :,
                :,
            ],
            axis=2,
        )

        total += float(
            np.sum(
                distances
            )
        )

        count += (
            end
            - start
        ) * n_samples

    return total / count


def compute_energy_loss_by_time(
    observations,
    forecast_samples,
):
    observations = np.asarray(
        observations,
        dtype=float,
    )

    forecast_samples = np.asarray(
        forecast_samples,
        dtype=float,
    )

    n_variables, T, S = forecast_samples.shape

    if observations.shape != (
        n_variables,
        T,
    ):
        raise ValueError(
            "observations must have shape "
            f"{(n_variables, T)}, got {observations.shape}"
        )

    losses = np.zeros(
        T,
        dtype=float,
    )

    for t in range(T):
        y_t = observations[
            :,
            t,
        ]

        samples_t = forecast_samples[
            :,
            t,
            :,
        ].T

        term_1 = np.mean(
            np.linalg.norm(
                samples_t
                - y_t[
                    None,
                    :,
                ],
                axis=1,
            )
        )

        term_2 = _mean_pairwise_euclidean_distance(
            samples_t
        )

        losses[
            t
        ] = (
            term_1
            - 0.5
            * term_2
        )

    return losses


def _crps_ensemble_1d(
    y,
    samples,
):
    samples = np.asarray(
        samples,
        dtype=float,
    )

    samples = samples[
        np.isfinite(
            samples
        )
    ]

    n = len(
        samples
    )

    if n == 0:
        return np.nan

    mean_abs_error = np.mean(
        np.abs(
            samples
            - y
        )
    )

    sorted_samples = np.sort(
        samples
    )

    indices = np.arange(
        1,
        n + 1,
        dtype=float,
    )

    half_pairwise_abs_mean = (
        np.sum(
            (
                2.0
                * indices
                - n
                - 1.0
            )
            * sorted_samples
        )
        / (
            n ** 2
        )
    )

    return float(
        mean_abs_error
        - half_pairwise_abs_mean
    )


def compute_crps_loss_by_time(
    observations,
    forecast_samples,
):
    observations = np.asarray(
        observations,
        dtype=float,
    )

    forecast_samples = np.asarray(
        forecast_samples,
        dtype=float,
    )

    n_variables, T, S = forecast_samples.shape

    if observations.shape != (
        n_variables,
        T,
    ):
        raise ValueError(
            "observations must have shape "
            f"{(n_variables, T)}, got {observations.shape}"
        )

    losses = np.zeros(
        T,
        dtype=float,
    )

    for t in range(T):
        variable_losses = []

        for variable_index in range(
            n_variables
        ):
            variable_losses.append(
                _crps_ensemble_1d(
                    y=observations[
                        variable_index,
                        t,
                    ],
                    samples=forecast_samples[
                        variable_index,
                        t,
                        :,
                    ],
                )
            )

        losses[
            t
        ] = np.nanmean(
            variable_losses
        )

    return losses


def compute_crps_loss_by_variable_and_time(
    observations,
    forecast_samples,
):
    observations = np.asarray(
        observations,
        dtype=float,
    )

    forecast_samples = np.asarray(
        forecast_samples,
        dtype=float,
    )

    n_variables, T, S = forecast_samples.shape

    if observations.shape != (
        n_variables,
        T,
    ):
        raise ValueError(
            "observations must have shape "
            f"{(n_variables, T)}, got {observations.shape}"
        )

    losses = np.full(
        (
            n_variables,
            T,
        ),
        np.nan,
        dtype=float,
    )

    for variable_index in range(
        n_variables
    ):
        for t in range(T):
            losses[
                variable_index,
                t,
            ] = _crps_ensemble_1d(
                y=observations[
                    variable_index,
                    t,
                ],
                samples=forecast_samples[
                    variable_index,
                    t,
                    :,
                ],
            )

    return losses


def append_loss_rows_for_case(
    loss_rows,
    crps_series_rows,
    dimension,
    n_samples,
    surface,
    forecasts,
    observations,
):
    for method, arr in forecasts.items():

        energy_losses = compute_energy_loss_by_time(
            observations=observations,
            forecast_samples=arr,
        )

        crps_losses = compute_crps_loss_by_time(
            observations=observations,
            forecast_samples=arr,
        )

        crps_series_losses = compute_crps_loss_by_variable_and_time(
            observations=observations,
            forecast_samples=arr,
        )

        for t, loss_value in enumerate(
            energy_losses
        ):
            loss_rows.append(
                {
                    "dimension": dimension,
                    "n_samples": n_samples,
                    "surface": surface,
                    "method": method,
                    "score": "energy_score",
                    "t": t,
                    "loss": loss_value,
                }
            )

        for t, loss_value in enumerate(
            crps_losses
        ):
            loss_rows.append(
                {
                    "dimension": dimension,
                    "n_samples": n_samples,
                    "surface": surface,
                    "method": method,
                    "score": "crps",
                    "t": t,
                    "loss": loss_value,
                }
            )

        n_variables, T = crps_series_losses.shape

        for variable_index in range(
            n_variables
        ):
            for t in range(T):
                crps_series_rows.append(
                    {
                        "dimension": dimension,
                        "n_samples": n_samples,
                        "surface": surface,
                        "method": method,
                        "score": "crps",
                        "series_index": variable_index,
                        "t": t,
                        "loss": crps_series_losses[
                            variable_index,
                            t,
                        ],
                    }
                )


# ============================================================
# CHECKPOINTS
# ============================================================

def save_loss_checkpoint(
    loss_rows,
    loss_file,
):
    if not loss_rows:
        return

    loss_df = pd.DataFrame(
        loss_rows
    )

    loss_df = loss_df.sort_values(
        [
            "dimension",
            "n_samples",
            "surface",
            "score",
            "method",
            "t",
        ]
    )

    loss_df.to_csv(
        loss_file,
        index=False,
    )


def save_crps_series_checkpoint(
    crps_series_rows,
    crps_series_loss_file,
):
    if not crps_series_rows:
        return

    crps_series_df = pd.DataFrame(
        crps_series_rows
    )

    crps_series_df = crps_series_df.sort_values(
        [
            "dimension",
            "n_samples",
            "surface",
            "score",
            "method",
            "series_index",
            "t",
        ]
    )

    crps_series_df.to_csv(
        crps_series_loss_file,
        index=False,
    )


def save_score_checkpoint(
    score_rows,
    score_file,
):
    if not score_rows:
        return

    score_df = pd.DataFrame(
        score_rows
    )

    score_df = score_df.sort_values(
        [
            "dimension",
            "n_samples",
            "surface",
            "method",
        ]
    )

    score_df.to_csv(
        score_file,
        index=False,
    )


def save_score_summary_checkpoint(
    score_summary_rows,
    score_summary_file,
):
    if not score_summary_rows:
        return

    score_summary_df = pd.DataFrame(
        score_summary_rows
    )

    score_summary_df = score_summary_df.sort_values(
        [
            "dimension",
            "n_samples",
            "surface",
            "score",
            "method",
        ]
    )

    score_summary_df.to_csv(
        score_summary_file,
        index=False,
    )


# ============================================================
# LATEX SCORE TABLE EXPORT
# ============================================================

def latex_escape(
    text,
):
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


def format_score_cell(
    value,
    is_minimum,
    decimals=2,
):
    if pd.isna(
        value
    ):
        return "--"

    text = f"{float(value):.{decimals}f}"

    if is_minimum:
        return rf"\textbf{{{text}}}"

    return text


def add_minimum_flags(
    score_df,
    decimals=2,
):
    """
    Bold minima are determined after rounding to the displayed precision.
    """

    score_df = score_df.copy()

    score_df[
        "display_relative_score"
    ] = score_df[
        "relative_score"
    ].round(
        decimals
    )

    score_df[
        "is_display_minimum"
    ] = False

    for (
        dimension,
        n_samples,
        surface,
        score,
    ), group in score_df.groupby(
        [
            "dimension",
            "n_samples",
            "surface",
            "score",
        ],
        dropna=False,
    ):

        minimum_display_score = group[
            "display_relative_score"
        ].min()

        minimum_indices = group.index[
            group[
                "display_relative_score"
            ]
            == minimum_display_score
        ]

        score_df.loc[
            minimum_indices,
            "is_display_minimum",
        ] = True

    return score_df


def surface_display_name(
    surface,
):
    return SURFACE_LABELS.get(
        surface,
        str(
            surface
        ).title(),
    )


def write_latex_table_for_dimension_and_sample_size(
    lines,
    score_df,
    dimension,
    n_samples,
):
    subset_df = score_df[
        (
            score_df[
                "dimension"
            ]
            == dimension
        )
        & (
            score_df[
                "n_samples"
            ]
            == n_samples
        )
    ].copy()

    available_surfaces = list(
        subset_df[
            "surface"
        ].dropna().unique()
    )

    surface_order = [
        surface
        for surface in SURFACE_ORDER
        if surface in available_surfaces
    ]

    extra_surfaces = sorted(
        surface
        for surface in available_surfaces
        if surface not in surface_order
    )

    surface_order = (
        surface_order
        + extra_surfaces
    )

    method_groups = [
        (
            "Baseline",
            [
                "base",
                "pbu",
            ],
        ),
        (
            "Projection",
            [
                "ols",
                "wls",
                "full",
            ],
        ),
        (
            r"\textit{Conditioning}",
            [
                "ukf",
            ],
        ),
    ]

    def get_cell(
        score,
        surface,
        method,
    ):
        match = subset_df[
            (
                subset_df[
                    "score"
                ]
                == score
            )
            & (
                subset_df[
                    "surface"
                ]
                == surface
            )
            & (
                subset_df[
                    "method"
                ]
                == method
            )
        ]

        if match.empty:
            return "--"

        row = match.iloc[
            0
        ]

        return format_score_cell(
            value=row[
                "relative_score"
            ],
            is_minimum=row[
                "is_display_minimum"
            ],
            decimals=2,
        )

    column_spec = (
        "ll"
        + "c" * len(surface_order)
        + "c" * len(surface_order)
    )

    crps_span = len(
        surface_order
    )

    energy_start = (
        3
        + crps_span
    )

    energy_end = (
        2
        + 2
        * crps_span
    )

    lines.append(
        r"\begin{table}[h!]"
    )
    lines.append(
        r"    \centering"
    )
    lines.append(
        f"    \\begin{{tabular}}{{{column_spec}}}"
    )
    lines.append(
        r"    \toprule"
    )

    lines.append(
        r"    \multicolumn{2}{c}{\multirow{2}{*}{\textit{Methods}}} & "
        + rf"\multicolumn{{{crps_span}}}{{c}}{{\textit{{CRPS}}}} & "
        + rf"\multicolumn{{{crps_span}}}{{c}}{{\textit{{Energy Score}}}} \\"
    )

    lines.append(
        rf"    \cmidrule(l){{3-{2 + crps_span}}} "
        rf"\cmidrule(l){{{energy_start}-{energy_end}}}"
    )

    surface_headers = [
        rf"\textbf{{{latex_escape(surface_display_name(surface))}}}"
        for surface in surface_order
    ]

    lines.append(
        "    & & "
        + " & ".join(
            surface_headers
        )
        + " & "
        + " & ".join(
            surface_headers
        )
        + r" \\"
    )

    lines.append(
        r"    \midrule"
    )

    first_group = True

    for group_label, methods in method_groups:

        if not first_group:
            lines.append(
                r"    \midrule"
            )

        first_group = False

        for method_index, method in enumerate(
            methods
        ):

            if len(
                methods
            ) > 1:
                if method_index == 0:
                    group_cell = (
                        rf"\multirow{{{len(methods)}}}{{*}}{{{group_label}}}"
                    )
                else:
                    group_cell = ""
            else:
                group_cell = group_label

            method_cell = METHOD_LABELS.get(
                method,
                method,
            )

            crps_cells = [
                get_cell(
                    score="crps",
                    surface=surface,
                    method=method,
                )
                for surface in surface_order
            ]

            energy_cells = [
                get_cell(
                    score="energy",
                    surface=surface,
                    method=method,
                )
                for surface in surface_order
            ]

            row_values = [
                group_cell,
                method_cell,
            ] + crps_cells + energy_cells

            lines.append(
                "    "
                + " & ".join(
                    row_values
                )
                + r" \\"
            )

    lines.append(
        r"    \bottomrule"
    )

    lines.append(
        r"    \end{tabular}"
    )

    lines.append(
        r"    \caption{Relative CRPS and Energy Score for the high-dimensional "
        rf"simulation study with \(d={dimension}\) and \(S={n_samples}\) forecast samples. "
        r"Scores are normalized with respect to the base forecast. The best displayed "
        r"value for each surface and score is highlighted in bold.}"
    )

    lines.append(
        rf"    \label{{tab:simulation_study_d{dimension}_s{n_samples}}}"
    )

    lines.append(
        r"\end{table}"
    )

    lines.append("")


def write_simulation_score_latex_tables(
    score_summary_rows,
):
    if not score_summary_rows:
        return

    score_df = pd.DataFrame(
        score_summary_rows
    )

    score_df = add_minimum_flags(
        score_df,
        decimals=2,
    )

    lines = []

    grouped = score_df[
        [
            "dimension",
            "n_samples",
        ]
    ].drop_duplicates().sort_values(
        [
            "dimension",
            "n_samples",
        ]
    )

    for row in grouped.itertuples(
        index=False
    ):
        write_latex_table_for_dimension_and_sample_size(
            lines=lines,
            score_df=score_df,
            dimension=row.dimension,
            n_samples=row.n_samples,
        )

    latex_text = "\n".join(
        lines
    )

    with open(
        SIMULATION_SCORE_LATEX_TABLE_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            latex_text
        )

    print()
    print(
        f"Saved simulation LaTeX score tables: {SIMULATION_SCORE_LATEX_TABLE_FILE}"
    )


# ============================================================
# SCORE REPORTING
# ============================================================

def print_final_score_summary(
    score_rows,
):
    if not score_rows:
        print(
            "No scores were computed."
        )
        return

    score_df = pd.DataFrame(
        score_rows
    )

    score_df[
        "method"
    ] = pd.Categorical(
        score_df[
            "method"
        ],
        categories=METHOD_ORDER,
        ordered=True,
    )

    score_df = score_df.sort_values(
        [
            "dimension",
            "n_samples",
            "surface",
            "method",
        ]
    )

    print()
    print("=" * 100)
    print("FINAL SCORE SUMMARY")
    print("=" * 100)
    print()

    print(
        score_df.to_string(
            index=False,
            formatters={
                "energy_score": lambda x: f"{x:.6f}",
                "energy_relative": lambda x: f"{x:.4f}",
                "crps": lambda x: f"{x:.6f}",
                "crps_relative": lambda x: f"{x:.4f}",
            },
        )
    )


# ============================================================
# FILE HELPERS
# ============================================================

def find_forecast_file(
    fc_folder,
    kind,
    surface,
    dimension,
    n_samples,
    dataset_tag="indep",
):
    """
    Find files produced by the base-forecast script.

    Main filenames:

        {kind}_{surface}_d{dimension}_{n_samples}.pkl

    Optional tagged filenames:

        {kind}_{surface}_{dataset_tag}_d{dimension}_{n_samples}.pkl
    """

    candidates = [
        os.path.join(
            fc_folder,
            f"{kind}_{surface}_d{dimension}_{n_samples}.pkl",
        ),
        os.path.join(
            fc_folder,
            f"{kind}_{surface}_{dataset_tag}_d{dimension}_{n_samples}.pkl",
        ),
    ]

    for candidate in candidates:
        if os.path.exists(
            candidate
        ):
            return candidate

    raise FileNotFoundError(
        "Could not find forecast file. Tried:\n"
        + "\n".join(
            candidates
        )
    )


# ============================================================
# MAIN LOOP
# ============================================================

def main():
    os.makedirs(
        RESULTS_FOLDER,
        exist_ok=True,
    )

    dataset_tag = "indep"

    dimensions = [
        10,
        20,
        50,
    ]

    surfaces = [
        "paraboloid",
        "saddle",
        "ripples",
    ]

    sample_sizes = [
        2000,
    ]

    projection_workers = 4

    score_rows = []
    score_summary_rows = []
    loss_rows = []
    crps_series_rows = []

    for dimension in dimensions:
        print()
        print(
            "==================================================="
        )
        print(
            f"Bottom-level dimension: d={dimension}"
        )
        print(
            "==================================================="
        )

        for n_samples in sample_sizes:
            print()
            print(
                "---------------------------------------------------"
            )
            print(
                f"Forecast sample size: S={n_samples}"
            )
            print(
                "---------------------------------------------------"
            )

            for surface in surfaces:
                print()
                print(
                    "==================================================="
                )
                print(
                    f"Running reconciliation for "
                    f"surface={surface}, d={dimension}, S={n_samples}"
                )
                print(
                    "==================================================="
                )
                print()

                base_path = find_forecast_file(
                    fc_folder=FORECASTS_FOLDER,
                    kind="base_fc",
                    surface=surface,
                    dimension=dimension,
                    n_samples=n_samples,
                    dataset_tag=dataset_tag,
                )

                res_path = find_forecast_file(
                    fc_folder=FORECASTS_FOLDER,
                    kind="residuals",
                    surface=surface,
                    dimension=dimension,
                    n_samples=n_samples,
                    dataset_tag=dataset_tag,
                )

                te_path = find_forecast_file(
                    fc_folder=FORECASTS_FOLDER,
                    kind="test_data",
                    surface=surface,
                    dimension=dimension,
                    n_samples=n_samples,
                    dataset_tag=dataset_tag,
                )

                print(
                    f"Loading base forecast: {base_path}"
                )

                print(
                    f"Loading residuals:     {res_path}"
                )

                print(
                    f"Loading test data:     {te_path}"
                )

                base_fc = pd.read_pickle(
                    base_path
                )

                base_fc = np.asarray(
                    base_fc,
                    dtype=np.float64,
                )

                T = base_fc.shape[
                    1
                ]

                S = base_fc.shape[
                    2
                ]

                if S != n_samples:
                    raise ValueError(
                        f"File name indicates n_samples={n_samples}, "
                        f"but loaded base forecast has S={S}."
                    )

                tr_res = pd.read_pickle(
                    res_path
                )

                tr_res = np.asarray(
                    tr_res,
                    dtype=np.float64,
                )

                df_te = pd.read_pickle(
                    te_path
                )

                expected_variables = (
                    dimension
                    + 1
                )

                if base_fc.shape[
                    0
                ] != expected_variables:
                    raise ValueError(
                        f"Expected {expected_variables} forecast variables "
                        f"for d={dimension}, got shape {base_fc.shape}"
                    )

                if tr_res.shape[
                    0
                ] != expected_variables:
                    raise ValueError(
                        f"Expected {expected_variables} residual variables "
                        f"for d={dimension}, got shape {tr_res.shape}"
                    )

                # y_hat at origin t forecasts the next observation t+1.
                gt_test = df_te.iloc[
                    1:
                ].values

                if gt_test.shape != (
                    T,
                    expected_variables,
                ):
                    raise ValueError(
                        "Ground-truth shape does not match forecasts: "
                        f"gt_test={gt_test.shape}, "
                        f"base_fc={base_fc.shape}"
                    )

                print(
                    f"Base forecast shape: {base_fc.shape}"
                )

                print(
                    f"Residual shape: {tr_res.shape}"
                )

                print(
                    f"Ground-truth shape: {gt_test.shape}"
                )

                # ============================================================
                # PROBABILISTIC BOTTOM-UP
                # ============================================================

                print()
                print(
                    "PBU started."
                )

                bot_base = base_fc[
                    1:,
                    :,
                    :,
                ]

                bu_fc = pbu(
                    B=bot_base,
                    surface=surface,
                )

                print(
                    "PBU completed."
                )

                # ============================================================
                # OLS/WLS/FULL PROJECTION
                # ============================================================

                print()
                print(
                    "Projection started."
                )

                (
                    ols_fc,
                    full_fc,
                    wls_fc,
                ) = run_projection_parallel(
                    base_fc=base_fc,
                    tr_res=tr_res,
                    surface=surface,
                    dimension=dimension,
                    n_iter=20,
                    seed=42,
                    max_workers=projection_workers,
                )

                print(
                    "Projection completed."
                )

                # ============================================================
                # UKF
                # ============================================================

                print()
                print(
                    "UKF started."
                )

                def f_ukf_vec(b):
                    return f_surface(
                        surface=surface,
                        B=b,
                        axis=0,
                    )

                def f_ukf_mult(bmat):
                    return f_surface(
                        surface=surface,
                        B=bmat,
                        axis=1,
                    )

                bot_res = tr_res[
                    1:,
                    :,
                ]

                R = schafer_strimmer_cov(
                    tr_res.T
                )["shrink_cov"][
                    0,
                    0,
                ]

                ukf_dict = {}

                for t in range(
                    T
                ):
                    print(
                        f"UKF t={t + 1}/{T}",
                        flush=True,
                    )

                    u_obs = np.mean(
                        base_fc[
                            0,
                            t,
                            :,
                        ]
                    ).reshape(
                        1,
                    )

                    bot_list = []

                    for bottom_index in range(
                        dimension
                    ):
                        bot_list.append(
                            {
                                "samples": bot_base[
                                    bottom_index,
                                    t,
                                    :,
                                ],
                                "residuals": bot_res[
                                    bottom_index,
                                    :,
                                ],
                            }
                        )

                    out = reconc_nl_ukf(
                        bottom_base_forecasts=bot_list,
                        in_type=[
                            "samples"
                        ],
                        distr=[
                            "gaussian"
                        ],
                        f=f_ukf_vec,
                        upper_base_forecasts=u_obs,
                        R=R,
                        num_samples=S,
                        seed=42,
                    )

                    Brec = out[
                        "bottom_reconciled_samples"
                    ]

                    Urec = f_ukf_mult(
                        Brec.T
                    )

                    ukf_dict[
                        t
                    ] = np.vstack(
                        [
                            Urec,
                            Brec,
                        ]
                    )

                ukf_fc = np.stack(
                    [
                        ukf_dict[
                            t
                        ]
                        for t in range(
                            T
                        )
                    ],
                    axis=1,
                )

                print(
                    "UKF completed."
                )

                # ============================================================
                # EVALUATION
                # ============================================================

                forecasts = {
                    "base": base_fc,
                    "pbu": bu_fc,
                    "ols": ols_fc,
                    "wls": wls_fc,
                    "full": full_fc,
                    "ukf": ukf_fc,
                }

                es_scores = {}

                for key, arr in forecasts.items():
                    es_scores[
                        key
                    ] = compute_es(
                        gt_test.T,
                        arr,
                    )

                base_es = es_scores[
                    "base"
                ]

                crps_scores = {}

                for key, arr in forecasts.items():
                    crps_scores[
                        key
                    ] = compute_crps_new(
                        gt_test.T,
                        arr,
                    )

                base_crps = crps_scores[
                    "base"
                ]

                for method in forecasts:
                    energy_score = es_scores[
                        method
                    ]

                    crps = crps_scores[
                        method
                    ]

                    energy_relative = (
                        energy_score
                        / base_es
                        if base_es != 0
                        else np.nan
                    )

                    crps_relative = (
                        crps
                        / base_crps
                        if base_crps != 0
                        else np.nan
                    )

                    score_rows.append(
                        {
                            "dimension": dimension,
                            "n_samples": n_samples,
                            "surface": surface,
                            "method": method,
                            "energy_score": float(
                                energy_score
                            ),
                            "energy_relative": float(
                                energy_relative
                            ),
                            "crps": float(
                                crps
                            ),
                            "crps_relative": float(
                                crps_relative
                            ),
                        }
                    )

                    score_summary_rows.append(
                        {
                            "dimension": dimension,
                            "n_samples": n_samples,
                            "surface": surface,
                            "score": "energy",
                            "method": method,
                            "absolute_score": float(
                                energy_score
                            ),
                            "relative_score": float(
                                energy_relative
                            ),
                        }
                    )

                    score_summary_rows.append(
                        {
                            "dimension": dimension,
                            "n_samples": n_samples,
                            "surface": surface,
                            "score": "crps",
                            "method": method,
                            "absolute_score": float(
                                crps
                            ),
                            "relative_score": float(
                                crps_relative
                            ),
                        }
                    )

                print()
                print(
                    "Saving per-time-step losses for DM/MCS tests."
                )

                append_loss_rows_for_case(
                    loss_rows=loss_rows,
                    crps_series_rows=crps_series_rows,
                    dimension=dimension,
                    n_samples=n_samples,
                    surface=surface,
                    forecasts=forecasts,
                    observations=gt_test.T,
                )

                print()
                print(
                    f"Scores for {surface}, d={dimension}, S={n_samples}"
                )

                print()
                print(
                    f"{'method':<8}"
                    f"{'ES rel.':>12}"
                    f"{'CRPS rel.':>12}"
                )

                print(
                    "-" * 32
                )

                for method in forecasts:
                    print(
                        f"{method:<8}"
                        f"{es_scores[method] / base_es:>12.4f}"
                        f"{crps_scores[method] / base_crps:>12.4f}"
                    )

                save_score_checkpoint(
                    score_rows=score_rows,
                    score_file=SCORE_FILE,
                )

                save_score_summary_checkpoint(
                    score_summary_rows=score_summary_rows,
                    score_summary_file=SIMULATION_SCORE_SUMMARY_FILE,
                )

                save_loss_checkpoint(
                    loss_rows=loss_rows,
                    loss_file=LOSS_FILE,
                )

                save_crps_series_checkpoint(
                    crps_series_rows=crps_series_rows,
                    crps_series_loss_file=CRPS_SERIES_LOSS_FILE,
                )

                write_simulation_score_latex_tables(
                    score_summary_rows=score_summary_rows,
                )

                print()
                print(
                    f"Score checkpoint saved to: {SCORE_FILE}"
                )

                print(
                    f"Score summary checkpoint saved to: {SIMULATION_SCORE_SUMMARY_FILE}"
                )

                print(
                    f"Loss checkpoint saved to: {LOSS_FILE}"
                )

                print(
                    f"CRPS series loss checkpoint saved to: {CRPS_SERIES_LOSS_FILE}"
                )

                print(
                    f"LaTeX score table saved to: {SIMULATION_SCORE_LATEX_TABLE_FILE}"
                )

    print()
    print(
        "All dimensions, sample sizes, and surfaces completed."
    )

    print_final_score_summary(
        score_rows
    )

    save_score_checkpoint(
        score_rows=score_rows,
        score_file=SCORE_FILE,
    )

    save_score_summary_checkpoint(
        score_summary_rows=score_summary_rows,
        score_summary_file=SIMULATION_SCORE_SUMMARY_FILE,
    )

    save_loss_checkpoint(
        loss_rows=loss_rows,
        loss_file=LOSS_FILE,
    )

    save_crps_series_checkpoint(
        crps_series_rows=crps_series_rows,
        crps_series_loss_file=CRPS_SERIES_LOSS_FILE,
    )

    write_simulation_score_latex_tables(
        score_summary_rows=score_summary_rows,
    )

    print()
    print(
        f"Final score file: {SCORE_FILE}"
    )

    print(
        f"Final score summary file: {SIMULATION_SCORE_SUMMARY_FILE}"
    )

    print(
        f"Final loss file: {LOSS_FILE}"
    )

    print(
        f"Final CRPS series loss file: {CRPS_SERIES_LOSS_FILE}"
    )

    print(
        f"Final LaTeX score table: {SIMULATION_SCORE_LATEX_TABLE_FILE}"
    )

    print()


if __name__ == "__main__":
    main()