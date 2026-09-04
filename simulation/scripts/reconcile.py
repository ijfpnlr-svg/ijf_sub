import warnings
warnings.filterwarnings("ignore")

import os
from multiprocessing import get_context
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import jax.numpy as jnp
from bayesreconpy.shrink_cov import _schafer_strimmer_cov as schafer_strimmer_cov

from reconc.reconc_nl_ols import reconc_nl_ols
from reconc.reconc_nl_ukf import reconc_nl_ukf
from score_functions import compute_es
from simulation.scripts.score_functions import compute_crps_new


# ============================================================
# CONFIG
# ============================================================

RESULTS_FOLDER = "../results"

MCS_CRPS_SERIES_LOSS_FILE = os.path.join(
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


def sync_any(x):
    if hasattr(x, "block_until_ready"):
        x.block_until_ready()
    elif isinstance(x, dict):
        for v in x.values():
            sync_any(v)
    elif isinstance(x, (list, tuple)):
        for v in x:
            sync_any(v)
    return x


# ============================================================
# PARALLEL PROJECTION
# ============================================================

def _project_at_time_step_worker(task):
    t, z_t, res_t, surface, n_iter, seed = task

    def f_ols(z):
        u = z[0]
        b1 = z[1]
        b2 = z[2]

        return jnp.array(
            [
                u
                - f_surface_jax(
                    surface,
                    b1,
                    b2,
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

    out_mint = reconc_nl_ols(
        z_t,
        f_ols,
        n_iter=n_iter,
        seed=seed,
        W=P,
    )
    sync_any(
        out_mint
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
        "mint": np.asarray(
            out_mint[
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
    indep_base_fc,
    indep_tr_res,
    surface,
    n_iter=20,
    seed=42,
    max_workers=None,
):
    T = indep_base_fc.shape[
        1
    ]

    tasks = [
        (
            t,
            indep_base_fc[
                :,
                t,
                :,
            ].T.copy(),
            indep_tr_res[
                :,
                t,
                :,
            ].copy(),
            surface,
            n_iter,
            seed,
        )
        for t in range(
            T
        )
    ]

    ols_dict = {}
    mint_dict = {}
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

        for fut in as_completed(
            futures
        ):
            result = fut.result()

            t = result[
                "t"
            ]

            ols_dict[
                t
            ] = result[
                "ols"
            ]

            mint_dict[
                t
            ] = result[
                "mint"
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
            for t in range(
                T
            )
        ],
        axis=1,
    )

    mint_fc = np.stack(
        [
            mint_dict[
                t
            ]
            for t in range(
                T
            )
        ],
        axis=1,
    )

    wls_fc = np.stack(
        [
            wls_dict[
                t
            ]
            for t in range(
                T
            )
        ],
        axis=1,
    )

    return (
        ols_fc,
        mint_fc,
        wls_fc,
    )


# ============================================================
# SURFACE FUNCTIONS
# ============================================================

def f_surface(
    surface,
    b1,
    b2,
):
    if surface == "paraboloid":
        return (
            b1 ** 2
            + b2 ** 2
        )

    if surface == "cone":
        return np.sqrt(
            np.maximum(
                b1 ** 2
                + b2 ** 2,
                0.0,
            )
        )

    if surface == "saddle":
        return (
            b1 ** 2
            - b2 ** 2
        )

    if surface == "ripples":
        return (
            np.sin(
                b1
            )
            + np.cos(
                b2
            )
        )

    if surface == "linear":
        return (
            b1
            + b2
        )

    raise ValueError(
        f"Unknown surface {surface}"
    )


def f_surface_jax(
    surface,
    b1,
    b2,
):
    if surface == "paraboloid":
        return (
            b1 ** 2
            + b2 ** 2
        )

    if surface == "cone":
        return jnp.sqrt(
            jnp.maximum(
                b1 ** 2
                + b2 ** 2,
                0.0,
            )
        )

    if surface == "saddle":
        return (
            b1 ** 2
            - b2 ** 2
        )

    if surface == "ripples":
        return (
            jnp.sin(
                b1
            )
            + jnp.cos(
                b2
            )
        )

    if surface == "linear":
        return (
            b1
            + b2
        )

    raise ValueError(
        f"Unknown surface {surface}"
    )


# ============================================================
# BOTTOM-UP CONSTRUCTION
# ============================================================

def pbu(
    B,
    surface,
):
    """
    B shape = (2, T, S) = (B1, B2)
    Output = (3, T, S) = (U, B1, B2)
    """

    b1 = B[
        0,
        :,
        :,
    ]

    b2 = B[
        1,
        :,
        :,
    ]

    U = f_surface(
        surface,
        b1,
        b2,
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
# PRECISION MATRIX UTIL
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
        / cov.shape[
            0
        ]
    )

    return np.linalg.pinv(
        cov
        + lam
        * np.eye(
            cov.shape[
                0
            ]
        )
    )


# ============================================================
# MCS LOSS EXPORT
# ============================================================

def crps_1d_from_samples(
    observation,
    samples,
):
    """
    Empirical CRPS for one scalar observation and one sample vector.

    CRPS(F, y) = E|X - y| - 0.5 E|X - X'|.

    Uses the sorted-sample identity for the pairwise term.
    """

    samples = np.asarray(
        samples,
        dtype=float,
    )

    n_samples = samples.size

    if n_samples == 0:
        return np.nan

    mean_abs_error = np.mean(
        np.abs(
            samples
            - observation
        )
    )

    sorted_samples = np.sort(
        samples
    )

    indices = np.arange(
        1,
        n_samples + 1,
        dtype=float,
    )

    coefficients = (
        2.0
        * indices
        - n_samples
        - 1.0
    )

    half_pairwise_distance = np.sum(
        coefficients
        * sorted_samples
    ) / (
        n_samples
        ** 2
    )

    crps = (
        mean_abs_error
        - half_pairwise_distance
    )

    return float(
        crps
    )


def append_crps_by_series_time_rows(
    rows,
    target,
    forecasts,
    observations,
):
    """
    Append CRPS losses for the simulation MCS script.

    Output columns:
        target, score, method, series_index, t, loss

    Here:
        target = surface name
        score = crps
        method = reconciliation method
        series_index = index of the time series in the full vector
        t = rolling forecast origin / test time index
        loss = scalar CRPS
    """

    observations = np.asarray(
        observations,
        dtype=float,
    )

    n_series = observations.shape[
        0
    ]

    n_time_steps = observations.shape[
        1
    ]

    for method, forecast_array in forecasts.items():

        forecast_array = np.asarray(
            forecast_array,
            dtype=float,
        )

        if forecast_array.shape[
            0
        ] != n_series:
            raise ValueError(
                f"Method {method} has incompatible number of series: "
                f"{forecast_array.shape[0]} != {n_series}"
            )

        if forecast_array.shape[
            1
        ] != n_time_steps:
            raise ValueError(
                f"Method {method} has incompatible number of time steps: "
                f"{forecast_array.shape[1]} != {n_time_steps}"
            )

        for series_index in range(
            n_series
        ):
            for t in range(
                n_time_steps
            ):
                loss = crps_1d_from_samples(
                    observation=observations[
                        series_index,
                        t,
                    ],
                    samples=forecast_array[
                        series_index,
                        t,
                        :,
                    ],
                )

                rows.append(
                    {
                        "target": target,
                        "score": "crps",
                        "method": method,
                        "series_index": series_index,
                        "t": t,
                        "loss": loss,
                    }
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

    This makes the LaTeX table visually consistent: if several methods print
    as 0.96 and 0.96 is the displayed minimum, all are bolded.
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
        score,
        surface,
    ), group in score_df.groupby(
        [
            "score",
            "surface",
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


def write_simulation_score_latex_table(
    score_summary_df,
):
    score_df = add_minimum_flags(
        score_summary_df,
        decimals=2,
    )

    surface_order = [
        "paraboloid",
        "saddle",
        "ripples",
    ]

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

    method_labels = {
        "base": "Base",
        "pbu": "PBU",
        "ols": "OLS",
        "wls": "WLS",
        "full": "FULL",
        "ukf": "UKF",
    }

    def get_cell(
        score,
        surface,
        method,
    ):
        match = score_df[
            (
                score_df[
                    "score"
                ]
                == score
            )
            & (
                score_df[
                    "surface"
                ]
                == surface
            )
            & (
                score_df[
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

    lines = []

    lines.append(
        r"\begin{table}[h!]"
    )
    lines.append(
        r"    \centering"
    )
    lines.append(
        r"    \begin{tabular}{llcccccc}"
    )
    lines.append(
        r"    \toprule"
    )
    lines.append(
        r"    \multicolumn{2}{c}{\multirow{2}{*}{\textit{Methods}}} & "
        r"\multicolumn{3}{c}{\textit{CRPS}} & "
        r"\multicolumn{3}{c}{\textit{Energy Score}} \\"
    )
    lines.append(
        r"    \cmidrule(l){3-5} \cmidrule(l){6-8}"
    )
    lines.append(
        r"    & & "
        r"\textbf{paraboloid} & \textbf{saddle} & \textbf{ripples} & "
        r"\textbf{paraboloid} & \textbf{saddle} & \textbf{ripples} \\"
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

            method_cell = method_labels.get(
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
        r"    \caption{Relative CRPS and Energy Score for the simulation study. "
        r"The best method is highlighted in bold.}"
    )
    lines.append(
        r"    \label{tab:simulation_study}"
    )
    lines.append(
        r"\end{table}"
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
        f"Saved simulation LaTeX score table: {SIMULATION_SCORE_LATEX_TABLE_FILE}"
    )


# ============================================================
# MAIN LOOP OVER ALL SURFACES
# ============================================================

def main():
    fc_folder = "../forecasts"

    os.makedirs(
        RESULTS_FOLDER,
        exist_ok=True,
    )

    crps_by_series_time_rows = []

    score_summary_rows = []

    surfaces = [
        "saddle",
        "ripples",
        "paraboloid",
        # "linear",
    ]

    n_samples = 2000

    projection_workers = None

    for surface in surfaces:
        print(
            "\n==================================================="
        )
        print(
            f"Running reconciliation for surface: {surface}"
        )
        print(
            "===================================================\n"
        )

        # ------------- LOAD FORECASTS -------------
        base_path = (
            f"{fc_folder}/base_fc_{surface}_indep_{n_samples}.pkl"
        )

        res_path = (
            f"{fc_folder}/residuals_{surface}_indep_{n_samples}.pkl"
        )

        te_path = (
            f"{fc_folder}/test_data_{surface}_indep_{n_samples}.pkl"
        )

        indep_base_fc = pd.read_pickle(
            base_path
        )

        T = indep_base_fc.shape[
            1
        ]

        S = indep_base_fc.shape[
            2
        ]

        indep_tr_res = pd.read_pickle(
            res_path
        )

        indep_tr_res = np.repeat(
            indep_tr_res[
                :,
                None,
                :,
            ],
            T,
            axis=1,
        )

        df_te = pd.read_pickle(
            te_path
        )

        gt_test = df_te.iloc[
            1:
        ].values

        indep_base_fc = np.array(
            indep_base_fc,
            dtype=np.float64,
        )

        # ============================================================
        # PROBABILISTIC BOTTOM-UP
        # ============================================================

        print(
            "PBU started."
        )

        bot_base = indep_base_fc[
            1:,
            :,
            :,
        ]

        bu_fc = pbu(
            bot_base,
            surface,
        )

        print(
            "PBU completed."
        )

        # ============================================================
        # OLS/WLS/FULL PROJECTION
        # ============================================================

        print(
            "Projection started."
        )

        ols_fc, mint_fc, wls_fc = run_projection_parallel(
            indep_base_fc=indep_base_fc,
            indep_tr_res=indep_tr_res,
            surface=surface,
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

        def f_ukf_vec(b):
            b1, b2 = b[
                0
            ], b[
                1
            ]

            return f_surface(
                surface,
                b1,
                b2,
            )

        def f_ukf_mult(bmat):
            b1 = bmat[
                :,
                0,
            ]

            b2 = bmat[
                :,
                1,
            ]

            return f_surface(
                surface,
                b1,
                b2,
            )

        bot_res = indep_tr_res[
            1:,
            :,
            :,
        ]

        ukf_dict = {}

        for t in range(
            T
        ):
            u_obs = np.mean(
                indep_base_fc[
                    0,
                    t,
                    :,
                ]
            ).reshape(
                1,
            )

            R = schafer_strimmer_cov(
                indep_tr_res[
                    :,
                    t,
                    :,
                ].T
            )[
                "shrink_cov"
            ][
                0,
                0,
            ]

            bot_list = []

            for s in range(
                2
            ):
                bot_list.append(
                    {
                        "samples": bot_base[
                            s,
                            t,
                            :,
                        ],
                        "residuals": bot_res[
                            s,
                            t,
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

        # ============================================================
        # EVALUATION
        # ============================================================

        forecasts = {
            "base": indep_base_fc,
            "pbu": bu_fc,
            "ols": ols_fc,
            "wls": wls_fc,
            "full": mint_fc,
            "ukf": ukf_fc,
        }

        append_crps_by_series_time_rows(
            rows=crps_by_series_time_rows,
            target=surface,
            forecasts=forecasts,
            observations=gt_test.T,
        )

        # ---- Energy Score: absolute + relative to base ----
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

        print(
            f"Energy Scores for {surface} (relative to base):"
        )

        for key in forecasts.keys():
            abs_es = es_scores[
                key
            ]

            rel_es = (
                abs_es
                / base_es
                if base_es != 0
                else np.nan
            )

            score_summary_rows.append(
                {
                    "surface": surface,
                    "score": "energy",
                    "method": key,
                    "absolute_score": float(
                        abs_es
                    ),
                    "relative_score": float(
                        rel_es
                    ),
                }
            )

            print(
                f"  {key:<6} :  {rel_es:.2f}"
            )

        # ---- CRPS: absolute + relative to base ----
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

        print(
            f"CRPS for {surface} (relative to base):"
        )

        for key in forecasts.keys():
            abs_crps = crps_scores[
                key
            ]

            rel_crps = (
                abs_crps
                / base_crps
                if base_crps != 0
                else np.nan
            )

            score_summary_rows.append(
                {
                    "surface": surface,
                    "score": "crps",
                    "method": key,
                    "absolute_score": float(
                        abs_crps
                    ),
                    "relative_score": float(
                        rel_crps
                    ),
                }
            )

            print(
                f"  {key:<6} :  {rel_crps:.2f}"
            )

    # ============================================================
    # EXPORT RESULTS
    # ============================================================

    crps_by_series_time_df = pd.DataFrame(
        crps_by_series_time_rows
    )

    crps_by_series_time_df.to_csv(
        MCS_CRPS_SERIES_LOSS_FILE,
        index=False,
    )

    print()
    print(
        f"Saved CRPS losses for MCS: {MCS_CRPS_SERIES_LOSS_FILE}"
    )

    score_summary_df = pd.DataFrame(
        score_summary_rows
    )

    score_summary_df.to_csv(
        SIMULATION_SCORE_SUMMARY_FILE,
        index=False,
    )

    print()
    print(
        f"Saved simulation relative score summary: {SIMULATION_SCORE_SUMMARY_FILE}"
    )

    write_simulation_score_latex_table(
        score_summary_df=score_summary_df,
    )

    print(
        "\nAll surfaces completed.\n"
    )


if __name__ == "__main__":
    main()