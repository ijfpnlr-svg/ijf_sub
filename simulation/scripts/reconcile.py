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
from score_functions import compute_es
from simulation.scripts.score_functions import compute_crps_new


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

def f_surface(surface, B, axis=0):
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


def f_surface_jax(surface, B, axis=0):
    B = jnp.asarray(B)

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

def _to_precision(cov, eps=1e-6):
    cov = 0.5 * (
        cov
        + cov.T
    )

    lam = (
        eps
        * np.trace(cov)
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
            np.diag(P)
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
            os.cpu_count() or 1,
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

            ols_dict[t] = result[
                "ols"
            ]

            full_dict[t] = result[
                "full"
            ]

            wls_dict[t] = result[
                "wls"
            ]

            print(
                f"finished projection task t={t}",
                flush=True,
            )

    ols_fc = np.stack(
        [
            ols_dict[t]
            for t in range(T)
        ],
        axis=1,
    )

    full_fc = np.stack(
        [
            full_dict[t]
            for t in range(T)
        ],
        axis=1,
    )

    wls_fc = np.stack(
        [
            wls_dict[t]
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

def pbu(B, surface):
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
# PER-TIME-STEP LOSS FUNCTIONS FOR DM TESTS
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
    """
    Compute one energy-score loss per forecast time step.

    observations shape:

        (n_variables, T)

    forecast_samples shape:

        (n_variables, T, S)
    """
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

    pairwise_abs_mean = (
        2.0
        * np.sum(
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
        - 0.5
        * pairwise_abs_mean
    )


def compute_crps_loss_by_time(
    observations,
    forecast_samples,
):
    """
    Compute one CRPS loss per forecast time step.

    At each time step, CRPS is averaged across all variables.
    """
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


def append_loss_rows_for_case(
    loss_rows,
    dimension,
    surface,
    forecasts,
    observations,
):
    """
    Append per-time-step ES and CRPS losses.

    These rows are later used by a separate Diebold-Mariano script.
    """
    for method, arr in forecasts.items():
        energy_losses = compute_energy_loss_by_time(
            observations=observations,
            forecast_samples=arr,
        )

        crps_losses = compute_crps_loss_by_time(
            observations=observations,
            forecast_samples=arr,
        )

        for t, loss_value in enumerate(
            energy_losses
        ):
            loss_rows.append(
                {
                    "dimension": dimension,
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
                    "surface": surface,
                    "method": method,
                    "score": "crps",
                    "t": t,
                    "loss": loss_value,
                }
            )


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


# ============================================================
# SCORE REPORTING
# ============================================================

def save_score_checkpoint(
    score_rows,
    score_file,
):
    score_df = pd.DataFrame(
        score_rows
    )

    score_df = score_df.sort_values(
        [
            "dimension",
            "surface",
            "method",
        ]
    )

    score_df.to_csv(
        score_file,
        index=False,
    )


def print_final_score_summary(score_rows):
    if not score_rows:
        print(
            "No scores were computed."
        )
        return

    score_df = pd.DataFrame(
        score_rows
    )

    method_order = [
        "base",
        "pbu",
        "ols",
        "wls",
        "full",
        "ukf",
    ]

    score_df["method"] = pd.Categorical(
        score_df["method"],
        categories=method_order,
        ordered=True,
    )

    score_df = score_df.sort_values(
        [
            "dimension",
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
                "energy_score": (
                    lambda x: f"{x:.6f}"
                ),
                "energy_relative": (
                    lambda x: f"{x:.4f}"
                ),
                "crps": (
                    lambda x: f"{x:.6f}"
                ),
                "crps_relative": (
                    lambda x: f"{x:.4f}"
                ),
            },
        )
    )

    es_pivot = score_df.pivot(
        index=[
            "dimension",
            "surface",
        ],
        columns="method",
        values="energy_relative",
    )

    es_pivot = es_pivot.reindex(
        columns=[
            method
            for method in method_order
            if method in es_pivot.columns
        ]
    )

    print()
    print("=" * 100)
    print("RELATIVE ENERGY SCORE")
    print("Base = 1.0; lower is better")
    print("=" * 100)
    print()

    print(
        es_pivot.to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )

    crps_pivot = score_df.pivot(
        index=[
            "dimension",
            "surface",
        ],
        columns="method",
        values="crps_relative",
    )

    crps_pivot = crps_pivot.reindex(
        columns=[
            method
            for method in method_order
            if method in crps_pivot.columns
        ]
    )

    print()
    print("=" * 100)
    print("RELATIVE CRPS")
    print("Base = 1.0; lower is better")
    print("=" * 100)
    print()

    print(
        crps_pivot.to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )


# ============================================================
# MAIN LOOP
# ============================================================

def main():
    fc_folder = "../forecasts"
    results_folder = "../results"

    os.makedirs(
        results_folder,
        exist_ok=True,
    )

    score_file = os.path.join(
        results_folder,
        "reconciliation_scores.csv",
    )

    loss_file = os.path.join(
        results_folder,
        "reconciliation_losses_by_time.csv",
    )

    dimensions = [
        2,
        10,
        20,
        50,
        100
    ]

    surfaces = [
        "saddle",
        "ripples",
        "paraboloid",
    ]

    n_samples = 2000

    projection_workers = 4

    score_rows = []
    loss_rows = []

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

        for surface in surfaces:
            print()
            print(
                "==================================================="
            )
            print(
                f"Running reconciliation for "
                f"surface={surface}, d={dimension}"
            )
            print(
                "==================================================="
            )
            print()

            base_path = os.path.join(
                fc_folder,
                f"base_fc_{surface}_"
                f"d{dimension}_"
                f"{n_samples}.pkl",
            )

            res_path = os.path.join(
                fc_folder,
                f"residuals_{surface}_"
                f"d{dimension}_"
                f"{n_samples}.pkl",
            )

            te_path = os.path.join(
                fc_folder,
                f"test_data_{surface}_"
                f"d{dimension}_"
                f"{n_samples}.pkl",
            )

            base_fc = pd.read_pickle(
                base_path
            )

            base_fc = np.asarray(
                base_fc,
                dtype=np.float64,
            )

            T = base_fc.shape[1]
            S = base_fc.shape[2]

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
                dimension + 1
            )

            if base_fc.shape[0] != expected_variables:
                raise ValueError(
                    f"Expected {expected_variables} forecast variables "
                    f"for d={dimension}, got shape {base_fc.shape}"
                )

            if tr_res.shape[0] != expected_variables:
                raise ValueError(
                    f"Expected {expected_variables} residual variables "
                    f"for d={dimension}, got shape {tr_res.shape}"
                )

            gt_test = df_te.iloc[
                :-1
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
                f"Base forecast shape: "
                f"{base_fc.shape}"
            )

            print(
                f"Residual shape: "
                f"{tr_res.shape}"
            )

            print(
                f"Ground-truth shape: "
                f"{gt_test.shape}"
            )

            print()
            print("BU started.")

            bot_base = base_fc[
                1:,
                :,
                :,
            ]

            bu_fc = pbu(
                B=bot_base,
                surface=surface,
            )

            print("BU completed.")

            print()
            print("Projection started.")

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

            print("Projection completed.")

            print()
            print("UKF started.")

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
            )["shrink_cov"][0, 0]

            ukf_dict = {}

            for t in range(T):
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

                ukf_dict[t] = np.vstack(
                    [
                        Urec,
                        Brec,
                    ]
                )

            ukf_fc = np.stack(
                [
                    ukf_dict[t]
                    for t in range(T)
                ],
                axis=1,
            )

            print("UKF completed.")

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
                es_scores[key] = compute_es(
                    gt_test.T,
                    arr,
                )

            base_es = es_scores[
                "base"
            ]

            crps_scores = {}

            for key, arr in forecasts.items():
                crps_scores[key] = compute_crps_new(
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
                    energy_score / base_es
                    if base_es != 0
                    else np.nan
                )

                crps_relative = (
                    crps / base_crps
                    if base_crps != 0
                    else np.nan
                )

                score_rows.append(
                    {
                        "dimension": dimension,
                        "surface": surface,
                        "method": method,
                        "energy_score": energy_score,
                        "energy_relative": energy_relative,
                        "crps": crps,
                        "crps_relative": crps_relative,
                    }
                )

            print()
            print("Saving per-time-step losses for DM tests.")

            append_loss_rows_for_case(
                loss_rows=loss_rows,
                dimension=dimension,
                surface=surface,
                forecasts=forecasts,
                observations=gt_test.T,
            )

            print()
            print(
                f"Scores for {surface}, d={dimension}"
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
                score_file=score_file,
            )

            save_loss_checkpoint(
                loss_rows=loss_rows,
                loss_file=loss_file,
            )

            print()
            print(
                f"Score checkpoint saved to: "
                f"{score_file}"
            )

            print(
                f"Loss checkpoint saved to: "
                f"{loss_file}"
            )

    print()
    print(
        "All dimensions and surfaces completed."
    )

    print_final_score_summary(
        score_rows
    )

    save_loss_checkpoint(
        loss_rows=loss_rows,
        loss_file=loss_file,
    )

    print()
    print(
        f"Final score file: {score_file}"
    )

    print(
        f"Final loss file: {loss_file}"
    )

    print()


if __name__ == "__main__":
    main()