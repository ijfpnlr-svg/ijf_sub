import os
import pickle
import argparse
import time

import numpy as np
import pandas as pd
import jax.numpy as jnp

from scipy.linalg import block_diag
from concurrent.futures import ProcessPoolExecutor, as_completed

from reconc.reconc_nl_buis import reconc_nl_buis
from reconc.reconc_nl_ukf import reconc_nl_ukf, _schafer_strimmer_cov
from reconc.reconc_nl_ols import reconc_nl_ols
from simulation.scripts.score_functions import compute_crps


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
# Precision matrix utility
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
# Projection constraint
# ============================================================

def make_f_ols(
    U,
    no_ind_pos,
):
    no_ind_pos = np.asarray(
        no_ind_pos
    )

    expected_dim = (
        3 * U
        + 2
    )

    def f_ols(z):
        z = jnp.asarray(
            z
        )

        if z.ndim != 1:
            raise ValueError(
                f"f_ols expects a 1D vector, got shape {z.shape}"
            )

        if z.shape[0] != expected_dim:
            raise ValueError(
                f"f_ols expects length {expected_dim}, got {z.shape[0]}"
            )

        top = z[
            0
        ]

        ratio_mid = z[
            1:U
        ]

        total_target = z[
            U
        ]

        pop_total = z[
            U + 1
        ]

        target_bottom = z[
            U + 2: 2 * U + 2
        ]

        pop_bottom = z[
            2 * U + 2: 3 * U + 2
        ]

        eps = 1e-8

        c_top = (
            top
            - total_target
            / (
                pop_total
                + eps
            )
        )

        c_mid = (
            ratio_mid
            - (
                jnp.delete(
                    target_bottom,
                    no_ind_pos,
                )
                / (
                    jnp.delete(
                        pop_bottom,
                        no_ind_pos,
                    )
                    + eps
                )
            )
        )

        c_total_target = (
            total_target
            - jnp.sum(
                target_bottom
            )
        )

        c_total_pop = (
            pop_total
            - jnp.sum(
                pop_bottom
            )
        )

        return jnp.concatenate(
            [
                jnp.array(
                    [
                        c_top
                    ]
                ),
                c_mid,
                jnp.array(
                    [
                        c_total_target,
                        c_total_pop,
                    ]
                ),
            ]
        )

    return f_ols


# ============================================================
# Parallel projection
# ============================================================

def _run_single_nl_ols_task(task):
    (
        i,
        ratio_top,
        ratio_mid,
        total_target,
        pop_total,
        bot_target,
        pop_bot,
        ratio_top_res,
        ratio_mid_res,
        total_target_res,
        pop_total_res,
        bot_target_res,
        pop_bot_res,
        U,
        no_ind_pos,
        n_iter,
        seed,
    ) = task

    f_ols = make_f_ols(
        U,
        no_ind_pos,
    )

    W = _schafer_strimmer_cov(
        np.concatenate(
            [
                ratio_top_res[
                    :,
                    i,
                    :,
                ].T,

                ratio_mid_res[
                    :,
                    i,
                    :,
                ].T,

                total_target_res[
                    :,
                    i,
                    :,
                ].T,

                pop_total_res[
                    :,
                    i,
                    :,
                ].T,

                bot_target_res[
                    :,
                    i,
                    :,
                ].T,

                pop_bot_res[
                    :,
                    i,
                    :,
                ].T,
            ],
            axis=1,
        )
    )[
        "shrink_cov"
    ]

    P = _to_precision(
        W
    )

    Z = np.vstack(
        [
            ratio_top[
                :,
                i,
                :,
            ],

            ratio_mid[
                :,
                i,
                :,
            ],

            total_target[
                :,
                i,
                :,
            ],

            pop_total[
                :,
                i,
                :,
            ],

            bot_target[
                :,
                i,
                :,
            ],

            pop_bot[
                :,
                i,
                :,
            ],
        ]
    ).T

    D = np.diag(
        np.diag(
            P
        )
    )

    B = block_diag(
        D[
            :54,
            :54,
        ],
        P[
            54:,
            54:,
        ],
    )

    res_ols = reconc_nl_ols(
        Z,
        f_ols,
        n_iter=n_iter,
        seed=seed,
    )

    sync_any(
        res_ols
    )

    res_wls = reconc_nl_ols(
        Z,
        f_ols,
        W=D,
        n_iter=n_iter,
        seed=seed,
    )

    sync_any(
        res_wls
    )

    t0_full = time.perf_counter()

    res_full = reconc_nl_ols(
        Z,
        f_ols,
        W=P,
        n_iter=n_iter,
        seed=seed,
    )

    sync_any(
        res_full
    )

    full_time = (
        time.perf_counter()
        - t0_full
    )

    res_block = reconc_nl_ols(
        Z,
        f_ols,
        W=B,
        n_iter=n_iter,
        seed=seed,
    )

    sync_any(
        res_block
    )

    return (
        i,
        res_ols[
            "reconciled_samples"
        ],
        res_wls[
            "reconciled_samples"
        ],
        res_full[
            "reconciled_samples"
        ],
        res_block[
            "reconciled_samples"
        ],
        full_time,
    )


def run_parallel_nl_ols_block(
    ratio_top,
    ratio_mid,
    total_target,
    pop_total,
    bot_target,
    pop_bot,
    ratio_top_res,
    ratio_mid_res,
    total_target_res,
    pop_total_res,
    bot_target_res,
    pop_bot_res,
    U,
    no_ind_pos,
    T,
    n_iter=25,
    seed=42,
    max_workers=None,
):
    tasks = [
        (
            i,
            ratio_top,
            ratio_mid,
            total_target,
            pop_total,
            bot_target,
            pop_bot,
            ratio_top_res,
            ratio_mid_res,
            total_target_res,
            pop_total_res,
            bot_target_res,
            pop_bot_res,
            U,
            no_ind_pos,
            n_iter,
            seed,
        )
        for i in range(
            T
        )
    ]

    ols = {}
    wls = {}
    full = {}
    block = {}

    full_time_total = 0.0

    with ProcessPoolExecutor(
        max_workers=max_workers
    ) as executor:

        futures = [
            executor.submit(
                _run_single_nl_ols_task,
                task,
            )
            for task in tasks
        ]

        for future in as_completed(
            futures
        ):
            (
                i,
                res_ols,
                res_wls,
                res_full,
                res_block,
                full_time,
            ) = future.result()

            ols[
                i
            ] = res_ols

            wls[
                i
            ] = res_wls

            full[
                i
            ] = res_full

            block[
                i
            ] = res_block

            full_time_total += full_time

    ols_arr = np.stack(
        [
            ols[
                t
            ]
            for t in range(
                T
            )
        ],
        axis=1,
    )

    wls_arr = np.stack(
        [
            wls[
                t
            ]
            for t in range(
                T
            )
        ],
        axis=1,
    )

    full_arr = np.stack(
        [
            full[
                t
            ]
            for t in range(
                T
            )
        ],
        axis=1,
    )

    block_arr = np.stack(
        [
            block[
                t
            ]
            for t in range(
                T
            )
        ],
        axis=1,
    )

    return (
        ols_arr,
        wls_arr,
        full_arr,
        block_arr,
        full_time_total,
    )


# ============================================================
# Hierarchy functions
# ============================================================

def f_upper_from_bottom(
    Bns,
    ref_id,
):
    """
    Vectorized map from bottom series to upper series.

    Bottom order:

        [target_bottom(U), population_bottom(U)]

    Output order:

        [top_ratio, middle_ratios_without_no_indication, target_total, population_total]
    """
    U = (
        Bns.shape[
            1
        ]
        // 2
    )

    target = Bns[
        :,
        :U,
    ]

    pop = Bns[
        :,
        U:,
    ]

    total_target = np.sum(
        target,
        axis=1,
        keepdims=True,
    )

    total_pop = np.sum(
        pop,
        axis=1,
        keepdims=True,
    )

    no_ind_pos = np.where(
        np.array(
            ref_id
        )
        == "No indication"
    )[0].tolist()

    mid_ratio = (
        np.delete(
            target,
            no_ind_pos,
            axis=1,
        )
        / (
            np.delete(
                pop,
                no_ind_pos,
                axis=1,
            )
            + 1e-8
        )
    )

    top = (
        total_target
        / (
            total_pop
            + 1e-8
        )
    )

    upper = np.concatenate(
        [
            top,
            mid_ratio,
            total_target,
            total_pop,
        ],
        axis=1,
    )

    return upper.T


def f_upper_from_bottom_single(
    z,
    no_ind_pos,
):
    U = (
        z.shape[
            0
        ]
        // 2
    )

    target = z[
        :U
    ]

    pop = z[
        U:
    ]

    total_target = np.atleast_1d(
        np.sum(
            target
        )
    )

    total_pop = np.atleast_1d(
        np.sum(
            pop
        )
    )

    mid_ratio = (
        np.delete(
            target,
            no_ind_pos,
        )
        / (
            np.delete(
                pop,
                no_ind_pos,
            )
            + 1e-8
        )
    )

    top = np.atleast_1d(
        np.sum(
            target
        )
        / (
            np.sum(
                pop
            )
            + 1e-8
        )
    )

    return np.concatenate(
        [
            top,
            mid_ratio,
            total_target,
            total_pop,
        ]
    )


def pbu_block(
    target_bot,
    pop_bot,
    no_ind_pos,
):
    U, T, M = pop_bot.shape

    total_target = np.sum(
        target_bot,
        axis=0,
    ).reshape(
        (
            1,
            T,
            M,
        )
    )

    total_pop = np.sum(
        pop_bot,
        axis=0,
    ).reshape(
        (
            1,
            T,
            M,
        )
    )

    mid_ratio = (
        np.delete(
            target_bot,
            no_ind_pos,
            axis=0,
        )
        / (
            np.delete(
                pop_bot,
                no_ind_pos,
                axis=0,
            )
            + 1e-8
        )
    )

    top = (
        total_target
        / (
            total_pop
            + 1e-8
        )
    )

    return np.concatenate(
        [
            top,
            mid_ratio,
            total_target,
            total_pop,
            target_bot,
            pop_bot,
        ],
        axis=0,
    )


# ============================================================
# Score functions
# ============================================================

def compute_es(
    y_true,
    y_samples,
):
    """
    Mean multivariate Energy Score across time.
    """
    n_series, n_splits, n_samples = y_samples.shape

    es_total = 0.0

    for t in range(
        n_splits
    ):
        x = y_samples[
            :,
            t,
            :,
        ].T

        y = y_true[
            :,
            t,
        ]

        term_1 = np.mean(
            np.linalg.norm(
                x
                - y[
                    None,
                    :
                ],
                axis=1,
            )
        )

        term_2 = 0.5 * np.mean(
            np.linalg.norm(
                x[
                    :,
                    None,
                    :,
                ]
                - x[
                    None,
                    :,
                    :,
                ],
                axis=2,
            )
        )

        es_total += (
            term_1
            - term_2
        )

    return (
        es_total
        / n_splits
    )


def compute_es_weighted(
    y_true,
    y_samples,
    w_rows,
):
    R, T, M = y_samples.shape

    w = np.asarray(
        w_rows,
        dtype=float,
    )

    sqrt_w = np.sqrt(
        w
    )

    es_total = 0.0

    for t in range(
        T
    ):
        X = y_samples[
            :,
            t,
            :,
        ].T

        y = y_true[
            :,
            t,
        ]

        Xw = (
            X
            * sqrt_w
        )

        yw = (
            y
            * sqrt_w
        )

        d1 = np.mean(
            np.linalg.norm(
                Xw
                - yw,
                axis=1,
            )
        )

        d2 = 0.5 * np.mean(
            np.linalg.norm(
                Xw[
                    :,
                    None,
                    :,
                ]
                - Xw[
                    None,
                    :,
                    :,
                ],
                axis=2,
            )
        )

        es_total += (
            d1
            - d2
        )

    return (
        es_total
        / T
    )


# ============================================================
# Loss export utilities for DM / MCS
# ============================================================

def _mean_pairwise_euclidean_distance(
    samples,
    chunk_size=256,
):
    samples = np.asarray(
        samples,
        dtype=float,
    )

    n_samples = samples.shape[
        0
    ]

    if n_samples == 0:
        return np.nan

    total = 0.0
    count = 0

    for start in range(
        0,
        n_samples,
        chunk_size,
    ):
        end = min(
            start
            + chunk_size,
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

    return (
        total
        / count
    )


def compute_energy_loss_by_time(
    y_true,
    y_samples,
    rows_idx=None,
):
    """
    Compute one multivariate Energy Score loss per forecast time step.
    """
    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_samples = np.asarray(
        y_samples,
        dtype=float,
    )

    if y_samples.ndim != 3:
        raise ValueError(
            f"y_samples must have shape (R,T,M), got {y_samples.shape}"
        )

    R, T, M = y_samples.shape

    if y_true.shape != (
        R,
        T,
    ):
        raise ValueError(
            f"y_true must have shape {(R, T)}, got {y_true.shape}"
        )

    if rows_idx is None:
        rows = np.arange(
            R
        )
    else:
        rows = np.asarray(
            rows_idx,
            dtype=int,
        )

    losses = np.full(
        T,
        np.nan,
        dtype=float,
    )

    for t in range(
        T
    ):
        X = y_samples[
            rows,
            t,
            :,
        ].T

        y = y_true[
            rows,
            t,
        ]

        term_1 = np.mean(
            np.linalg.norm(
                X
                - y[
                    None,
                    :
                ],
                axis=1,
            )
        )

        term_2 = _mean_pairwise_euclidean_distance(
            X
        )

        losses[
            t
        ] = (
            term_1
            - 0.5
            * term_2
        )

    return losses


def compute_crps_loss_by_time(
    y_true,
    y_samples,
    rows_idx=None,
):
    """
    Compute one average CRPS loss per time step.

    CRPS is averaged over the selected rows.
    """
    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_samples = np.asarray(
        y_samples,
        dtype=float,
    )

    if y_samples.ndim != 3:
        raise ValueError(
            f"y_samples must have shape (R,T,M), got {y_samples.shape}"
        )

    R, T, M = y_samples.shape

    if y_true.shape != (
        R,
        T,
    ):
        raise ValueError(
            f"y_true must have shape {(R, T)}, got {y_true.shape}"
        )

    if rows_idx is None:
        rows = np.arange(
            R
        )
    else:
        rows = np.asarray(
            rows_idx,
            dtype=int,
        )

    losses = np.full(
        T,
        np.nan,
        dtype=float,
    )

    for t in range(
        T
    ):
        crps_rows = compute_crps(
            y_true[
                rows,
                t,
            ],
            y_samples[
                rows,
                t,
                :,
            ],
        )

        losses[
            t
        ] = np.nanmean(
            crps_rows
        )

    return losses


def compute_crps_loss_by_series_and_time(
    y_true,
    y_samples,
    rows_idx=None,
):
    """
    Compute CRPS for each selected series and each forecast time step.

    Output:

        losses: shape (n_selected_series, T)
        rows: original row indices

    This is needed for the paper metric:

        RelCRPS_m
        =
        exp(mean_j log(mean_t CRPS_{m,j,t} / mean_t CRPS_{base,j,t}))
    """
    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_samples = np.asarray(
        y_samples,
        dtype=float,
    )

    if y_samples.ndim != 3:
        raise ValueError(
            f"y_samples must have shape (R,T,M), got {y_samples.shape}"
        )

    R, T, M = y_samples.shape

    if y_true.shape != (
        R,
        T,
    ):
        raise ValueError(
            f"y_true must have shape {(R, T)}, got {y_true.shape}"
        )

    if rows_idx is None:
        rows = np.arange(
            R
        )
    else:
        rows = np.asarray(
            rows_idx,
            dtype=int,
        )

    losses = np.full(
        (
            len(
                rows
            ),
            T,
        ),
        np.nan,
        dtype=float,
    )

    for t in range(
        T
    ):
        crps_rows = compute_crps(
            y_true[
                rows,
                t,
            ],
            y_samples[
                rows,
                t,
                :,
            ],
        )

        losses[
            :,
            t,
        ] = np.asarray(
            crps_rows,
            dtype=float,
        )

    return (
        losses,
        rows,
    )


def append_dm_loss_rows(
    loss_rows,
    target,
    forecast_methods,
    ground_truth,
    levels,
    crps_series_rows=None,
):
    """
    Append losses for later DM/MCS-style tests.

    Main file:

        target, level, score, method, t, loss

    Optional per-series CRPS file:

        target, level, score, method, series_index, t, loss
    """
    for level_name, rows_idx in levels.items():
        for method, y_hat in forecast_methods.items():
            crps_losses = compute_crps_loss_by_time(
                y_true=ground_truth,
                y_samples=y_hat,
                rows_idx=rows_idx,
            )

            es_losses = compute_energy_loss_by_time(
                y_true=ground_truth,
                y_samples=y_hat,
                rows_idx=rows_idx,
            )

            for t, loss_value in enumerate(
                crps_losses
            ):
                loss_rows.append(
                    {
                        "target": target,
                        "level": level_name,
                        "score": "crps",
                        "method": method,
                        "t": t,
                        "loss": loss_value,
                    }
                )

            for t, loss_value in enumerate(
                es_losses
            ):
                loss_rows.append(
                    {
                        "target": target,
                        "level": level_name,
                        "score": "energy_score",
                        "method": method,
                        "t": t,
                        "loss": loss_value,
                    }
                )

            if crps_series_rows is not None:
                (
                    crps_series_losses,
                    selected_rows,
                ) = compute_crps_loss_by_series_and_time(
                    y_true=ground_truth,
                    y_samples=y_hat,
                    rows_idx=rows_idx,
                )

                for row_pos, series_index in enumerate(
                    selected_rows
                ):
                    for t in range(
                        crps_series_losses.shape[
                            1
                        ]
                    ):
                        crps_series_rows.append(
                            {
                                "target": target,
                                "level": level_name,
                                "score": "crps",
                                "method": method,
                                "series_index": int(
                                    series_index
                                ),
                                "t": t,
                                "loss": crps_series_losses[
                                    row_pos,
                                    t,
                                ],
                            }
                        )


def save_dm_loss_rows(
    loss_rows,
    loss_file,
):
    if not loss_rows:
        print(
            "No DM losses to save."
        )
        return

    loss_df = pd.DataFrame(
        loss_rows
    )

    loss_df = loss_df.sort_values(
        [
            "target",
            "level",
            "score",
            "method",
            "t",
        ]
    )

    loss_df.to_csv(
        loss_file,
        index=False,
    )


def save_crps_series_loss_rows(
    crps_series_rows,
    crps_series_loss_file,
):
    if not crps_series_rows:
        print(
            "No per-series CRPS losses to save."
        )
        return

    crps_series_df = pd.DataFrame(
        crps_series_rows
    )

    crps_series_df = crps_series_df.sort_values(
        [
            "target",
            "level",
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


# ============================================================
# Reporting utilities
# ============================================================

def compute_crps_over_level(
    y_true,
    y_samples,
    rows_idx,
    average_within_level=True,
):
    rows = np.asarray(
        rows_idx,
        dtype=int,
    )

    T = y_true.shape[
        1
    ]

    per_time = []

    for t in range(
        T
    ):
        crps_rows = compute_crps(
            y_true[
                rows,
                t,
            ],
            y_samples[
                rows,
                t,
                :,
            ],
        )

        if average_within_level:
            per_time.append(
                np.nanmean(
                    crps_rows
                )
            )

        else:
            per_time.append(
                crps_rows
            )

    per_time = np.array(
        per_time
    )

    if average_within_level:
        return float(
            np.nanmean(
                per_time
            )
        )

    return per_time.T


def crps_table_and_relative(
    forecast_methods,
    gt,
    U,
    show_levels=None,
):
    if show_levels is None:
        show_levels = {
            "full": np.arange(
                gt.shape[
                    0
                ]
            ),
            "top": [
                0
            ],
            "middle_ratios": list(
                range(
                    1,
                    U,
                )
            ),
            "swiss_totals": list(
                range(
                    U,
                    U + 2,
                )
            ),
            "bottom": list(
                range(
                    U + 2,
                    gt.shape[
                        0
                    ],
                )
            ),
        }

    abs_crps = {}

    for name, y_hat in forecast_methods.items():
        level_scores = {}

        for level_name, indices in show_levels.items():
            level_scores[
                level_name
            ] = compute_crps_over_level(
                gt,
                y_hat,
                indices,
                average_within_level=True,
            )

        abs_crps[
            name
        ] = level_scores

    base_scores = abs_crps.get(
        "Base",
        next(
            iter(
                abs_crps.values()
            )
        ),
    )

    rel_crps = {}

    for name, scores in abs_crps.items():
        rel = {}

        for level_name, value in scores.items():
            base_value = base_scores.get(
                level_name,
                np.nan,
            )

            rel[
                level_name
            ] = (
                value
                / base_value
                if np.isfinite(
                    base_value
                )
                and base_value != 0
                else np.nan
            )

        rel_crps[
            name
        ] = rel

    print(
        f"{'Method':<12s} | "
        f"{'Full':>10s} | "
        f"{'Top':>10s} | "
        f"{'Ratios':>10s} | "
        f"{'Totals':>10s} | "
        f"{'Bottom':>10s}"
    )

    print(
        "-" * 68
    )

    for name in abs_crps:
        absolute = abs_crps[
            name
        ]

        relative = rel_crps[
            name
        ]

        print(
            f"{name:<12s} | "
            f"{absolute['full']:.4g} ({relative['full']:.3f}x) | "
            f"{absolute['top']:.4g} ({relative['top']:.3f}x) | "
            f"{absolute['middle_ratios']:.4g} ({relative['middle_ratios']:.3f}x) | "
            f"{absolute['swiss_totals']:.4g} ({relative['swiss_totals']:.3f}x) | "
            f"{absolute['bottom']:.4g} ({relative['bottom']:.3f}x)"
        )


def crps_relative_geomean_over_series(
    forecast_methods,
    gt,
    baseline_name="Base",
    eps=1e-12,
):
    if baseline_name in forecast_methods:
        base_key = baseline_name

    else:
        base_key = next(
            iter(
                forecast_methods.keys()
            )
        )

    R, T = gt.shape

    per_series_avg = {}

    for name, y_hat in forecast_methods.items():
        if y_hat.shape[
            :2
        ] != (
            R,
            T,
        ):
            raise ValueError(
                f"{name}: expected shape (R,T,M) with (R,T)=({R},{T}), got {y_hat.shape}"
            )

        avg_j = np.full(
            R,
            np.nan,
            dtype=float,
        )

        for j in range(
            R
        ):
            crps_t = []

            for t in range(
                T
            ):
                value = compute_crps(
                    gt[
                        j:j + 1,
                        t,
                    ],
                    y_hat[
                        j:j + 1,
                        t,
                        :,
                    ],
                )[0]

                crps_t.append(
                    value
                )

            avg_j[
                j
            ] = np.nanmean(
                crps_t
            )

        per_series_avg[
            name
        ] = avg_j

    base_avg = per_series_avg[
        base_key
    ]

    base_safe = np.where(
        np.isfinite(
            base_avg
        )
        & (
            base_avg
            > 0
        ),
        base_avg,
        np.nan,
    )

    per_series_ratio = {}
    gm_ratio = {}

    for name, avg_j in per_series_avg.items():
        ratio = (
            avg_j
            / base_safe
        )

        ratio = np.where(
            np.isfinite(
                ratio
            ),
            np.maximum(
                ratio,
                eps,
            ),
            np.nan,
        )

        per_series_ratio[
            name
        ] = ratio

        valid = np.isfinite(
            ratio
        )

        if np.any(
            valid
        ):
            gm_ratio[
                name
            ] = float(
                np.exp(
                    np.nanmean(
                        np.log(
                            ratio[
                                valid
                            ]
                        )
                    )
                )
            )

        else:
            gm_ratio[
                name
            ] = np.nan

    return (
        per_series_avg,
        per_series_ratio,
        gm_ratio,
    )


def crps_gm_table(
    forecast_methods,
    gt,
    baseline_name="Base",
):
    (
        per_series_avg,
        per_series_ratio,
        gm_ratio,
    ) = crps_relative_geomean_over_series(
        forecast_methods,
        gt,
        baseline_name=baseline_name,
    )

    if baseline_name in forecast_methods:
        base_key = baseline_name

    else:
        base_key = next(
            iter(
                forecast_methods.keys()
            )
        )

    print(
        f"Baseline for ratios: {base_key}"
    )

    print(
        f"{'Method':<12s} | {'GM(CRPS/BASE)':>14s}"
    )

    print(
        "-" * 30
    )

    for name in forecast_methods.keys():
        value = gm_ratio.get(
            name,
            np.nan,
        )

        if np.isfinite(
            value
        ):
            print(
                f"{name:<12s} | {value:>14.3f}x"
            )

        else:
            print(
                f"{name:<12s} | {'nan':>14s}"
            )

    return (
        per_series_avg,
        per_series_ratio,
        gm_ratio,
    )


def extract_relative_scores_for_middle_level(
    forecast_methods,
    ground_truth,
    U,
    base_key="Base",
    pbu_key="PBU",
):
    middle_idx = list(
        range(
            1,
            U,
        )
    )

    y_true = ground_truth[
        middle_idx,
        :,
    ]

    def middle_view(x):
        return x[
            middle_idx,
            :,
            :,
        ]

    base_score = np.mean(
        [
            compute_crps(
                y_true[
                    :,
                    t,
                ],
                middle_view(
                    forecast_methods[
                        base_key
                    ]
                )[
                    :,
                    t,
                    :,
                ],
            )
            for t in range(
                y_true.shape[
                    1
                ]
            )
        ]
    )

    pbu_score = np.mean(
        [
            compute_crps(
                y_true[
                    :,
                    t,
                ],
                middle_view(
                    forecast_methods[
                        pbu_key
                    ]
                )[
                    :,
                    t,
                    :,
                ],
            )
            for t in range(
                y_true.shape[
                    1
                ]
            )
        ]
    )

    out = {}

    for name, yhat in forecast_methods.items():
        if name not in [
            base_key,
            pbu_key,
        ]:
            model_score = np.mean(
                [
                    compute_crps(
                        y_true[
                            :,
                            t,
                        ],
                        middle_view(
                            yhat
                        )[
                            :,
                            t,
                            :,
                        ],
                    )
                    for t in range(
                        y_true.shape[
                            1
                        ]
                    )
                ]
            )

            out[
                name
            ] = {
                "rel_to_base": (
                    model_score
                    / base_score
                    if base_score
                    else np.nan
                ),
                "rel_to_pbu": (
                    model_score
                    / pbu_score
                    if pbu_score
                    else np.nan
                ),
            }

    return out


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base_pkl",
        default="../forecasts/fc_imm_cit_autoarima2_30.pkl",
    )

    parser.add_argument(
        "--base_2_pkl",
        default="../forecasts/fc_imm_cit_autoarima_30.pkl",
    )

    parser.add_argument(
        "--train_pkl",
        default="../forecasts/train_data_new.pkl",
    )

    parser.add_argument(
        "--test_pkl",
        default="../forecasts/test_autoarima_30.pkl",
    )

    parser.add_argument(
        "--test_2_pkl",
        default="../forecasts/test_autoarima2_30.pkl",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--iters",
        type=int,
        default=20,
    )

    args = parser.parse_args()

    results_folder = "../results"

    os.makedirs(
        results_folder,
        exist_ok=True,
    )

    loss_file = os.path.join(
        results_folder,
        "swiss_demo_losses_by_time.csv",
    )

    crps_series_loss_file = os.path.join(
        results_folder,
        "swiss_demo_crps_by_series_time.csv",
    )

    with open(
        args.base_2_pkl,
        "rb",
    ) as file:
        base_2 = pickle.load(
            file
        )

    with open(
        args.base_pkl,
        "rb",
    ) as file:
        base = pickle.load(
            file
        )

    with open(
        args.train_pkl,
        "rb",
    ) as file:
        train = pickle.load(
            file
        )

    with open(
        args.test_pkl,
        "rb",
    ) as file:
        test_data = pickle.load(
            file
        )

    with open(
        args.test_2_pkl,
        "rb",
    ) as file:
        test_data_2 = pickle.load(
            file
        )

    def pick_block(
        block_key,
        base_dict=base,
    ):
        samples = base_dict[
            block_key
        ][
            "samples"
        ]

        residuals = base_dict[
            block_key
        ][
            "residuals"
        ]

        return (
            samples,
            residuals,
        )

    pop_bot, pop_bot_res = pick_block(
        "population",
        base,
    )

    imm_bot, imm_bot_res = pick_block(
        "immigration",
        base,
    )

    cit_bot, cit_bot_res = pick_block(
        "citizenship",
        base,
    )

    U, T, M = pop_bot.shape

    ref_uids = base[
        "population"
    ][
        "uids"
    ]

    no_ind_pos = np.where(
        np.array(
            ref_uids
        )
        == "No indication"
    )[0].tolist()

    pop_total, pop_total_res = pick_block(
        "Switzerland_population",
        base_2,
    )

    imm_total, imm_total_res = pick_block(
        "Switzerland_immigration",
        base_2,
    )

    cit_total, cit_total_res = pick_block(
        "Switzerland_citizenship",
        base_2,
    )

    imm_ratio, imm_ratio_res = pick_block(
        "immigration_ratio",
        base,
    )

    cit_ratio, cit_ratio_res = pick_block(
        "citizenship_ratio",
        base,
    )

    rat_uids = base[
        "immigration_ratio"
    ][
        "uids"
    ]

    ch_uid_pos = np.where(
        np.array(
            rat_uids
        )
        == "Switzerland"
    )[0]

    imm_ratio_mid = np.delete(
        imm_ratio,
        ch_uid_pos,
        axis=0,
    )

    imm_ratio_mid_res = np.delete(
        imm_ratio_res,
        ch_uid_pos,
        axis=0,
    )

    imm_ratio_top = imm_ratio[
        ch_uid_pos,
        :,
        :,
    ]

    imm_ratio_top_res = imm_ratio_res[
        ch_uid_pos,
        :,
        :,
    ]

    cit_ratio_mid = np.delete(
        cit_ratio,
        ch_uid_pos,
        axis=0,
    )

    cit_ratio_mid_res = np.delete(
        cit_ratio_res,
        ch_uid_pos,
        axis=0,
    )

    cit_ratio_top = cit_ratio[
        ch_uid_pos,
        :,
        :,
    ]

    cit_ratio_top_res = cit_ratio_res[
        ch_uid_pos,
        :,
        :,
    ]

    timing_summary = {
        "immigration": {},
        "citizenship": {},
    }

    loss_rows = []
    crps_series_rows = []

    # ========================================================
    # 1) Probabilistic bottom-up
    # ========================================================

    pbu_imm = pbu_block(
        imm_bot,
        pop_bot,
        no_ind_pos,
    )

    pbu_cit = pbu_block(
        cit_bot,
        pop_bot,
        no_ind_pos,
    )

    print(
        "PBU immigration:",
        pbu_imm.shape,
        "PBU citizenship:",
        pbu_cit.shape,
    )

    # ========================================================
    # 2) NL-UKF
    # ========================================================

    print(
        "— Running NL-UKF…"
    )

    t0_ukf_imm = time.perf_counter()

    nlukf_imm = {}

    for t in range(
        T
    ):
        u_obs = np.mean(
            np.vstack(
                [
                    imm_ratio_top[
                        :,
                        t,
                        :,
                    ],
                    imm_ratio_mid[
                        :,
                        t,
                        :,
                    ],
                    imm_total[
                        :,
                        t,
                        :,
                    ],
                    pop_total[
                        :,
                        t,
                        :,
                    ],
                ]
            ),
            axis=1,
        )

        R = _schafer_strimmer_cov(
            np.vstack(
                [
                    imm_ratio_top_res[
                        :,
                        t,
                        :,
                    ],
                    imm_ratio_mid_res[
                        :,
                        t,
                        :,
                    ],
                    imm_total_res[
                        :,
                        t,
                        :,
                    ],
                    pop_total_res[
                        :,
                        t,
                        :,
                    ],
                ]
            ).T
        )[
            "shrink_cov"
        ]

        bot_list = []

        for k in range(
            U
        ):
            bot_list.append(
                {
                    "samples": imm_bot[
                        k,
                        t,
                        :,
                    ],
                    "residuals": imm_bot_res[
                        k,
                        t,
                        :,
                    ],
                }
            )

        for k in range(
            U
        ):
            bot_list.append(
                {
                    "samples": pop_bot[
                        k,
                        t,
                        :,
                    ],
                    "residuals": pop_bot_res[
                        k,
                        t,
                        :,
                    ],
                }
            )

        def f_single(z):
            return f_upper_from_bottom_single(
                z,
                no_ind_pos,
            )

        out = reconc_nl_ukf(
            bottom_base_forecasts=bot_list,
            in_type=[
                "samples"
            ]
            * (
                2
                * U
            ),
            distr=[
                "gaussian"
            ]
            * (
                2
                * U
            ),
            f=f_single,
            upper_base_forecasts=u_obs,
            R=R,
            num_samples=M,
            seed=args.seed,
        )

        sync_any(
            out
        )

        Brec = out[
            "bottom_reconciled_samples"
        ]

        Urec = f_upper_from_bottom(
            Brec.T,
            ref_id=ref_uids,
        )

        nlukf_imm[
            t
        ] = np.vstack(
            [
                Urec,
                Brec,
            ]
        )

    nlukf_imm = np.stack(
        [
            nlukf_imm[
                t
            ]
            for t in range(
                T
            )
        ],
        axis=1,
    )

    timing_summary[
        "immigration"
    ][
        "UKF"
    ] = (
        time.perf_counter()
        - t0_ukf_imm
    )

    print(
        "NL-UKF immigration:",
        nlukf_imm.shape,
    )

    t0_ukf_cit = time.perf_counter()

    nlukf_cit = {}

    for t in range(
        T
    ):
        u_obs = np.mean(
            np.vstack(
                [
                    cit_ratio_top[
                        :,
                        t,
                        :,
                    ],
                    cit_ratio_mid[
                        :,
                        t,
                        :,
                    ],
                    cit_total[
                        :,
                        t,
                        :,
                    ],
                    pop_total[
                        :,
                        t,
                        :,
                    ],
                ]
            ),
            axis=1,
        )

        R = _schafer_strimmer_cov(
            np.vstack(
                [
                    cit_ratio_top_res[
                        :,
                        t,
                        :,
                    ],
                    cit_ratio_mid_res[
                        :,
                        t,
                        :,
                    ],
                    cit_total_res[
                        :,
                        t,
                        :,
                    ],
                    pop_total_res[
                        :,
                        t,
                        :,
                    ],
                ]
            ).T
        )[
            "shrink_cov"
        ]

        bot_list = []

        for k in range(
            U
        ):
            bot_list.append(
                {
                    "samples": cit_bot[
                        k,
                        t,
                        :,
                    ],
                    "residuals": cit_bot_res[
                        k,
                        t,
                        :,
                    ],
                }
            )

        for k in range(
            U
        ):
            bot_list.append(
                {
                    "samples": pop_bot[
                        k,
                        t,
                        :,
                    ],
                    "residuals": pop_bot_res[
                        k,
                        t,
                        :,
                    ],
                }
            )

        def f_single(z):
            return f_upper_from_bottom_single(
                z,
                no_ind_pos,
            )

        out = reconc_nl_ukf(
            bottom_base_forecasts=bot_list,
            in_type=[
                "samples"
            ]
            * (
                2
                * U
            ),
            distr=[
                "gaussian"
            ]
            * (
                2
                * U
            ),
            f=f_single,
            upper_base_forecasts=u_obs,
            R=R,
            num_samples=M,
            seed=args.seed,
        )

        sync_any(
            out
        )

        Brec = out[
            "bottom_reconciled_samples"
        ]

        Urec = f_upper_from_bottom(
            Brec.T,
            ref_id=ref_uids,
        )

        nlukf_cit[
            t
        ] = np.vstack(
            [
                Urec,
                Brec,
            ]
        )

    nlukf_cit = np.stack(
        [
            nlukf_cit[
                t
            ]
            for t in range(
                T
            )
        ],
        axis=1,
    )

    timing_summary[
        "citizenship"
    ][
        "UKF"
    ] = (
        time.perf_counter()
        - t0_ukf_cit
    )

    print(
        "NL-UKF citizenship:",
        nlukf_cit.shape,
    )

    # ========================================================
    # 3) NL-OLS / WLS / FULL / BLOCK
    # ========================================================

    print(
        "— Running NL-OLS…"
    )

    ols_cit, wls_cit, full_cit, block_cit, full_time_cit = run_parallel_nl_ols_block(
        ratio_top=cit_ratio_top,
        ratio_mid=cit_ratio_mid,
        total_target=cit_total,
        pop_total=pop_total,
        bot_target=cit_bot,
        pop_bot=pop_bot,
        ratio_top_res=cit_ratio_top_res,
        ratio_mid_res=cit_ratio_mid_res,
        total_target_res=cit_total_res,
        pop_total_res=pop_total_res,
        bot_target_res=cit_bot_res,
        pop_bot_res=pop_bot_res,
        U=U,
        no_ind_pos=no_ind_pos,
        T=T,
        n_iter=args.iters,
        seed=args.seed,
        max_workers=None,
    )

    timing_summary[
        "citizenship"
    ][
        "full"
    ] = full_time_cit

    print(
        "OLS citizenship:",
        ols_cit.shape,
    )

    print(
        "WLS citizenship:",
        wls_cit.shape,
    )

    print(
        "FULL citizenship:",
        full_cit.shape,
    )

    print(
        "BLOCK citizenship:",
        block_cit.shape,
    )

    ols_imm, wls_imm, full_imm, block_imm, full_time_imm = run_parallel_nl_ols_block(
        ratio_top=imm_ratio_top,
        ratio_mid=imm_ratio_mid,
        total_target=imm_total,
        pop_total=pop_total,
        bot_target=imm_bot,
        pop_bot=pop_bot,
        ratio_top_res=imm_ratio_top_res,
        ratio_mid_res=imm_ratio_mid_res,
        total_target_res=imm_total_res,
        pop_total_res=pop_total_res,
        bot_target_res=imm_bot_res,
        pop_bot_res=pop_bot_res,
        U=U,
        no_ind_pos=no_ind_pos,
        T=T,
        n_iter=args.iters,
        seed=args.seed,
        max_workers=None,
    )

    timing_summary[
        "immigration"
    ][
        "full"
    ] = full_time_imm

    print(
        "OLS immigration:",
        ols_imm.shape,
    )

    print(
        "WLS immigration:",
        wls_imm.shape,
    )

    print(
        "FULL immigration:",
        full_imm.shape,
    )

    print(
        "BLOCK immigration:",
        block_imm.shape,
    )

    # ========================================================
    # 4) P + NL-BUIS
    # ========================================================

    print(
        "— Running IS…"
    )

    def f_buis(B):
        P = (
            B.shape[
                1
            ]
            // 2
        )

        target = B[
            :,
            :P,
        ]

        pop = B[
            :,
            P:,
        ]

        total_target = np.sum(
            target,
            axis=1,
            keepdims=True,
        )

        total_pop = np.sum(
            pop,
            axis=1,
            keepdims=True,
        )

        mid_ratio = (
            np.delete(
                target,
                no_ind_pos,
                axis=1,
            )
            / (
                np.delete(
                    pop,
                    no_ind_pos,
                    axis=1,
                )
                + 1e-8
            )
        )

        top = (
            total_target
            / (
                total_pop
                + 1e-8
            )
        )

        upper = np.concatenate(
            [
                top,
                mid_ratio,
                total_target,
                total_pop,
            ],
            axis=1,
        )

        return upper.T

    nl_buis_imm = {}

    for t in range(
        T
    ):
        fc_bot_arr = np.vstack(
            [
                imm_bot[
                    :,
                    t,
                    :,
                ],
                pop_bot[
                    :,
                    t,
                    :,
                ],
            ]
        )

        fc_upp_arr = np.vstack(
            [
                imm_ratio_top[
                    :,
                    t,
                    :,
                ],
                imm_ratio_mid[
                    :,
                    t,
                    :,
                ],
                imm_total[
                    :,
                    t,
                    :,
                ],
                pop_total[
                    :,
                    t,
                    :,
                ],
            ]
        )

        n_bottom = fc_bot_arr.shape[
            0
        ]

        joint_base = np.vstack(
            [
                fc_upp_arr,
                fc_bot_arr,
            ]
        )

        joint_resid = np.concatenate(
            [
                imm_ratio_top_res[
                    :,
                    t,
                    :,
                ].T,
                imm_ratio_mid_res[
                    :,
                    t,
                    :,
                ].T,
                imm_total_res[
                    :,
                    t,
                    :,
                ].T,
                pop_total_res[
                    :,
                    t,
                    :,
                ].T,
                imm_bot_res[
                    :,
                    t,
                    :,
                ].T,
                pop_bot_res[
                    :,
                    t,
                    :,
                ].T,
            ],
            axis=1,
        )

        joint_cov = _schafer_strimmer_cov(
            joint_resid
        )[
            "shrink_cov"
        ]

        joint_mean = np.mean(
            joint_base,
            axis=1,
        )

        buis_res = reconc_nl_buis(
            assume_independent=False,
            joint_mean=joint_mean,
            joint_cov=joint_cov,
            n_bot=n_bottom,
            f=f_buis,
            num_samples=M,
            seed=args.seed,
        )

        nl_buis_imm[
            t
        ] = buis_res[
            "reconciled_samples"
        ]

    buis_imm = np.stack(
        [
            nl_buis_imm[
                t
            ]
            for t in range(
                T
            )
        ],
        axis=1,
    )

    print(
        "NL-BUIS immigration:",
        buis_imm.shape,
    )

    nl_buis_cit = {}

    for t in range(
        T
    ):
        fc_bot_arr = np.vstack(
            [
                cit_bot[
                    :,
                    t,
                    :,
                ],
                pop_bot[
                    :,
                    t,
                    :,
                ],
            ]
        )

        fc_upp_arr = np.vstack(
            [
                cit_ratio_top[
                    :,
                    t,
                    :,
                ],
                cit_ratio_mid[
                    :,
                    t,
                    :,
                ],
                cit_total[
                    :,
                    t,
                    :,
                ],
                pop_total[
                    :,
                    t,
                    :,
                ],
            ]
        )

        n_bottom = fc_bot_arr.shape[
            0
        ]

        joint_base = np.vstack(
            [
                fc_upp_arr,
                fc_bot_arr,
            ]
        )

        joint_resid = np.concatenate(
            [
                cit_ratio_top_res[
                    :,
                    t,
                    :,
                ].T,
                cit_ratio_mid_res[
                    :,
                    t,
                    :,
                ].T,
                cit_total_res[
                    :,
                    t,
                    :,
                ].T,
                pop_total_res[
                    :,
                    t,
                    :,
                ].T,
                cit_bot_res[
                    :,
                    t,
                    :,
                ].T,
                pop_bot_res[
                    :,
                    t,
                    :,
                ].T,
            ],
            axis=1,
        )

        joint_cov = _schafer_strimmer_cov(
            joint_resid
        )[
            "shrink_cov"
        ]

        joint_mean = np.mean(
            joint_base,
            axis=1,
        )

        buis_res = reconc_nl_buis(
            assume_independent=False,
            joint_mean=joint_mean,
            joint_cov=joint_cov,
            n_bot=n_bottom,
            f=f_buis,
            num_samples=M,
            seed=args.seed,
        )

        nl_buis_cit[
            t
        ] = buis_res[
            "reconciled_samples"
        ]

    buis_cit = np.stack(
        [
            nl_buis_cit[
                t
            ]
            for t in range(
                T
            )
        ],
        axis=1,
    )

    print(
        "NL-BUIS citizenship:",
        buis_cit.shape,
    )

    # ========================================================
    # Ground truth and forecast stacks
    # ========================================================

    test_imm_ratio = test_data_2[
        "immigration_ratio"
    ][
        "y_true"
    ]

    test_cit_ratio = test_data_2[
        "citizenship_ratio"
    ][
        "y_true"
    ]

    test_imm_total = test_data[
        "Switzerland_immigration"
    ][
        "y_true"
    ].reshape(
        1,
        T,
    )

    test_pop_total = test_data[
        "Switzerland_population"
    ][
        "y_true"
    ].reshape(
        1,
        T,
    )

    test_cit_total = test_data[
        "Switzerland_citizenship"
    ][
        "y_true"
    ].reshape(
        1,
        T,
    )

    test_imm = test_data_2[
        "immigration"
    ][
        "y_true"
    ]

    test_pop = test_data_2[
        "population"
    ][
        "y_true"
    ]

    test_cit = test_data_2[
        "citizenship"
    ][
        "y_true"
    ]

    test_imm_data = np.vstack(
        [
            test_imm_ratio[
                ch_uid_pos,
                :,
            ],
            np.delete(
                test_imm_ratio,
                ch_uid_pos,
                axis=0,
            ),
            test_imm_total,
            test_pop_total,
            test_imm,
            test_pop,
        ]
    )

    test_cit_data = np.vstack(
        [
            test_cit_ratio[
                ch_uid_pos,
                :,
            ],
            np.delete(
                test_cit_ratio,
                ch_uid_pos,
                axis=0,
            ),
            test_cit_total,
            test_pop_total,
            test_cit,
            test_pop,
        ]
    )

    base_imm = np.vstack(
        [
            imm_ratio_top,
            imm_ratio_mid,
            imm_total,
            pop_total,
            imm_bot,
            pop_bot,
        ]
    )

    base_cit = np.vstack(
        [
            cit_ratio_top,
            cit_ratio_mid,
            cit_total,
            pop_total,
            cit_bot,
            pop_bot,
        ]
    )

    methods_imm = {
        "Base": base_imm,
        "PBU": pbu_imm,
        "UKF": nlukf_imm,
        "OLS": ols_imm,
        "WLS": wls_imm,
        "full": full_imm,
        # "block": block_imm,
        # "IS": buis_imm,
    }

    methods_cit = {
        "Base": base_cit,
        "PBU": pbu_cit,
        "UKF": nlukf_cit,
        "OLS": ols_cit,
        "WLS": wls_cit,
        "full": full_cit,
        # "block": block_cit,
        # "IS": buis_cit,
    }

    show_levels = {
        "full": np.arange(
            test_imm_data.shape[
                0
            ]
        ),
        "top": [
            0
        ],
        "middle_ratios": list(
            range(
                1,
                U,
            )
        ),
        "swiss_totals": list(
            range(
                U,
                U + 2,
            )
        ),
        "bottom": list(
            range(
                U + 2,
                test_imm_data.shape[
                    0
                ],
            )
        ),
    }

    # ========================================================
    # Save losses for DM / MCS
    # ========================================================

    print(
        "\n🔹 Saving per-time-step losses for DM/MCS tests"
    )

    append_dm_loss_rows(
        loss_rows=loss_rows,
        crps_series_rows=crps_series_rows,
        target="immigration",
        forecast_methods=methods_imm,
        ground_truth=test_imm_data,
        levels=show_levels,
    )

    append_dm_loss_rows(
        loss_rows=loss_rows,
        crps_series_rows=crps_series_rows,
        target="citizenship",
        forecast_methods=methods_cit,
        ground_truth=test_cit_data,
        levels=show_levels,
    )

    save_dm_loss_rows(
        loss_rows=loss_rows,
        loss_file=loss_file,
    )

    save_crps_series_loss_rows(
        crps_series_rows=crps_series_rows,
        crps_series_loss_file=crps_series_loss_file,
    )

    print(
        f"DM loss file saved to: {loss_file}"
    )

    print(
        f"Per-series CRPS loss file saved to: {crps_series_loss_file}"
    )

    # ========================================================
    # Print summaries
    # ========================================================

    print(
        "\n🔹 Computation times (seconds)"
    )

    print(
        f"Immigration  - UKF :  {timing_summary['immigration']['UKF']:.4f}"
    )

    print(
        f"Immigration  - full:  {timing_summary['immigration']['full']:.4f}"
    )

    print(
        f"Citizenship  - UKF :  {timing_summary['citizenship']['UKF']:.4f}"
    )

    print(
        f"Citizenship  - full:  {timing_summary['citizenship']['full']:.4f}"
    )

    print(
        "\n🔹 CRPS — Immigration"
    )

    crps_table_and_relative(
        methods_imm,
        test_imm_data,
        U,
        show_levels,
    )

    crps_gm_table(
        methods_imm,
        test_imm_data,
        baseline_name="Base",
    )

    crps_relative_geomean_over_series(
        methods_imm,
        test_imm_data,
        baseline_name="Base",
    )

    print(
        "\n🔹 CRPS — Citizenship"
    )

    crps_table_and_relative(
        methods_cit,
        test_cit_data,
        U,
        show_levels,
    )

    crps_gm_table(
        methods_cit,
        test_cit_data,
        baseline_name="Base",
    )

    crps_relative_geomean_over_series(
        methods_cit,
        test_cit_data,
        baseline_name="Base",
    )

    rel_scores_imm = extract_relative_scores_for_middle_level(
        methods_imm,
        test_imm_data,
        U=U,
    )

    rel_scores_cit = extract_relative_scores_for_middle_level(
        methods_cit,
        test_cit_data,
        U=U,
    )

    with open(
        "../forecasts/relative_scores_imm_2.pkl",
        "wb",
    ) as file:
        pickle.dump(
            rel_scores_imm,
            file,
        )

    with open(
        "../forecasts/relative_scores_cit_2.pkl",
        "wb",
    ) as file:
        pickle.dump(
            rel_scores_cit,
            file,
        )

    print(
        0
    )


if __name__ == "__main__":
    main()