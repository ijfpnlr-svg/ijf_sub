import os
import pickle
import argparse

import numpy as np
import pandas as pd
import jax.numpy as jnp

from scipy.linalg import block_diag
import concurrent.futures

from reconc.reconc_nl_buis import reconc_nl_buis
from reconc.reconc_nl_ukf import reconc_nl_ukf, _schafer_strimmer_cov
from reconc.reconc_nl_ols import reconc_nl_ols
from simulation.scripts.score_functions import compute_crps
from CH.scripts.reconcile_hybrid import _to_precision


# ============================================================
# Constraint functions
# ============================================================

def make_f_ols(B):
    """
    Fixed-shape nonlinear constraint function for projection.

    Layout of z:

        z = [total, ratio_1, ..., ratio_B, bot_1, ..., bot_B]

    so len(z) = 1 + B + B.
    """
    expected_dim = 1 + B + B

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

        total = z[
            0
        ]

        ratio = z[
            1:1 + B
        ]

        bot = z[
            1 + B:1 + 2 * B
        ]

        eps = 1e-8

        c_mid = (
            ratio
            - bot
            / (
                total
                + eps
            )
        )

        c_total = (
            total
            - jnp.sum(
                bot
            )
        )

        return jnp.concatenate(
            [
                jnp.array(
                    [
                        c_total
                    ]
                ),
                c_mid,
            ]
        )

    return f_ols


def process_t(
    t,
    trips_total_res,
    ratio_state_res,
    trips_bottom_res,
    trips_total,
    ratio_state,
    trips_bottom,
    args,
):
    """
    Process one time index for projection-based reconciliation.
    """
    B = trips_bottom.shape[
        0
    ]

    f_ols = make_f_ols(
        B
    )

    W = _schafer_strimmer_cov(
        np.concatenate(
            [
                trips_total_res[
                    :,
                    t,
                    :,
                ].T,

                ratio_state_res[
                    :,
                    t,
                    :,
                ].T,

                trips_bottom_res[
                    :,
                    t,
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
            trips_total[
                :,
                t,
                :,
            ],
            ratio_state[
                :,
                t,
                :,
            ],
            trips_bottom[
                :,
                t,
                :,
            ],
        ]
    ).T

    D = np.diag(
        np.diag(
            P
        )
    )

    block_W = block_diag(
        D[
            :1 + B,
            :1 + B,
        ],
        P[
            1 + B:,
            1 + B:,
        ],
    )

    ols_out = reconc_nl_ols(
        Z,
        f_ols,
        n_iter=args.iters,
        seed=args.seed,
    )

    wls_out = reconc_nl_ols(
        Z,
        f_ols,
        W=D,
        n_iter=args.iters,
        seed=args.seed,
    )

    block_out = reconc_nl_ols(
        Z,
        f_ols,
        W=block_W,
        n_iter=args.iters,
        seed=args.seed,
    )

    mint_out = reconc_nl_ols(
        Z,
        f_ols,
        W=P,
        n_iter=args.iters,
        seed=args.seed,
    )

    return (
        ols_out[
            "reconciled_samples"
        ],
        wls_out[
            "reconciled_samples"
        ],
        block_out[
            "reconciled_samples"
        ],
        mint_out[
            "reconciled_samples"
        ],
    )


# ============================================================
# Reconciliation maps
# ============================================================

def pbu(bot):
    B, T, M = bot.shape

    trips_total = np.sum(
        bot,
        axis=0,
    ).reshape(
        (
            1,
            T,
            M,
        )
    )

    ratio_state = (
        bot
        / (
            trips_total
            + 1e-8
        )
    )

    return np.concatenate(
        [
            trips_total,
            ratio_state,
            bot,
        ],
        axis=0,
    )


def f_upper_from_bottom(bot):
    trip_tot = np.sum(
        bot,
        axis=1,
        keepdims=True,
    )

    ratio_state = (
        bot
        / (
            trip_tot
            + 1e-8
        )
    )

    return np.concatenate(
        [
            trip_tot,
            ratio_state,
        ],
        axis=1,
    ).T


def f_upper_to_bottom_single(bot):
    trip_total = np.atleast_1d(
        np.sum(
            bot
        )
    )

    ratio_state = (
        bot
        / (
            trip_total
            + 1e-8
        )
    )

    return np.concatenate(
        [
            trip_total,
            ratio_state,
        ]
    )


# ============================================================
# Energy Score
# ============================================================

def compute_es(
    y_true,
    y_samples,
):
    """
    Compute the multivariate Energy Score.
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
                - y,
                axis=1,
            )
        )

        term_2 = 0.5 * np.mean(
            np.linalg.norm(
                x[
                    :,
                    None,
                    :
                ]
                - x[
                    None,
                    :,
                    :
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


# ============================================================
# Loss export utilities for DM / MCS
# ============================================================

def _mean_pairwise_euclidean_distance(
    samples,
    chunk_size=256,
):
    """
    Mean pairwise Euclidean distance for the Energy Score.

    samples shape:

        (M, R)
    """
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
                :
            ]
            - samples[
                None,
                :,
                :
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
    Compute one multivariate Energy Score loss per time step.

    y_true shape:

        (R, T)

    y_samples shape:

        (R, T, M)
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

    At each time step, CRPS is averaged over the selected rows.
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

    This is required by the paper-style metric:

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
    Append per-time-step losses for later tests.

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
# Reporting CRPS utilities
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

    if rows.size == 0:
        return (
            np.nan
            if average_within_level
            else np.empty(
                (
                    0,
                    y_true.shape[
                        1
                    ],
                )
            )
        )

    T = y_true.shape[
        1
    ]

    per_time = []

    for t in range(
        T
    ):
        y_t = np.atleast_1d(
            y_true[
                rows,
                t,
            ]
        )

        samples_t = np.atleast_2d(
            y_samples[
                rows,
                t,
                :,
            ]
        )

        crps_rows = np.atleast_1d(
            compute_crps(
                y_t,
                samples_t,
            )
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
    show_levels=None,
):
    if show_levels is None:
        show_levels = {
            "full": np.arange(
                gt.shape[
                    0
                ]
            )
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

    header_keys = list(
        show_levels.keys()
    )

    cols = " | ".join(
        [
            f"{key:^12s}"
            for key in [
                "Method"
            ]
            + header_keys
        ]
    )

    print(
        cols
    )

    print(
        "-" * (
            14
            * (
                1
                + len(
                    header_keys
                )
            )
        )
    )

    for name in abs_crps:
        row = [
            f"{name:<12s}"
        ]

        for key in header_keys:
            absolute = abs_crps[
                name
            ][
                key
            ]

            relative = rel_crps[
                name
            ][
                key
            ]

            row.append(
                f"{absolute:.4g} ({relative:.3f}x)"
            )

        print(
            " | ".join(
                row
            )
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
            0
        ] != R or y_hat.shape[
            1
        ] != T:
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

        gm_ratio[
            name
        ] = (
            float(
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
            if np.any(
                valid
            )
            else np.nan
        )

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

    base_key = (
        baseline_name
        if baseline_name in forecast_methods
        else next(
            iter(
                forecast_methods.keys()
            )
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


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base_pkl",
        type=str,
        default="forecasts/fc_tourism_autoarima.pkl",
    )

    parser.add_argument(
        "--test_pkl",
        type=str,
        default="forecasts/test_tourism_autoarima.pkl",
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

    results_folder = "results"

    os.makedirs(
        results_folder,
        exist_ok=True,
    )

    loss_file = os.path.join(
        results_folder,
        "australian_tourism_losses_by_time.csv",
    )

    crps_series_loss_file = os.path.join(
        results_folder,
        "australian_tourism_crps_by_series_time.csv",
    )

    loss_rows = []
    crps_series_rows = []

    with open(
        args.base_pkl,
        "rb",
    ) as file:
        base = pickle.load(
            file
        )

    with open(
        args.test_pkl,
        "rb",
    ) as file:
        test_data = pickle.load(
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

    trips, trips_res = pick_block(
        "Trips",
        base,
    )

    ratio, ratio_res = pick_block(
        "Tourism_Ratio",
        base,
    )

    uids = list(
        base[
            "Trips"
        ][
            "uids"
        ]
    )

    try:
        total_idx = uids.index(
            "Total"
        )

    except ValueError:
        total_idx = None

    if total_idx is None:
        raise ValueError(
            "Could not find 'Total' in Trips uids."
        )

    mask = (
        np.arange(
            len(
                uids
            )
        )
        != total_idx
    )

    trips_bottom = trips[
        mask
    ]

    trips_bottom_res = trips_res[
        mask
    ]

    trips_total = np.expand_dims(
        trips[
            total_idx
        ],
        axis=0,
    )

    trips_total_res = np.expand_dims(
        trips_res[
            total_idx
        ],
        axis=0,
    )

    bottom_uids = [
        uid
        for index, uid in enumerate(
            uids
        )
        if index != total_idx
    ]

    total_uid = uids[
        total_idx
    ]

    print(
        "bottoms:",
        len(
            bottom_uids
        ),
        "total found:",
        total_uid is not None,
    )

    print(
        "trips_bottom.shape =",
        trips_bottom.shape,
        "trips_total.shape =",
        trips_total.shape,
    )

    ratio_uids = list(
        base[
            "Tourism_Ratio"
        ][
            "uids"
        ]
    )

    try:
        ratio_total_idx = ratio_uids.index(
            "Total"
        )

    except ValueError:
        ratio_total_idx = None

    if ratio_total_idx is not None:
        mask_ratio = (
            np.arange(
                len(
                    ratio_uids
                )
            )
            != ratio_total_idx
        )

    else:
        mask_ratio = np.ones(
            len(
                ratio_uids
            ),
            dtype=bool,
        )

    ratio_state = ratio[
        mask_ratio
    ]

    ratio_state_res = ratio_res[
        mask_ratio
    ]

    ratio_state_uids = [
        uid
        for index, uid in enumerate(
            ratio_uids
        )
        if mask_ratio[
            index
        ]
    ]

    print(
        "ratio_state.shape =",
        ratio_state.shape,
    )

    B, T, M = trips_bottom.shape

    print(
        "Running reconciliations..."
    )

    # ======================================================
    # PBU
    # ======================================================

    print(
        "----Running PBU-----"
    )

    pbu_tourism = pbu(
        trips_bottom
    )

    print(
        "PBU complete. Final Shape =",
        pbu_tourism.shape,
    )

    # ======================================================
    # UKF
    # ======================================================

    print(
        "----Running UKF-----"
    )

    ukf_tourism = {}

    for t in range(
        T
    ):
        u_obs = np.mean(
            np.vstack(
                [
                    trips_total[
                        :,
                        t,
                        :,
                    ],
                    ratio_state[
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
                    trips_total_res[
                        :,
                        t,
                        :,
                    ],
                    ratio_state_res[
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
            B
        ):
            bot_list.append(
                {
                    "samples": trips_bottom[
                        k,
                        t,
                        :,
                    ],
                    "residuals": trips_bottom_res[
                        k,
                        t,
                        :,
                    ],
                }
            )

        out = reconc_nl_ukf(
            bottom_base_forecasts=bot_list,
            in_type=[
                "samples"
            ]
            * B,
            distr=[
                "normal"
            ]
            * B,
            f=f_upper_to_bottom_single,
            upper_base_forecasts=u_obs,
            R=R,
            num_samples=M,
            seed=args.seed,
        )

        Brec = out[
            "bottom_reconciled_samples"
        ]

        Urec = f_upper_from_bottom(
            Brec.T
        )

        ukf_tourism[
            t
        ] = np.vstack(
            [
                Urec,
                Brec,
            ]
        )

    ukf_tourism = np.stack(
        [
            ukf_tourism[
                t
            ]
            for t in range(
                T
            )
        ],
        axis=1,
    )

    print(
        "UKF complete. Final Shape =",
        ukf_tourism.shape,
    )

    # ======================================================
    # Projection
    # ======================================================

    print(
        "---Running projection---"
    )

    ols = {}
    wls = {}
    mint = {}
    block = {}

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                process_t,
                t,
                trips_total_res,
                ratio_state_res,
                trips_bottom_res,
                trips_total,
                ratio_state,
                trips_bottom,
                args,
            )
            for t in range(
                T
            )
        ]

        for t, future in enumerate(
            futures
        ):
            (
                ols_out,
                wls_out,
                block_out,
                mint_out,
            ) = future.result()

            ols[
                t
            ] = ols_out

            wls[
                t
            ] = wls_out

            block[
                t
            ] = block_out

            mint[
                t
            ] = mint_out

    ols_tourism = np.stack(
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

    wls_tourism = np.stack(
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

    block_tourism = np.stack(
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

    mint_tourism = np.stack(
        [
            mint[
                t
            ]
            for t in range(
                T
            )
        ],
        axis=1,
    )

    print(
        "Projection complete"
    )

    print(
        "OLS Shape =",
        ols_tourism.shape,
    )

    print(
        "WLS Shape =",
        wls_tourism.shape,
    )

    print(
        "Mint Shape =",
        mint_tourism.shape,
    )

    # ======================================================
    # BUIS
    # ======================================================

    print(
        "---Running BUIS---"
    )

    buis_res = {}

    for t in range(
        T
    ):
        fc_bot_arr = trips_bottom[
            :,
            t,
            :,
        ]

        fc_upp_arr = np.vstack(
            [
                trips_total[
                    :,
                    t,
                    :,
                ],
                ratio_state[
                    :,
                    t,
                    :,
                ],
            ]
        )

        n_bottom = fc_bot_arr.shape[
            0
        ]

        buis_out = reconc_nl_buis(
            f=f_upper_from_bottom,
            n_bot=n_bottom,
            num_samples=M,
            seed=args.seed,
            assume_independent=False,
            joint_mean=np.mean(
                np.vstack(
                    [
                        fc_upp_arr,
                        fc_bot_arr,
                    ]
                ),
                axis=1,
            ),
            joint_cov=_schafer_strimmer_cov(
                np.vstack(
                    [
                        trips_total_res[
                            :,
                            t,
                            :,
                        ],
                        ratio_state_res[
                            :,
                            t,
                            :,
                        ],
                        trips_bottom_res[
                            :,
                            t,
                            :,
                        ],
                    ]
                ).T
            )[
                "shrink_cov"
            ],
        )

        buis_res[
            t
        ] = buis_out[
            "reconciled_samples"
        ]

    buis_tourism = np.stack(
        [
            buis_res[
                t
            ]
            for t in range(
                T
            )
        ],
        axis=1,
    )

    print(
        "BUIS complete. Final Shape =",
        buis_tourism.shape,
    )

    # ======================================================
    # Ground truth and forecast dictionaries
    # ======================================================

    test_total = test_data[
        "Trips"
    ][
        "y_true"
    ][
        total_idx
    ]

    test_total = test_total.reshape(
        1,
        T,
    )

    test_ratio = test_data[
        "Tourism_Ratio"
    ][
        "y_true"
    ][
        mask_ratio
    ]

    test_bot = test_data[
        "Trips"
    ][
        "y_true"
    ][
        mask
    ]

    tourism_data = np.vstack(
        [
            test_total,
            test_ratio,
            test_bot,
        ]
    )

    base_tourism = np.vstack(
        [
            trips_total,
            ratio_state,
            trips_bottom,
        ]
    )

    forecast_methods = {
        "Base": base_tourism,
        "PBU": pbu_tourism,
        "UKF": ukf_tourism,
        "OLS": ols_tourism,
        "WLS": wls_tourism,
        "FULL": mint_tourism,
        # "Block": block_tourism,
        # "BUIS": buis_tourism,
    }

    # ======================================================
    # Energy Score
    # ======================================================

    print(
        "\n Energy scores for Australian Tourism"
    )

    for method, y_hat in forecast_methods.items():
        score = compute_es(
            tourism_data,
            y_hat,
        )

        print(
            f"{method}: ES = {score:.4f}"
        )

    # ======================================================
    # Levels
    # ======================================================

    ratio_count = ratio_state.shape[
        0
    ]

    total_row_index = 0

    ratio_start = 1

    ratio_end = (
        1
        + ratio_count
    )

    bottom_start = ratio_end

    bottom_end = (
        bottom_start
        + B
    )

    show_levels_tourism = {
        "full": list(
            range(
                tourism_data.shape[
                    0
                ]
            )
        ),
        "top_total": [
            total_row_index
        ],
        "ratios": list(
            range(
                ratio_start,
                ratio_end,
            )
        ),
        "bottoms": list(
            range(
                bottom_start,
                bottom_end,
            )
        ),
    }

    # ======================================================
    # Save losses for DM / MCS
    # ======================================================

    print(
        "\n🔹 Saving per-time-step losses for DM/MCS tests"
    )

    append_dm_loss_rows(
        loss_rows=loss_rows,
        crps_series_rows=crps_series_rows,
        target="tourism",
        forecast_methods=forecast_methods,
        ground_truth=tourism_data,
        levels=show_levels_tourism,
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

    # ======================================================
    # CRPS output
    # ======================================================

    print(
        "\n🔹 CRPS — Tourism"
    )

    crps_table_and_relative(
        forecast_methods,
        tourism_data,
        show_levels=show_levels_tourism,
    )

    crps_gm_table(
        forecast_methods,
        tourism_data,
        baseline_name="Base",
    )

    crps_relative_geomean_over_series(
        forecast_methods,
        tourism_data,
        baseline_name="Base",
    )


if __name__ == "__main__":
    main()