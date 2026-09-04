import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor


# ================================
# Helper function to create lagged features
# ================================

def make_lagged_xy(
    series: pd.Series,
    p: int,
):
    """
    Prepare lagged data for one-step-ahead training.
    """

    X = pd.concat(
        [
            series.shift(
                l
            )
            for l in range(
                0,
                p,
            )
        ],
        axis=1,
    )

    X.columns = [
        f"{series.name}_lag{l}"
        for l in range(
            0,
            p,
        )
    ]

    y = series.shift(
        -1
    )

    valid = (
        X.notna().all(
            axis=1
        )
        & y.notna()
    )

    return (
        X.loc[
            valid
        ],
        y.loc[
            valid
        ],
    )


# ================================
# 3D Constraint Surface Plotting
# ================================

def plot_constraint_surface(
    fig,
    surface,
    x_range=(-0.6, 0.6),
    n_grid=100,
):
    """
    Plot the nonlinear constraint surface in 3D.
    """

    x = np.linspace(
        x_range[0],
        x_range[1],
        n_grid,
    )

    y = np.linspace(
        x_range[0],
        x_range[1],
        n_grid,
    )

    X, Y = np.meshgrid(
        x,
        y,
    )

    if surface == "paraboloid":
        Z = X**2 + Y**2

    elif surface == "linear":
        Z = X + Y

    elif surface == "cone":
        Z = np.sqrt(
            np.maximum(
                X**2 + Y**2,
                0.0,
            )
        )

    elif surface == "saddle":
        Z = X**2 - Y**2

    elif surface == "ripples":
        Z = (
            np.sin(
                X
            )
            + np.cos(
                Y
            )
        )

    elif surface == "sphere_cap":
        Z = np.sqrt(
            np.maximum(
                1.0
                - (
                    X**2
                    + Y**2
                ),
                0.0,
            )
        )

    elif surface == "ratio":
        num = np.exp(
            X
        )

        den = (
            num
            + np.exp(
                Y
            )
            + 1e-12
        )

        Z = num / den

    else:
        return

    fig.add_trace(
        go.Surface(
            x=X,
            y=Y,
            z=Z,
            opacity=0.3,
            colorscale="Blues",
            showscale=False,
        )
    )


# ============================================
# Model fitting with Random Forest Regressor
# ============================================

def fit_predictive_model(
    data,
    surface,
    dataset_tag,
    fig_folder,
    tr_ratio=0.8,
    n_estimators=100,
    criterion="absolute_error",
    n_samples=1000,
    random_state=42,
):
    """
    Fixed pipeline using joint residual bootstrapping.

    The residual bootstrap samples whole residual rows, so the dependence among
    U, B1, and B2 residuals is preserved.
    """

    # ---------------------------
    # Train / test split
    # ---------------------------

    data = data[
        [
            "U",
            "B1",
            "B2",
        ]
    ]

    split_idx = int(
        tr_ratio
        * len(
            data
        )
    )

    df_tr = data.iloc[
        :split_idx
    ]

    df_te = data.iloc[
        split_idx:
    ]

    # DataFrames to store point predictions
    y_hat_te = pd.DataFrame(
        index=df_te.index[:-1],
        columns=data.columns,
    )

    y_hat_tr = pd.DataFrame(
        index=df_tr.index[:-1],
        columns=data.columns,
    )

    y_tr_actual = pd.DataFrame(
        index=y_hat_tr.index,
        columns=data.columns,
    )

    # ---------------------------
    # Fit one model per time series
    # ---------------------------

    for col in data.columns:
        X_tr, y_tr = make_lagged_xy(
            df_tr[
                col
            ],
            p=1,
        )

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            criterion=criterion,
            random_state=random_state,
        )

        model.fit(
            X_tr,
            y_tr,
        )

        X_te, _ = make_lagged_xy(
            df_te[
                col
            ],
            p=1,
        )

        valid_te = X_te.index.intersection(
            y_hat_te.index
        )

        y_hat_tr.loc[
            X_tr.index,
            col,
        ] = model.predict(
            X_tr
        )

        y_tr_actual.loc[
            y_tr.index,
            col,
        ] = y_tr

        y_hat_te.loc[
            valid_te,
            col,
        ] = model.predict(
            X_te.loc[
                valid_te
            ]
        )

    # ---------------------------
    # Joint residual bootstrap
    # ---------------------------
    # y_hat_tr at origin t forecasts the next observation t+1.

    tr_errors = (
        y_tr_actual.values
        - y_hat_tr.values
    ).astype(
        float
    )

    tr_errors = tr_errors[
        ~np.isnan(
            tr_errors
        ).any(
            axis=1
        )
    ]

    if len(
        tr_errors
    ) == 0:
        raise ValueError(
            "No valid training residuals available for bootstrap."
        )

    # Joint residual matrix: shape (n_train_steps, 3)
    res_mat = tr_errors

    # Generate reproducible random indices to pick whole residual rows.
    # This preserves the joint dependence between U, B1, and B2 errors.
    n_test_steps = len(
        y_hat_te
    )

    rng = np.random.default_rng(
        random_state
    )

    idx = rng.integers(
        0,
        len(
            res_mat
        ),
        size=(
            n_test_steps,
            n_samples,
        ),
    )

    # eps shape: (n_test_steps, n_samples, 3)
    eps = res_mat[
        idx
    ]

    # ---------------------------
    # Forecast samples
    # ---------------------------

    y_hat_te_point = y_hat_te.values[
        :,
        None,
        :,
    ]

    # Shape: (n_test_steps, n_samples, 3)
    y_hat_te_samples_tm = (
        y_hat_te_point
        + eps
    )

    # Desired output shape: (3, n_test_steps, n_samples)
    y_hat_te_samples = np.transpose(
        y_hat_te_samples_tm,
        (
            2,
            0,
            1,
        ),
    )

    # Residuals for reconciliation script: shape (3, n_valid_train_steps)
    rf_residuals = res_mat.T

    # ---------------------------
    # 2D prediction plot
    # ---------------------------

    fig, ax = plt.subplots(
        figsize=(
            10,
            5,
        )
    )

    ax.set_title(
        f"Surface: {surface}"
    )

    colors = plt.get_cmap(
        "tab10"
    )

    for i, col in enumerate(
        data.columns
    ):
        ax.plot(
            df_te.index[:-1],
            df_te[
                col
            ].iloc[
                1:
            ],
            label=f"{col} (GT)",
            color=colors(
                i
            ),
        )

        ax.plot(
            df_te.index[:-1],
            y_hat_te[
                col
            ],
            "--",
            label=f"{col} (Pred)",
            color=colors(
                i
            ),
        )

    ax.legend()

    fig.savefig(
        os.path.join(
            fig_folder,
            f"{surface}_{dataset_tag}_pred.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    # ---------------------------
    # 3D sample plot
    # ---------------------------

    fig = go.Figure()

    max_steps = min(
        20,
        y_hat_te_samples_tm.shape[0],
    )

    for t in range(
        max_steps
    ):
        fig.add_trace(
            go.Scatter3d(
                x=y_hat_te_samples_tm[
                    t,
                    :,
                    1,
                ],
                y=y_hat_te_samples_tm[
                    t,
                    :,
                    2,
                ],
                z=y_hat_te_samples_tm[
                    t,
                    :,
                    0,
                ],
                mode="markers",
                marker=dict(
                    size=2,
                    color="blue",
                    opacity=0.5,
                ),
                name=(
                    "samples"
                    if t == 0
                    else None
                ),
            )
        )

    plot_constraint_surface(
        fig,
        surface,
    )

    fig.write_html(
        os.path.join(
            fig_folder,
            f"{surface}_{dataset_tag}_3d.html",
        )
    )

    return (
        y_hat_te_samples,
        rf_residuals,
        y_hat_te,
        df_te,
    )


# ==========================
# MAIN PIPELINE
# ==========================

def main():
    data_folder = "../data/"
    fc_folder = "../forecasts/"
    fig_folder = "../fig/"

    os.makedirs(
        fc_folder,
        exist_ok=True,
    )

    os.makedirs(
        fig_folder,
        exist_ok=True,
    )

    surfaces = [
        "paraboloid",
        "saddle",
        "ripples",
        # "ratio",
        # "linear",
    ]

    n_samples = 2000
    random_state = 42

    for surface in surfaces:
        print(
            f"\n=== Surface {surface} — independent ==="
        )

        data_indep = pd.read_pickle(
            os.path.join(
                data_folder,
                f"{surface}_data_indep.pkl",
            )
        )

        base_fc, res, det, df_te = fit_predictive_model(
            data=data_indep,
            surface=surface,
            dataset_tag="indep",
            fig_folder=fig_folder,
            n_samples=n_samples,
            random_state=random_state,
        )

        with open(
            os.path.join(
                fc_folder,
                f"base_fc_{surface}_indep_{n_samples}.pkl",
            ),
            "wb",
        ) as file:
            pickle.dump(
                base_fc,
                file,
            )

        with open(
            os.path.join(
                fc_folder,
                f"residuals_{surface}_indep_{n_samples}.pkl",
            ),
            "wb",
        ) as file:
            pickle.dump(
                res,
                file,
            )

        with open(
            os.path.join(
                fc_folder,
                f"test_data_{surface}_indep_{n_samples}.pkl",
            ),
            "wb",
        ) as file:
            pickle.dump(
                df_te,
                file,
            )

        with open(
            os.path.join(
                fc_folder,
                f"det_forecasts_{surface}_indep_{n_samples}.pkl",
            ),
            "wb",
        ) as file:
            pickle.dump(
                det,
                file,
            )


if __name__ == "__main__":
    main()