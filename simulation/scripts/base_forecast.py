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

def make_lagged_xy(series: pd.Series, p: int):
    """
    Prepare lagged data for training.
    """
    X = pd.concat(
        [
            series.shift(l)
            for l in range(1, p + 1)
        ],
        axis=1,
    )

    X.columns = [
        f"{series.name}_lag{l}"
        for l in range(1, p + 1)
    ]

    y = series.shift(-1)

    return X, y


# ================================
# Data validation
# ================================

def get_expected_columns(dimension: int):
    """
    Return the expected dataset columns:

        U, B1, B2, ..., Bd
    """
    return [
        "U",
        *[
            f"B{i}"
            for i in range(1, dimension + 1)
        ],
    ]


def validate_dataset(
    data: pd.DataFrame,
    dimension: int,
):
    """
    Validate and select the expected columns for a dataset.

    Expected columns:

        U, B1, B2, ..., Bd
    """
    expected_columns = get_expected_columns(
        dimension
    )

    missing_columns = [
        col
        for col in expected_columns
        if col not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset for d={dimension} is missing columns: "
            f"{missing_columns}"
        )

    return data[
        expected_columns
    ].copy()


# ================================
# 3D constraint surface plotting
# Only meaningful for d=2
# ================================

def plot_constraint_surface(
    fig,
    surface,
    x_range=(-0.6, 0.6),
    n_grid=100,
):
    """
    Plot the d=2 constraint surface in 3D.

    This function is used only for visualization of datasets
    with two bottom-level series.
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
        Z = X ** 2 + Y ** 2

    elif surface == "linear":
        Z = X + Y

    elif surface == "saddle":
        Z = X ** 2 - Y ** 2

    elif surface == "ripples":
        Z = np.sin(X) + np.cos(Y)

    else:
        raise ValueError(
            f"Unknown surface '{surface}'"
        )

    fig.add_trace(
        go.Surface(
            x=X,
            y=Y,
            z=Z,
            opacity=0.3,
            colorscale="Blues",
            showscale=False,
            name="constraint surface",
        )
    )


# ============================================
# Prediction plotting
# ============================================

def plot_predictions(
    data,
    df_te,
    y_hat_te,
    surface,
    dimension,
    fig_folder,
):
    """
    Plot ground truth and deterministic predictions.

    For d=2, all series are shown:

        U, B1, B2

    For d>2, only the following are shown to keep the figure readable:

        U, B1, B2
    """
    if dimension == 2:
        plot_columns = list(
            data.columns
        )

    else:
        plot_columns = [
            "U",
            "B1",
            "B2",
        ]

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.set_title(
        f"Surface: {surface}, d={dimension}"
    )

    colors = plt.get_cmap(
        "tab10"
    )

    for i, col in enumerate(
        plot_columns
    ):
        ax.plot(
            df_te.index[:-1],
            df_te[col].iloc[:-1],
            label=f"{col} (GT)",
            color=colors(i),
        )

        ax.plot(
            df_te.index[:-1],
            y_hat_te[col],
            "--",
            label=f"{col} (Pred)",
            color=colors(i),
        )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            fig_folder,
            f"{surface}_d{dimension}_pred.png",
        ),
        dpi=300,
    )

    plt.close(fig)


# ============================================
# 3D sample plotting
# Only meaningful for d=2
# ============================================

def plot_3d_samples(
    y_hat_te_samples_tm,
    surface,
    dimension,
    fig_folder,
):
    """
    Plot forecast samples in 3D.

    The input array has shape:

        (n_test_steps, n_samples, d + 1)

    with variable order:

        U, B1, ..., Bd

    This function is called only when d=2.
    """
    if dimension != 2:
        return

    fig = go.Figure()

    max_steps = min(
        20,
        y_hat_te_samples_tm.shape[0],
    )

    for t in range(max_steps):
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
                showlegend=(t == 0),
            )
        )

    plot_constraint_surface(
        fig=fig,
        surface=surface,
    )

    fig.update_layout(
        title=f"{surface}, d={dimension}",
        scene=dict(
            xaxis_title="B1",
            yaxis_title="B2",
            zaxis_title="U",
        ),
    )

    fig.write_html(
        os.path.join(
            fig_folder,
            f"{surface}_d{dimension}_3d.html",
        )
    )


# ============================================
# Model fitting with Random Forest Regressor
# ============================================

def fit_predictive_model(
    data,
    surface,
    dimension,
    fig_folder,
    tr_ratio=0.8,
    n_estimators=100,
    criterion="absolute_error",
    n_samples=1000,
):
    """
    Fit one Random Forest model for every series and generate
    forecast samples using joint residual bootstrapping.

    The input data contain:

        U, B1, ..., Bd

    The residual bootstrap samples complete residual vectors:

        (e_U, e_B1, ..., e_Bd)

    so contemporaneous residual dependence across all variables
    is preserved.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset containing U and all bottom-level series.

    surface : str
        Surface name.

    dimension : int
        Number of bottom-level series.

    fig_folder : str
        Folder where figures are saved.

    tr_ratio : float
        Fraction of observations used for training.

    n_estimators : int
        Number of trees in each Random Forest model.

    criterion : str
        Random Forest splitting criterion.

    n_samples : int
        Number of bootstrap forecast samples.

    Returns
    -------
    y_hat_te_samples : np.ndarray
        Forecast samples with shape:

            (d + 1, n_test_steps, n_samples)

    rf_residuals : np.ndarray
        Training residuals with shape:

            (d + 1, n_valid_train_steps)

    y_hat_te : pd.DataFrame
        Deterministic test forecasts.

    df_te : pd.DataFrame
        Test data.
    """
    # ---------------------------
    # Validate and select columns
    # ---------------------------

    data = validate_dataset(
        data=data,
        dimension=dimension,
    )

    n_variables = len(
        data.columns
    )

    print(
        f"Fitting {n_variables} models "
        f"for surface={surface}, d={dimension}"
    )

    # ---------------------------
    # Train / test split
    # ---------------------------

    split_idx = int(
        tr_ratio * len(data)
    )

    df_tr = data.iloc[
        :split_idx
    ]

    df_te = data.iloc[
        split_idx:
    ]

    # DataFrames to store point predictions.
    y_hat_te = pd.DataFrame(
        index=df_te.index[:-1],
        columns=data.columns,
        dtype=float,
    )

    y_hat_tr = pd.DataFrame(
        index=df_tr.index[:-1],
        columns=data.columns,
        dtype=float,
    )

    # ---------------------------
    # Fit one model per variable
    # ---------------------------

    for model_index, col in enumerate(
        data.columns,
        start=1,
    ):
        print(
            f"  [{model_index}/{n_variables}] "
            f"Fitting {col}"
        )

        X_tr, y_tr = make_lagged_xy(
            df_tr[col],
            p=1,
        )

        valid_tr = X_tr.index[
            X_tr.notna().all(axis=1)
            & y_tr.notna()
        ]

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            criterion=criterion,
            random_state=42,
        )

        model.fit(
            X_tr.loc[valid_tr],
            y_tr.loc[valid_tr],
        )

        X_te, _ = make_lagged_xy(
            df_te[col],
            p=1,
        )

        valid_te = (
            X_te.index[
                X_te.notna().all(axis=1)
            ]
            .intersection(
                y_hat_te.index
            )
        )

        y_hat_tr.loc[
            valid_tr,
            col,
        ] = model.predict(
            X_tr.loc[valid_tr]
        )

        y_hat_te.loc[
            valid_te,
            col,
        ] = model.predict(
            X_te.loc[valid_te]
        )

    # ---------------------------
    # Preserve original first-forecast alignment
    # ---------------------------

    for col in data.columns:
        y_hat_tr.loc[
            y_hat_tr.index[0],
            col,
        ] = df_tr[col].iloc[0]

        y_hat_te.loc[
            y_hat_te.index[0],
            col,
        ] = df_tr[col].iloc[-1]

    # ---------------------------
    # Joint residual bootstrap
    # ---------------------------

    # Calculate residuals for all variables together.
    #
    # Shape:
    #
    #     (n_train_steps, d + 1)
    #
    tr_errors = (
        df_tr.iloc[:-1].values
        - y_hat_tr.values
    ).astype(float)

    # Remove rows containing any NaN.
    tr_errors = tr_errors[
        ~np.isnan(
            tr_errors
        ).any(axis=1)
    ]

    if len(tr_errors) == 0:
        raise RuntimeError(
            "No valid training residuals are available "
            f"for surface={surface}, d={dimension}"
        )

    res_mat = tr_errors

    print(
        f"Residual matrix shape: "
        f"{res_mat.shape}"
    )

    # ---------------------------
    # Sample complete residual vectors
    # ---------------------------

    n_test_steps = len(
        y_hat_te
    )

    idx = np.random.randint(
        low=0,
        high=len(res_mat),
        size=(
            n_test_steps,
            n_samples,
        ),
    )

    # Sample complete residual rows.
    #
    # Shape:
    #
    #     (n_test_steps, n_samples, d + 1)
    #
    y_hat_te_samples_tm = res_mat[
        idx
    ].copy()

    # Add deterministic point forecasts.
    #
    # y_hat_te.values[:, None, :] has shape:
    #
    #     (n_test_steps, 1, d + 1)
    #
    # Broadcasting gives:
    #
    #     (n_test_steps, n_samples, d + 1)
    #
    y_hat_te_samples_tm += (
        y_hat_te.values[
            :,
            None,
            :,
        ]
    )

    # Output format:
    #
    #     (d + 1, n_test_steps, n_samples)
    #
    y_hat_te_samples = np.transpose(
        y_hat_te_samples_tm,
        (
            2,
            0,
            1,
        ),
    )

    # Residual output format:
    #
    #     (d + 1, n_valid_train_steps)
    #
    rf_residuals = res_mat.T

    print(
        f"Forecast sample shape: "
        f"{y_hat_te_samples.shape}"
    )

    print(
        f"Residual output shape: "
        f"{rf_residuals.shape}"
    )

    # ---------------------------
    # Prediction plot
    # ---------------------------

    plot_predictions(
        data=data,
        df_te=df_te,
        y_hat_te=y_hat_te,
        surface=surface,
        dimension=dimension,
        fig_folder=fig_folder,
    )

    # ---------------------------
    # 3D plot only for d=2
    # ---------------------------

    if dimension == 2:
        plot_3d_samples(
            y_hat_te_samples_tm=(
                y_hat_te_samples_tm
            ),
            surface=surface,
            dimension=dimension,
            fig_folder=fig_folder,
        )

    return (
        y_hat_te_samples,
        rf_residuals,
        y_hat_te,
        df_te,
    )


# ==========================
# Pickle helper
# ==========================

def save_pickle(
    obj,
    file_path,
):
    """
    Save an object as a pickle file.
    """
    with open(
        file_path,
        "wb",
    ) as file:
        pickle.dump(
            obj,
            file,
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

    dimensions = [
        2,
        10,
        20,
        50,
        100,
        200
    ]

    surfaces = [
        "paraboloid",
        "saddle",
        "ripples",
        "linear",
    ]

    n_samples = 20000

    for dimension in dimensions:
        print()
        print("=" * 70)
        print(
            f"Bottom-level dimension: "
            f"d={dimension}"
        )
        print("=" * 70)

        for surface in surfaces:
            print()
            print(
                f"=== Surface {surface}, "
                f"d={dimension} ==="
            )

            data_file = os.path.join(
                data_folder,
                f"{surface}_data_d{dimension}.pkl",
            )

            data = pd.read_pickle(
                data_file
            )

            (
                base_fc,
                residuals,
                deterministic_forecasts,
                test_data,
            ) = fit_predictive_model(
                data=data,
                surface=surface,
                dimension=dimension,
                fig_folder=fig_folder,
                n_samples=n_samples,
            )

            # ---------------------------
            # Save forecast samples
            # ---------------------------

            save_pickle(
                base_fc,
                os.path.join(
                    fc_folder,
                    f"base_fc_{surface}_"
                    f"d{dimension}_"
                    f"{n_samples}.pkl",
                ),
            )

            # ---------------------------
            # Save residuals
            # ---------------------------

            save_pickle(
                residuals,
                os.path.join(
                    fc_folder,
                    f"residuals_{surface}_"
                    f"d{dimension}_"
                    f"{n_samples}.pkl",
                ),
            )

            # ---------------------------
            # Save test data
            # ---------------------------

            save_pickle(
                test_data,
                os.path.join(
                    fc_folder,
                    f"test_data_{surface}_"
                    f"d{dimension}_"
                    f"{n_samples}.pkl",
                ),
            )

            # ---------------------------
            # Save deterministic forecasts
            # ---------------------------

            save_pickle(
                deterministic_forecasts,
                os.path.join(
                    fc_folder,
                    f"det_forecasts_{surface}_"
                    f"d{dimension}_"
                    f"{n_samples}.pkl",
                ),
            )

            print(
                f"Completed surface={surface}, "
                f"d={dimension}"
            )


if __name__ == "__main__":
    main()