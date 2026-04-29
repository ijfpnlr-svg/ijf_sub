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
    X = pd.concat([series.shift(l) for l in range(1, p + 1)], axis=1)
    X.columns = [f"{series.name}_lag{l}" for l in range(1, p + 1)]
    y = series.shift(-1)
    return X, y

# ================================
#  3D Constraint Surface Plotting (just for visualization)
# ================================
def plot_constraint_surface(fig, surface, x_range=(-0.6, 0.6), n_grid=100):
    """
    Function to plot the constraint surface in 3D for various surfaces.
    """
    x = np.linspace(x_range[0], x_range[1], n_grid)
    y = np.linspace(x_range[0], x_range[1], n_grid)
    X, Y = np.meshgrid(x, y)

    if surface == "paraboloid":
        Z = X**2 + Y**2
    if surface == "linear":
        Z = X + Y
    elif surface == "cone":
        Z = np.sqrt(np.maximum(X**2 + Y**2, 0.0))
    elif surface == "saddle":
        Z = X**2 - Y**2
    elif surface == "ripples":
        Z = np.sin(X) + np.cos(Y)
    elif surface == "sphere_cap":
        Z = np.sqrt(np.maximum(1.0 - (X**2 + Y**2), 0.0))
    elif surface == "ratio":
        num = np.exp(X)
        den = num + np.exp(Y) + 1e-12
        Z = num / den
    else:
        return

    fig.add_trace(go.Surface(x=X, y=Y, z=Z, opacity=0.3, colorscale="Blues", showscale=False))

# ============================================
# Model fitting with Random Forest Regressor for all series (lag 1)
# ============================================
def fit_predictive_model(data, surface, dataset_tag, fig_folder, tr_ratio=0.8, n_estimators=100,
                         criterion="absolute_error", n_samples=1000):
    """
    Fixed pipeline: Uses Joint Residual Bootstrapping to preserve correlations
    between U, B1, and B2.
    """
    # ---------------------------
    # Train / test split
    # ---------------------------
    data = data[["U", "B1", "B2"]]
    split_idx = int(tr_ratio * len(data))
    df_tr = data.iloc[:split_idx]
    df_te = data.iloc[split_idx:]

    # DataFrames to store point predictions
    y_hat_te = pd.DataFrame(index=df_te.index[:-1], columns=data.columns)
    y_hat_tr = pd.DataFrame(index=df_tr.index[:-1], columns=data.columns)

    # 1. Fit models and get point predictions
    for col in data.columns:
        X_tr, y_tr = make_lagged_xy(df_tr[col], p=1)
        valid_tr = X_tr.index[X_tr.notna().all(axis=1) & y_tr.notna()]

        model = RandomForestRegressor(n_estimators=n_estimators, criterion=criterion, random_state=42)
        model.fit(X_tr.loc[valid_tr], y_tr.loc[valid_tr])

        X_te, _ = make_lagged_xy(df_te[col], p=1)
        valid_te = X_te.index[X_te.notna().all(axis=1)].intersection(y_hat_te.index)

        y_hat_tr.loc[valid_tr, col] = model.predict(X_tr.loc[valid_tr])
        y_hat_te.loc[valid_te, col] = model.predict(X_te.loc[valid_te])

    # Fix the first forecast alignment
    for col in data.columns:
        y_hat_tr.loc[y_hat_tr.index[0], col] = df_tr[col].iloc[0]
        y_hat_te.loc[y_hat_te.index[0], col] = df_tr[col].iloc[-1]

    # ---------------------------
    # JOINT RESIDUAL BOOTSTRAP (The Fix)
    # ---------------------------
    # Calculate residuals for all columns at once to keep them aligned
    # We use df_tr.iloc[1:] because lag-1 removes the first point
    tr_errors = (df_tr.iloc[:-1].values - y_hat_tr.values).astype(float)
    tr_errors = tr_errors[~np.isnan(tr_errors).any(axis=1)]  # Remove any rows with NaNs

    # Create the Joint Matrix (n_train_obs, 3)
    res_mat = tr_errors

    # Generate random indices to pick WHOLE ROWS from res_mat
    # This ensures that if U has a high error at time T, we pick the corresponding B1/B2 errors at time T
    n_test_steps = len(y_hat_te)
    idx = np.random.randint(0, len(res_mat), size=(n_test_steps, n_samples))

    # eps shape will be (n_test_steps, n_samples, 3)
    eps = res_mat[idx]

    # ---------------------------
    # Forecast samples (Broadcasting)
    # ---------------------------
    # y_hat_te.values is (n_test_steps, 3) -> we need (n_test_steps, 1, 3) for broadcasting
    y_hat_te_point = y_hat_te.values[:, None, :]

    # y_hat_te_samples_tm shape: (n_test_steps, n_samples, 3)
    y_hat_te_samples_tm = y_hat_te_point + eps

    # Transpose to your desired output: (3, n_test_steps, n_samples)
    y_hat_te_samples = np.transpose(y_hat_te_samples_tm, (2, 0, 1))

    # ---------------------------
    # Residuals for return (Transposed)
    # ---------------------------
    rf_residuals = res_mat.T  # (3, n_valid_train_steps)
    # ---------------------------
    # Plots
    # ---------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.title(f"Surface: {surface}")
    colors = plt.get_cmap("tab10")
    for i, col in enumerate(data.columns):
        ax.plot(df_te.index[:-1], df_te[col].iloc[:-1],
                label=f"{col} (GT)", color=colors(i))
        ax.plot(df_te.index[:-1], y_hat_te[col],
                "--", label=f"{col} (Pred)", color=colors(i))
    ax.legend()
    fig.savefig(os.path.join(fig_folder, f"{surface}_{dataset_tag}_pred.png"), dpi=300)
    plt.close()

    fig = go.Figure()
    max_steps = min(20, y_hat_te_samples_tm.shape[0])
    for t in range(max_steps):
        fig.add_trace(go.Scatter3d(
            x=y_hat_te_samples_tm[t, :, 1],
            y=y_hat_te_samples_tm[t, :, 2],
            z=y_hat_te_samples_tm[t, :, 0],
            mode="markers",
            marker=dict(size=2, color="blue", opacity=0.5),
            name=("samples" if t == 0 else None)
        ))

    plot_constraint_surface(fig, surface)
    fig.write_html(os.path.join(fig_folder, f"{surface}_{dataset_tag}_3d.html"))

    return y_hat_te_samples, rf_residuals, y_hat_te, df_te


# ==========================
#  MAIN PIPELINE
# ==========================
def main():
    data_folder = "../data/"
    fc_folder = "../forecasts/"
    fig_folder = "../fig/"

    os.makedirs(fc_folder, exist_ok=True)
    os.makedirs(fig_folder, exist_ok=True)

    surfaces = [
        "paraboloid",
        "saddle",
        "ripples",
        "ratio",
        "linear",
    ]

    n_samples = 2000

    for surface in surfaces:
        print(f"\n=== Surface {surface} — independent ===")
        data_indep = pd.read_pickle(
            os.path.join(data_folder, f"{surface}_data_indep.pkl")
        )

        base_fc, res, det, df_te = fit_predictive_model(
            data_indep, surface, "indep", fig_folder, n_samples=n_samples
        )

        pickle.dump(base_fc,
                    open(os.path.join(fc_folder,
                         f"base_fc_{surface}_indep_{n_samples}.pkl"), "wb"))
        pickle.dump(res,
                    open(os.path.join(fc_folder,
                         f"residuals_{surface}_indep_{n_samples}.pkl"), "wb"))
        pickle.dump(df_te,
                    open(os.path.join(fc_folder,
                         f"test_data_{surface}_indep_{n_samples}.pkl"), "wb"))
        pickle.dump(det,
                    open(os.path.join(fc_folder,
                         f"det_forecasts_{surface}_indep_{n_samples}.pkl"), "wb"))


if __name__ == "__main__":
    main()
