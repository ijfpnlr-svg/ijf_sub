import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import jax
import matplotlib.pyplot as plt
import numpy as np


def main():
    # Set random seed for reproducibility
    np.random.seed(42)

    # Define the paraboloid function f(y, x1, x2) = y - (x1^2 + x2^2)
    def f(Z):
        y, x1, x2 = Z
        return y - (x1 ** 2 + x2 ** 2)

    # Jacobian of f using JAX
    J = jax.grad(f)
    # Hessian of f using JAX
    H = jax.hessian(f)

    # Step 1: Data Generation Process (points on the manifold f = 0)
    def generate_true_data(n_samples=100):
        x1 = np.random.uniform(-2, 2, n_samples)
        x2 = np.random.uniform(-2, 2, n_samples)
        y = x1 ** 2 + x2 ** 2
        return np.stack([y, x1, x2], axis=1)

    # Step 2: Prediction Generation Process (noisy forecasts off the manifold)
    def generate_predictions(true_data, noise_scale=0.5):
        noise = np.random.normal(0, noise_scale, true_data.shape)
        predictions = true_data + noise
        return predictions

    # Step 3: Orthogonal Projection onto f(Z) = 0
    def project_to_manifold(Z_hat, max_iter=100, tol=1e-6):
        Z = Z_hat.copy()
        for i in range(max_iter):
            f_val = f(Z)
            J_val = np.array(J(Z))
            delta = f_val / np.dot(J_val, J_val) * J_val
            Z = Z - delta
            if np.abs(f_val) < tol:
                break
        if i == max_iter - 1:
            print('steps{}, err:{:0.2e}'.format(i, np.abs(f_val)))
        return Z

    # Step 4: Compute RMSEs and compare
    def compute_rmse(true_data, predictions, reconciled):
        rmse_unrecon = np.sqrt(np.mean((true_data - predictions) ** 2))
        rmse_recon = np.sqrt(np.mean((true_data - reconciled) ** 2))
        return rmse_unrecon, rmse_recon

    # Step 5: Main Experiment
    n_max = 1000
    scale_factor = 10
    base_fc = pd.read_pickle('../forecasts/cor_det_base_fc_arima_hor_1.pkl')
    fc_array = np.concatenate(
        [base_fc['U'], base_fc['B1'], base_fc['B2']], axis=1
    ).squeeze(-1)
    fc_array = fc_array[:n_max]

    reconciled = np.array([project_to_manifold(pred, max_iter=100) for pred in fc_array.copy()])
    np.mean(np.abs(np.hstack([r[0]-(r[1]**2+r[2]**2) for r in reconciled])))
    test_data = pd.read_pickle('../data/corr_ar_process_0.1.pkl')
    true_data = test_data[['U', 'B1', 'B2']].to_numpy()[300:300+n_max,:]

    # Compute RMSEs
    predictions = fc_array
    rmse_unrecon, rmse_recon = compute_rmse(true_data, predictions, reconciled)
    print(f"RMSE Unreconciled: {rmse_unrecon:.4f}")
    print(f"RMSE Reconciled: {rmse_recon:.4f}")


    # plot results

    labels = ['U', 'B1', 'B2']
    n_dims = 3
    n_samples = 100
    x = np.arange(n_samples)

    fig, axes = plt.subplots(n_dims, 1, figsize=(14, 9), sharex=True)

    for i in range(n_dims):
        ax = axes[i]
        ax.plot(x, true_data[:100, i], label='True', color='black', linewidth=1.5)
        ax.plot(x, predictions[:100, i], label='Predicted', color='orange', alpha=0.7)
        ax.plot(x, reconciled[:100, i], label='Reconciled', color='green', alpha=0.7)
        ax.set_ylabel(labels[i])
        ax.legend(loc='upper right')
        ax.grid(True)

    axes[-1].set_xlabel("Sample index")
    plt.suptitle("Line Plots of True vs Predicted vs Reconciled for U, B1, B2", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()

    # Compute per-sample RMSE difference (squared error difference)
    rmse_diff = np.mean((true_data - predictions)**2, axis=1) - np.mean((true_data - reconciled)**2, axis=1)

    import plotly.graph_objects as go
    for i in np.arange(1, 50):
        true_filtered = true_data[i, :]
        pred_filtered = predictions[i, :]
        recon_filtered = reconciled[i, :]

        # Get coordinates of predicted point
        x1_pred = pred_filtered[1]
        x2_pred = pred_filtered[2]
        z_pred = pred_filtered[0]

        # Compute surface value at (x1_pred, x2_pred)
        surface_z = x1_pred ** 2 + x2_pred ** 2

        # Plot only if the point lies outside (either above or below the surface)
        if not np.isclose(z_pred, surface_z, atol=1e-3):  # you can adjust tolerance
            x1_grid, x2_grid = np.meshgrid(np.linspace(-2, 2, 20), np.linspace(-2, 2, 20))
            y_grid = x1_grid ** 2 + x2_grid ** 2

            fig = go.Figure()

            # Surface
            fig.add_trace(go.Surface(
                z=y_grid, x=x1_grid, y=x2_grid,
                opacity=0.3,
                colorscale='Blues',
                showscale=False,
                name='f(Z) = 0'
            ))

            # True Point
            fig.add_trace(go.Scatter3d(
                x=[true_filtered[1]],
                y=[true_filtered[2]],
                z=[true_filtered[0]],
                mode='markers',
                marker=dict(size=5, color='green'),
                name='True'
            ))

            # Predicted Point
            fig.add_trace(go.Scatter3d(
                x=[x1_pred],
                y=[x2_pred],
                z=[z_pred],
                mode='markers',
                marker=dict(size=5, color='red'),
                name='Predicted'
            ))

            # Reconciled Point
            fig.add_trace(go.Scatter3d(
                x=[recon_filtered[1]],
                y=[recon_filtered[2]],
                z=[recon_filtered[0]],
                mode='markers',
                marker=dict(size=5, color='blue'),
                name='Reconciled'
            ))

            fig.update_layout(
                scene=dict(
                    xaxis_title="X1",
                    yaxis_title="X2",
                    zaxis_title="Y",
                    aspectmode='cube'
                ),
                title=f"3D Visualization - Sample {i}",
                margin=dict(l=0, r=0, b=0, t=30)
            )

            fig.show()



    # Step 6: Filter data around a level set (l = 2, eps = 0.5)
    for l in np.linspace(0, 3, 6):
        eps = 0.1
        mask = (predictions[:, 0] >= l - eps) & (predictions[:, 0] <= l + eps)
        true_filtered = true_data[mask]
        pred_filtered = predictions[mask]
        recon_filtered = reconciled[mask]
        rmse_diff_filtered = rmse_diff[mask]

        # Step 7: Visualization
        plt.figure(figsize=(10, 8))
        bad_idxs = rmse_diff_filtered < 0
        plt.scatter(pred_filtered[bad_idxs, 1], pred_filtered[bad_idxs, 2], c='red',label='Predictions')
        plt.scatter(pred_filtered[~bad_idxs, 1], pred_filtered[~bad_idxs, 2], c='green',label='Predictions')

        theta = np.linspace(0, 2*np.pi, 100)
        x1_circle = np.sqrt(l) * np.cos(theta)
        x2_circle = np.sqrt(l) * np.sin(theta)
        plt.fill(x1_circle, x2_circle, alpha=0.3, color='blue', label=f'Level Set y ≈ {l}')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title(f'Predictions and Level Set (y ≈ {l} ± {eps})')
        plt.legend()
        plt.axis('equal')
        plt.grid(True)
        plt.show()

        # Step 8: 3D Plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        x1_grid, x2_grid = np.meshgrid(np.linspace(-2, 2, 20), np.linspace(-2, 2, 20))
        y_grid = x1_grid**2 + x2_grid**2
        ax.plot_surface(x1_grid, x2_grid, y_grid, alpha=0.3, color='blue', label='f(Z) = 0')
        ax.scatter(true_filtered[:, 1], true_filtered[:, 2], true_filtered[:, 0], c='green', label='True')
        ax.scatter(pred_filtered[:, 1], pred_filtered[:, 2], pred_filtered[:, 0], c='red', label='Predicted')
        ax.scatter(recon_filtered[:, 1], recon_filtered[:, 2], recon_filtered[:, 0], c='blue', label='Reconciled')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('y')
        ax.set_title('3D View of Paraboloid and Points')
        ax.legend()
        plt.show()

    # Step 9: Verify Convexity at Projection Points
    for i, Z in enumerate(recon_filtered[:5]):
        H_val = np.array(H(Z))  # 3x3 matrix
        J_val = np.array(J(Z))  # 1x3 vector
        # Scalar normal curvature: J H J^T / (J J^T)
        H_normal = np.dot(J_val, H_val.dot(J_val.T)) / np.dot(J_val, J_val)
        print(f"Point {i+1}: Normal Curvature = {H_normal:.4f}, Convex: {H_normal > 0}")






if __name__ == "__main__":
    main()