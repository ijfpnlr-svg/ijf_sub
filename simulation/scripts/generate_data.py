import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)


# ----------------------------------------------------------------------
# Surface definitions
# ----------------------------------------------------------------------

def surface_paraboloid(b1, b2):
    """Paraboloid: U = B1^2 + B2^2."""
    return b1 ** 2 + b2 ** 2

def surface_linear(b1, b2):
    """Linear: U = B1 + B2."""
    return b1 + b2

def surface_cone(b1, b2):
    """Cone: U = sqrt(B1^2 + B2^2)."""
    r2 = b1 ** 2 + b2 ** 2
    return np.sqrt(np.maximum(r2, 0.0))


def surface_saddle(b1, b2):
    """Saddle: U = B1^2 - B2^2."""
    return b1 ** 2 - b2 ** 2


def surface_ripples(b1, b2):
    """Ripple surface: U = sin(B1) + cos(B2)."""
    return np.sin(b1) + np.cos(b2)


def _normalize_to_unit_square(b):
    """
    Affine-rescale a 1D array to [-1, 1], robustly.
    If values are (almost) constant, return zeros.
    """
    b = np.asarray(b)
    b_min = np.min(b)
    b_max = np.max(b)
    if np.isclose(b_max, b_min):
        return np.zeros_like(b)
    return 2.0 * (b - b_min) / (b_max - b_min) - 1.0


def surface_ratio(b1, b2, eps=1e-12):
    """
    Ratio surface in [0,1]:

        U = new_b1 / (new_b1 + new_b2),

    where new_b1 = exp(B1), new_b2 = exp(B2) > 0.
    This ensures U is strictly between 0 and 1 and behaves like a share.
    Now exp(B1) and exp(B2) are used directly as the new bottoms.
    """
    b1 = np.asarray(b1)
    b2 = np.asarray(b2)
    #new_b1 = np.exp(b1)  # New bottom 1 (exp(B1))
    #new_b2 = np.exp(b2)  # New bottom 2 (exp(B2))
    denom = b1 + b2 + eps  # small eps to avoid any possible 0
    return b1 / denom  # Surface ratio as expected


SURFACES = {
    "paraboloid": surface_paraboloid,
    "cone": surface_cone,
    "saddle": surface_saddle,
    "ripples": surface_ripples,
    "ratio": surface_ratio,
    "linear": surface_linear,
}

# ----------------------------------------------------------------------
# Bottom-level AR(1) generator
# ----------------------------------------------------------------------

def generate_ar_processes(phi_1, phi_2, T=1000, case='independent',
                          rho=0.95, scale=0.001, seed=42,
                          make_plots=True):
    """
    Generate two AR(1) processes B1, B2 with either independent or correlated noise.
    """
    np.random.seed(seed)

    b1 = np.zeros(T)
    b2 = np.zeros(T)

    if case == 'independent':
        eps_1 = np.random.normal(0, 0.1, T)
        eps_2 = np.random.normal(0, 0.1, T)
    elif case == 'correlated':
        mean = [0, 0]
        cov = [[1, rho], [rho, 1]]
        eps = scale * np.random.multivariate_normal(mean, cov, T)
        eps_1, eps_2 = eps[:, 0], eps[:, 1]
    else:
        raise ValueError("case must be either 'independent' or 'correlated'")

    for t in range(1, T):
        b1[t] = phi_1 * b1[t - 1] + eps_1[t]
        b2[t] = phi_2 * b2[t - 1] + eps_2[t]

    if case == 'correlated':
        # Just to make structure visible in scatter plots
        b1 *= 5e2
        b2 *= 5e2

    if make_plots:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        df = pd.DataFrame(np.vstack([b1, b2]).T, columns=['b1', 'b2'])
        df.plot(ax=ax[0])
        ax[0].set_title(f"Time series (case={case})")
        ax[0].set_xlabel("t")
        ax[0].set_ylabel("value")

        ax[1].scatter(df['b1'].values, df['b2'].values, s=1)
        ax[1].set_xlabel("b1")
        ax[1].set_ylabel("b2")
        ax[1].set_title("Phase plot")

        plt.tight_layout()
        plt.show()

    return b1, b2


# ----------------------------------------------------------------------
# Plotting utilities: robust 3D surface + scatter (matplotlib + Plotly)
# ----------------------------------------------------------------------

def _plotly_marker_size(scatter_size):
    """Map `scatter_size` to Plotly marker.size (pixels)."""
    s = float(scatter_size)
    if s <= 0:
        return 1
    # If user passed <1 (intended for Matplotlib), scale up modestly
    if s < 1.0:
        return max(1, int(round(s * 4)))
    return max(1, int(round(s)))


def plot_3d_surface(b1, b2, u, surface_name,
                    grid_n=80, surface_alpha=0.6, scatter_size=0.5,
                    use_plotly=True, pct_clip=(1, 99)):
    """
    Plot a single surface + data points.

    - If `use_plotly=True`: opens a Plotly viewer (title contains only the surface name).
    - Otherwise uses Matplotlib 3D.
    - `scatter_size` is interpreted flexibly:
      * for Matplotlib: `s = (max(0.1, scatter_size))**2` (points^2)
      * for Plotly: `marker.size = int(...)` (pixels)
    """
    surf_fun = SURFACES.get(surface_name)
    if surf_fun is None:
        raise ValueError(f"Unknown surface: {surface_name}")

    b1 = np.asarray(b1)
    b2 = np.asarray(b2)
    u = np.asarray(u)

    paired_samples = (u.ndim == 1 and b1.ndim == 1 and b2.ndim == 1 and u.size == b1.size == b2.size)

    # Build clipped grid
    x_lin = np.linspace(np.percentile(b1, pct_clip[0]), np.percentile(b1, pct_clip[1]), grid_n)
    y_lin = np.linspace(np.percentile(b2, pct_clip[0]), np.percentile(b2, pct_clip[1]), grid_n)
    Xg, Yg = np.meshgrid(x_lin, y_lin)

    # Compute Zg robustly
    try:
        Zg = surf_fun(Xg.ravel(), Yg.ravel()).reshape(Xg.shape)
    except Exception:
        Zg = surf_fun(Xg, Yg)

    # For ratio surface, use exp(b1) and exp(b2) for x and z axes
    if surface_name == "ratio":
        exp_b1 = np.exp(Xg)  # exp(B1) for the x-axis
        exp_b2 = np.exp(Yg)  # exp(B2) for the z-axis
    else:
        exp_b1 = Xg
        exp_b2 = Yg

    if use_plotly:
        try:
            import plotly.graph_objects as go
        except Exception as e:
            raise RuntimeError("Plotly not available; install plotly or set use_plotly=False") from e

        surf = go.Surface(x=exp_b1, y=exp_b2, z=Zg, colorscale='Viridis', opacity=surface_alpha, name='surface')
        data = [surf]
        if paired_samples:
            p_size = _plotly_marker_size(scatter_size)
            # For the ratio surface, use exp(b1) and exp(b2)
            if surface_name == "ratio":
                scatter = go.Scatter3d(x=np.exp(b1), y=np.exp(b2), z=u, mode='markers',
                                       marker=dict(size=p_size, color='black'),
                                       name='data')
            else:
                scatter = go.Scatter3d(x=b1, y=b2, z=u, mode='markers',
                                       marker=dict(size=p_size, color='black'),
                                       name='data')
            data.append(scatter)

        fig = go.Figure(data=data)
        fig.update_layout(scene=dict(xaxis_title='B1',
                                     yaxis_title='B2',
                                     zaxis_title='U'),
                          title=f"{surface_name}")
        fig.show()
        return fig




# ----------------------------------------------------------------------
# Main: generate AR(1) + multiple surfaces, save as pickles
# ----------------------------------------------------------------------

def main():
    phi_1 = 0.9
    phi_2 = 0.9
    T = 1000
    data_folder = '../data/'
    os.makedirs(data_folder, exist_ok=True)

    cases = ['independent']
    surfaces = ['paraboloid', 'saddle', 'ripples', 'ratio', 'linear']

    for case in cases:
        # One AR realization per case, reused for all surfaces
        b1, b2 = generate_ar_processes(
            phi_1=phi_1,
            phi_2=phi_2,
            T=T,
            case=case,
            seed=42,
            make_plots=False,  # Disable 2D plot here
        )

        case_tag = 'indep' if case == 'independent' else 'corr'

        u_for_plots = []
        for surface in surfaces:
            surf_fun = SURFACES[surface]
            # evaluate surface using broadcast-friendly inputs
            try:
                u = surf_fun(b1, b2)
            except Exception:
                # fallback: evaluate on meshgrid then ravel to match b1/b2 shapes if needed
                B1, B2 = np.meshgrid(b1, b2)
                u = surf_fun(B1, B2)

            # prepare dataframe rows (flattened)
            U_flat = np.ravel(u)
            B1_flat = np.ravel(np.broadcast_to(b1, U_flat.shape))
            B2_flat = np.ravel(np.broadcast_to(b2, U_flat.shape))
            if surface == 'ratio':
                # For ratio surface, save exp(b1), exp(b2)
                B1_flat = np.ravel(np.exp(b1))
                B2_flat = B1_flat + np.ravel(np.exp(b2))
                df = pd.DataFrame({'U': U_flat, 'B1': B1_flat, 'B2': B2_flat})

            else:
                df = pd.DataFrame({'U': U_flat, 'B1': B1_flat, 'B2': B2_flat})
            file_name = os.path.join(
                data_folder,
                f'{surface}_data_{case_tag}.pkl'
            )
            with open(file_name, 'wb') as f:
                pd.to_pickle(df, f)

            print(f"Saved {file_name}")

            # For plotting prefer paired 1D u matching b1/b2; otherwise None
            if np.asarray(u).ndim == 1 and np.asarray(u).size == b1.size == b2.size:
                u_for_plots.append(np.asarray(u))
            else:
                u_for_plots.append(None)

        # Plot the ratio surface separately (one interactive Plotly window)
        try:
            for i, surface in enumerate(surfaces):
                u_val = u_for_plots[i] if u_for_plots[i] is not None else np.zeros_like(b1)
                # use Plotly for each surface (small markers); adjust scatter_size as needed
                plot_3d_surface(b1, b2, u_val, surface,
                                grid_n=80, surface_alpha=0.6,
                                use_plotly=True, scatter_size=1.5)
        except Exception as e:
            # if Plotly not available or fails, fallback to Matplotlib separate plots
            print("Plotly unavailable or failed, falling back to matplotlib plots:", e)
            for i, surface in enumerate(surfaces):
                u_val = u_for_plots[i] if u_for_plots[i] is not None else np.zeros_like(b1)
                plot_3d_surface(b1, b2, u_val, surface,
                                grid_n=60, surface_alpha=0.6,
                                use_plotly=False, scatter_size=0.4)


if __name__ == '__main__':
    main()