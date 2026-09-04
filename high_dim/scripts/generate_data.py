import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# High-dimensional surface definitions
# ----------------------------------------------------------------------

def _validate_bottom_matrix(B):
    """
    Validate that B is a 2D array with shape (T, d).
    """

    B = np.asarray(
        B,
        dtype=float,
    )

    if B.ndim != 2:
        raise ValueError(
            f"B must be 2D with shape (T, d), got {B.shape}"
        )

    return B


def surface_paraboloid(B):
    """
    High-dimensional paraboloid:

        U = B1^2 + B2^2 + ... + Bd^2

    For d = 2:

        U = B1^2 + B2^2
    """

    B = _validate_bottom_matrix(
        B
    )

    return np.sum(
        B ** 2,
        axis=1,
    )


def surface_linear(B):
    """
    High-dimensional linear surface:

        U = B1 + B2 + ... + Bd
    """

    B = _validate_bottom_matrix(
        B
    )

    return np.sum(
        B,
        axis=1,
    )


def surface_saddle(B):
    """
    High-dimensional saddle:

        U = sum(first half of B_j^2)
            - sum(second half of B_j^2)

    For d = 2:

        U = B1^2 - B2^2
    """

    B = _validate_bottom_matrix(
        B
    )

    d = B.shape[
        1
    ]

    if d % 2 != 0:
        raise ValueError(
            f"Saddle surface requires an even number of dimensions, got d={d}"
        )

    half = d // 2

    positive_part = np.sum(
        B[
            :,
            :half,
        ] ** 2,
        axis=1,
    )

    negative_part = np.sum(
        B[
            :,
            half:,
        ] ** 2,
        axis=1,
    )

    return (
        positive_part
        - negative_part
    )


def surface_ripples(B):
    """
    High-dimensional ripple surface:

        U = sin(B1) + cos(B2) + sin(B3) + cos(B4) + ...

    For d = 2:

        U = sin(B1) + cos(B2)
    """

    B = _validate_bottom_matrix(
        B
    )

    U = np.zeros(
        B.shape[
            0
        ],
        dtype=float,
    )

    # B1, B3, B5, ... use sine.
    U += np.sum(
        np.sin(
            B[
                :,
                0::2,
            ]
        ),
        axis=1,
    )

    # B2, B4, B6, ... use cosine.
    U += np.sum(
        np.cos(
            B[
                :,
                1::2,
            ]
        ),
        axis=1,
    )

    return U


SURFACES = {
    "paraboloid": surface_paraboloid,
    "saddle": surface_saddle,
    "ripples": surface_ripples,
    # "linear": surface_linear,
}


# ----------------------------------------------------------------------
# Bottom-level AR(1) generator
# ----------------------------------------------------------------------

def generate_ar_processes(
    phi,
    n_dimensions,
    T=1000,
    case="independent",
    rho=0.95,
    scale=0.1,
    seed=42,
    make_plots=False,
):
    """
    Generate d bottom-level AR(1) processes.

    For case='independent', this intentionally uses np.random.seed(seed)
    and sequential calls to np.random.normal(0, 0.1, T), matching the
    original two-bottom script. Thus, for d=2 and seed=42, B1 and B2 are
    generated in the same way as the old script.

    Each bottom-level process follows:

        B_j,t = phi_j * B_j,t-1 + epsilon_j,t

    Parameters
    ----------
    phi : float or array-like
        AR(1) coefficient. If scalar, the same value is used for all
        bottom-level processes.

    n_dimensions : int
        Number of bottom-level time series.

    T : int
        Number of time steps.

    case : str
        Either 'independent' or 'correlated'.

    rho : float
        Equicorrelation parameter for correlated innovations.

    scale : float
        Innovation scale. For independent case, the default is 0.1 to match
        the original two-bottom script.

    seed : int
        Random seed.

    make_plots : bool
        If True, plot generated bottom-level processes.

    Returns
    -------
    B : np.ndarray
        Array with shape (T, n_dimensions).
    """

    if n_dimensions < 1:
        raise ValueError(
            "n_dimensions must be at least 1"
        )

    if T < 2:
        raise ValueError(
            "T must be at least 2"
        )

    if scale <= 0:
        raise ValueError(
            "scale must be positive"
        )

    phi = np.asarray(
        phi,
        dtype=float,
    )

    if phi.ndim == 0:
        phi = np.full(
            n_dimensions,
            float(
                phi
            ),
        )

    elif (
        phi.ndim == 1
        and phi.size == n_dimensions
    ):
        phi = phi.copy()

    else:
        raise ValueError(
            "phi must be either a scalar or a 1D array "
            f"of length {n_dimensions}"
        )

    B = np.zeros(
        (
            T,
            n_dimensions,
        ),
        dtype=float,
    )

    np.random.seed(
        seed
    )

    if case == "independent":

        innovations = np.zeros(
            (
                T,
                n_dimensions,
            ),
            dtype=float,
        )

        for j in range(
            n_dimensions
        ):
            innovations[
                :,
                j,
            ] = np.random.normal(
                0.0,
                scale,
                T,
            )

    elif case == "correlated":

        if not (
            -1.0
            < rho
            < 1.0
        ):
            raise ValueError(
                "rho must be between -1 and 1."
            )

        covariance = np.full(
            (
                n_dimensions,
                n_dimensions,
            ),
            rho,
            dtype=float,
        )

        np.fill_diagonal(
            covariance,
            1.0,
        )

        innovations = scale * np.random.multivariate_normal(
            mean=np.zeros(
                n_dimensions
            ),
            cov=covariance,
            size=T,
        )

    else:
        raise ValueError(
            "case must be either 'independent' or 'correlated'"
        )

    for t in range(
        1,
        T,
    ):
        B[
            t,
            :,
        ] = (
            phi
            * B[
                t - 1,
                :,
            ]
            + innovations[
                t,
                :,
            ]
        )

    if make_plots:
        plot_bottom_processes(
            B
        )

    return B


# ----------------------------------------------------------------------
# Bottom-level plotting
# ----------------------------------------------------------------------

def plot_bottom_processes(B):
    """
    Plot generated bottom-level processes.

    For d=2:
        - time-series plot
        - phase plot

    For d>2:
        - time-series plot only
    """

    B = _validate_bottom_matrix(
        B
    )

    _, d = B.shape

    columns = [
        f"B{i + 1}"
        for i in range(
            d
        )
    ]

    df = pd.DataFrame(
        B,
        columns=columns,
    )

    if d == 2:
        fig, ax = plt.subplots(
            1,
            2,
            figsize=(
                12,
                4,
            ),
        )

        df.plot(
            ax=ax[
                0
            ],
        )

        ax[
            0
        ].set_title(
            f"Bottom-level time series, d={d}"
        )

        ax[
            0
        ].set_xlabel(
            "t"
        )

        ax[
            0
        ].set_ylabel(
            "value"
        )

        ax[
            1
        ].scatter(
            df[
                "B1"
            ].values,
            df[
                "B2"
            ].values,
            s=2,
        )

        ax[
            1
        ].set_xlabel(
            "B1"
        )

        ax[
            1
        ].set_ylabel(
            "B2"
        )

        ax[
            1
        ].set_title(
            "Phase plot"
        )

    else:
        fig, ax = plt.subplots(
            figsize=(
                12,
                5,
            ),
        )

        df.plot(
            ax=ax,
            legend=(
                d <= 10
            ),
        )

        ax.set_title(
            f"Bottom-level time series, d={d}"
        )

        ax.set_xlabel(
            "t"
        )

        ax.set_ylabel(
            "value"
        )

    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# 2D surface evaluation for plotting
# ----------------------------------------------------------------------

def evaluate_2d_surface(
    b1,
    b2,
    surface_name,
):
    """
    Evaluate one of the high-dimensional surfaces in the special d=2 case.
    This helper is used only for visualization.
    """

    b1 = np.asarray(
        b1,
        dtype=float,
    )

    b2 = np.asarray(
        b2,
        dtype=float,
    )

    if b1.shape != b2.shape:
        raise ValueError(
            f"b1 and b2 must have the same shape, "
            f"got {b1.shape} and {b2.shape}"
        )

    B = np.column_stack(
        [
            b1.ravel(),
            b2.ravel(),
        ]
    )

    surface_function = SURFACES.get(
        surface_name
    )

    if surface_function is None:
        raise ValueError(
            f"Unknown surface '{surface_name}'. "
            f"Available surfaces: {list(SURFACES)}"
        )

    U = surface_function(
        B
    )

    return U.reshape(
        b1.shape
    )


# ----------------------------------------------------------------------
# Plotting utilities: 3D surface + scatter
# ----------------------------------------------------------------------

def _plotly_marker_size(
    scatter_size,
):
    """
    Map scatter_size to Plotly marker.size in pixels.
    """

    size = float(
        scatter_size
    )

    if size <= 0:
        return 1

    if size < 1.0:
        return max(
            1,
            int(
                round(
                    size
                    * 4
                )
            ),
        )

    return max(
        1,
        int(
            round(
                size
            )
        ),
    )


def plot_3d_surface(
    b1,
    b2,
    u,
    surface_name,
    grid_n=80,
    surface_alpha=0.6,
    scatter_size=1.5,
    use_plotly=True,
    pct_clip=(
        1,
        99,
    ),
):
    """
    Plot a 2D-bottom / 1D-upper surface.

    Used only when d=2.
    """

    if surface_name not in SURFACES:
        raise ValueError(
            f"Unknown surface '{surface_name}'. "
            f"Available surfaces: {list(SURFACES)}"
        )

    b1 = np.asarray(
        b1,
        dtype=float,
    )

    b2 = np.asarray(
        b2,
        dtype=float,
    )

    u = np.asarray(
        u,
        dtype=float,
    )

    if not (
        b1.ndim == 1
        and b2.ndim == 1
        and u.ndim == 1
        and b1.size == b2.size == u.size
    ):
        raise ValueError(
            "b1, b2, and u must be paired 1D arrays "
            "with the same length"
        )

    x_lin = np.linspace(
        np.percentile(
            b1,
            pct_clip[
                0
            ],
        ),
        np.percentile(
            b1,
            pct_clip[
                1
            ],
        ),
        grid_n,
    )

    y_lin = np.linspace(
        np.percentile(
            b2,
            pct_clip[
                0
            ],
        ),
        np.percentile(
            b2,
            pct_clip[
                1
            ],
        ),
        grid_n,
    )

    Xg, Yg = np.meshgrid(
        x_lin,
        y_lin,
    )

    Zg = evaluate_2d_surface(
        Xg,
        Yg,
        surface_name,
    )

    if use_plotly:
        try:
            import plotly.graph_objects as go

        except ImportError as exc:
            raise RuntimeError(
                "Plotly is not available. "
                "Install plotly or set use_plotly=False."
            ) from exc

        surface = go.Surface(
            x=Xg,
            y=Yg,
            z=Zg,
            colorscale="Viridis",
            opacity=surface_alpha,
            name="surface",
        )

        scatter = go.Scatter3d(
            x=b1,
            y=b2,
            z=u,
            mode="markers",
            marker=dict(
                size=_plotly_marker_size(
                    scatter_size
                ),
                color="black",
            ),
            name="data",
        )

        fig = go.Figure(
            data=[
                surface,
                scatter,
            ]
        )

        fig.update_layout(
            title=surface_name,
            scene=dict(
                xaxis_title="B1",
                yaxis_title="B2",
                zaxis_title="U",
            ),
        )

        fig.show()

        return fig

    fig = plt.figure(
        figsize=(
            9,
            7,
        )
    )

    ax = fig.add_subplot(
        111,
        projection="3d",
    )

    ax.plot_surface(
        Xg,
        Yg,
        Zg,
        alpha=surface_alpha,
    )

    ax.scatter(
        b1,
        b2,
        u,
        s=max(
            0.1,
            float(
                scatter_size
            ),
        ) ** 2,
    )

    ax.set_xlabel(
        "B1"
    )

    ax.set_ylabel(
        "B2"
    )

    ax.set_zlabel(
        "U"
    )

    ax.set_title(
        surface_name
    )

    plt.tight_layout()
    plt.show()

    return fig


# ----------------------------------------------------------------------
# Dataframe creation
# ----------------------------------------------------------------------

def create_dataset(
    B,
    U,
):
    """
    Create a dataframe with columns:

        U, B1, B2, ..., Bd
    """

    B = _validate_bottom_matrix(
        B
    )

    U = np.asarray(
        U,
        dtype=float,
    )

    if U.ndim != 1:
        raise ValueError(
            f"U must be 1D, got {U.shape}"
        )

    if B.shape[
        0
    ] != U.size:
        raise ValueError(
            "B and U must contain the same number "
            f"of observations: {B.shape[0]} != {U.size}"
        )

    data = {
        "U": U,
    }

    for j in range(
        B.shape[
            1
        ]
    ):
        data[
            f"B{j + 1}"
        ] = B[
            :,
            j,
        ]

    return pd.DataFrame(
        data
    )


# ----------------------------------------------------------------------
# Main simulation
# ----------------------------------------------------------------------

def main():
    phi = 0.9
    T = 1000
    scale = 0.1
    seed = 42

    data_folder = "../data/"

    os.makedirs(
        data_folder,
        exist_ok=True,
    )

    cases = [
        "independent",
        # "correlated",
    ]

    dimensions = [
        10,
        20,
        50,
        100,
    ]

    surfaces = [
        "paraboloid",
        "saddle",
        "ripples",
        # "linear",
    ]

    save_legacy_d2_names = True

    for case in cases:
        case_tag = (
            "indep"
            if case == "independent"
            else "corr"
        )

        for d in dimensions:
            print()
            print("=" * 70)
            print(
                f"Case={case}, bottom-level dimension d={d}"
            )
            print("=" * 70)

            # One AR realization per case and dimension.
            # The same realization is reused for all surfaces.
            B = generate_ar_processes(
                phi=phi,
                n_dimensions=d,
                T=T,
                scale=scale,
                seed=seed,
                case=case,
                make_plots=False,
            )

            U_for_plots = {}

            for surface_name in surfaces:
                surface_function = SURFACES[
                    surface_name
                ]

                U = surface_function(
                    B
                )

                df = create_dataset(
                    B=B,
                    U=U,
                )

                # Main high-dimensional file name.
                high_dim_file_name = os.path.join(
                    data_folder,
                    f"{surface_name}_data_{case_tag}_d{d}.pkl",
                )

                df.to_pickle(
                    high_dim_file_name
                )

                print(
                    f"Saved {high_dim_file_name} "
                    f"with shape {df.shape}"
                )

                # Optional legacy d=2 file name, compatible with the old
                # two-bottom scripts.
                if (
                    save_legacy_d2_names
                    and d == 2
                ):
                    legacy_file_name = os.path.join(
                        data_folder,
                        f"{surface_name}_data_{case_tag}.pkl",
                    )

                    df.to_pickle(
                        legacy_file_name
                    )

                    print(
                        f"Saved legacy d=2 file {legacy_file_name} "
                        f"with shape {df.shape}"
                    )

                if d == 2:
                    U_for_plots[
                        surface_name
                    ] = U

            # 3D plots only for d=2.
            if d == 2:
                b1 = B[
                    :,
                    0,
                ]

                b2 = B[
                    :,
                    1,
                ]

                try:
                    for surface_name in surfaces:
                        plot_3d_surface(
                            b1=b1,
                            b2=b2,
                            u=U_for_plots[
                                surface_name
                            ],
                            surface_name=surface_name,
                            grid_n=80,
                            surface_alpha=0.6,
                            use_plotly=True,
                            scatter_size=1.5,
                        )

                except Exception as exc:
                    print(
                        "Plotly unavailable or failed. "
                        "Falling back to Matplotlib:",
                        exc,
                    )

                    for surface_name in surfaces:
                        plot_3d_surface(
                            b1=b1,
                            b2=b2,
                            u=U_for_plots[
                                surface_name
                            ],
                            surface_name=surface_name,
                            grid_n=60,
                            surface_alpha=0.6,
                            use_plotly=False,
                            scatter_size=0.4,
                        )


if __name__ == "__main__":
    main()