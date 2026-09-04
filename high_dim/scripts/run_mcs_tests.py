
# ============================================================
# High-dimensional Sequential Studentized MCS
# ============================================================

import os
import math
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

RESULTS_FOLDER = "../results"

CRPS_SERIES_LOSS_FILE = os.path.join(
    RESULTS_FOLDER,
    "reconciliation_crps_by_series_time.csv",
)

OUTPUT_PREFIX = "simulation"

MCS_SUMMARY_FILE = os.path.join(
    RESULTS_FOLDER,
    f"{OUTPUT_PREFIX}_sequential_studentized_mcs_crps_summary.csv",
)

MCS_TEXT_SUMMARY_FILE = os.path.join(
    RESULTS_FOLDER,
    f"{OUTPUT_PREFIX}_sequential_studentized_mcs_crps_summary.txt",
)

MCS_LATEX_FILE = os.path.join(
    RESULTS_FOLDER,
    f"{OUTPUT_PREFIX}_sequential_studentized_mcs_tables.tex",
)

MCS_PLOT_FOLDER = os.path.join(
    RESULTS_FOLDER,
    f"{OUTPUT_PREFIX}_sequential_studentized_mcs_plots",
)


METHOD_ORDER = [
    "base",
    "pbu",
    "ols",
    "wls",
    "full",
    "ukf",
]


METHOD_LABELS = {
    "base": "Base",
    "pbu": "PBU",
    "ols": "OLS",
    "wls": "WLS",
    "full": "FULL",
    "ukf": "UKF",
}


SURFACE_ORDER = [
    "paraboloid",
    "saddle",
    "ripples",
]


SURFACE_LABELS = {
    "paraboloid": "Paraboloid",
    "saddle": "Saddle",
    "ripples": "Ripples",
}


REFERENCE_METHOD = "base"

ALPHA = 0.05

N_BOOTSTRAP = 1000

BLOCK_LENGTH = "auto"

RANDOM_SEED = 42

EPSILON = 1e-12

PLOT_DPI = 300


# ============================================================
# LOADING
# ============================================================

def load_crps_series_losses():

    if not os.path.exists(
        CRPS_SERIES_LOSS_FILE
    ):
        raise FileNotFoundError(
            CRPS_SERIES_LOSS_FILE
        )

    df = pd.read_csv(
        CRPS_SERIES_LOSS_FILE
    )

    required_columns = {
        "dimension",
        "n_samples",
        "surface",
        "method",
        "score",
        "series_index",
        "t",
        "loss",
    }

    missing = (
        required_columns
        -
        set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    df["surface"] = (
        df["surface"]
        .astype(str)
        .str.lower()
    )

    df["method"] = (
        df["method"]
        .astype(str)
        .str.lower()
    )

    df = df[
        df["score"]
        .str.lower()
        ==
        "crps"
    ].copy()

    df["dimension"] = (
        df["dimension"]
        .astype(int)
    )

    df["n_samples"] = (
        df["n_samples"]
        .astype(int)
    )

    df["series_index"] = (
        df["series_index"]
        .astype(int)
    )

    df["t"] = (
        df["t"]
        .astype(int)
    )

    df["loss"] = (
        df["loss"]
        .astype(float)
    )

    return df


# ============================================================
# UTILITIES
# ============================================================

def sanitize_for_filename(text):

    text = str(text)

    text = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        text,
    )

    return text.strip("_")


def surface_display_name(surface):

    return SURFACE_LABELS.get(
        surface,
        surface.title(),
    )


def method_display_name(method):

    return METHOD_LABELS.get(
        method,
        method,
    )


def order_methods(methods):

    ordered = [
        m
        for m in METHOD_ORDER
        if m in methods
    ]

    extra = sorted(
        set(methods)
        -
        set(ordered)
    )

    return ordered + extra


def resolve_block_length(n):

    if BLOCK_LENGTH == "auto":
        return max(
            2,
            int(
                round(
                    math.sqrt(n)
                )
            ),
        )

    return int(
        BLOCK_LENGTH
    )


def moving_block_bootstrap_indices(
    n,
    block,
    rng,
):

    n_blocks = int(
        math.ceil(
            n / block
        )
    )

    starts = rng.integers(
        0,
        n - block + 1,
        size=n_blocks,
    )

    indices = []

    for start in starts:
        indices.extend(
            range(
                start,
                start + block,
            )
        )

    return np.asarray(
        indices[:n],
        dtype=int,
    )


# ============================================================
# CRPS STATISTIC
# ============================================================

def build_crps_tensor(group):

    methods = order_methods(
        group["method"].unique()
    )

    times = sorted(
        group["t"].unique()
    )

    series = sorted(
        group["series_index"].unique()
    )


    tensor = np.full(
        (
            len(times),
            len(series),
            len(methods),
        ),
        np.nan,
        dtype=float,
    )


    t_map = {
        x:i
        for i,x in enumerate(times)
    }

    s_map = {
        x:i
        for i,x in enumerate(series)
    }

    m_map = {
        x:i
        for i,x in enumerate(methods)
    }


    for row in group.itertuples(
        index=False
    ):

        tensor[
            t_map[row.t],
            s_map[row.series_index],
            m_map[row.method],
        ] = row.loss


    valid = np.all(
        np.isfinite(
            tensor
        ),
        axis=(1,2),
    )


    tensor = tensor[
        valid
    ]


    return (
        tensor,
        methods,
    )


def compute_relative_crps(
    tensor,
    methods,
):

    base_index = methods.index(
        REFERENCE_METHOD
    )


    mean_scores = np.mean(
        tensor,
        axis=0,
    )


    base_scores = mean_scores[
        :,
        base_index,
    ]


    ratios = (
        mean_scores
        + EPSILON
    ) / (
        base_scores[:,None]
        + EPSILON
    )


    return np.exp(
        np.mean(
            np.log(
                ratios
            ),
            axis=0,
        )
    )


# ============================================================
# STUDENTIZED T_R MCS
# ============================================================

def compute_pairwise_statistics(
    observed,
    bootstrap,
):

    D = (
        observed[:,None]
        -
        observed[None,:]
    )


    D_boot = (
        bootstrap[:,:,None]
        -
        bootstrap[:,None,:]
    )


    se = np.std(
        D_boot,
        axis=0,
        ddof=1,
    )


    se = np.maximum(
        se,
        EPSILON,
    )


    np.fill_diagonal(
        se,
        np.inf,
    )


    t_obs = D / se


    np.fill_diagonal(
        t_obs,
        0,
    )


    centered = (
        D_boot
        -
        D[None,:,:]
    )


    t_boot = centered / se[None,:,:]


    for b in range(
        t_boot.shape[0]
    ):

        np.fill_diagonal(
            t_boot[b],
            0,
        )


    return (
        t_obs,
        t_boot,
    )


def sequential_studentized_mcs(
    observed,
    bootstrap,
    methods,
):

    t_obs, t_boot = compute_pairwise_statistics(
        observed,
        bootstrap,
    )


    active = np.ones(
        len(methods),
        dtype=bool,
    )


    elimination = []

    step = 1


    while active.sum() > 1:

        idx = np.where(
            active
        )[0]


        obs = t_obs[
            np.ix_(
                idx,
                idx,
            )
        ]


        boot = t_boot[
            :,
            idx,
            :
        ][
            :,
            :,
            idx
        ]


        mask = ~np.eye(
            len(idx),
            dtype=bool,
        )


        TR = np.max(
            np.abs(
                obs[mask]
            )
        )


        TR_boot = np.max(
            np.abs(
                boot[:,mask]
            ),
            axis=1,
        )


        critical = np.quantile(
            TR_boot,
            1-ALPHA,
        )


        p = np.mean(
            TR_boot >= TR
        )


        if TR <= critical:
            break


        worst = np.argmax(
            np.max(
                obs,
                axis=1,
            )
        )


        remove = idx[
            worst
        ]


        elimination.append(
            {
                "step": step,
                "method": methods[remove],
                "TR": TR,
                "critical": critical,
                "p": p,
            }
        )


        active[
            remove
        ] = False


        step += 1


    return (
        active,
        elimination,
    )

# ============================================================
# OUTPUT + PLOTTING
# ============================================================

def plot_mcs_crps(
    result,
    dimension,
    n_samples,
    surface,
):

    os.makedirs(
        MCS_PLOT_FOLDER,
        exist_ok=True,
    )


    df = result.sort_values(
        "relative_crps"
    )


    fig, ax = plt.subplots(
        figsize=(8,4)
    )


    for i, row in enumerate(
        df.itertuples(
            index=False
        )
    ):

        label = row.method.upper()

        if row.in_mcs:
            label += " ✓"
        else:
            label += " ×"


        ax.errorbar(
            row.relative_crps,
            i,
            xerr=row.relative_crps_ci,
            fmt="o" if row.in_mcs else "x",
            capsize=4,
        )


    ax.set_yticks(
        np.arange(
            len(df)
        )
    )

    ax.set_yticklabels(
        [
            r.method.upper()
            + (" ✓" if r.in_mcs else " ×")
            for r in df.itertuples()
        ]
    )

    ax.invert_yaxis()

    ax.set_xlabel(
        "GM relative CRPS"
    )

    ax.set_title(
        f"d={dimension}, S={n_samples}, {surface}"
    )

    ax.grid(
        True,
        axis="x",
        alpha=0.3,
    )


    fig.tight_layout()


    filename = (
        f"{OUTPUT_PREFIX}_mcs_crps_"
        f"d{dimension}_"
        f"S{n_samples}_"
        f"{surface}.png"
    )


    path = os.path.join(
        MCS_PLOT_FOLDER,
        filename,
    )


    fig.savefig(
        path,
        dpi=PLOT_DPI,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


    return path



def plot_mcs_test_path(
    elimination,
    dimension,
    n_samples,
    surface,
):

    if len(elimination) == 0:
        return None


    steps = [
        e["step"]
        for e in elimination
    ]

    TR = [
        e["TR"]
        for e in elimination
    ]

    critical = [
        e["critical"]
        for e in elimination
    ]


    fig, ax = plt.subplots(
        figsize=(7,4)
    )


    ax.plot(
        steps,
        TR,
        marker="o",
        label=r"$T_R$",
    )


    ax.plot(
        steps,
        critical,
        marker="s",
        linestyle="--",
        label="critical value",
    )


    for e in elimination:

        ax.annotate(
            e["method"].upper(),
            (
                e["step"],
                e["TR"],
            ),
            xytext=(5,5),
            textcoords="offset points",
        )


    ax.set_xlabel(
        "MCS elimination step"
    )

    ax.set_ylabel(
        "Statistic"
    )


    ax.set_title(
        f"MCS elimination path: d={dimension}, "
        f"S={n_samples}, {surface}"
    )


    ax.legend()

    ax.grid(
        True,
        alpha=0.3,
    )


    fig.tight_layout()


    filename = (
        f"{OUTPUT_PREFIX}_mcs_path_"
        f"d{dimension}_"
        f"S{n_samples}_"
        f"{surface}.png"
    )


    path = os.path.join(
        MCS_PLOT_FOLDER,
        filename,
    )


    fig.savefig(
        path,
        dpi=PLOT_DPI,
        bbox_inches="tight",
    )


    plt.close(
        fig
    )


    return path



# ============================================================
# LATEX OUTPUT
# ============================================================

def latex_escape(
    text
):

    return (
        str(text)
        .replace("_", r"\_")
    )



def write_latex_tables(
    summary_df
):

    lines = []


    # --------------------------------------------------------
    # Elimination tables
    # --------------------------------------------------------

    for (
        dimension,
        n_samples,
        surface,
    ), group in summary_df.groupby(
        [
            "dimension",
            "n_samples",
            "surface",
        ]
    ):


        lines.append(
            r"\begin{table}[h!]"
        )

        lines.append(
            r"\centering"
        )

        lines.append(
            r"\begin{tabular}{lcccc}"
        )

        lines.append(
            r"\toprule"
        )

        lines.append(
            r"Step & Eliminated method & $T_R$ & Critical value & $p$-value \\"
        )

        lines.append(
            r"\midrule"
        )


        eliminated = group[
            ~group.in_mcs
        ].sort_values(
            "elimination_step"
        )


        for row in eliminated.itertuples():

            lines.append(
                f"{int(row.elimination_step)} & "
                f"{row.method.upper()} & "
                f"{row.TR:.4f} & "
                f"{row.critical:.4f} & "
                f"{row.p_value:.4f} \\\\"
            )


        lines.append(
            r"\bottomrule"
        )


        lines.append(
            r"\end{tabular}"
        )


        lines.append(
            rf"\caption{{Sequential studentized MCS elimination "
            rf"for {surface}, $d={dimension}$, $S={n_samples}$.}}"
        )


        lines.append(
            rf"\label{{tab:mcs_steps_{dimension}_{n_samples}_{surface}}}"
        )


        lines.append(
            r"\end{table}"
        )


        lines.append("")



    # --------------------------------------------------------
    # Final membership tables
    # --------------------------------------------------------

    for (
        dimension,
        n_samples,
    ), group in summary_df.groupby(
        [
            "dimension",
            "n_samples",
        ]
    ):

        surfaces = sorted(
            group.surface.unique()
        )


        lines.append(
            r"\begin{table}[h!]"
        )

        lines.append(
            r"\centering"
        )

        lines.append(
            "\\begin{tabular}{"
            + "c" * len(surfaces)
            + "}"
        )

        lines.append(
            r"\toprule"
        )

        lines.append(
            " & ".join(
                [
                    s.title()
                    for s in surfaces
                ]
            )
            + r" \\"
        )


        lines.append(
            r"\midrule"
        )


        ordered = {}

        max_len = 0


        for surface in surfaces:

            tmp = group[
                group.surface == surface
            ].copy()


            retained = tmp[
                tmp.in_mcs
            ].sort_values(
                "relative_crps"
            )


            eliminated = tmp[
                ~tmp.in_mcs
            ].sort_values(
                "elimination_step",
                ascending=False,
            )


            ordered[surface] = pd.concat(
                [
                    retained,
                    eliminated,
                ]
            )


            max_len = max(
                max_len,
                len(
                    ordered[surface]
                )
            )


        for i in range(
            max_len
        ):

            row = []

            for surface in surfaces:

                data = ordered[surface]

                if i >= len(data):

                    row.append(
                        ""
                    )

                    continue


                r = data.iloc[i]


                if r.in_mcs:

                    row.append(
                        rf"\color{{brown}}{{\textbf{{{r.method.upper()}}}}}"
                    )

                else:

                    row.append(
                        r.method.upper()
                    )


            lines.append(
                " & ".join(row)
                + r" \\"
            )


        lines.append(
            r"\bottomrule"
        )


        lines.append(
            r"\end{tabular}"
        )


        lines.append(
            rf"\caption{{Final sequential studentized MCS membership "
            rf"for $d={dimension}$ and $S={n_samples}$. "
            r"Bold brown entries indicate methods retained in the final "
            r"MCS set. Eliminated methods are ordered according to the "
            r"sequential elimination path, with the first eliminated "
            r"method shown at the bottom.}}"
        )


        lines.append(
            rf"\label{{tab:mcs_membership_d{dimension}_S{n_samples}}}"
        )


        lines.append(
            r"\end{table}"
        )


        lines.append("")


    with open(
        os.path.join(
            RESULTS_FOLDER,
            f"{OUTPUT_PREFIX}_mcs_tables.tex",
        ),
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "\n".join(lines)
        )



# ============================================================
# MAIN
# ============================================================

def main():

    df = load_crps_series_losses()


    results = []


    grouped = df.groupby(
        [
            "dimension",
            "n_samples",
            "surface",
        ]
    )


    rng = np.random.default_rng(
        RANDOM_SEED
    )


    for (
        dimension,
        n_samples,
        surface,
    ), group in grouped:


        print(
            f"Running MCS: d={dimension}, "
            f"S={n_samples}, {surface}"
        )


        tensor, methods = build_crps_tensor(
            group
        )


        observed = compute_relative_crps(
            tensor,
            methods,
        )


        bootstrap = np.zeros(
            (
                N_BOOTSTRAP,
                len(methods),
            )
        )


        block = (
            int(
                round(
                    math.sqrt(
                        tensor.shape[0]
                    )
                )
            )
            if BLOCK_LENGTH == "auto"
            else BLOCK_LENGTH
        )


        for b in range(
            N_BOOTSTRAP
        ):

            idx = moving_block_bootstrap_indices(
                tensor.shape[0],
                block,
                rng,
            )


            bootstrap[b] = compute_relative_crps(
                tensor[idx],
                methods,
            )



        retained, elimination = sequential_studentized_mcs(
            observed,
            bootstrap,
            methods,
        )


        plot_mcs_crps(
            pd.DataFrame(
                {
                    "method": methods,
                    "relative_crps": observed,
                    "in_mcs": retained,
                    "relative_crps_ci": np.std(
                        bootstrap,
                        axis=0,
                    ),
                }
            ),
            dimension,
            n_samples,
            surface,
        )


        plot_mcs_test_path(
            elimination,
            dimension,
            n_samples,
            surface,
        )


        elimination_map = {
            e["method"]: e
            for e in elimination
        }


        for i, method in enumerate(methods):

            e = elimination_map.get(
                method,
                {}
            )


            results.append(
                {
                    "dimension": dimension,
                    "n_samples": n_samples,
                    "surface": surface,
                    "method": method,
                    "relative_crps": observed[i],
                    "in_mcs": bool(
                        retained[i]
                    ),
                    "elimination_step": e.get(
                        "step",
                        np.nan,
                    ),
                    "TR": e.get(
                        "TR",
                        np.nan,
                    ),
                    "critical": e.get(
                        "critical",
                        np.nan,
                    ),
                    "p_value": e.get(
                        "p",
                        np.nan,
                    ),
                }
            )


    summary = pd.DataFrame(
        results
    )


    summary.to_csv(
        os.path.join(
            RESULTS_FOLDER,
            f"{OUTPUT_PREFIX}_mcs_summary.csv",
        ),
        index=False,
    )


    write_latex_tables(
        summary
    )


    print(
        "MCS analysis completed."
    )



if __name__ == "__main__":
    main()
