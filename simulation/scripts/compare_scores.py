import numpy as np
import matplotlib.pyplot as plt
from score_functions import compute_crps
import pandas as pd
import os
import jax

def compare_crps(
    test_data: dict,
    base_fc: dict,
    reconciled_fc: dict,
    num_horizons: int,
    levels=('U', 'B1', 'B2'),
    plot=True
):
    """
    Compare CRPS between base and reconciled forecasts.

    Parameters:
        test_data: dict of true values per level
        base_fc: dict of 3D arrays (N, H, S) for each level
        reconciled_fc: dict with per-horizon DataFrames for each level
        num_horizons: number of forecast steps
        levels: which levels to evaluate
        plot: whether to visualize

    Returns:
        crps_results: dict with keys ['base', 'reconciled'], values are dict[level][h] = mean CRPS
    """
    crps_results = {'base': {level: {} for level in levels},
                    'reconciled': {level: {} for level in levels}}

    for h in range(1, num_horizons + 1):
        h_key = f"h={h}"
        for level in levels:
            y_true = test_data[level][h_key].values                    # shape (N,)
            y_base = base_fc[level][:, h-1, :]                         # shape (N, n_samples)
            y_recon = reconciled_fc[h_key][level].values              # shape (N, n_samples)

            base_crps = compute_crps(y_true, y_base)
            recon_crps = compute_crps(y_true, y_recon)

            crps_results['base'][level][h_key] = np.mean(base_crps)
            crps_results['reconciled'][level][h_key] = np.mean(recon_crps)

    if plot:
        horizons = np.arange(1, num_horizons + 1)
        fig, axes = plt.subplots(len(levels), 1, figsize=(10, 4 * len(levels)), sharex=True)

        for i, level in enumerate(levels):
            axes[i].plot(horizons, [crps_results['base'][level][f"h={h}"] for h in horizons],
                         label='Base', marker='o')
            axes[i].plot(horizons, [crps_results['reconciled'][level][f"h={h}"] for h in horizons],
                         label='Reconciled', marker='x')
            axes[i].set_title(f"CRPS Comparison for {level}")
            axes[i].set_ylabel("Mean CRPS")
            axes[i].grid(True, linestyle='--', alpha=0.6)

        axes[-1].set_xlabel("Forecast Horizon")
        axes[0].legend()
        plt.tight_layout()
        plt.show()

    return crps_results

def main():
    fc_folder = '../forecasts'
    hor = 1
    test_data = pd.read_pickle(os.path.join(fc_folder, f'indep_test_dict_hor_{hor}.pkl'))
    ets_fc = pd.read_pickle(os.path.join(fc_folder, f'indep_base_fc_arima_hor_{hor}.pkl'))
    ets_rec_fc = pd.read_pickle(os.path.join(fc_folder, f'rec_fc_arima_hor_{hor}.pkl'))
    crps_comparison = compare_crps(
        test_data=test_data,
        base_fc=ets_fc,
        reconciled_fc=ets_rec_fc,
        num_horizons=1
    )
    crps_comparison.to_pickle(os.path.join(fc_folder, 'crps_comparison.pkl'))




if __name__ == "__main__":
    main()