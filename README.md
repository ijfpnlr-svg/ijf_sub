# Non-Linear-Reconciliation
*Submission snapshot*

Nonlinear forecast reconciliation becomes necessary when the relationships among time series variables are governed by non-additive constraints — for instance, **ratios**, **maxima**, or physical laws  (**power flow equations**) in energy systems. These nonlinear structures frequently arise in energy networks, transport systems, and engineering domains, where ensuring physically consistent forecasts improves operational reliability and supports better planning and control. Current probabilistic reconciliation methods are although extensive - with optimal solutions for Gaussian distributions, various algorithms for discrete and mixed distributions, the literature is limited to linear contraints. 

In this study, we focus on nonlinear reconciliation, in which the time series are linked by a nonlinear function f. We assume f to be known. We propose four probabilistic methods for nonlinear forecast reconciliation, which extend the existing linear reconciliation methods for the nonlinear case.

## Reconciliation Methods

Two approaches to nonlinear probabilistic forecast reconciliation - (i) **via projection**, and (ii) **via conditioning** are proposed. The different methods for the approaches are coded as reconciliation fucntions in the folder `reconc`.

### Installation

* on linux
sudo apt update
sudo apt install -y libtirpc-dev
sudo apt install r-base-dev

* install uv and set up environment
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

## Simulation

Run the scripts `generate_data.py` ➡️ `base_forecast.py` ➡️ `reconcile.py` in the given order from the folder `simulation/scripts` to reproduce the Relative CRPS score for the simulated surfaces.

To reproduce the runtimes for different reconciliation methods, run the script `simulation/scripts/runtime.py`.

## Demographic Rates of CH


The data is already downloaded and base forecasts are available. To reproduce the Relative CRPS run the script `CH/scripts/reconcile_hybrid.py` and to reproduce the runtimes run the script `CH/scripts/runtime.py`

## Australian tourism rate

The data is already downloaded and base forecasts are available. To reproduce the Relative CRPS run the script `aus_tourism/scripts/reconcile_hybrid.py` and to reproduce the runtimes run the script `aus_tourism/scripts/runtime.py`

