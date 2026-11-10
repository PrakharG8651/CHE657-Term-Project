# Single-reactant rate-law discovery

This project contains a synthetic reaction dataset and a symbolic-regression workflow for laws of the form

\[
r=-\frac{dC_A}{dt}=f(C_A).
\]

The current `Analysis.ipynb` is a research comparison, not a pre-trained universal classifier. Its final reported experiment is only second-order kinetics at 5% simulated noise. The six stored benchmark laws are documented in `reaction_data/rate_law_parameters.csv`.

## Run the upload site

Use Python 3.10 or newer. PySR also initializes its Julia backend on first use; install Julia and allow the first PySR run to complete its backend setup.

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

### CSV format and column order

Recommended order is **time, concentration, trajectory ID**. Use one observation per row, with one run/replicate ID repeated down that trajectory. The ID is optional for a file containing only one trajectory. The app lets you select columns by name, so a different order is also accepted.

```csv
time_min,CA,trajectory_id
0.00,0.0500,run_1
0.25,0.0488,run_1
0.50,0.0475,run_1
0.00,0.0750,run_2
0.25,0.0724,run_2
0.50,0.0700,run_2
```

In the app, choose `time_min` as the time column, `CA` as concentration, and `trajectory_id` as the trajectory/replicate column. Choose “single trajectory” if the file has no ID column. Names are examples; your column names can differ.

The project dataset `reaction_data/reaction_dataset.csv` has `time_min` and `CA_observed` columns, but it contains several reactions, starting concentrations, noise levels, and replicates together. For analysis, first filter it to the reaction and conditions you want. To retain multiple trajectories in one upload, make a `trajectory_id` from the identifying fields (for example `reaction`, `initial_concentration_mol_L`, `noise_level`, and `replicate`) and arrange columns as `time_min,CA_observed,trajectory_id`. Concentrations must be nonnegative, each trajectory must contain at least 12 observations, and time must be strictly increasing within each trajectory. Values and units are interpreted as supplied.

The app now compares raw differentiation, Savitzky–Golay smoothing/differentiation, and GCV smoothing splines. It trains a separate one-variable PySR search for each method, refits coefficients on training observations, and checks simplified candidates by integrating their rate law over held-out concentration trajectories. A one-standard-error rule favors compact equations whose validation performance is close to the best candidate; when preprocessing methods are statistically tied, Savitzky–Golay is preferred, following the notebook's second-order noisy-data result. With only one trajectory, uncertainty cannot be estimated from independent runs, so the app uses a disclosed 5% tie window to favor Savitzky–Golay. This runs up to three PySR searches per upload, so discovery takes longer than a single search. The results page shows measured and smoothed concentration-versus-time trajectories, rates from the selected preprocessing method versus the selected law, validation metrics, a compact analysis summary, candidate equations, and warnings. It can also simulate a concentration trajectory from the discovered law. The equation and metrics are downloadable as a JSON model card. A downstream Python service can evaluate the exported equation with `predict_rate(equation, CA)` or simulate concentration with `integrate_rate_law(equation, times, CA0)` from `rate_law_discovery.py`.

## What “export the model” means here

The reusable site model is the discovery pipeline plus a JSON equation result for each upload. PySR is a symbolic-search algorithm, so a model fitted to one file is not a fixed predictor for an unrelated file. The equation JSON is portable and does not require PySR for evaluation, while discovering a law from a new upload does require PySR and its Julia backend.

Rate RMSE, MAE, and R² describe agreement with derivative estimates on training observations. The app also reports ODE trajectory RMSE on held-out data: with multiple trajectory IDs, whole runs are held out; with only one run, the final time block is held out, which is weaker than testing an independent experiment. For reliable deployment, upload several independent trajectories over a useful concentration range and avoid trusting equations outside the reported range. The notebook's strongest demonstrated result selected Savitzky–Golay preprocessing by trajectory validation and obtained a clean-reference test RMSE of about 0.00193 for the specific second-order, 5%-noise synthetic case. That result does not establish performance on every law or experimental dataset.

## Notebook handoff

`Analysis.ipynb` already covers raw finite differences, Savitzky–Golay and GCV spline preprocessing, PySR libraries, coefficient refitting, and an ODE-based held-out trajectory comparison. The final experiment is a single second-order/noise condition. For deployment, the site calls the standalone `rate_law_discovery.py` module so it can discover a fresh law from each uploaded dataset.
