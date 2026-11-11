# Single-reactant rate-law discovery

The project studies equations of the form `r = -dCA/dt = f(CA)`, with every discovered equation restricted to `CA` alone. `Data_generator.ipynb` creates six benchmark reaction laws at three noise levels. `Analysis.ipynb` compares preprocessing methods, PySR equation search, equation simplification, and held-out trajectory accuracy.

## Run the upload site

Use Python 3.10 or newer. PySR initializes its Julia backend when discovery starts; install Julia and let the initial backend setup finish.

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The site accepts a CSV with one observation per row. The recommended column order is time, concentration, trajectory ID; the ID is optional for a single run. Column names and order can vary because the site lets you select the columns.

```csv
time_min,CA,trajectory_id
0.00,0.0500,run_1
0.25,0.0488,run_1
0.50,0.0475,run_1
0.00,0.0750,run_2
0.25,0.0724,run_2
0.50,0.0700,run_2
```

Each trajectory needs at least 12 observations with unique, increasing times. Concentrations must be nonnegative. Savitzky-Golay differentiation requires equally spaced times; raw differentiation and GCV splines can handle nonuniform sampling. Values and units are interpreted as supplied.

The site compares raw differentiation, Savitzky-Golay, and GCV smoothing splines. For every PySR equation it generates bounded algebraic simplifications (including additive and rational numerator/denominator reductions), refits linear coefficients on training rates, and ranks candidates by held-out ODE trajectory reconstruction. A one-standard-error rule favors simpler candidates whose validation performance is statistically close to the best. This follows the analysis notebook's noisy-data selection approach; validation, rather than an in-sample complexity score, guards against overfitting from the larger simplification candidate set. The site returns metrics, plots, candidate equations, warnings, and a downloadable JSON equation card. With several runs, entire trajectories are reserved for validation; with one run, a final time block is held out, which is weaker validation. Equation evaluation does not require PySR, but discovering a law from a new upload does.

## Full benchmark in `Analysis.ipynb`

The previous noisy-data section was a pilot configured for one reaction and one noise level. The revised notebook defaults to all six reactions (`First_Order`, `Second_Order`, `Fractional_Order`, `Saturation`, `Fourth_Order`, `Zero_Order`) at 1%, 3%, and 5% noise: 18 reaction/noise conditions. For each condition it searches with all three preprocessing methods, for up to 54 noisy-data PySR fits. The clean-data baseline separately fits six models, so a full run can perform up to 60 PySR searches. Expect this to take a while.

For each condition, the split is:

- Training: initial concentrations 0.025 and 0.050 M, replicates 1-4.
- Validation: replicate 5 at 0.025 and 0.050 M; used to select preprocessing and a compact equation by ODE trajectory RMSE and the one-standard-error rule. The summary lists all preprocessing methods within one standard error, so near-ties are visible instead of being presented as a decisive method ranking.
- Final test: all five replicates at 0.075 M; used only for the reported accuracy comparison.

The noisy-data simplifier generates candidates and refits coefficients using training observations only. The final tables compare the validation-selected candidate with its original PySR source on the untouched 0.075 M test set. To avoid fitting the noise-dominated tail, training points below three standard deviations of the known synthetic measurement noise are excluded using measured CA; this also drops near-depleted zero-order points. Ground-truth depletion masking is used only for retrospective rate-estimation evaluation.

To try one condition first, set `RUN_REACTIONS = ["First_Order"]` and `RUN_NOISE_LEVELS = [0.05]` in Section 14.1, then run the notebook cells from the imports/data load through Section 15. To run the full study, restore all six reaction names and all three noise levels, then rerun Sections 14 and 15. Results are written to `outputs/all_laws_three_noise_levels/` as three CSV summaries. Do not use final-test results to revise candidate selection.

The saved first-order 5% pilot selected GCV and reported `rate = 0.0985864 * CA`, mean observed-data RMSE 0.004076, and mean clean-reference RMSE 0.001303. This is one pilot condition, not the all-law benchmark; the full grid must be rerun to produce comparable results for all 18 conditions.

## Interpreting the results

The notebook uses known synthetic rates only to retrospectively score preprocessing. Model fitting uses measured concentration data and estimated rates. Validation and final test use trajectory reconstruction by integrating `dCA/dt = -f(CA)`. The final test changes the initial concentration, so it checks prediction beyond the training concentration range. These experiments evaluate equation discovery for each labeled law; they do not train a classifier that identifies an unknown reaction type from a mixed file.

`rate_law_discovery.py` exposes `discover_rate_law`, `predict_rate`, `integrate_rate_law`, and `export_result` for the site and downstream Python use.
