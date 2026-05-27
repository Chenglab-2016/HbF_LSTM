# HbF LSTM Analysis

This repository contains notebooks and helper code for modeling longitudinal fetal hemoglobin (HbF) measurements with recurrent neural networks and related statistical analyses.


## Repository structure

```text
HbF_LSTM/
+-- codes/
    +-- models.py
    +-- utilities.py
    +-- LSTM_*.ipynb
    +-- GEE_no_CTXN_subgroup_3model.ipynb
    +-- raw_hbf_changes_analysis.ipynb
    +-- chort_summary.ipynb
    +-- plot_*.ipynb
```

## Main files

| File | Purpose |
| --- | --- |
| `codes/models.py` | Shared model and data-splitting utilities for the LSTM workflow. Defines `CombinedLSTMModel`, sequence construction helpers, normalization, train/validation/test splitting by patient ID, prediction helpers, and automatic CPU/GPU device selection. |
| `codes/utilities.py` | Shared data utilities for reading SAS files, decoding byte columns, separating background and dynamic datasets, merging datasets, converting dates, encoding object columns, creating directories, and summarizing absolute SHAP values. |
| `codes/LSTM_Cmodel.ipynb` | Trains/evaluates the C model for HbF prediction. |
| `codes/LSTM_Tmodel.ipynb` | Trains/evaluates the T model for HbF prediction. |
| `codes/LSTM_MixModel.ipynb` | Trains/evaluates the mixed model. |
| `codes/LSTM_Cmodel_shap.ipynb` | Computes SHAP-based interpretation outputs for the C model. |
| `codes/LSTM_Tmodel_shap.ipynb` | Computes SHAP-based interpretation outputs for the T model. |
| `codes/LSTM_MixModel_shap.ipynb` | Computes SHAP-based interpretation outputs for the mixed model. |
| `codes/LSTM_loop_ctxn_compare.ipynb` | Compares model behavior w/wo ctxn inputs. |
| `codes/LSTM_Mix_loop_visits_change.ipynb` | Runs mixed-model experiments across visit-count settings. |
| `codes/two_model_prediction_loop.ipynb` | Runs repeated prediction loops for treatment/control model. |
| `codes/GEE_no_CTXN_subgroup_3model.ipynb` | Performs generalized estimating equation (GEE) analyses for 3 subgroup/model comparison without CTXN inputs. |
| `codes/raw_hbf_changes_analysis.ipynb` | Explores raw HbF changes and related descriptive plots. |
| `codes/chort_summary.ipynb` | Summarizes cohort/background/dynamic data characteristics. |
| `codes/plot_R2.ipynb` | Creates model R2 comparison plots. |
| `codes/plot_number_of_visits_R2.ipynb` | Plots model performance by number of visits. |
| `codes/plot_SHAP_impact.ipynb` | Aggregates and visualizes SHAP impact summaries. |

## Expected external folders

Several notebooks use relative paths that expect data and generated artifacts outside the `codes/` folder:

```text
HbF_LSTM/
+-- codes/
+-- data/
|   +-- Apply_MI_Zheng_2023_01/
|       +-- bloodtrans_final.sas7bdat
|       +-- ctxn.sas7bdat
|       +-- edhosp.sas7bdat
|       +-- hu.sas7bdat
|       +-- othertreatment_summary.sas7bdat
+-- results/
    +-- data/
    +-- figures/
    +-- models/
```

Common derived files referenced by the notebooks include:

- `results/data/LSTM_processed_data_dy_filter.pkl`
- `results/data/LSTM_processed_data_ba_filter.pkl`
- `results/data/SCCRIP_ID_dict.pkl`
- model outputs under `results/models/*.h5`
- SHAP, R2, and prediction outputs under `results/data/`
- generated figures under `results/figures/`

These files may contain sensitive or large research artifacts and should be managed according to the relevant data-use and privacy requirements.

## Python dependencies

The notebooks and helper modules use the following core Python packages:

- `pandas`
- `numpy`
- `scikit-learn`
- `tensorflow`
- `matplotlib`
- `seaborn`
- `shap`
- `statsmodels`
- `jupyter`

Install them in a Python environment before running the notebooks. For example:

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn shap statsmodels jupyter
```

TensorFlow will use a GPU if one is available and configured; otherwise the shared model code falls back to CPU.

## Suggested workflow

1. Place the required raw SAS datasets in `data/Apply_MI_Zheng_2023_01/`.
2. Generate or provide the processed longitudinal/background pickle files in `results/data/`.
3. Run the model notebooks for the desired prediction setting:
   - `LSTM_Cmodel.ipynb`
   - `LSTM_Tmodel.ipynb`
   - `LSTM_MixModel.ipynb`
4. Run the SHAP notebooks to compute feature-importance summaries.
5. Run the plotting notebooks to produce R2, visit-count, SHAP-impact, and raw-HbF summary figures.
6. Use the GEE notebook for statistical subgroup comparisons.
