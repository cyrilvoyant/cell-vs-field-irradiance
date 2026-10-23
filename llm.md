# Cell versus field irradiance: scientific use contract

## Scientific question and scope

How much spatial information helps short-term irradiance forecasting when model cost is counted as well as error? The study keeps a common forecasting architecture for its main comparison and changes the representation: cell, neighborhood, whole field or compressed field. It does not claim spatial information is useless or that a small model always wins. Its methodological point is to justify added spatial complexity against a matched per-cell reference.

Authority: the author's manuscript *How Much Space Is Enough for Short-Term Satellite Irradiance Forecasting?*, especially Data, Forecast set-up and evaluation set, and Materials and Methods, compared with source commit `5e13a7577247e8f96f7410edc61be3ef2b97be86`. The manuscript is in preparation, not a published journal article. This contract covers the publicly distributed **hourly HelioClim-3/Corsica benchmark**. A newer local manuscript contains additional experiments; do not claim this snapshot reproduces the entire evolving manuscript.

## Entry points and execution cost

`run_all.ps1` orchestrates MATLAB and optional Python training. Its default `tables` mode only reconstructs two LaTeX result tables from archived results. `light` trains models with reduced repetition/settings; `full` runs the larger campaign. A table rebuild is not a model retraining or independent validation. Do not launch expensive training merely to inspect a result.

```powershell
.\run_all.ps1 -Mode tables -Matlab 'C:\Program Files\MATLAB\R2025b\bin\matlab.exe'
```

The path is an example: use the installed MATLAB executable. See README for dependencies, modes and parameters. `Draws`, `Payload`, `Codes`, `UnetBase`, `LstmReps`, `Patch` and `Only` configure light mode; full mode replaces the light settings. MATLAB's scalar metric/field functions are not interchangeable remote API calls.

## Inputs and units

- `Basic/GHI_HC3.mat`: 17,520 hourly observations at 1,148 points in the distributed two-year extract. GHI is in W/m². Preserve the hourly averaging convention; do not silently substitute hourly energy without conversion.
- `Basic/geopoint.mat`: point latitude and longitude, degrees. The gridded field uses rows for latitude (south first) and columns for longitude (west first).
- The experiment interpolates onto a 32 by 32 grid; the 722 cells inside the data support are evaluated. Data-validity masks and solar-elevation evaluation masks are distinct.
- `results/hyper_frozen.mat`: historical hyperparameter choices from the training-year sweep. `results/scale_ref.mat`: test-set mean used for error normalization, not a model fitting input.
- `paper_results/results`: frozen result archives for table mode. Keep archive provenance and software commit alongside any exported table.

The code is MIT licensed; HelioClim-3 data have separate SoDa terms. A software license is not a data-redistribution license. See Basic/README.md.

## Temporal protocol and causality

The published repository configuration fits on 2005 and evaluates on 2006. Each origin reads the preceding 24 hourly samples and predicts the next 24, directly, without feeding forecasts back as observations. Histories and targets remain within their chronological split. `hms_codex_origins([1 8760],24,24)` gives 8,713 origins, from 24 through 8,736: it excludes cross-year history as well as targets beyond the split.

Model settings are selected on the training year without an internal validation holdout. This is a stated experimental choice, not nested cross-validation. The time origin has a declared residual uncertainty of about one day in `hms_config.m`; do not erase that qualification when interpreting solar masks. Changing time resolution changes the physical meaning of a 24-step horizon.

## Scoring, quantities and interpretation

All models are scored on the same geometry-based, per-cell target mask: solar elevation strictly above 5 degrees. Forecasts are forced to zero below 0 degrees. Those night values are outside the scored subset; this forcing does not improve the reported score.

Errors are pooled over the selected cells and times. RMSE and MAE have unit W/m²; nRMSE, nMAE and nMBE divide by the same documented mean of scored test-year observations (407.9003 W/m² in the frozen extract). Do not confuse this denominator with standard deviation, peak irradiance, a per-cell mean or the mean of model-level normalized errors. The test mean normalizes reported scores; it is not a predictor feature.

NICE ratios use **simple persistence at the same horizon**, with pooled L1, L2 and L3 errors and the same mask. Cyclic persistence is another competing forecast, not a replacement NICE denominator. `hms_metrics_h` reports horizons 1,2,3,6,12,24 hours; the pooled 1–24-hour nRMSE and the mean of those six horizon summaries are different aggregations. Gamma uses this study's spatial tolerance of one pixel, intensity tolerance 50 W/m² and no temporal search. Do not transplant different temporal-gamma tolerances from another repository.

Stored parameter counts measure model storage, not training energy or total runtime. A compression ratio or a good reconstruction does not by itself imply better forecasts. The public results concern this dataset, protocol and tested model budget; no universal ranking follows.

## Minimal verification without fitting

```matlab
o = hms_codex_origins([1 8760],24,24);
assert(numel(o)==8713 && o(1)==24 && o(end)==8736);
```

For table reconstruction, run `hms_results_tables(output_directory)` with `paper_results` as current directory and the repository on the MATLAB path. It reads `paper_results/results` and writes `ladder.tex` and `routes.tex` to the chosen output directory. Compare reconstructed tables to the distributed tables; do not claim that this reruns the underlying fits.

On 23 October 2026, reconstruction succeeded (42 collected model configurations). Comparison with the current working manuscript found the compression-route table equal except for the generator comment, but the main table had revised labels and LSTM-PCA storage counts. The archived counts are therefore version-specific; reconciliation with the newer accounting remains open. Do not report all tables as identical to the latest manuscript. The generated table files are outputs, not tracked files shipped under `paper_results/tables` in this snapshot.

## Attribution and status

Manuscript authors: Cyril Voyant, Andreas Kazantzidis and Jan Kleissl. Status: manuscript in preparation. Software concept DOI: https://doi.org/10.5281/zenodo.23562557 ; archived v0.1.0: https://doi.org/10.5281/zenodo.23562558 . These are software identifiers, not a journal-article DOI. Cite the actual software commit used. Later documentation is not automatically included in the existing archive.

`websemantic.json` and `ontology/` document discovery and scientific interpretation. They do not deploy an adapter, grant data rights or certify every model. No original study file is modified by this documentation.
