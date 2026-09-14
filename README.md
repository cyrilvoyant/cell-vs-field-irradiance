# cell-vs-field-irradiance

**How much space is enough for short-term satellite irradiance forecasting?** This repository runs the benchmark of the paper of that name on two years of hourly HelioClim-3 irradiance over Corsica. One forecaster is held fixed while its input changes: a single cell, a small patch, the whole field, or the field compressed by one of seven methods. Every route is scored on the same cells, and the number of parameters each one stores is counted next to its error.

**These are the paper's own files.** MATLAB fits and scores every model, Python trains the U-Net and the LSTM, and one PowerShell script runs the lot. Nothing was rewritten: the MATLAB files were renamed from `rt_` to `hms_` and nothing else changed (see [Checks](#checks-done-on-this-repository)).

> C. Voyant, *How Much Space Is Enough for Short-Term Satellite Irradiance Forecasting?*, manuscript in preparation.

---

## Contents

- [Requirements](#requirements)
- [Quick start](#quick-start)
- [The three modes](#the-three-modes)
- [Parameters of run_all.ps1](#parameters-of-run_allps1)
- [What each step fits](#what-each-step-fits)
- [Data and frozen inputs](#data-and-frozen-inputs)
- [How a forecast is scored](#how-a-forecast-is-scored)
- [Results of the paper](#results-of-the-paper)
- [Results of a light run](#results-of-a-light-run)
- [A check of the full protocol](#a-check-of-the-full-protocol)
- [Files](#files)
- [Checks done on this repository](#checks-done-on-this-repository)
- [Citation and licence](#citation-and-licence)

---

## Requirements

| tool | tested with | needed for |
|---|---|---|
| MATLAB | R2025b Update 1 | everything |
| Image Processing Toolbox | | Radon transform, field builder |
| Wavelet Toolbox | | wavelet compression |
| Deep Learning Toolbox | | autoencoder compression |
| Python | 3.12.9 | U-Net and LSTM only |
| numpy, scipy, h5py, torch | 2.1.3, 1.15.2, 3.13.0, 2.6.0 (CPU) | U-Net and LSTM only |
| Windows PowerShell | 5.1 | `run_all.ps1` |

The toolbox list is the one MATLAB's `requiredFilesAndProducts` returns for the six entry files.

```bash
python -m pip install -r python/requirements.txt
```

MATLAB calls the interpreter by the name `python`. If several are installed, pass the right one with `-Python`, and `run_all.ps1` puts it first on the PATH.

---

## Quick start

```bash
git clone https://github.com/cyrilvoyant/cell-vs-field-irradiance.git
```

Rebuild the paper's result tables from its archives. No model is fitted and it takes about 12 seconds:

```bash
powershell -ExecutionPolicy Bypass -File .\run_all.ps1
```

Run every model of the paper once, on a light setting:

```bash
powershell -ExecutionPolicy Bypass -File .\run_all.ps1 -Mode light
```

Run only some steps, or other settings; lists are separated by commas:

```bash
powershell -ExecutionPolicy Bypass -File .\run_all.ps1 -Mode light -Only bench,blend -Codes dct,pca -Payload 49
```

`-ExecutionPolicy Bypass` applies to that one PowerShell process and changes no system setting.

---

## The three modes

| mode | what it does | output | time |
|---|---|---|---|
| `tables` (default) | reads the paper's archives in `paper_results\results` and writes its two result tables | `paper_results\tables\ladder.tex`, `routes.tex` | 0.2 min on a laptop |
| `light` | fits and scores every model of the paper once, with the settings below | `results\*.mat`, `results\tables\*.tex`, `results\logs\` | see [Results of a light run](#results-of-a-light-run) |
| `full` | the campaign as the paper ran it | same as `light` | the archives record about 58 hours of fitting for these models on the author's desktop |

The bench, patch, U-Net and LSTM steps resume: a step that finds its output file in `results\` skips the models that file already holds. BLEND is recomputed each time. Move the old files away for a fresh run.

---

## Parameters of run_all.ps1

Every value is set by hand. Nothing is tuned by the script.

| parameter | default | paper (`full`) | meaning |
|---|---|---|---|
| `-Mode` | `tables` | | `tables`, `light` or `full` |
| `-Draws` | 1 | 50 | random hidden layers drawn per ELM model; the median one is kept |
| `-Payload` | 196 | 49, 196, 392 | numbers kept per map by every compression method, and by the compressed LSTM |
| `-Codes` | radon, dct, wavelet, pca, randproj, subsample, autoencoder | same | the compression methods feeding the ELM |
| `-UnetBase` | 8 | 8, 16, 24 | channels of the first U-Net level |
| `-LstmReps` | field, dct | field, radon, dct, pca | what the LSTM reads |
| `-Patch` | 1, 3 | 1, 3, 5, 7 | patch sides of the neighbourhood sweep |
| `-Only` | all | all | any of `bench`, `blend`, `patch`, `unet`, `lstm` |
| `-Matlab` | `matlab` | | the MATLAB executable |
| `-Python` | `python` | | the Python interpreter |

In `full` mode the light parameters are ignored.

Values fixed inside the files, the same in every mode:

| where | value |
|---|---|
| all steps | 32 × 32 grid, 24 lags, 24 hours ahead, days 1-365 (2005) train, days 366-730 (2006) test, cells scored above 5° of solar elevation |
| `results\hyper_frozen.mat` | ELM width 2048; ridge penalty 1e-8 of the mean of diag(HᵀH) up to 64 inputs (one cell), 1e-6 above (patch, field, compressed field) |
| `hms_forecast_bench.m` | the linear ridge penalty is picked among eight ratios, 1e-8 to 10, on the training year: the largest whose score is within 1e-4 × the mean score of the best |
| `hms_latent.m` | autoencoder: `p` tanh units, linear decoder, 120 epochs, batch 256, learning rate 1e-3, seed 7 |
| `python/unet_bench.py` | 15 epochs, batch 32, learning rate 1e-3, seed 1 |
| `python/lstm_bench.py` | one layer of 256 units, 15 epochs, batch 64, learning rate 1e-3, seed 1 |

The ridge, the ELM, the U-Net and the LSTM read the 24 past hourly values of what they forecast, with no calendar or other input, and forecast the 24 lead times at once: one model, one output per lead time, no forecast fed back as an input.

---

## What each step fits

| step | MATLAB call | models | archive |
|---|---|---|---|
| `bench` | `hms_forecast_bench` | persistence, cyclic persistence, linear ridge per cell, ELM per cell, linear and tanh ELM on the whole field, and the ELM on each compression method at each payload | `results\bench.mat` |
| `blend` | `hms_blend_arm` | BLEND, a mix of the two persistences with one weight per hour of the day and lead time | `results\blend_bench.mat` |
| `patch` | `hms_neighbourhood` | the per-cell ELM reading a k × k patch | `results\neighbourhood.mat` |
| `unet` | `hms_unet_arm` | U-Net on the whole field; MATLAB writes the inputs, `python\unet_bench.py` trains | `results\unet_bench.mat` |
| `lstm` | `hms_lstm_arm` | LSTM on the field cells or on a compressed field; `python\lstm_bench.py` trains | `results\lstm_bench.mat` |
| `tables` | `hms_results_tables` | reads the archives and writes `ladder.tex` and `routes.tex` | |

Python never computes a score. It returns predictions, and MATLAB scores them with the same function as every other model.

---

## Data and frozen inputs

`Basic\` holds the input data, described in [Basic/README.md](Basic/README.md):

- `GHI_HC3.mat`: HelioClim-3 global horizontal irradiance on 1148 points over Corsica, one cell per hour, the first 17 520 hours (2005 and 2006) of the author's eight-year archive;
- `geopoint.mat`: latitude and longitude of the 1148 points.

`hms_build_field` interpolates each hour onto a 32 × 32 grid (natural neighbour, no extrapolation). 722 cells lie inside the convex hull of the points; only they are scored.

`results\` starts with two frozen inputs:

- `hyper_frozen.mat`: the ELM width and penalties, frozen once by the paper's sweep over widths 16 to 2048 and eight penalties on the training year. 2048 is the edge of that sweep, and the file says so;
- `scale_ref.mat`: the mean of the scored test-year observations, 407.9003 W/m² over 2 908 263 values, which every nRMSE divides by. `hms_scale_ref` recomputes it if the file is missing.

---

## How a forecast is scored

1. A cell is scored at a target hour only where its own solar elevation exceeds 5° (`hms_eval_mask`, `hms_solar`).
2. Negative forecasts are set to 0 in every step. The bench and U-Net steps also set to 0 every forecast where the sun is at or below the horizon. BLEND, the patch sweep and the LSTM do not, and their files say why: no such cell is ever scored, so the numbers do not change.
3. Errors are pooled over every scored cell of every map (`hms_score`).
4. `hms_metrics_h` computes, at 1, 2, 3, 6, 12 and 24 hours: nRMSE and nMAE divided by the mean above, nMBE, the NICE family against simple persistence, and the gamma index with its pass rate. The tables give NICE^Σ, the mean gamma and the gamma pass rate as the mean of these six lead times, the same way for every model.

---

## Results of the paper

From `paper_results\tables\ladder.tex` and `routes.tex`, nRMSE pooled over 1 to 24 hours and at two lead times. Parameters are the stored values, in thousands (k) or millions (M); GPR is the gamma pass rate, mean of the six lead times.

| model | nRMSE | 1 h | 24 h | parameters | GPR |
|---|---:|---:|---:|---:|---:|
| ELM-pixel | **0.308** | 0.156 | 0.325 | 100.4k | 58.6 % |
| LSTM-field | 0.326 | 0.182 | 0.343 | 5.5M | 54.2 % |
| U-Net-field (8 channels) | 0.343 | 0.205 | 0.346 | 31.2k | 42.7 % |
| AE-ELM-field (p = 392), best compression | 0.348 | 0.181 | | 39.4M | |
| Ridge-pixel | 0.354 | 0.175 | 0.353 | 600 | 57.1 % |
| ELM-field | 0.355 | 0.174 | 0.376 | 100.7M | 48.6 % |
| BLEND | 0.376 | 0.226 | 0.396 | 576 | 50.6 % |
| Pers-24h | 0.396 | 0.396 | 0.396 | 0 | 61.2 % |
| Pers | 0.998 | 0.320 | 0.396 | 0 | 19.8 % |

The per-cell model stores a thousandth of the whole-field model and has the lowest error. No compression method reaches it.

---

## Results of a light run

`run_all.ps1 -Mode light` with the defaults above, on a laptop with an Intel Core i7-1365U, CPU only. The run log spans 06:49 to 10:15, restarts included.

nRMSE pooled over 1 to 24 hours, from the tables of that run, next to the paper's:

| model | light run | paper |
|---|---:|---:|
| ELM-pixel | 0.308 | 0.308 |
| ELM-patch, k = 1 | 0.3077 | 0.3077 |
| ELM-patch, k = 3 | 0.3087 | 0.3087 |
| LSTM-field | 0.326 | 0.326 |
| U-Net-field (8 channels) | 0.343 | 0.343 |
| AE-ELM-field (p = 196) | 0.352 | 0.351 |
| ELM-field | 0.353 | 0.355 |
| Ridge-pixel | 0.354 | 0.354 |
| ELM-field (linear) | 0.364 | 0.368 |
| Wav-ELM-field (p = 196) | 0.374 | 0.374 |
| BLEND | 0.376 | 0.376 |
| LSTM-DCT-field (p = 196) | 0.383 | 0.383 |
| Pers-24h | 0.396 | 0.396 |
| DCT-ELM-field (p = 196) | 0.398 | 0.399 |
| Radon-ELM-field (p = 196) | 0.408 | 0.410 |
| Sub-ELM-field (p = 196) | 0.417 | 0.416 |
| PCA-ELM-field (p = 196) | 0.504 | 0.501 |
| RP-ELM-field (p = 196) | 0.949 | 0.950 |
| Pers | 0.998 | 0.998 |

The models the paper fits once give its values. The whole-field and compressed ELM routes differ by at most 0.004: the paper keeps the median of 50 random hidden layers, and a light run draws one.

---

## A check of the full protocol

The paper's protocol, the median of 50 random hidden layers, was run on the same laptop for the bench step and one compression method:

```bash
powershell -ExecutionPolicy Bypass -File .\run_all.ps1 -Mode light -Draws 50 -Only bench -Codes dct -Payload 196
```

It ran in 3.7 hours without an error. Its tables match the paper's in every column except fit time, which depends on the machine: the pooled error, the six lead times, NICE^Σ, the mean gamma and the pass rate.

| model | full protocol | paper |
|---|---:|---:|
| ELM-pixel | 0.308 | 0.308 |
| Ridge-pixel | 0.354 | 0.354 |
| ELM-field | 0.355 | 0.355 |
| ELM-field (linear) | 0.368 | 0.368 |
| DCT-ELM-field (p = 196) | 0.399 | 0.399 |
| Pers-24h | 0.396 | 0.396 |
| Pers | 0.998 | 0.998 |

---

## Files

| file | first line of its header |
|---|---|
| `run_all.ps1` | (PowerShell) checks the tools, runs the steps, keeps a log of each |
| `hms_forecast_bench.m` | The forecasting protocol, exactly as specified. |
| `hms_blend_arm.m` | The BLEND persistence operator, scored like every other arm. |
| `hms_blend.m` | The simplified cyclostationary BLEND persistence operator, on a field. |
| `hms_neighbourhood.m` | Where does spatial information die? A sweep over patch radius. |
| `hms_unet_arm.m` | The spatial reference, through a network that shares its weights. |
| `hms_lstm_arm.m` | The trained sequence model, on the raw field and on each latent. |
| `hms_results_tables.m` | The result tables of the paper, generated from the archives. |
| `hms_config.m` | Single source of truth for the Radon-transform benchmark. |
| `hms_build_field.m` | Build the gridded spatio-temporal field and its validity mask. |
| `hms_codex_origins.m` | Strict split-local histories and targets; no cross-split use. |
| `hms_elm.m` | Extreme learning machine, MIMO multi-horizon, memory-streamed. |
| `hms_elm_config.m` | The width and penalty of every extreme learning machine, once. |
| `hms_latent.m` | Build one latent encoder, at a prescribed latent dimension. |
| `hms_operator.m` | Build a linear measurement operator and its EXACT adjoint. |
| `hms_binops.m` | A coarser detector, as a pair of matrices. |
| `hms_eval_mask.m` | The one evaluation mask, built from geometry before any prediction. |
| `hms_solar.m` | Geometric solar elevation and azimuth, in degrees. |
| `hms_epoch.m` | UTC seconds at the middle of every hour of the given days. |
| `hms_horizons.m` | The horizons this paper reports, in one place. |
| `hms_metrics_h.m` | The six metrics, at every reported horizon, computed in one place. |
| `hms_gamma.m` | Spatio-temporal gamma index for gridded forecasts. |
| `hms_score.m` | The one error convention of this study: pooled over cells and time. |
| `hms_scale_ref.m` | The normalising constant of this study, computed once and stated. |
| `hms_push_row.m` | Append one result to a campaign, whatever order its fields came in. |
| `hms_resid.m` | Keep the per-origin errors the pooled score throws away. |
| `python/unet_bench.py` | U-Net forecaster: the spatial reference the comparison was missing. |
| `python/lstm_bench.py` | LSTM forecaster: the arm in which compressing the input actually pays. |
| `tools/make_basic_2years.m` | how `Basic\GHI_HC3.mat` was cut from the eight-year archive |

`hms_config.m` still calls the study the "Radon-transform benchmark": it began as one, and the header was kept as written. "Arm" in the file headers means one model of the benchmark.

---

## Checks done on this repository

- **Same code.** The `rt_` prefix was replaced by `hms_` on whole words only. Applying the reverse replacement gives back the project's files byte for byte.
- **Same data.** `Basic\GHI_HC3.mat` equals hours 1 to 17 520 of the original archive, missing values included (MATLAB `isequaln`).
- **Same field.** `hms_build_field` on this repository and `rt_build_field` on the original project return identical fields and masks for days 1, 2, 365, 366, 729 and 730.
- **Same tables.** `run_all.ps1` in `tables` mode writes `ladder.tex` and `routes.tex` identical to the paper's, apart from the generator's name in their first comment line.
- **Same numbers.** A light run gives the paper's errors for every model the paper fits once, and the full protocol gives the paper's tables for the models it runs (tables above).
- **One fix.** The light run showed a bug in the table generator. A new archive stores its normaliser inside its metrics block only, and the generator fell back to an old constant: every lead-time column of a new run came out 0.655 times too small, while the pooled column was right. The generator now reads the metrics block first, in the project and here; the paper's tables come out unchanged.

---

## Citation and licence

See [CITATION.cff](CITATION.cff). The code is under the MIT licence ([LICENSE](LICENSE)). HelioClim-3 data are distributed by the SoDa service under its own terms of use; check them before any use of `Basic\GHI_HC3.mat` beyond running this benchmark.

**Keywords:** solar irradiance forecasting, satellite irradiance, HelioClim-3, GHI, spatio-temporal forecasting, extreme learning machine, U-Net, LSTM, Radon transform, DCT, wavelet, PCA, random projection, autoencoder, compression, number of parameters, frugal machine learning, MATLAB, benchmark, Corsica.
