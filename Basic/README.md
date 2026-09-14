# Basic: the input data

Two files, read by `hms_build_field.m` through the folder `hms_config.m` points to.

| file | variable | content |
|---|---|---|
| `GHI_HC3.mat` | `GHI_HC3` | 1 × 17 520 cell array; cell k holds the 1148 HelioClim-3 irradiance values of hour k, in W/m², as doubles |
| `geopoint.mat` | `geopoint` | one row per point, the same 1148 points; column 1 latitude, column 2 longitude |

## Time

Hour k of day d is cell `(d-1)*24 + k`. It holds the UTC mean over the hour [k-1, k). Day 1 is 1 January 2005 (`hms_config.m` gives the evidence for that origin and its one-day uncertainty). The benchmark trains on days 1-365 and tests on days 366-730.

## Where it comes from

HelioClim-3 surface irradiance, retrieved from Meteosat images by the Heliosat-2 method and distributed by the SoDa service. The author's archive covers eight years, 70 080 hours. `tools/make_basic_2years.m` kept the first 17 520 hours and checked that the copy equals them, missing values included.

## Missing values

132 020 of the 20 112 960 values are NaN. `hms_build_field` interpolates what exists; a grid cell it cannot interpolate is set to 0 and marked invalid in the mask it returns. The paper declares the gaps of this record in its appendix.

## Terms of use

HelioClim-3 data are distributed by the SoDa service under its own terms of use. The author includes this two-year extract so that the benchmark can be run as published. Check the SoDa terms before any other use.
