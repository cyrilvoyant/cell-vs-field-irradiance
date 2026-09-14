function S = hms_scale_ref(force)
%HMS_SCALE_REF The normalising constant of this study, computed once and stated.
%
%   S = HMS_SCALE_REF()        read it, computing it the first time
%   S = HMS_SCALE_REF(true)    recompute it
%
%   WHY THIS FILE REPLACES SIGMA_REF. Every normalised error in this study was
%   divided by the standard deviation of the scored irradiance, and that is not
%   the convention. nRMSE is RMSE normalised by the MEAN of the observations,
%   which is what the solar forecasting literature reports and what the author's
%   own published work uses. Dividing by the spread instead inflated every
%   figure by a factor of 1.53 and made the numbers incomparable with anything
%   else in the field. The ratios between arms were unaffected, so no conclusion
%   moved, but no number was right.
%
%   The old file, results/sigma_ref.mat, held one variable called sig with no
%   record of what it was, where it came from, or which script produced it --
%   nothing in the current code creates it. A name that does not say what it
%   means is how the error survived: sig reads as "the scale", and a caller
%   dividing by "the scale" has no reason to look further.
%
%   WHAT IS RETURNED
%     S.mu      mean of the observations over the scored set, the normaliser
%     S.sigma   their standard deviation, kept because the reconstruction half
%               of the study reports variance-referenced quantities and because
%               a reader converting old numbers needs the ratio
%     S.n       how many values it was computed on
%     S.set     which set, in words, so it cannot be confused with another
%
%   THE SCORED SET IS THE TEST YEAR under the evaluation mask of the study:
%   local solar elevation above five degrees, on the cells the product reports.
%   Using the training year instead would change the constant by a fraction of a
%   per cent and would be defensible; using the test year is what the reported
%   errors are computed on, so it is what they are normalised by.

if nargin < 1, force = false; end
f = fullfile(pwd, 'results', 'scale_ref.mat');

if isfile(f) && ~force
    q = load(f);  S = q.S;  return
end

n = 32;  cfg = hms_config('n', n);  d = n*n;  ELEV = 5;  dte = 366:730;

[F, M, info] = hms_build_field(dte, cfg);
V = reshape(F, d, []);  clear F
[Lon, Lat] = meshgrid(info.lon_grid, info.lat_grid);
cal = struct('resolved', true, 'days_in_year', 365, ...
             'source', sprintf('HelioClim-3, origin %s', cfg.absolute_start_date));
K = reshape(hms_eval_mask(Lat, Lon, hms_epoch(dte, cfg), cal, ...
                         struct('eval_deg', ELEV)), d, []);
msk = mean(M,3) > 0.5;
K = K & msk(:);

v = V(K);
S = struct('mu', mean(v), 'sigma', std(v), 'n', numel(v), ...
           'set', sprintf(['HelioClim-3 Corsica, test year (days %d-%d), ' ...
                           'cells reported by the product, local solar ' ...
                           'elevation above %d degrees'], ...
                          dte(1), dte(end), ELEV), ...
           'computed', datestr(now, 'yyyy-mm-dd HH:MM'));

if ~isfolder(fileparts(f)), mkdir(fileparts(f)); end
save(f, 'S');
fprintf('\nnormalising constant of the study\n');
fprintf('  mean   %.3f W/m2   <- nRMSE divides by this\n', S.mu);
fprintf('  sigma  %.3f W/m2   (kept for reference only)\n', S.sigma);
fprintf('  on     %d scored values\n', S.n);
fprintf('  ratio  sigma/mu = %.4f, the factor by which the old figures erred\n', ...
        S.sigma / S.mu);
fprintf('written %s\n', f);
end
