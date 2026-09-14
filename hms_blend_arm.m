function R = hms_blend_arm(outfile)
%HMS_BLEND_ARM The BLEND persistence operator, scored like every other arm.
%
%   R = HMS_BLEND_ARM(outfile)
%
%   WHY A SEPARATE FILE AND NOT A ROW IN BENCH.MAT. The metric migration added
%   seven fields to every row of results/bench.mat, and HMS_PUSH_ROW refuses a
%   row whose fields differ -- correctly, since a campaign whose rows carry
%   different fields is not one campaign. Adding an arm to that archive would
%   mean running the bench against the pre-migration backup and migrating
%   again, which touches the published numbers to add something beside them.
%   The convolutional and recurrent arms already live in their own archives and
%   are merged when the tables are generated; BLEND follows the same path, and
%   nothing published is opened to write.
%
%   THE OPERATOR IS THE AUTHOR'S OWN. See HMS_BLEND: Voyant et al., Applied
%   Mathematical Modelling 157 (2026) 116988, ported from
%   github.com/cyrilvoyant/cyclostationary-forecasting-matlab.
%
%   WHAT IT COSTS TO STORE, AND WHY THAT IS THE INTERESTING PART HERE. The
%   coefficient is a table of Td by H numbers, 576 at an hourly step. That
%   places the operator between the two free references and the ridge, which
%   stores 600, and it is the first arm in this study whose cost is neither
%   zero nor dominated by a weight matrix. A paper about whether information
%   repays its storage should be able to say what a near-free operator buys.
%
%   THE COEFFICIENT IS ESTIMATED ON THE TRAINING YEAR AND APPLIED TO THE TEST
%   YEAR. HMS_BLEND takes the two fields separately for exactly this reason.

if nargin < 1 || isempty(outfile)
    outfile = fullfile(pwd, 'results', 'blend_bench.mat');
end
if ~isfolder(fileparts(outfile)), mkdir(fileparts(outfile)); end

n = 32;  cfg = hms_config('n', n);
L = 24;  H = 24;  Td = 24;  ELEV = 5;  DT_MIN = 60;
dtr = 1:365;  dte = 366:730;

R = struct('name',{},'scale_name',{},'scale_value',{}, ...
           'rmse',{},'rmse_h',{},'n_par',{},'met',{}, ...
           'secs',{},'secs_met',{},'lambda',{},'period_steps',{}, ...
           'dt_min',{},'estimator',{});

fprintf('\n=== BLEND arm, the cyclostationary persistence operator ===\n');
t0 = tic;
[Ftr, Mtr, info] = hms_build_field(dtr, cfg);
[Fte, ~, ~]      = hms_build_field(dte, cfg);
Vtr = reshape(Ftr, n*n, []);  Vte = reshape(Fte, n*n, []);
clear Ftr Fte
msk = mean(Mtr, 3) > 0.5;  msk = msk(:);  d = numel(msk);
fprintf('fields built in %.0f s, island %d of %d cells\n', toc(t0), nnz(msk), d);

[Lon, Lat] = meshgrid(info.lon_grid, info.lat_grid);
cal = struct('resolved', true, 'days_in_year', 365, ...
             'source', sprintf('HelioClim-3, origin %s', cfg.absolute_start_date));
em = @(dd) reshape(hms_eval_mask(Lat, Lon, epoch_of(dd, cfg), cal, ...
                                struct('eval_deg', ELEV)), n*n, []);
day_tr = em(dtr);  day_te = em(dte);

% THE SAME ORIGINS AS EVERY OTHER ARM. BLEND reads no lag block, so it would
% tolerate earlier origins than the fitted arms; using its own would make its
% column of the results table describe a different test set from the one
% beside it, and would break the paired test outright.
ote = hms_codex_origins([1 size(Vte,2)], max(L, Td), H);
hms_resid('origins', ote);        % for the paired test; no-op when capture is off
Yte = tgt(Vte, ote, H);
Kte = flg(day_te, ote, H, d) & repmat(msk, H, 1).';
SC = hms_scale_ref();  mu = SC.mu;
fprintf('origins  test %d | nRMSE divides by the mean, mu = %.4f\n', ...
        numel(ote), mu);

% ---- the NICE denominator, this arm's own simple persistence
HSU = hms_horizons();
Pp  = max(repmat(Vte(:, ote).', 1, H), 0);
[~, LREF] = hms_metrics_h(@(ih) slice_h(Pp, Yte, Kte, HSU(ih), d, n), mu, ...
    struct('horizons', HSU, 'res', 3.4, 'dt', DT_MIN, 'isref', true));
clear Pp

% ---- the operator
tb = tic;
Ktr = day_tr & repmat(msk, 1, size(Vtr, 2));
[Pb, lam, npar, Tdo] = hms_blend(Vte, ote, DT_MIN, H, Vtr, Ktr);
assert(Tdo == Td, 'hms_blend_arm:period', ...
    'The operator derived a period of %d steps where this archive has %d.', ...
    Tdo, Td);
% THE SAME REPAIR EVERY ARM GETS. Negative predictions are clipped at zero.
% Night forcing is not applied: Kte already excludes every cell below the
% elevation threshold, so a night cell is never scored.
Pb = max(Pb, 0);
secs = toc(tb);

[MET, ~, tmet] = hms_metrics_h(@(ih) slice_h(Pb, Yte, Kte, HSU(ih), d, n), mu, ...
    struct('horizons', HSU, 'res', 3.4, 'dt', DT_MIN, 'niceref', LREF));

% THE ROW DECLARES ITS OWN SCALE. HMS_RESULTS_TABLES asks each archived row
% what its met block was divided by, and treats a row without scale_name as
% legacy, rescaling it by 267.1184/mu. This arm was scored on the study
% convention from the start, so without these two fields the ladder would
% correct a correction and print BLEND wrong by a factor of 1.53.
R = hms_push_row(R, struct( ...
    'name', 'BLEND', 'rmse', errm(Pb, Yte, Kte), ...
    'scale_name', 'oos_mean_ghi', 'scale_value', mu, ...
    'rmse_h', errh(Pb, Yte, Kte, d, H), 'n_par', npar, 'met', MET, ...
    'secs', secs, 'secs_met', tmet, 'lambda', lam, ...
    'period_steps', Tdo, 'dt_min', DT_MIN, ...
    'estimator', ['least squares, one coefficient per phase and per horizon, ' ...
                  'fitted on the training year; NOT the closed form ' ...
                  'lambda = (1+rho)/2 of the reference, which assumes a ' ...
                  'detrended signal this study does not build']));

fprintf('\n  %-16s RMSE %7.2f W/m2 | nRMSE %.4f | %d stored | %.1f s\n', ...
        'BLEND', R(end).rmse, R(end).rmse / mu, npar, secs);
fprintf('  h=3: nRMSE %.4f  NICE^S %.3f  GPR %.1f%%\n', ...
        MET.nrmse(3), MET.nicesig(3), MET.gpr(3));

save(outfile, 'R', 'L', 'H', 'ELEV', 'dtr', 'dte', 'DT_MIN', '-v7.3');
fprintf('\ntotal %.1f min\nwritten %s\n', toc(t0)/60, outfile);
end

% ---------------------------------------------------------------------------
function Y = tgt(Z, o, H)
d = size(Z,1);  Y = zeros(numel(o), d*H);
for h = 1:H, Y(:, (h-1)*d+(1:d)) = Z(:, o+h).'; end
end

function K = flg(F, o, H, d)
K = false(numel(o), d*H);
for h = 1:H, K(:, (h-1)*d+(1:d)) = F(:, o(:)+h).'; end
end

function e = errm(P, Y, K)
% POOLED, the one convention of this study; see HMS_SCORE.
D = P - Y;
e = hms_score(sum((D.^2).*K, 2), sum(K, 2));
end

function e = errh(P, Y, K, d, H)
e = zeros(1, H);
for h = 1:H
    c = (h-1)*d+1:h*d;  D = P(:,c)-Y(:,c);  k = K(:,c);
    if any(k(:)), e(h) = hms_score(sum((D.^2).*k,2), sum(k,2)); end
end
end

function [Pv, Yv, Kv] = slice_h(P, Y, K, h, d, nside)
c = (h-1)*d + (1:d);
Pv = reshape(P(:,c).', nside, nside, []);
Yv = reshape(Y(:,c).', nside, nside, []);
Kv = reshape(K(:,c).', nside, nside, []);
end

function t = epoch_of(days, cfg)
d0 = datetime(cfg.absolute_start_date,'InputFormat','yyyy-MM-dd','TimeZone','UTC');
y0 = year(d0);  all_days = NaT(0,1,'TimeZone','UTC');
while numel(all_days) < cfg.n_days
    dd = (datetime(y0,1,1,'TimeZone','UTC'):datetime(y0,12,31,'TimeZone','UTC')).';
    dd = dd(~(month(dd)==2 & day(dd)==29));
    all_days = [all_days; dd]; %#ok<AGROW>
    y0 = y0 + 1;
end
sel = all_days(days(:));  Hh = cfg.hours_per_day;
t = posixtime(repelem(sel,1,Hh)) + repmat((0:Hh-1)*3600 + 1800, numel(sel), 1);
t = reshape(t.',1,[]);
end
