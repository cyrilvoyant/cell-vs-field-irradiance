function cfg = hms_config(varargin)
%HMS_CONFIG Single source of truth for the Radon-transform benchmark.
%
%   cfg = HMS_CONFIG() returns the default configuration.
%   cfg = HMS_CONFIG('name',value,...) overrides individual fields.
%
%   Every script of the benchmark reads its settings from here and nowhere
%   else. This answers finding A8 of the code audit (Discussion.txt): the
%   legacy scripts in Basic/ each used a different interpolation method, a
%   different day and a different grid, so their figures were not computed
%   on the same field.
%
%   DESIGN RULE -- RAW ENCODER, DOMAIN-SPECIFIC EVALUATION KEPT SEPARATE.
%   The encoder receives any scalar field unchanged: solar, medical, or
%   otherwise. No clear-sky model, clear-sky index, top-of-atmosphere
%   irradiance or night mask changes that input. Domain physics may define an
%   evaluation subset downstream; for GHI this is a solar mask evaluated per
%   pixel and horizon, explicitly distinct from data validity.
%
%   The protection against a field whose variance is dominated by a
%   deterministic cycle comes from the naive references (HMS_BASELINES,
%   HMS_NICE), never from touching the signal: the trivial encoder reproduces
%   night zeros as cheaply as any transform, and cyclic persistence carries
%   the diurnal envelope forward for free. Fix the reference, not the signal.
%
%   Only the loader is dataset-specific. Everything downstream sees a field
%   F and a validity mask M, so a new modality costs one adapter and not a
%   line of the rest.
%
%   AXIS CONVENTION (finding A7). Throughout the benchmark:
%       rows    = latitude   (index 1 = southernmost)
%       columns = longitude  (index 1 = westernmost)
%   built with [Lon,Lat] = meshgrid(lon_grid, lat_grid), which is the
%   convention imagesc(lon_grid, lat_grid, M) already assumes. The legacy
%   scripts used meshgrid(lat_grid, lon_grid) and displayed the transpose.
%
%   Cyril Voyant, O.I.E. Mines Paris-PSL -- Radon compression study.

p = inputParser;
isTextScalar = @(x) ischar(x) || (isstring(x) && isscalar(x));
isLogicalScalar = @(x) islogical(x) && isscalar(x);

% -- paths ---------------------------------------------------------------
here = fileparts(mfilename('fullpath'));
p.addParameter('root',       here,                        isTextScalar);
p.addParameter('dir_basic',  fullfile(here,'Basic'),      isTextScalar);
p.addParameter('dir_treat',  fullfile(here,'traitement'), isTextScalar);
p.addParameter('dir_out',    fullfile(here,'results'),    isTextScalar);

% -- spatial grid --------------------------------------------------------
% Square grid: iradon returns a square image, so num_lat ~= num_lon is not
% supported by the reconstruction path (finding A7).
p.addParameter('n',          50,     @(x)is_int_scalar(x,8));

% -- temporal ------------------------------------------------------------
p.addParameter('hours_per_day', 24,   @(x)is_int_scalar(x,1));
p.addParameter('n_days',        2920, @(x)is_int_scalar(x,1));

% HC3 TIME METADATA, NOW MEASURED RATHER THAN UNKNOWN.
%
% This block used to read calendar_resolved = false, which was honest and which
% blocked every solar computation on this archive: HMS_SOLAR refuses an
% unresolved calendar rather than silently interpreting day 1 as 1 January. The
% origin has since been established the same way the Ajaccio timestamp
% convention was, by scanning it against solar geometry instead of assuming it.
%
% THE EVIDENCE. The archive holds 70080 hourly values on 1148 points whose
% centre is 42.146 N, 9.096 E, that is 2920 days. Eight calendar years from 2005
% to 2012 with the two leap days removed give exactly 2920. The domain-mean
% series was correlated against max(0, sin(elevation)) over the whole record,
% and the offset scanned:
%   HOUR CONVENTION, SHARP. Correlation 0.9595 at zero offset, falling to 0.921
%   and 0.913 at minus and plus one hour. Hour index h therefore holds the mean
%   over [h-1, h) UTC, exactly as the file already assumed.
%   DAY ORIGIN, FLAT TO ABOUT ONE DAY. 0.959479 at 2005-01-01 and 0.959544 one
%   day later, a difference of 6e-5; the annual cycle is symmetric about the
%   solstice, so this criterion cannot do better. Coarse offsets are decisive:
%   0.9349 at thirty days, 0.7656 at ninety.
%
% THE RESIDUAL UNCERTAINTY IS DECLARED, NOT HIDDEN. A one-day error changes 112
% of 32282 retained steps under the five-degree evaluation rule, that is 0.35
% per cent, and 0.14 per cent of the night mask. That is the uncertainty every
% number computed on this archive carries, and it belongs in the paper.
p.addParameter('calendar_resolved', true, isLogicalScalar);
p.addParameter('day1_doy', 1, @(x) isnumeric(x) && isreal(x) && isscalar(x) && ...
    (isnan(x) || (isfinite(x) && x == fix(x) && x >= 1 && x <= 366)));
p.addParameter('absolute_start_date', '2005-01-01', isTextScalar);
p.addParameter('calendar_origin_uncertainty_days', 1, @(x)isnumeric(x) && isscalar(x) && x >= 0);
p.addParameter('calendar_evidence', ...
    'correlation of domain mean against max(0,sin(elevation)): 0.9595 at zero hour offset, 0.9214/0.9134 at -1/+1 h; day offset flat within +-1 day (0.959479 vs 0.959544), 0.9349 at 30 d', ...
    isTextScalar);
p.addParameter('leap_day_policy', '29 February removed in 2008 and 2012', isTextScalar);

% -- masks ---------------------------------------------------------------
% Data existence defines the primary spatial domain. Domain-specific masks
% affect only their metrics, and reconstruction masks are sensitivities.

% -- Radon operator ------------------------------------------------------
% theta = (0:Na-1)*180/Na. Never linspace(0,360,N): that expression yields
% (N-1)/gcd(N-1,2) distinct directions modulo 180, so a sweep over N
% alternates between two redundancy regimes (finding A5).
p.addParameter('Na',        24,       @(x)is_int_scalar(x,1));
p.addParameter('interp',    'linear', isTextScalar);      % iradon interpolation
p.addParameter('filter',    'Ram-Lak',isTextScalar);      % iradon filter

% -- interpolation of the scattered points onto the grid ------------------
% One method, used everywhere, documented in the paper (finding A8).
p.addParameter('scatter_method', 'natural', isTextScalar);
p.addParameter('scatter_extrap', 'none',    isTextScalar);  % 'none' -> NaN outside

% -- chronological split -------------------------------------------------
% Strictly chronological, never shuffled: any statistic used by the encoder
% or by a comparator must be estimated on the training days only.
p.addParameter('days_train', 1:1825,    @isnumeric);   % years 1-5
p.addParameter('days_val',   1826:2190, @isnumeric);   % year 6
p.addParameter('days_test',  2191:2920, @isnumeric);   % years 7-8

% -- evaluation ----------------------------------------------------------
% Three domains remain distinct: data validity is primary; the solar mask is
% used only by GHI metrics and is pixel-by-horizon; the inscribed disk is a
% reconstruction sensitivity, never the primary score.
p.addParameter('reconstruction_disk_sensitivity', true, isLogicalScalar);
p.addParameter('seed', 42, @(x)is_int_scalar(x,0) && x <= double(intmax('uint32')));

p.parse(varargin{:});
cfg = p.Results;

% Normalize accepted MATLAB string scalars to char for stable serialization.
text_fields = {'root','dir_basic','dir_treat','dir_out','absolute_start_date','calendar_evidence', ...
    'leap_day_policy','interp','filter','scatter_method','scatter_extrap'};
for i = 1:numel(text_fields)
    cfg.(text_fields{i}) = char(cfg.(text_fields{i}));
end

% A split may leave chronological guard gaps, so complete coverage is not
% required. Each subset must nevertheless be ordered, bounded and disjoint,
% and the three subsets must occur in strict chronological order.
cfg.days_train = validate_days(cfg.days_train, cfg.n_days, 'days_train');
cfg.days_val   = validate_days(cfg.days_val,   cfg.n_days, 'days_val');
cfg.days_test  = validate_days(cfg.days_test,  cfg.n_days, 'days_test');
assert(max(cfg.days_train) < min(cfg.days_val) && ...
       max(cfg.days_val)   < min(cfg.days_test), ...
    'hms_config:splitOrder', ...
    'Require max(train) < min(validation) < max(validation) < min(test).');
assert(isempty(intersect(cfg.days_train,cfg.days_val)) && ...
       isempty(intersect(cfg.days_train,cfg.days_test)) && ...
       isempty(intersect(cfg.days_val,cfg.days_test)), ...
    'hms_config:splitOverlap', 'Training, validation and test days must be disjoint.');

if cfg.calendar_resolved
    assert(isfinite(cfg.day1_doy), 'hms_config:calendarOrigin', ...
        'calendar_resolved=true requires an integer day1_doy.');
else
    assert(isnan(cfg.day1_doy), 'hms_config:calendarContradiction', ...
        'Keep day1_doy=NaN while calendar_resolved=false.');
end

% -- derived, never set by hand ------------------------------------------
assert(exist('radon','file') == 2, 'hms_config:missingRadon', ...
    ['radon is unavailable. Install/enable Image Processing Toolbox before ' ...
     'building a Radon benchmark configuration.']);
cfg.theta   = (0:cfg.Na-1) * (180/cfg.Na);
cfg.ns      = size(radon(zeros(cfg.n), 0), 1);   % exact, not sqrt(2)*n
cfg.n_enc   = cfg.ns * cfg.Na;                   % encoded values per map
cfg.coefficient_ratio_map = (cfg.n^2) / cfg.n_enc;
cfg.cr_map  = cfg.coefficient_ratio_map;          % compatibility alias only

cfg.mask_policy = struct( ...
    'data',           'primary_validity_domain', ...
    'solar',          'ghi_metrics_pixel_by_horizon_only', ...
    'reconstruction', 'inscribed_disk_sensitivity_only');
step_hours = 24 / cfg.hours_per_day;
cfg.time = struct( ...
    'calendar_resolved', cfg.calendar_resolved, ...
    'day1_doy',          cfg.day1_doy, ...
    'absolute_start_date',cfg.absolute_start_date, ...
    'leap_day_policy',   cfg.leap_day_policy, ...
    'step_hours',        step_hours, ...
    'interval_convention','sample h is the UTC mean over [(h-1)*step,h*step)', ...
    'midpoint_formula',  '(h-0.5)*step_hours');
used = [cfg.days_train cfg.days_val cfg.days_test];
cfg.split = struct('coverage_required',false, ...
    'unassigned_days',setdiff(1:cfg.n_days,used));

cfg.cost_definition = struct( ...
    'coefficient_ratio_map','n^2/(ns*Na); excludes metadata, bytes and model cost', ...
    'bytes','measure separately', ...
    'parameters','measure separately', ...
    'time','measure separately', ...
    'memory','measure separately');
cfg.version = 'rt-2.0-audit1';

signature_fields = struct( ...
    'version',cfg.version,'n',cfg.n,'hours_per_day',cfg.hours_per_day, ...
    'n_days',cfg.n_days,'Na',cfg.Na,'theta',cfg.theta,'ns',cfg.ns, ...
    'interp',cfg.interp,'filter',cfg.filter, ...
    'scatter_method',cfg.scatter_method,'scatter_extrap',cfg.scatter_extrap, ...
    'days_train',cfg.days_train,'days_val',cfg.days_val,'days_test',cfg.days_test, ...
    'seed',cfg.seed,'calendar_resolved',cfg.calendar_resolved, ...
    'day1_doy',cfg.day1_doy,'absolute_start_date',cfg.absolute_start_date, ...
    'calendar_origin_uncertainty_days',cfg.calendar_origin_uncertainty_days, ...
    'leap_day_policy',cfg.leap_day_policy, ...
    'reconstruction_disk_sensitivity',cfg.reconstruction_disk_sensitivity);
cfg.signature_payload = jsonencode(signature_fields);
cfg.signature_sha256 = sha256_text(cfg.signature_payload);

end

% =========================================================================
function tf = is_int_scalar(x, lower_bound)
tf = isnumeric(x) && isreal(x) && isscalar(x) && isfinite(x) && ...
     x == fix(x) && x >= lower_bound;
end

function days = validate_days(days, n_days, name)
assert(isnumeric(days) && isreal(days) && isvector(days) && ~isempty(days), ...
    'hms_config:badSplit', '%s must be a non-empty real numeric vector.', name);
days = days(:).';
assert(all(isfinite(days)) && all(days == fix(days)), ...
    'hms_config:badSplit', '%s must contain finite integer day indices.', name);
assert(all(days >= 1 & days <= n_days), ...
    'hms_config:badSplit', '%s must lie in 1..n_days.', name);
assert(all(diff(days) > 0), ...
    'hms_config:badSplit', '%s must be unique and strictly increasing.', name);
end

function hex = sha256_text(payload)
md = java.security.MessageDigest.getInstance('SHA-256');
md.update(uint8(unicode2native(payload,'UTF-8')));
raw = typecast(md.digest(),'uint8');
hex = lower(reshape(dec2hex(raw,2).',1,[]));
end
