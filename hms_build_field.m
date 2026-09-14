function [F, M, info] = hms_build_field(days, cfg)
%HMS_BUILD_FIELD Build the gridded spatio-temporal field and its validity mask.
%
%   [F, M, info] = HMS_BUILD_FIELD(days, cfg)
%
%   F    : n x n x T double field, rows = latitude, cols = longitude
%   M    : n x n x T logical validity mask
%   info : grid, support, diagnostics
%
%   NOTHING SOLAR ENTERS THIS FILE. No clear-sky model, no clear-sky index,
%   no top-of-atmosphere irradiance, no night mask. Two earlier versions used
%   first a clear-sky index and then a TOA-derived illumination mask; both
%   were removed because both tie the benchmark to solar physics and destroy
%   the claim that the encoder applies to any spatio-temporal field. A
%   protocol that needs a domain model is a solar protocol.
%
%   The raw field is encoded as it stands, night included. The protection
%   against a field whose variance is dominated by a deterministic cycle
%   comes entirely from the naive references of HMS_BASELINES and HMS_NICE:
%   the trivial encoder reproduces night zeros just as cheaply as any
%   transform, and cyclic persistence carries the diurnal envelope forward
%   for free, so neither can be beaten by exploiting the trivial part of the
%   signal. Fix the reference, not the signal.
%
%   The only mask is therefore about DATA EXISTENCE, which every field has:
%       info.support_geo  convex hull of the measurement points, static
%       info.finite       the interpolated value exists at that cell
%   and M is their intersection.
%
%   THE FULL HOURLY AXIS IS ALWAYS RETURNED, nothing is dropped. Removing
%   maps would make a sunset hour and the following sunrise adjacent in the
%   series and corrupt any temporal model.
%
%   AXIS CONVENTION: [Lon,Lat] = meshgrid(lon_grid, lat_grid), so rows index
%   latitude and columns index longitude, matching imagesc(lon,lat,F).
%
%   TIME CONVENTION: within a day, hourly index h = 1..24 holds the mean over
%   the UTC interval [h-1, h), so its effective instant is h - 0.5 UTC.
%   Obtained by fitting the barycentre of the diurnal extraterrestrial
%   profile against true solar noon at the domain centre, agreeing to 1.2
%   minutes at both solstices and the equinox. That fit establishes the HOUR
%   CONVENTION only; it does not establish the absolute start date or which
%   leap days were removed, which remain open (see Discussion.txt). The fit
%   was a one-off calibration and is not part of this pipeline.

persistent CACHE

if nargin < 2, cfg = hms_config(); end
days = days(:).';
H    = cfg.hours_per_day;
assert(~isempty(days) && all(days >= 1 & days <= cfg.n_days), ...
    'hms_build_field:badDays', 'days must lie in 1..%d.', cfg.n_days);

key = cachekey(cfg);
if isempty(CACHE) || ~isfield(CACHE,'key') || ~isequal(CACHE.key, key)
    CACHE = struct();
    CACHE.geo = loadvar(fullfile(cfg.dir_basic,'geopoint.mat'), 'geopoint');
    CACHE.GHI = loadvar(fullfile(cfg.dir_basic,'GHI_HC3.mat'),  'GHI_HC3');
    CACHE.key = key;
end
geo = CACHE.geo;  GHI = CACHE.GHI;

assert(size(geo,1) == numel(GHI{1}), 'hms_build_field:pointMismatch', ...
    'geopoint has %d rows but GHI_HC3 cells hold %d values.', ...
    size(geo,1), numel(GHI{1}));

lat_pts = geo(:,1);  lon_pts = geo(:,2);
lat_grid = linspace(min(lat_pts), max(lat_pts), cfg.n);   % rows
lon_grid = linspace(min(lon_pts), max(lon_pts), cfg.n);   % cols
[Lon, Lat] = meshgrid(lon_grid, lat_grid);

kh = convhull(lon_pts, lat_pts);          % a genuine failure must raise
support_geo = inpolygon(Lon, Lat, lon_pts(kh), lat_pts(kh));

T = numel(days) * H;
F       = zeros(cfg.n, cfg.n, T);
finite_ = false(cfg.n, cfg.n, T);

kk = 0;
for d = days
    for h = 1:H
        kk  = kk + 1;
        g   = double(GHI{1,(d-1)*H + h});
        Fi  = scatteredInterpolant(lon_pts, lat_pts, g, ...
                 cfg.scatter_method, cfg.scatter_extrap);
        gg  = Fi(Lon, Lat);
        finite_(:,:,kk) = isfinite(gg) & support_geo;
        gg(~isfinite(gg)) = 0;            % outside the hull, masked out below
        F(:,:,kk) = gg;
    end
end

M = repmat(support_geo, 1, 1, T) & finite_;

% Purely descriptive, computed from the data and used by nothing: the share
% of supported cells that are exactly zero at each step. Reported so that the
% trivial part of the record is visible in the paper rather than hidden, and
% never used to select, weight or exclude anything.
nsup = max(1, sum(support_geo(:)));
zero_fraction = squeeze(sum(sum((F == 0) & repmat(support_geo,1,1,T), 1), 2)).' / nsup;

info = struct( ...
    'lat_grid',     lat_grid, ...
    'lon_grid',     lon_grid, ...
    'support_geo',  support_geo, ...
    'finite',       finite_, ...
    'n_points',     numel(lat_pts), ...
    'n_pixels',     cfg.n^2, ...
    'oversampling', cfg.n^2 / numel(lat_pts), ...
    'days',         days, ...
    'hour_index',   'within a day, h = 1..24 holds the UTC mean over [h-1,h); instant = h-0.5 UTC', ...
    'zero_fraction',zero_fraction, ...
    'note',         'no clear-sky model, no clear-sky index, no TOA, no night mask', ...
    'cfg',          cfg);
end

% =========================================================================
function v = loadvar(path, expected)
S = load(path);
f = fieldnames(S);
if any(strcmp(f, expected))
    v = S.(expected);
else
    assert(numel(f) == 1, 'hms_build_field:ambiguousMat', ...
        '%s holds %d variables and none named %s.', path, numel(f), expected);
    v = S.(f{1});
end
end

function k = cachekey(cfg)
p = {fullfile(cfg.dir_basic,'geopoint.mat'), fullfile(cfg.dir_basic,'GHI_HC3.mat')};
k = cell(1, numel(p));
for i = 1:numel(p)
    d = dir(p{i});
    assert(~isempty(d), 'hms_build_field:missingFile', 'Not found: %s', p{i});
    k{i} = sprintf('%s|%d|%.6f', p{i}, d.bytes, d.datenum);
end
end
