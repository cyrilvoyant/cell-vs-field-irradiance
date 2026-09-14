function [elev, azim, singular] = hms_solar(doy, hour_utc, lat, lon, calendar)
%HMS_SOLAR Geometric solar elevation and azimuth, in degrees.
%
%   [elev, azim] = HMS_SOLAR(doy, hour_utc, lat, lon, calendar)
%
%   doy      : day of year, 1..days_in_year
%   hour_utc : hour UTC, 0..24, may be fractional
%   lat, lon : degrees, north and east positive. Longitude is taken in
%              [-180, 180]; an earlier version also accepted up to 360, which
%              is a second convention silently overlapping the first.
%   calendar : struct, REQUIRED, describing where doy comes from
%                .resolved      logical, must be true
%                .days_in_year  365 or 366
%                .source        non-empty scalar text, for traceability
%                .singular_azimuth  optional, 'nan' (default) or 'fallback'
%
%   elev : GEOMETRIC elevation above the horizon, degrees, negative at night.
%          Atmospheric refraction is NOT applied, so near the horizon this is
%          about half a degree below what an observer sees. That matters at a
%          one-degree threshold and is stated rather than left to be discovered.
%   azim : azimuth, degrees clockwise from north. NaN by default where it is
%          undefined: the sun exactly overhead, or a geographic pole.
%   singular : logical, true exactly where the azimuth is undefined, so a
%          caller can tell an absent value from a computed one without testing
%          for NaN.
%
%   THE SINGULAR AZIMUTH IS A CHOICE, AND IT IS THE CALLER'S. Setting
%   calendar.singular_azimuth = 'fallback' returns a conventional meridional
%   placeholder (0 or 180 degrees) instead of NaN. It is offered only for code
%   that requires a finite number; it is NOT a physical azimuth and NOT a
%   path-independent limit. The `singular` output remains true and the fallback
%   must not be used as a physical input or in a score.
%
%   WHY THE CALENDAR IS AN ARGUMENT AND NOT A DEFAULT
%   A day-of-year is meaningless without knowing which calendar produced it,
%   and a wrong one fails silently: the elevation comes out perfectly plausible
%   and is simply wrong by a day or two. Two failures are prevented here.
%
%   First, LEAP YEARS. An earlier version divided the fractional year by 365
%   always, while accepting a doy up to 366. On 2020, which is the year of both
%   the CAMS and the ERA5 data used in this study, that shifts the declination
%   by up to 0.294 degrees, and on 149 days of the year it moves the crossing
%   of a one-degree threshold by more than a minute. The effect is small, it is
%   systematic, and it silently changes which steps enter an evaluation set.
%
%   Second, AN UNRESOLVED ORIGIN. The HelioClim archive holds 2920 days with
%   leap days removed and no established start date, so its day index is not a
%   day-of-year at all. HMS_CONFIG records that honestly as
%   calendar_resolved = false. This function therefore REFUSES an unresolved
%   calendar rather than computing a confident answer about an unknown date.
%
%   SOURCE OF THE EQUATIONS. The equation of time and the declination are the
%   Fourier series of Spencer (1971), as given in the NOAA solar calculation
%   sheet at https://gml.noaa.gov/grad/solcalc/solareqns.PDF. An earlier header
%   attributed them to the Astronomical Almanac and claimed about 0.01 degrees
%   of accuracy: BOTH WERE WRONG. That accuracy figure belongs to NREL's
%   SOLPOS, a different algorithm. No accuracy is claimed here; what is known
%   is measured against an independent implementation and reported as a result.
%
%   SHAPES. Each input is real and finite. Inputs are either scalar or of one
%   common size; a scalar is broadcast against that size and nothing else is.
%   Two non-scalar inputs of different sizes are an error rather than an
%   implicit expansion, because an accidental transpose would otherwise produce
%   an outer product that looks like a valid field.

% ------------------------------------------------------------ the calendar
assert(nargin >= 5 && isstruct(calendar) && isscalar(calendar), ...
    'hms_solar:calendarRequired', ...
    ['A calendar struct is required: .resolved, .days_in_year, .source. ' ...
     'A day-of-year without its calendar cannot be interpreted.']);
for f = {'resolved','days_in_year','source'}
    assert(isfield(calendar, f{1}), 'hms_solar:calendarField', ...
        'calendar.%s is missing.', f{1});
end
assert(islogical(calendar.resolved) && isscalar(calendar.resolved), ...
    'hms_solar:calendarResolvedType', 'calendar.resolved must be a logical scalar.');
assert(calendar.resolved, 'hms_solar:calendarUnresolved', ...
    ['calendar.resolved is false: the origin of this day index is not ' ...
     'established, so no date can be attached to it and no solar position ' ...
     'can be computed. Establish the origin, or use a dataset whose time ' ...
     'axis is absolute.']);
D = calendar.days_in_year;
assert(isscalar(D) && isnumeric(D) && isreal(D) && isfinite(D) && ...
    any(D == [365 366]), ...
    'hms_solar:daysInYear', 'calendar.days_in_year must be the real scalar 365 or 366.');
src = calendar.source;
assert((ischar(src) && isrow(src)) || (isstring(src) && isscalar(src)), ...
    'hms_solar:calendarSource', ...
    'calendar.source must be a single char row or a scalar string.');
assert(strlength(strtrim(string(src))) > 0, 'hms_solar:calendarSource', ...
    'calendar.source must not be blank: it exists to make a run traceable.');
sing_mode = 'nan';
if isfield(calendar, 'singular_azimuth')
    raw_mode = calendar.singular_azimuth;
    assert((ischar(raw_mode) && isrow(raw_mode)) || ...
           (isstring(raw_mode) && isscalar(raw_mode)), ...
        'hms_solar:singularMode', ...
        'calendar.singular_azimuth must be a char row or scalar string.');
    sing_mode = lower(strtrim(char(raw_mode)));
    assert(any(strcmp(sing_mode, {'nan','fallback'})), 'hms_solar:singularMode', ...
        'calendar.singular_azimuth must be ''nan'' or ''fallback'', got ''%s''.', sing_mode);
end

% -------------------------------------------------------------- the shapes
nm = {'doy','hour_utc','lat','lon'};
v  = {doy, hour_utc, lat, lon};
lo = [1 0 -90 -180];
hi = [D 24 90 180];
sz = cell(1, 4);
for k = 1:4
    x = v{k};
    assert(isnumeric(x) && isreal(x) && ~isempty(x), 'hms_solar:type', ...
        '%s must be a non-empty real numeric array.', nm{k});
    assert(all(isfinite(x(:))), 'hms_solar:finite', ...
        '%s contains %d non-finite values.', nm{k}, sum(~isfinite(x(:))));
    assert(all(x(:) >= lo(k) & x(:) <= hi(k)), 'hms_solar:range', ...
        '%s must lie in [%g, %g]; got [%g, %g].', nm{k}, lo(k), hi(k), ...
        min(x(:)), max(x(:)));
    if k == 1
        assert(all(x(:) == round(x(:))), 'hms_solar:doyInteger', ...
            ['doy is a whole day index in 1..%d; the fraction of a day is ' ...
             'carried by hour_utc. A fractional doy would add it twice.'], D);
    end
    sz{k} = size(x);
    v{k} = double(x);
end
big = sz(~cellfun(@(s) prod(s) == 1, sz));
if ~isempty(big)
    for k = 2:numel(big)
        assert(isequal(big{k}, big{1}), 'hms_solar:shape', ...
            ['Non-scalar inputs must share one size: got %s and %s. ' ...
             'Broadcast a scalar if that is what you mean; two different ' ...
             'sizes are refused because a stray transpose would otherwise ' ...
             'produce a plausible-looking outer product.'], ...
            mat2str(big{1}), mat2str(big{k}));
    end
    out = big{1};
else
    out = [1 1];
end
[doy, hour_utc, lat, lon] = deal(v{:});
ex = @(x) (prod(size(x)) == 1) * ones(out) .* x + (prod(size(x)) ~= 1) * x; %#ok<PSIZE>
doy = ex(doy);  hour_utc = ex(hour_utc);  lat = ex(lat);  lon = ex(lon);

% ------------------------------------------------------------ the geometry
% Fractional year, using the ACTUAL length of the year.
g = 2*pi/D .* (doy - 1 + (hour_utc - 12)/24);

% Equation of time in minutes and declination in radians, Spencer (1971).
eqt = 229.18 * (0.000075 + 0.001868*cos(g) - 0.032077*sin(g) ...
                - 0.014615*cos(2*g) - 0.040849*sin(2*g));
dec = 0.006918 - 0.399912*cos(g) + 0.070257*sin(g) ...
      - 0.006758*cos(2*g) + 0.000907*sin(2*g) ...
      - 0.002697*cos(3*g) + 0.00148*sin(3*g);

tst = mod(hour_utc*60 + eqt + 4*lon, 1440);          % true solar time, minutes
ha  = deg2rad(tst/4 - 180);                          % hour angle
phi = deg2rad(lat);

sinel = min(1, max(-1, sin(phi).*sin(dec) + cos(phi).*cos(dec).*cos(ha)));
elev  = rad2deg(asin(sinel));

if nargout > 1
    % ATAN2 RATHER THAN ACOS. The arccosine form divides by cos(elevation),
    % which vanishes at the zenith and at the poles; an earlier version guarded
    % that with eps and so returned a fabricated azimuth exactly where none
    % exists. Building the azimuth from two components removes the division and
    % keeps the quadrant.
    cosel = sqrt(max(0, 1 - sinel.^2));
    sx = -cos(dec) .* sin(ha);
    cx =  sin(dec) .* cos(phi) - cos(dec) .* sin(phi) .* cos(ha);
    azim = mod(rad2deg(atan2(sx, cx)), 360);

    % WHERE THE AZIMUTH DOES NOT EXIST. The sun exactly overhead, or a
    % geographic pole: every direction points at the same place. NaN is
    % returned, because a number here would flow downstream indistinguishable
    % from a measurement. The caller may explicitly request a conventional
    % finite fallback instead; `singular` marks the affected entries either way.
    singular = (cosel < 1e-9) | (abs(abs(lat) - 90) < 1e-9);
    if any(singular(:))
        if strcmp(sing_mode, 'fallback')
            azim(singular) = 180 * (lat(singular) > rad2deg(dec(singular)));
        else
            azim(singular) = NaN;
        end
    end
end
end
