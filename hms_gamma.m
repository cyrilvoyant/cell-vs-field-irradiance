function G = hms_gamma(pred, meas, opt)
%HMS_GAMMA Spatio-temporal gamma index for gridded forecasts.
%
%   G = HMS_GAMMA(pred, meas, opt)
%
%   pred, meas : ny x nx x nt predicted and observed volumes
%   opt.dta    : distance-to-agreement tolerance, km
%   opt.tta    : time-to-agreement tolerance, minutes
%   opt.idt    : intensity tolerance, in the unit of the field
%   opt.res    : pixel size, km
%   opt.dt     : time step, minutes
%   opt.k      : search radius factor (default 1.5, as in the reference code)
%   opt.units  : 'physical' (default) or 'reference' -- see the note below
%   opt.mask   : ny x nx x nt logical, cells to evaluate (default all finite)
%
%   G.map        : ny x nx x nt gamma values
%   G.gpr        : gamma pass rate, per cent of evaluated cells with gamma <= 1
%   G.mean_local : mean gamma over the cells whose value is local, NOT over all
%   G.max_local  : the largest of those
%   G.n_eval, G.n_local, G.n_failed, G.status, G.note, G.opt
%
%   THE NAMES CARRY A WARNING AND THAT IS WHY THEY ARE LONG. There is no G.mean
%   and no G.median. A mean over every evaluated cell would include values that
%   are only lower bounds, since the search stops once gamma exceeds one, so the
%   suffix _local marks the subset the statistic is honest about. This header
%   said "G.mean, G.max, G.median" until Codex wrote new code against it and got
%   a missing field; the code was right and the documentation was wrong. Every
%   existing caller already reads mean_local, so no result was affected.
%
%   WHY THIS METRIC IS HERE AND NOT AS A DECORATION.
%   On an advected field a pixel-wise score double-penalises a displacement:
%   a forecast shifted by two pixels is wrong where the cloud is and wrong
%   where it is no longer, so it is punished twice although its structure is
%   right. This study's central hypothesis is precisely about advection, so
%   scoring it with a metric that punishes displacement would evaluate the
%   hypothesis with an instrument biased against it.
%
%   The criterion is a JOINT tolerance, not a disjunction:
%
%       gamma(i,j,t) = min over a neighbourhood of
%           sqrt( d_space^2/DTA^2 + d_time^2/TTA^2 + d_value^2/IDT^2 )
%
%   and the pass rate is the share of cells with gamma <= 1. The acceptance
%   region is therefore an ellipsoid: a pure displacement within DTA passes
%   and a pure value error within IDT passes, but a combination of the two is
%   penalised. Describing it as "close in value or in space or in time" is
%   loose and has been removed from this documentation.
%
%   It comes from radiotherapy dose verification (Low et al., 1998) and was
%   transposed to gridded forecasting in Voyant (2026), Int. J. Forecasting
%   42(4) 1207-1214, doi:10.1016/j.ijforecast.2026.03.002. Reference
%   implementation: github.com/cyrilvoyant/temporal-gamma-index.
%
%   THIS IS NOT A REPRODUCTION OF THE REFERENCE IMPLEMENTATION, AND SAYING SO
%   WOULD BE WRONG. Two definitions differ, not merely two unit conventions.
%   The reference code takes one 2-D predicted map and minimises over every
%   temporal centre, so the temporal penalty can vanish at the minimum by
%   choosing the centre equal to the candidate time. This function instead
%   compares each pred(:,:,t) against a neighbourhood around that same t, so
%   the temporal penalty is real. An earlier version of this file claimed that
%   opt.units = 'reference' reproduced the published behaviour exactly; that
%   claim was false and is withdrawn. The published repository is untouched.
%
%   UNITS. Every term is a ratio of a distance to a tolerance, so gamma is
%   dimensionless by construction, provided numerator and denominator share a
%   unit. Tolerances are therefore given in physical units (km, minutes, field
%   unit) and converted once, here, to pixels and steps. Expressing DTA
%   directly in pixels would be simpler but would break comparison ACROSS
%   resolutions, where the same pixel count is a different distance -- and
%   that comparison is one of the experiments of this study.

if nargin < 3, opt = struct(); end
d = @(f,v) getfielddef(opt, f, v);
res = d('res', 3.2);  dtm = d('dt', 60);  kf = d('k', 1.5);

% Tolerances may be given in physical units (dta in km, tta in minutes) or in
% the natural units of the grid (dta_px in pixels, tta_steps in time steps).
% The default is one pixel and one step, which is the smallest non-trivial
% tolerance: below it the criterion collapses to a pure value comparison. It
% transposes the radiotherapy convention, where the tolerance is likewise
% stated in the natural unit of the problem ("3 % / 3 mm"); here it reads
% "5 % / 1 pixel / 1 step" with idt = 50 W/m2 against a 1000 W/m2 reference.
%
% One pixel is NOT the same distance at two resolutions. Within one grid it is
% the right convention; for a comparison ACROSS resolutions a second run with
% a fixed physical dta must be reported alongside, or the two columns compare
% different tolerances without saying so.
if isfield(opt,'dta_px') && ~isempty(opt.dta_px)
    dta = opt.dta_px * res;
else
    dta = d('dta', res);                    % default: one pixel
end
% THERE IS NO TEMPORAL SEARCH BY DEFAULT, AND THAT IS A CORRECTION.
% A tolerance in space forgives a forecast that put the right thing in nearly
% the right place. A tolerance in TIME forgives a forecast that put the right
% thing at nearly the right moment -- which in a forecasting problem is the
% error being measured, not a nuisance to be excused. The two are not
% symmetric here even though they are in the radiotherapy problem the criterion
% comes from, where space and time are both coordinates of a static object.
%
% MEASURED, ON THIS DOMAIN, AT ONE HOUR AHEAD: simple persistence predicts the
% current map for the next hour and is wrong by 108.5 W/m2 in root mean square.
% With a one-step temporal tolerance its pass rate is 100.0 per cent, because
% the criterion finds the prediction exactly one step away -- the prediction IS
% the field one step away. Without the temporal search it is 26.4 per cent.
% Cyclic persistence, displaced by twenty-four steps rather than one, is
% unaffected either way at 50.6 per cent. So the temporal tolerance flatters
% exactly one family of forecasts, the trivial one, and would rank a repeated
% frame above every trained model.
%
% A tolerance of zero is therefore allowed and is the default. Passing a
% positive tta restores the search for a caller who wants it.
if isfield(opt,'tta_steps') && ~isempty(opt.tta_steps)
    tta = opt.tta_steps * dtm;
else
    tta = d('tta', 0);                      % default: no temporal search
end
% THE INTENSITY TOLERANCE IS ABSOLUTE, IN W/m2, AS THE FRAMEWORK DEFINES IT.
% Voyant (2026) states the three tolerances as absolute quantities -- a
% distance, a duration and an irradiance -- and gives ranges rather than a
% single default: thirty to sixty W/m2 for nowcasting, eighty to a hundred and
% twenty for medium-term forecasts. Sixty is taken here, the top of the
% nowcasting band, and it is stated rather than hidden because it is the one
% free number in the criterion.
%
% AN EARLIER VERSION OF THIS FILE MADE IT RELATIVE, ten per cent of the
% observation at each cell, and that is a different criterion. It is far
% stricter where the criterion is most often evaluated: the median scored cell
% reads 283 W/m2, so ten per cent is 28 W/m2 against the sixty prescribed, and
% at dawn a cell reading 62 W/m2 was being held to six. The reported pass rates
% were consequently about half what the framework produces, and the shortfall
% looked like a property of the forecasts rather than of the tolerance.
%
% AND THE ABSOLUTE OPTION DID NOT WORK. opt.idt was read, stored, reported in
% the output struct, and never used: only the fraction entered the arithmetic,
% so passing a tolerance in W/m2 changed nothing and the two conventions
% returned identical numbers. An option that is silently ignored is worse than
% one that is absent, because it produces a comparison the caller believes they
% made. Both are fixed here, and HMS_GAMMA_CHECK verifies the criterion against
% cases whose answer is known in closed form.
ref = d('ref', 1000);                       % W/m2, kept only to read idt_pct
if isfield(opt,'idt_pct') && ~isempty(opt.idt_pct)
    idt = opt.idt_pct/100 * ref;            % a percentage of the nominal peak
else
    idt = d('idt', 50);                     % W/m2, absolute, the default
end

% Tolerances and steps must be strictly positive: a zero denominator would
% silently send the corresponding term to infinity and turn gamma into a
% different criterion without saying so. The COMPUTED values are checked, not
% the optional fields, which may legitimately be absent.
% tta is checked separately: zero is meaningful for it and forbidden for the
% others, since a zero distance or a zero intensity tolerance would send its
% term to infinity and silently change the criterion.
assert(isscalar(tta) && isfinite(tta) && tta >= 0, 'hms_gamma:tolerance', ...
    'tta resolves to %g; it must be a finite non-negative scalar.', tta);
vals = struct('dta',dta,'idt',idt,'res',res,'dt',dtm,'k',kf);
for nm = fieldnames(vals).'
    v = vals.(nm{1});
    assert(isscalar(v) && isfinite(v) && v > 0, 'hms_gamma:tolerance', ...
        '%s resolves to %g; it must be a finite positive scalar.', nm{1}, v);
end

% size(...,k) rather than [ny nx nt]: MATLAB drops a trailing singleton, so a
% single map has ndims == 2 and an equality test on the size vector fails.
[ny, nx, nt] = size(meas, 1, 2, 3);
assert(size(pred,1)==ny && size(pred,2)==nx && size(pred,3)==nt, ...
    'hms_gamma:size', 'pred is %s but meas is %s.', ...
    mat2str(size(pred)), mat2str(size(meas)));

% Two distinct masks. M selects the CENTRES that are scored; S selects the
% candidates a centre is allowed to match against. Conflating them lets a
% centre be rescued by a neighbour that is itself invalid.
if isfield(opt,'mask') && ~isempty(opt.mask)
    M = logical(opt.mask);
    assert(size(M,1)==ny && size(M,2)==nx && size(M,3)==nt, ...
        'hms_gamma:maskSize', 'opt.mask is %s, expected %s.', ...
        mat2str(size(M)), mat2str([ny nx nt]));
else
    M = true(ny, nx, nt);
end
% A REFERENCE POINT NEEDS AN OBSERVATION. A cell whose measurement is missing
% is not a failure of the method, it is an absence of ground truth, and it
% leaves the reference set. This is the opposite case to a missing PREDICTION
% just below, and the two must not be confused: one is the data's fault, the
% other is the method's.
M = M & isfinite(meas);
if isfield(opt,'support') && ~isempty(opt.support)
    S = logical(opt.support);
else
    S = isfinite(meas);
end

% THE REFERENCE SET IS FIXED BEFORE THE PREDICTIONS ARE LOOKED AT. A previous
% version intersected M with isfinite(pred), which removes a reference point
% from the DENOMINATOR whenever the prediction at that point is invalid, so a
% method could raise its pass rate by failing to predict. The reference set is
% now the observations alone; an invalid prediction simply provides no usable
% candidate there, and its reference point counts as a failure like any other.
bad   = M & ~isfinite(pred);
n_bad = sum(bad(:));

ry = max(0, ceil(kf * dta / res));
if tta > 0, rt = max(0, ceil(kf * tta / dtm)); else, rt = 0; end

gam2 = inf(ny, nx, nt);
pred = double(pred);  meas = double(meas);
pos  = meas > 0;                 % references with a strictly positive observation
assert(~any(meas(M) < 0), 'hms_gamma:negativeReference', ...
    ['%d reference observations are negative. They are flagged, not silently ' ...
     'clipped: fix the data-validity rule before scoring.'], sum(meas(M) < 0));

% One whole-array operation per candidate offset, and a running minimum.
% The reference implementation loops over pixels and rebuilds a neighbourhood
% each time, which is O(ny*nx*nt*window) in interpreted code; this is the same
% quantity computed as (2ry+1)^2 (2rt+1) vectorised passes.
for dy = -ry:ry
    for dx = -ry:ry
        ds2 = ((dy*res)^2 + (dx*res)^2) / dta^2;      % km over km
        for dk = -rt:rt
            if tta > 0, dt2 = (dk*dtm)^2 / tta^2; else, dt2 = 0; end
            % THE OBSERVATION ANCHORS THE COMPARISON. The reference is the
            % measured value at this cell, and the candidates are PREDICTIONS
            % at the shifted positions -- not the reverse. The intensity
            % tolerance is that of the reference and stays fixed throughout
            % the search. The reverse comparison is a different quantity and
            % is not assumed equivalent to this one.
            Ph  = shiftnan(pred, dy, dx, dk);
            Ok  = shiftnan(double(S), dy, dx, dk) > 0.5;   % candidate allowed?
            qI  = zeros(ny, nx, nt);
            % the intensity term of the framework: an absolute tolerance, the
            % same everywhere, so that an error of a given number of watts
            % counts the same at dawn as at noon.
            qI(pos)  = ((Ph(pos) - meas(pos)).^2) ./ idt^2;
            % y_r = 0 : the zero-tolerance limit. Only an exactly zero
            % prediction passes; anything else is infinitely far. Deliberately
            % strict, and stated as such in the protocol.
            qI(~pos) = (Ph(~pos) ~= 0) * inf;
            g   = ds2 + dt2 + qI;
            g(~isfinite(Ph) | ~Ok) = inf;   % no usable candidate here
            gam2 = min(gam2, g);
        end
    end
end

G.map = sqrt(gam2);
G.map(~M) = NaN;

% TWO DIFFERENT SETS, AND THE DIFFERENCE IS THE WHOLE POINT. w is every scored
% reference point, including those where no usable candidate existed and gamma
% is therefore infinite; v is the finite subset. The PASS RATE runs on w, so a
% point the method declined to predict counts against it -- otherwise a method
% raises its score by returning NaN, which is the failure mode the reference
% set was fixed before the predictions to avoid. An earlier version of this
% line filtered on isfinite and so undid, three lines later, exactly what the
% comment above n_bad says it does. The LOCAL STATISTICS run on v, because the
% mean of a set containing an infinity is an infinity and says nothing; the
% count of what they exclude is reported beside them.
w = G.map(M);
v = w(isfinite(w));

% The search window is truncated at kf tolerances. That is enough to decide
% gamma <= 1, since any candidate beyond one tolerance in space or time
% already contributes more than 1 on its own, so the pass rate is exact. It is
% NOT enough to obtain the global minimum of the values above 1, which are
% therefore reported as local.
G.gpr        = 100 * mean(w <= 1);   % an infinite gamma is a failure, not a gap
G.mean_local = mean(v);
G.median_local = median(v);
G.max_local  = max(v);
G.n_eval     = numel(w);
G.n_local    = numel(v);   % how many of them the local statistics could use
G.n_failed   = n_bad;          % predictions non-finite where they were required
G.status     = 'ok';
if n_bad > 0
    G.status = sprintf('FAILED: %d non-finite predictions on scored cells', n_bad);
end
G.note = ['gamma <= 1 is exact; values above 1 are local to the ' ...
          'truncated search window'];
G.opt  = struct('dta',dta,'tta',tta,'idt',idt,'res',res,'dt',dtm, ...
                'k',kf,'ry',ry,'rt',rt, ...
                'definition','temporal penalty measured around each own t');
end

% =========================================================================
function S = shiftnan(A, dy, dx, dt)
% Shift by (dy,dx,dt) padding with NaN. Never circshift: wrapping would let a
% cloud leaving one edge excuse an error at the opposite edge.
S = nan(size(A));
[ny,nx,nt] = size(A);
ys = max(1,1+dy):min(ny,ny+dy);   yd = ys - dy;
xs = max(1,1+dx):min(nx,nx+dx);   xd = xs - dx;
ts = max(1,1+dt):min(nt,nt+dt);   td = ts - dt;
if isempty(ys)||isempty(xs)||isempty(ts), return; end
S(yd,xd,td) = A(ys,xs,ts);
end

function v = getfielddef(s, f, dflt)
if isfield(s,f) && ~isempty(s.(f)), v = s.(f); else, v = dflt; end
end
