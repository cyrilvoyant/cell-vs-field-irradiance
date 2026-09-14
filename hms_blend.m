function [P, lam, npar, Td] = hms_blend(Vte, ote, dt_min, H, Vtr, Ktr, verbose)
%HMS_BLEND The simplified cyclostationary BLEND persistence operator, on a field.
%
%   [P, lam, npar, Td] = HMS_BLEND(Vte, ote, dt_min, H, Vtr, Ktr)
%
%   Vte     d x nte  the field the forecast is made on
%   ote     1 x m    origin columns into Vte
%   dt_min  the sampling step in MINUTES: 60 here, 15 on quarter-hourly CAMS
%   H       horizons to produce, 1..H
%   Vtr     d x ntr  the field the coefficient is estimated on
%   Ktr     d x ntr  scored mask for Vtr
%   P       m x (d*H) forecast, the layout every arm in HMS_FORECAST_BENCH uses
%   lam     Td x H coefficients, by phase and horizon
%   npar    stored values the receiver needs, Td*H
%   Td      the period the function derived, in steps
%
%   WHAT THIS IS. The operator P~o_BLEND of Voyant et al., Applied Mathematical
%   Modelling 157 (2026) 116988, ported from the author's own MATLAB at
%   github.com/cyrilvoyant/cyclostationary-forecasting-matlab (Zenodo
%   10.5281/zenodo.18812334), files PREDICT_P_BLEND_TILDE and CYCLIC_PARAMETERS.
%   It forecasts as a convex combination of cyclic and simple persistence,
%
%       P_BLEND(I)(t+h) = (1 - lambda) I(t+h-T) + lambda I(t),
%
%   It therefore interpolates between the two references this study already
%   reports: lambda = 0 is cyclic persistence and lambda = 1 is simple
%   persistence.
%
%   THE OPERATOR IS THE PUBLISHED ONE. THE COEFFICIENT RULE IS NOT, AND THE
%   PAPER MUST SAY SO. The reference derives lambda in closed form,
%   lambda~ = (1 + rho)/2 with rho a phase-dependent correlation, and its
%   real-data section applies the operators to the clear-sky index
%   k_r = I / I_clr. This study feeds raw irradiance and takes no clear-sky
%   index anywhere, by an explicit decision recorded in HMS_CONFIG,
%   HMS_BUILD_FIELD and HMS_BASELINES. Under that convention the closed form
%   returns about 0.85 through the day and the operator scores 0.613, worse
%   than the 0.396 of the cyclic persistence it is built from.
%
%   Here lambda is ESTIMATED BY LEAST SQUARES, ONE VALUE PER PHASE AND PER
%   HORIZON, on the training year. The arm keeps the name BLEND, and every
%   place it is reported states the estimator, because a reader who sees only
%   the name would otherwise credit the published closed form with numbers it
%   did not produce.
%
%   THE TRAINING FIELD IS A SEPARATE ARGUMENT SO THAT LEAKAGE IS IMPOSSIBLE.
%   Lambda is a fitted quantity: it is a correlation estimated from data. A
%   signature taking one field would let a caller estimate it on the year it
%   then reports, and nothing in the output would show it. Requiring Vtr and
%   Vte separately makes the honest use the only use.
%
%   THE PERIOD IS A DAY, NOT TWENTY-FOUR STEPS, AND THAT IS WHY THE ARGUMENT
%   IS A DURATION. An earlier signature took the period in steps. On the
%   quarter-hourly archive a caller passing 24 would have reached back six
%   hours instead of a day, and the operator would have blended the wrong two
%   things without any symptom: the forecast stays finite, the shapes match,
%   and only the numbers are wrong. Deriving Td = 1440/dt_min inside removes
%   the chance to get it wrong. It also means the stored cost grows with the
%   cadence: Td*H is 24*H at an hourly step and 96*H at a quarter-hourly one.
%
%   ONLY THE SIMPLIFIED CYCLOSTATIONARY VARIANT IS IMPLEMENTED. The source
%   repository also carries a stationary form, whose coefficient comes from
%   Corollary 7 and costs H stored values instead of Td*H. It was written,
%   checked against the reference, and removed at Cyril's instruction: two
%   operators under one name is how a benchmark stops meaning anything.
%
%   LAMBDA SITS ON SIMPLE PERSISTENCE, AND THE PAPER STATES IT BOTH WAYS.
%   Eq. (C.1) and the mean-squared-error expansion put the coefficient there;
%   Eq. (D.1) puts it on the lagged term. Both reference implementations write
%   (1-lambda)*cyclic + lambda*simple, so that is the convention here. Reading
%   (D.1) into this code would silently return the complement.
%
%   THE FIT USES SCORED TARGETS ONLY, and a phase and horizon with too few
%   samples keeps lambda = 0, which is cyclic persistence. That is the right
%   default and not merely the safe one: lambda weights the present
%   observation, and when the present is night it carries nothing about a
%   daytime target, so its weight should vanish.
%
%   CLIPPING TO [0,1] IS THE REFERENCE BEHAVIOUR, not a repair invented here.
%   Lemma 6 shows the optimum can leave the interval as the periodic structure
%   weakens, and Theorem 11 shows the operator is only stable inside it.

if nargin < 7, verbose = true; end

assert(isscalar(dt_min) && dt_min > 0 && mod(1440, dt_min) == 0, ...
    'hms_blend:step', ...
    ['The sampling step must divide a day: %g minutes does not. The period ' ...
     'of this operator is one day, and a period that is not a whole number ' ...
     'of steps has no meaning here.'], dt_min);
Td = round(1440 / dt_min);

[d, nte] = size(Vte);
assert(size(Vtr, 1) == d, 'hms_blend:cells', ...
    'The training field has %d cells and the test field %d.', size(Vtr, 1), d);
if isempty(Ktr), Ktr = true(size(Vtr)); end
Ktr = logical(Ktr);
assert(isequal(size(Ktr), size(Vtr)), 'hms_blend:mask', ...
    'The training mask does not match the training field.');

ote = ote(:).';
m = numel(ote);
assert(all(ote + H <= nte), 'hms_blend:short', ...
    'An origin runs past the end of the record.');
assert(all(ote + 1 - Td >= 1), 'hms_blend:early', ...
    ['An origin has no full cycle behind it: with a %g minute step the ' ...
     'period is %d steps, so the first usable origin is %d.'], ...
    dt_min, Td, Td);

% ---- the coefficient, from the TRAINING year only
lam = phase_lambda(Vtr, Ktr, Td, H);
npar = numel(lam);

% ---- the forecast, on the test year
P = zeros(m, d * H);
ph = mod(ote - 1, Td) + 1;                     % phase of each origin
simple = Vte(:, ote).';                        % m x d, I(t), same at every h
for h = 1:H
    c = (h-1)*d + 1 : h*d;
    cyc = Vte(:, ote + h - Td).';              % m x d, I(t + h - T)
    L = lam(sub2ind(size(lam), ph(:), repmat(h, m, 1)));
    P(:, c) = (1 - L) .* cyc + L .* simple;
end

if verbose
    fprintf('\nBLEND: %d stored values (%d phases x %d horizons)\n', ...
            npar, Td, H);
    fprintf('  step %g min, period %d steps = %g h\n', ...
            dt_min, Td, Td * dt_min / 60);
    fprintf('  lambda  min %.3f  median %.3f  max %.3f\n', ...
            min(lam(:)), median(lam(:)), max(lam(:)));
    nb = nnz(lam == 0 | lam == 1);
    if nb
        fprintf('  %d of %d coefficients at a bound\n', nb, npar);
    end
end
end

% ---------------------------------------------------------------------------
function [lam, nfit] = phase_lambda(V, K, Td, H)
%PHASE_LAMBDA The least-squares blend weight, one per phase and horizon.
%
%   For each phase p and horizon h, over every training origin at that phase
%   and every scored cell, the operator predicts
%
%       yhat = (1 - lam) c + lam s,   c = I(t+h-T),  s = I(t),
%
%   so the residual is (y - c) - lam (s - c) and the coefficient that
%   minimises the squared error has the closed form
%
%       lam = sum(e .* g) / sum(g .^ 2),   e = y - c,   g = s - c.
%
%   WHY THIS REPLACED THE CORRELATION FORM. The published coefficient is
%   (1 + rho)/2, derived under assumptions the reference paper states plainly:
%   its real-data section applies the operators to the clear-sky index
%   k_r = I / I_clr, and its own demonstration runs on a clean 24 h sinusoid
%   with no season. This study feeds raw irradiance and takes no clear-sky
%   index anywhere, by an explicit decision recorded in HMS_CONFIG,
%   HMS_BUILD_FIELD and HMS_BASELINES. On raw irradiance rho is the correlation
%   of anomalies between two hours of the day across days; it is high, because
%   a cloudy day is cloudy all day, so (1 + rho)/2 came out near 0.85 and put
%   most of the weight on an observation whose LEVEL is wrong three hours
%   later. The operator scored 0.613 against 0.396 for the cyclic persistence
%   it is built from.
%
%   Fitting the same coefficient by least squares asks the question directly
%   and needs no stationarity assumption, no clear-sky model and no
%   deseasonalising. It is the same operator family: lam = 0 is cyclic
%   persistence, lam = 1 is simple persistence, and both remain reachable.
%
%   IT CANNOT DO WORSE THAN CYCLIC PERSISTENCE ON THE TRAINING YEAR, because
%   lam = 0 is in the feasible set and the objective is the training error
%   itself. On the test year it can, and that gap is the honest measure of
%   what the twenty-four by twenty-four table of coefficients buys.
%
%   THE FIT IS POOLED OVER CELLS, matching how the arm is scored: one error
%   over every scored (cell, step) pair, not a mean of per-cell errors.
[d, n] = size(V);
lam = zeros(Td, H);
nfit = zeros(Td, H);

% Every origin with a full cycle behind it and h steps ahead of it.
o = (Td : n - H);
assert(~isempty(o), 'hms_blend:cycles', ...
    'The training field is too short to hold one cycle and one horizon.');
ph = mod(o - 1, Td) + 1;

for p = 1:Td
    op = o(ph == p);
    if isempty(op), continue; end
    for h = 1:H
        y = V(:, op + h);                      % d x m target
        c = V(:, op + h - Td);                 % d x m cyclic candidate
        s = V(:, op);                          % d x m simple candidate
        m = K(:, op + h);                      % scored where the target is
        m = m & isfinite(y) & isfinite(c) & isfinite(s);
        if nnz(m) < 30, continue; end          % too few samples: keep 0
        e = y(m) - c(m);
        g = s(m) - c(m);
        den = sum(g .^ 2);
        if ~isfinite(den) || den <= 0, continue; end   % candidates coincide
        nfit(p, h) = nnz(m);
        % CLIPPED TO [0,1], WHICH IS THE OPERATOR'S OWN CONDITION. Theorem 11
        % of the reference bounds the forecast by the two candidates only
        % inside that interval; outside it the blend extrapolates, and an
        % extrapolating persistence is no longer a persistence.
        lam(p, h) = min(max(sum(e .* g) / den, 0), 1);
    end
end
end
