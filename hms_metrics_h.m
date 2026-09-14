function [M, Lref, secs] = hms_metrics_h(get, mu, opt)
%HMS_METRICS_H The six metrics, at every reported horizon, computed in one place.
%
%   [M, Lref, secs] = HMS_METRICS_H(get, mu, opt)
%
%   WHY THIS FILE EXISTS. The Corsica campaign reported six metrics broken down
%   by lead time; every other campaign reported the error alone. So a reader who
%   wanted the pass rate of the ELM on the Sahara, or the bias of the U-Net at
%   twelve hours, could not have it without running the campaign again -- and a
%   campaign that has to be rerun to answer an obvious question was not finished
%   the first time. The metric block is therefore extracted here and every arm
%   calls it, so that all of them carry the same columns and nobody has to
%   choose in advance which question the results will be asked.
%
%   THE ARGUMENT IS A GETTER, NOT AN ARRAY, because the volumes are large and
%   only one horizon is needed at a time. get(ih) returns the three volumes for
%   the ih-th reported horizon,
%
%       [Pv, Yv, Kv] = get(ih)
%
%   each of size (ny, nx, nt): the prediction, the observation and the logical
%   evaluation mask, in the grid layout the gamma index requires. Holding all
%   six horizons of a thirteen-domain campaign at once would cost gigabytes for
%   no reason, and a getter lets each caller keep its own storage private --
%   some hold predictions per origin, some per cell.
%
%   WHAT IS RETURNED. M carries, at each horizon of HMS_HORIZONS: nRMSE, nMAE and
%   nMBE, all normalised by mu; the NICE family at orders one to three and its
%   mean NICE^Sigma; the mean local gamma and the gamma pass rate. Lref carries
%   this arm's L^1, L^2 and L^3 error norms, 3 by nH, which the caller stores
%   when this arm is the NICE reference and passes back as opt.niceref for every
%   later arm. secs is what the block cost, because the gamma index dominates it
%   and the study records its computation times.
%
%   THE ERROR CONVENTION IS HMS_SCORE, POOLED, without exception. NICE is a ratio
%   of expectations by its published definition, so it is a ratio of pooled
%   norms here, which is the same convention read at a different order.

if nargin < 3, opt = struct(); end
g = @(f,dv) hms_getfield_or(opt, f, dv);
HS  = g('horizons', hms_horizons());
res = g('res', 3.4);          % km per pixel; DTA is one pixel, so it cancels
dtm = g('dt', 60);            % minutes per step
NR  = g('niceref', []);       % 3 by nH, the reference L-norms, or empty
% THE REFERENCE SCORES ITSELF AS ONE. NICE is this arm's error norms over the
% reference's, so when this arm IS the reference the ratio is one by
% construction, not missing. Leaving it NaN would put a hole in the column
% that a reader would have to be told how to fill.
isref = g('isref', false);

nH = numel(HS);
M = struct('h', HS, 'rmse', nan(1,nH), 'mae', nan(1,nH), 'mbe', nan(1,nH), ...
           'nrmse', nan(1,nH), 'nmae', nan(1,nH), 'nmbe', nan(1,nH), ...
           'nice1', nan(1,nH), 'nice2', nan(1,nH), 'nice3', nan(1,nH), ...
           'nicesig', nan(1,nH), 'gamma', nan(1,nH), 'gpr', nan(1,nH), ...
           'n_eval', zeros(1,nH), 'scale_name', 'oos_mean_ghi', ...
           'scale_value', mu);
Lref = nan(3, nH);

t0 = tic;
for ih = 1:nH
    [Pv, Yv, Kv] = get(ih);
    Kv = logical(Kv);
    if ~any(Kv(:)), continue; end
    e = double(Pv) - double(Yv);
    v = e(Kv);                       % every scored value, pooled, one bag

    M.rmse(ih)   = sqrt(mean(v.^2));
    M.mae(ih)    = mean(abs(v));
    M.mbe(ih)    = mean(v);
    M.nrmse(ih)  = M.rmse(ih) / mu;
    M.nmae(ih)   = M.mae(ih)  / mu;
    M.nmbe(ih)   = M.mbe(ih)  / mu;
    M.n_eval(ih) = numel(v);

    % THE NICE FAMILY. L^k of the error, this arm over the reference, the
    % reference being simple persistence. A ratio of pooled norms: the pooled
    % norm IS the expectation its definition takes.
    Lk = arrayfun(@(q) mean(abs(v).^q)^(1/q), 1:3);
    Lref(:,ih) = Lk(:);
    if isref
        M.nice1(ih) = 1;  M.nice2(ih) = 1;  M.nice3(ih) = 1;
        M.nicesig(ih) = 1;
    elseif ~isempty(NR) && all(isfinite(NR(:,ih)))
        r = Lk(:) ./ NR(:,ih);
        M.nice1(ih) = r(1);  M.nice2(ih) = r(2);  M.nice3(ih) = r(3);
        M.nicesig(ih) = mean(r);
    end

    % THE GAMMA INDEX, on the volume this horizon actually is. One pixel of
    % spatial tolerance, 50 W/m2 absolute in intensity, and no temporal search
    % -- the protocol of HMS_GAMMA, which a one-step temporal tolerance would
    % have turned into a free pass for persistence.
    G = hms_gamma(Pv, Yv, struct('res', res, 'dt', dtm, 'mask', Kv));
    M.gamma(ih) = G.mean_local;
    M.gpr(ih)   = G.gpr;
end
secs = toc(t0);
end

function v = hms_getfield_or(s, f, dv)
if isfield(s, f) && ~isempty(s.(f)), v = s.(f); else, v = dv; end
end
