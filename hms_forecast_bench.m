function R = hms_forecast_bench(outfile, ncand, latents, ps)
%HMS_FORECAST_BENCH The forecasting protocol, exactly as specified.
%
%   R = HMS_FORECAST_BENCH(outfile, ncand)
%
%   THE PROTOCOL, IN THE AUTHOR'S OWN TERMS. Standing at 12 October 12:00, the
%   model forecasts the following 24 hours, to 13 October 12:00. The first
%   output column is h = 1 h, the second h = 2 h, and so on to h = 24 h. That
%   window contains night as well as day; a cell whose LOCAL solar elevation is
%   at or below 5 degrees is NOT scored. The reported quantity is the error PER
%   HORIZON, averaged per map over the period. One year in sample for training,
%   one year out of sample for test, hourly sliding origins throughout.
%
%   WHY SOLAR ELEVATION AND NOT THE DATA. The evaluation set has to be defined
%   without looking at the quantity being evaluated. Elevation is deterministic,
%   known arbitrarily far in advance, and identical for every method including
%   the naive references. It defines WHERE we look; it never transforms what
%   the model sees, and no clear-sky or top-of-atmosphere normalisation is
%   reintroduced anywhere.
%
%   THE ARMS, chosen so that each comparison isolates one thing.
%     persistence         Vhat(t+h) = V(t)
%     cyclic persistence  Vhat(t+h) = V(t+h-24)
%     ridge AR per pixel  each series on its own L lags, its ridge penalty
%                         swept on the same split, no spatial information at all
%     linear random proj  the ELM with the tanh REMOVED: same inputs, same
%                         random weights, same ridge. The only difference from
%                         the ELM is the nonlinearity, so the gap between the
%                         two IS the effect of the nonlinearity and nothing else
%     ELM                 the nonlinear model, ncand candidate hidden layers,
%                         the best kept on validation
%
%   Nothing is stationarised and no residual is taken: the models see raw GHI
%   and predict raw GHI, as instructed.

if nargin < 1 || isempty(outfile), outfile = fullfile(pwd,'results','bench.mat'); end
if nargin < 2 || isempty(ncand),   ncand   = 100; end
if nargin < 3, latents = {}; end
if nargin < 4 || isempty(ps), ps = [49 98 196 392]; end
if ~isfolder(fileparts(outfile)), mkdir(fileparts(outfile)); end

n  = 32;  cfg = hms_config('n', n);
L  = 24;  H = 24;  Td = 24;
ELEV_MIN = 5;                      % degrees, the project's scoring threshold

% THE SPLIT IS CYRIL'S, AND IT DOES NOT MOVE. One year trains, the same year
% selects, the next year tests. The ninety-day validation window this file used
% to carry was retired everywhere else in the project months of work ago; left
% here it meant two things at once, both wrong. The hyper-parameters were frozen
% against a criterion computed on days 1..365, and were then applied by a bench
% that scored its draws on 366..455. And the test period was 456..820, so the
% forecasting half of the paper was reporting a different year from the
% reconstruction half.
%
% Selecting on the training year is the declared choice, with its consequence
% stated in the paper: a criterion evaluated on the data that fitted the model
% cannot penalise capacity.
dtr = 1:365;        % one year in sample
dva = dtr;          % the same year selects, as the protocol declares
dte = 366:730;      % the next year, out of sample, touched once

t0 = tic;
fprintf('\n=== forecasting bench ===\n');
fprintf('train %d d | val %d d | test %d d | L=%d | H=%d | scored above %g deg\n', ...
        numel(dtr), numel(dva), numel(dte), L, H, ELEV_MIN);

[Ftr, Mtr, info] = hms_build_field(dtr, cfg);
[Fva, ~, ~]      = hms_build_field(dva, cfg);
[Fte, ~, ~]      = hms_build_field(dte, cfg);
Vtr = reshape(Ftr,n*n,[]); Vva = reshape(Fva,n*n,[]); Vte = reshape(Fte,n*n,[]);
clear Ftr Fva Fte
% THE SUPPORT IS THE ISLAND, DEFINED THE SAME WAY THE GRID DEFINED IT. Taking
% the validity of the first time step alone treats as permanent whatever
% happened to be reported that hour; measured over a year the best coverage any
% cell reaches is 0.9885, and the missing instants are whole maps rather than
% scattered cells. HMS_GRIDSEARCH chose the hyper-parameters on cells with more
% than half coverage, and a benchmark that used them on a different support
% would not be using them at all.
msk = mean(Mtr, 3) > 0.5;  msk = msk(:);  d = numel(msk);
fprintf('island support %d of %d cells\n', nnz(msk), d);
latc = mean(info.lat_grid); lonc = mean(info.lon_grid);
fprintf('fields built in %.0f s\n', toc(t0));

% THE EVALUATION SET IS THE PROJECT'S RULE, PER PIXEL, NOT A CENTRE.
% This used to call HMS_SOLAR once at the domain centre and replicate the answer
% across every cell. Two things were wrong with it. The threshold was one degree
% where the protocol of the paper says five. And a single centre on a domain
% 110 by 216 km scores cells that are still dark and skips cells already lit ---
% the per-cell and centre-only masks disagree on 0.77% of cells, which is small
% and is not nothing. It also called HMS_SOLAR with the signature it had before a
% calendar became mandatory, so this function had not run since.
[Lon, Lat] = meshgrid(info.lon_grid, info.lat_grid);
cal = struct('resolved', true, 'days_in_year', 365, ...
             'source', sprintf('HelioClim-3, origin %s', cfg.absolute_start_date));
elmask = @(dd) reshape(hms_eval_mask(Lat, Lon, epoch_of(dd, cfg), cal, ...
                                    struct('eval_deg', ELEV_MIN)), n*n, []);
day_tr = elmask(dtr);
day_va = elmask(dva);
day_te = elmask(dte);

% NIGHT IS FORCED TO ZERO, BECAUSE THE PROTOCOL SAYS SO. A forecast that
% predicts irradiance where the sun is below the horizon is wrong whatever the
% score says, and the score here does not say: the evaluation set already
% excludes everything below five degrees, so not one night cell is ever counted
% --- measured, zero scored cells fall below 1 W/m2 and the smallest is 19. The
% forcing therefore changes no number in this paper. It is applied anyway, so
% that the prediction this code delivers is the prediction the paper describes.
% HMS_EVAL_MASK ALREADY RETURNS IT, as its second output, so there is nothing to
% compute twice. A first attempt called it again with eval_deg = 0, which its own
% assertion rejects: the evaluation threshold must sit strictly above the night
% threshold, and zero is not above zero.
[K5te, NTte] = hms_eval_mask(Lat, Lon, epoch_of(dte, cfg), cal, ...
                            struct('eval_deg', ELEV_MIN));
day_te = reshape(K5te, n*n, []);
sun_te = ~reshape(NTte, n*n, []);        % true where the sun is at or above 0
fprintf('lit cells  train %.1f%% | val %.1f%% | test %.1f%%\n', ...
        100*mean(day_tr(:)), 100*mean(day_va(:)), 100*mean(day_te(:)));

otr = hms_codex_origins([1 size(Vtr,2)], max(L,Td), H);
ova = hms_codex_origins([1 size(Vva,2)], max(L,Td), H);
ote = hms_codex_origins([1 size(Vte,2)], max(L,Td), H);
hms_resid('origins', ote);   % for the paired test; no-op when capture is off
fprintf('origins  train %d | val %d | test %d\n', numel(otr), numel(ova), numel(ote));

Ytr = tgt(Vtr,otr,H);  Yva = tgt(Vva,ova,H);  Yte = tgt(Vte,ote,H);
Ktr = flags(day_tr,otr,H,d) & repmat(msk,H,1).'; %#ok<NASGU>
Kva = flags(day_va,ova,H,d) & repmat(msk,H,1).';
Kte = flags(day_te,ote,H,d) & repmat(msk,H,1).';
NIGHT = ~flags(sun_te,ote,H,d);          % elevation at or below zero

R = struct('name',{},'rmse',{},'rmse_h',{},'n_par',{},'spread',{},'secs',{}, ...
           'met',{},'secs_met',{});
% A CAMPAIGN RESUMES INSTEAD OF RESTARTING. The reference arms cost four and
% a quarter hours between them and do not change when a representation is
% added, so an outfile that already holds them is read back and every arm it
% names is skipped. Extending a comparison then costs only the new arms, and
% the ones that carry over are bit-identical because they were not recomputed.
if isfile(outfile)
    prev = load(outfile);
    if isfield(prev,'R') && ~isempty(prev.R)
        R = prev.R;
        % A CAMPAIGN WRITTEN BEFORE A FIELD EXISTED IS STILL A VALID CAMPAIGN.
        % MATLAB refuses to append a struct carrying more fields than the array
        % it joins, so an outfile from before the metric block would make the
        % first new arm fail on an assignment rather than on anything real. The
        % missing fields are added empty, which is also the honest record: those
        % arms do not have those numbers and nothing should pretend they do.
        for fn = {'met','secs_met'}
            if ~isfield(R, fn{1}), [R.(fn{1})] = deal([]); end
        end
        fprintf('resuming: %d arms already in %s\n', numel(R), outfile);
    end
    clear prev
end
if ~have(R,'persistence'), R = push(R,'persistence',    repmat(Vte(:,ote).',1,H), Yte,Kte,d,H,0,0,0,NIGHT); end
if ~have(R,'cyclic'), R = push(R,'cyclic',         tgt(Vte,ote-Td,H),        Yte,Kte,d,H,0,0,0,NIGHT); end

% ---------------------------------------------------- ridge AR, per pixel
% THE PENALTY IS SWEPT HERE TOO, AND IT WAS NOT. This arm used to be fitted at
% lambda = 1e-2, a value nobody chose, while every ELM arm received a penalty
% frozen from a joint grid. Comparing them then compares tuning efforts rather
% than methods, which is the fault this study exists to avoid. The sweep is
% almost free: the Gram matrix A'A does not depend on lambda, so one
% accumulation per pixel serves every penalty, exactly as for the ELM.
%
% The tie is broken by the rule HMS_FREEZE uses, for the same reason -- the
% surface is flat, so the bare argmin returns whichever equivalent cell sorts
% first. Among penalties within TOL of the minimum, the largest is kept.
% ONE SHARED LINEAR MODEL, MATCHING THE SHARED NONLINEAR ONE. This arm used to
% fit one ridge PER CELL, seven hundred and twenty-two of them. Against a
% nonlinear arm that shares one model across every cell, that comparison varies
% the activation AND the sharing at once, so the gap between them cannot be read
% as the worth of the nonlinearity --- which is the whole purpose of the design.
%
% Both per-pixel arms now use the same samples, the same twenty-four lags, the
% same targets and the same kind of penalty; only the tanh differs. The penalty
% is a RATIO of the mean diagonal of the Gram matrix, as in the grid, because an
% absolute one is swamped by a sample count in the millions.
tj = tic;
[Gm, GtY, N_ar] = lin_accum(Vtr, otr, L, H, msk);
RAT = [1e-8 1e-6 1e-4 1e-3 1e-2 1e-1 1e0 1e1];
sc = mean(diag(Gm));
ear = nan(1, numel(RAT));
for ia = 1:numel(RAT)
    Bi = lin_solve(Gm, GtY, RAT(ia)*sc, L);
    ear(ia) = errm(clip(lin_pred(Bi, Vva, ova, L, H, msk)), Yva, Kva);
end
[~, ib] = min(ear);
TOL_AR = 1e-4 * mean(ear);
near = find(ear <= min(ear) + TOL_AR);
lam_ar = max(RAT(near));
fprintf(['      shared ridge: %d samples, %d of %d ratios within %.3g of the ' ...
         'minimum, keeping %g (bare argmin %g)\n'], N_ar, numel(near), ...
        numel(RAT), TOL_AR, lam_ar, RAT(ib));
Bar = lin_solve(Gm, GtY, lam_ar*sc, L);
Pte = lin_pred(Bar, Vte, ote, L, H, msk);
if ~have(R,'ridge, one shared model')
    R = push(R, sprintf('ridge, one shared model, ratio=%g', lam_ar), ...
             Pte, Yte,Kte,d,H, (L+1)*H, max(ear)-min(ear), toc(tj), NIGHT);
end
clear Gm GtY


% ------------------------------------------------ the per-pixel nonlinear arm
% The fourth cell of the design. With the per-pixel linear model already in
% place and both field-wide models measured, this one completes a 2x2 in which
% scope (one pixel or the whole field) and activation (linear or not) vary
% independently, so that their separate effects and their interaction can be
% read rather than inferred.
tj = tic;
% ONE MODEL AND ONE PENALTY, WHICH IS THE ONLY TRACTABLE CHOICE. An earlier
% version fitted one small machine PER CELL, seven hundred and twenty-two of
% them, each with its own output weights. Two things were wrong with that. It is
% not the object the grid chose its hyper-parameters for --- HMS_GRIDSEARCH sweeps
% a single shared model over every pair of an origin and a cell --- so the frozen
% width and penalty did not apply to it. And it makes the per-pixel arm differ
% from the whole-field arm in two ways at once, sharing as well as scope, so the
% gap between them stops measuring what the design says it measures.
%
% The arm is now one model shared by every cell, reading twenty-four of that
% cell's own lags, at the width and penalty the freeze holds. Fitting it costs
% one accumulation instead of seven hundred, and every cell is bound by one law.
if ~have(R,'ELM per pixel')
    [Pnl, npar_siso, tsiso] = elm_siso(Vtr, Vte, otr, ote, L, H, msk);
    R = push(R, 'ELM per pixel (one shared model)', Pnl, Yte, Kte, d, H, ...
             npar_siso, 0, tsiso, NIGHT);
    clear Pnl
end

% ------------------------------- linear random projection, then the ELM
Xtr = win(Vtr,otr,L);  Xva = win(Vva,ova,L);  Xte = win(Vte,ote,L);
score = @(P) errm(clip(P), Yva, Kva);
for act = {'linear','tanh'}
    if have(R, sprintf('%s  Nh=', act{1})), continue; end
    tj = tic;
    [M,inf1] = elm_select(Xtr, Ytr, Xva, score, act{1}, ncand, ...
                          [0 1e0 1e1 1e2 1e3]);
    Pte = elm_pred(M, Xte);
    R = push(R, sprintf('%s  Nh=%d lam=%g', act{1}, M.Nh, M.lambda), ...
             Pte, Yte,Kte,d,H, M.n_stored, inf1.spread, toc(tj), NIGHT);
    save(outfile,'R','cfg','L','H','ELEV_MIN','dtr','dva','dte','ncand');
end


% ------------------------------------------- the latent representations
% Each encoder replaces the raw field by p numbers, the forecaster runs in that
% space, and the prediction is decoded back with the encoder's OWN inverse
% before being scored in field space, against the same targets, the same mask
% and the same references. Only the representation changes: the forecaster, the
% selection procedure and the metric are the ones used above.
for ik = 1:numel(latents)
    for ip = 1:numel(ps)
        nm = sprintf('%s p=%d', latents{ik}, ps(ip));
        if any(contains({R.name}, nm)), continue; end
        tj = tic;
        try
            E = hms_latent(latents{ik}, n, ps(ip), cfg, Vtr);
            Ztr = E.encode(Vtr); Zva = E.encode(Vva); Zte = E.encode(Vte);
            Ltr = win(Ztr,otr,L); Lva = win(Zva,ova,L); Lte = win(Zte,ote,L);
            Gtr = tgt(Ztr,otr,H);
            sc = @(P) errm(clip(decodeH(P, E.decode, E.p, H)), Yva, Kva);
            [Mk,i2] = elm_select(Ltr, Gtr, Lva, sc, 'tanh', ncand, [0 1e0 1e1 1e2 1e3]);
            Pk = decodeH(elm_pred(Mk, Lte), E.decode, E.p, H);
            npar = Mk.n_stored + E.n_basis + E.n_decode;
            R = push(R, sprintf('%s p=%d lam=%g', latents{ik}, E.p, Mk.lambda), ...
                     Pk, Yte, Kte, d, H, npar, i2.spread, toc(tj), NIGHT);
            save(outfile,'R','cfg','L','H','ELEV_MIN','dtr','dva','dte','ncand','ps');
        catch ME
            fprintf('  %-26s FAILED: %s\n', nm, ME.message);
        end
    end
end

report(R, H);
fprintf('total %.0f min\n', toc(t0)/60);
end

% =========================================================================
function V = decodeH(P, dec, p, H)
%DECODEH Decode each horizon block of a latent-space prediction back to field
% space, with the encoder's own inverse: no fitted decoder is allowed to
% rescue a representation and then have the gain attributed to its geometry.
N = size(P,1);  d = size(dec(zeros(p,1)), 1);
V = zeros(N, d*H);
for h = 1:H
    V(:, (h-1)*d+1 : h*d) = dec(P(:, (h-1)*p+1 : h*p).').';
end
end

function t = have(R, name)
%HAVE Is an arm with this name already in the campaign, AND is it complete?
%   Matching is by prefix, because the recorded names carry the width and the
%   penalty chosen at run time, which the caller does not know.
%
%   COMPLETE MEANS COMPLETE FOR WHAT THIS RUN COMPUTES. A campaign written
%   before the metric block exists as a row with an error and no metrics, and
%   skipping it would leave the paper reporting one metric where it promises
%   six. Worse, the reference arm is the denominator of the NICE family: skip
%   it and every later arm divides by an empty reference and returns NaN
%   without complaining. An arm that lacks the metrics is therefore recomputed,
%   which is the whole point of resuming rather than restarting -- the arms
%   that ARE complete still cost nothing.
if isempty(R), t = false; return; end
k = startsWith({R.name}, name);
if ~any(k), t = false; return; end
if ~isfield(R, 'met'), t = false; return; end
t = any(arrayfun(@(x) ~isempty(x.met), R(k)));
end

function R = push(R, name, P, Y, K, d, H, npar, spread, secs, night)
%PUSH Score one arm, after the two repairs every arm gets identically.
%   Negative predictions are clipped at zero, and predictions where the local
%   solar elevation is at or below zero are set to zero. Both are applied to
%   every arm including the naive references, so no method is credited for a
%   repair the others do not get.
%
%   THE FULL METRIC SET IS COMPUTED HERE, ONCE, at the horizons HMS_HORIZONS
%   names. It is computed in this one place for the same reason the repairs
%   are: an arm that got its numbers somewhere else is not comparable, however
%   carefully the other file was written. The set is the one this group
%   reports -- nRMSE, nMAE, nMBE, the NICE family and the gamma index with its
%   pass rate -- and every one of them is normalised or referenced the same
%   way for every arm.
%
%   NICE IS A RATIO TO SIMPLE PERSISTENCE, and that is the definition, not a
%   choice made here. Voyant et al. (2025) fix the denominator as the error of
%   the persistence forecast, and Eq. (nice) in this paper writes it that way.
%   An earlier version of this block used CYCLIC persistence instead, which is
%   a different and much stronger reference: on the test year simple
%   persistence reaches 360.9 W/m2 against 130.5 for cyclic, so every NICE it
%   produced was roughly three times too large and none of them was the
%   published metric. Cyclic persistence is reported ALONGSIDE, as the
%   discriminating reference a method must actually beat; it is not the NICE
%   denominator.
%
%   The reference therefore has to be scored before anything else. Persistence
%   is the first arm the campaign pushes, and its per-horizon error norms are
%   kept here in a persistent so that every later arm divides by the same
%   denominators. A campaign that somehow scored another arm first finds the
%   field empty and leaves NICE as NaN, rather than quietly reporting a ratio
%   of one.
%
%   GAMMA COSTS ABOUT SEVENTY SECONDS PER ARM PER HORIZON on this grid,
%   measured, and that is the reason the horizon list is six long and not
%   twenty-four.
persistent NICEREF
if nargin >= 11 && ~isempty(night), P(night) = 0; end
P = clip(P);

HS = hms_horizons();
SC = hms_scale_ref();  mu = SC.mu;   % nRMSE divides by the MEAN of the
                                     % scored observations, not by their
                                     % spread; see HMS_SCALE_REF.
nside = round(sqrt(d));
isref = strcmp(name, 'persistence');
if isref, NICEREF = []; end
if isempty(NICEREF) && ~isref
    warning('hms_forecast_bench:noNiceRef', ...
        ['%s is being scored before the persistence reference, so the NICE ' ...
         'family has no denominator and is left as NaN. The reference must ' ...
         'be the first arm the campaign pushes.'], name);
end

% THE METRIC BLOCK LIVES IN HMS_METRICS_H AND NOT HERE. It used to be inline in
% this function and in no other, which is exactly why this campaign reported
% six metrics by horizon and every other campaign reported the error alone: a
% reader who wanted the pass rate of the ELM on the Sahara, or the bias of the
% U-Net at twelve hours, had to have the campaign run again. One implementation,
% called by everything, is the only way those columns can be compared at all.
%
% SIMPLE PERSISTENCE IS THE NICE DENOMINATOR and is therefore the first arm the
% campaign pushes; its error norms are kept in the persistent above so that
% every later arm divides by the same numbers.
[M, Lref, tmet] = hms_metrics_h(@(ih) slice_h(P, Y, K, HS(ih), d, nside), mu, ...
    struct('horizons', HS, 'res', 3.4, 'dt', 60, ...
           'niceref', NICEREF, 'isref', isref));
if isref, NICEREF = Lref; end

% The per-origin errors, kept only when HMS_RESID has been switched on. Off by
% default, in which case this line does nothing and costs nothing, so the code
% path is the one that produced the published numbers.
hms_resid('add', name, P, Y, K, d, H);

R = hms_push_row(R, struct('name',name,'rmse',errm(P,Y,K), ...
    'rmse_h',errh(P,Y,K,d,H),'n_par',npar,'spread',spread,'secs',secs, ...
    'met',M,'secs_met',tmet));
i3 = find(HS == 3, 1);
fprintf(['  %-26s RMSE %7.2f W/m2 %5.0fs | h=3: nRMSE %.4f  ' ...
         'NICE^S %.3f  GPR %.1f%%  [%.0fs]\n'], ...
        name, R(end).rmse, secs, M.nrmse(i3), M.nicesig(i3), M.gpr(i3), tmet);
end

function [Pv, Yv, Kv] = slice_h(P, Y, K, h, d, nside)
%SLICE_H The three volumes at one horizon, in the grid layout gamma needs.
%   At a FIXED horizon each origin contributes exactly one map, so the volume
%   is a grid by time and carries no factor of H.
c = (h-1)*d + (1:d);
Pv = reshape(P(:,c).', nside, nside, []);
Yv = reshape(Y(:,c).', nside, nside, []);
Kv = reshape(K(:,c).', nside, nside, []);
end

function report(R, H)
fprintf('\n%-26s %9s', 'method', 'RMSE');
hs = [1 2 3 6 12 24];
for h = hs, fprintf(' %7s', sprintf('h=%d',h)); end
fprintf(' %11s\n', 'parameters');
fprintf('%s\n', repmat('-', 1, 26+9+7*numel(hs)+12));
[~,o] = sort([R.rmse]);
for k = o
    fprintf('%-26s %9.2f', R(k).name, R(k).rmse);
    for h = hs, fprintf(' %7.2f', R(k).rmse_h(h)); end
    fprintf(' %11d\n', R(k).n_par);
end
fprintf(['\nRMSE is POOLED over every scored cell of every map, the one\n' ...
         'convention of this study (see HMS_SCORE), scored only where the\n' ...
         'solar elevation exceeds the threshold.\n']);
end

% ------------------------------------------------------------- the models
function [G, GtY, N] = lin_accum(V, o, L, H, msk)
%LIN_ACCUM One shared linear model: every pair of an origin and a cell is a row.
%   The Gram matrix is 25 by 25 whatever the field size, so the fit is instant
%   once the rows are formed, and the penalty can be swept for nothing.
ip = find(msk).';  N = numel(o)*numel(ip);
G = zeros(L+1);  GtY = zeros(L+1, H);
for q = ip
    A = zeros(numel(o), L);
    for k = 1:L, A(:,k) = V(q, o+k-L).'; end
    A = [A ones(numel(o),1)]; %#ok<AGROW>
    B = zeros(numel(o), H);
    for h = 1:H, B(:,h) = V(q, o+h).'; end
    G = G + A.'*A;  GtY = GtY + A.'*B;
end
end

function B = lin_solve(G, GtY, lam_abs, L)
%LIN_SOLVE The intercept is not penalised, hence the leading block only.
G(1:L,1:L) = G(1:L,1:L) + lam_abs*eye(L);
B = G \ GtY;
end

function P = lin_pred(B, V, o, L, H, msk)
d = size(V,1);  P = zeros(numel(o), d*H);
for q = find(msk).'
    A = zeros(numel(o), L);
    for k = 1:L, A(:,k) = V(q, o+k-L).'; end
    A = [A ones(numel(o),1)]; %#ok<AGROW>
    Q = A * B;
    for h = 1:H, P(:, (h-1)*d+q) = Q(:,h); end
end
end

function ACC = ar_accum(V, o, L, H, msk)
%AR_ACCUM The expensive half of the per-pixel ridge, which lambda does not touch.
%   One ridge per pixel on its OWN lags: the classical linear benchmark, using
%   no spatial information whatever, which is what makes it the reference a
%   spatial model has to justify itself against. The Gram matrix and the
%   cross-product are accumulated once here; AR_SOLVE then applies any penalty
%   for the cost of one small solve.
d = size(V,1);
ACC = struct('G', zeros(L+1, L+1, d), 'AtB', zeros(L+1, H, d), 'msk', msk);
for p = find(msk).'
    A = zeros(numel(o), L);
    for k = 1:numel(o), A(k,:) = V(p, o(k)-L+1:o(k)); end
    A = [A ones(numel(o),1)];                                    %#ok<AGROW>
    B = zeros(numel(o), H);
    for k = 1:numel(o), B(k,:) = V(p, o(k)+1:o(k)+H); end
    ACC.G(:,:,p)   = A.'*A;
    ACC.AtB(:,:,p) = A.'*B;
end
end

function M = ar_solve(ACC, lam, L, H, d)
%AR_SOLVE The cheap half: one penalty, one solve per pixel.
%   The intercept is not penalised, which is why the ridge is added to the
%   leading L by L block alone.
M = zeros(L+1, H, d);
for p = find(ACC.msk).'
    G = ACC.G(:,:,p);
    G(1:L,1:L) = G(1:L,1:L) + lam*eye(L);
    M(:,:,p) = G \ ACC.AtB(:,:,p);
end
end

function [P, npar, secs] = elm_siso(Vtr, Vte, otr, ote, L, H, msk)
%ELM_SISO ONE extreme learning machine shared by every cell, on its own lags.
%
%   Every pair of an origin and a cell is one training sample, so the machine
%   sees twenty-four inputs and is fitted on millions of rows. That is exactly
%   the object HMS_GRIDSEARCH sweeps as its per-pixel family, which is why the
%   frozen width and penalty can be read here rather than invented.
%
%   THE PENALTY BARELY MOVES THIS ARM, AND THE REASON IS THE SAMPLE COUNT. H'H
%   is accumulated over those millions of rows, so its diagonal is of order N
%   and any modest ridge is a negligible perturbation of it. The grid therefore
%   sweeps the penalty as a RATIO of that diagonal; even so, this arm is close
%   to unregularised over most of the range, and the paper says so rather than
%   presenting a chosen penalty that changes nothing.
t0 = tic;
[Nh, lam] = frozen_hyper(L);
ip = find(msk).';  d = size(Vtr,1);  N = numel(otr)*numel(ip);
Xtr = zeros(N, L);  Ytr = zeros(N, H);
r = 0;
for q = ip
    idx = r + (1:numel(otr));
    for k = 1:L, Xtr(idx,k) = Vtr(q, otr+k-L).'; end
    for h = 1:H, Ytr(idx,h) = Vtr(q, otr+h).'; end
    r = r + numel(otr);
end
% THE HIDDEN MATRIX IS NEVER MATERIALISED. Forming tanh(X W) in one go for six
% million rows and two thousand units asks for a hundred gigabytes; HMS_ELM
% streams it in blocks and accumulates H''H and H''Y, which is the same
% arithmetic at bounded memory. A first version of this function wrote the
% product out and died on it.
M = hms_elm('fit', Xtr, Ytr, Nh, 0, 1);
A = M.acc;
M.lambda = lam * mean(diag(A.HtH));
M.beta = (A.HtH + M.lambda*eye(Nh)) \ A.HtY;
clear Xtr Ytr
% THE SAME COUNT AS EVERY OTHER ARM, which this line used not to be. It read
% Nh*(L+H) + Nh, the random layer, its biases and the fitted output weights,
% and stopped there. HMS_ELM also counts the two standardisation vectors, 2*d_in
% values, without which the model cannot be applied to a new input; the whole
% field arm therefore carried 2*24576 values this one did not carry 2*24.
%
% The difference is 48 values out of a hundred thousand and changes nothing.
% What it changed was that Eq. 2 of the paper reproduced neither arm, which is
% the one thing a paper about honest accounting cannot afford. Found in the
% referee pass of 10 September 2026.
npar = M.n_stored;

P = zeros(numel(ote), d*H);
for q = ip
    T = zeros(numel(ote), L);
    for k = 1:L, T(:,k) = Vte(q, ote+k-L).'; end
    Q = hms_elm('predict', M, T);
    for h = 1:H, P(:, (h-1)*d+q) = Q(:,h); end
end
secs = toc(t0);
fprintf('      one shared model: Nh %d, lambda ratio %g, %d samples, %.0f s\n', ...
        Nh, lam, N, secs);
end

function P = ar_pred(M, V, o, L, H)
d = size(V,1);  P = zeros(numel(o), d*H);
for p = 1:d
    if ~any(M(:,:,p),'all'), continue; end
    A = zeros(numel(o), L);
    for k = 1:numel(o), A(k,:) = V(p, o(k)-L+1:o(k)); end
    Q = [A ones(numel(o),1)] * M(:,:,p);
    for h = 1:H, P(:, (h-1)*d+p) = Q(:,h); end
end
end

function [M, info] = elm_select(X, Y, Xva, score, act, ncand, lams)
% ncand random layers, and lambda chosen on the SAME validation split. Nothing
% here ever sees the test data.
%
% H'H AND H'Y DO NOT DEPEND ON LAMBDA. The expensive part of a fit is the pass
% over the samples that accumulates them; the ridge itself is one 1024x1024
% solve. Accumulating once per candidate and solving for every lambda turns
% ncand x numel(lams) fits into ncand accumulations, here a fivefold saving,
% and the result is identical.
% TWO PROTOCOL FAULTS WERE FIXED HERE, AND BOTH WERE INVISIBLE IN A RESULTS
% TABLE.
%
% First, this function fixed Nh = 1024 in its own body while a grid search ran
% elsewhere to choose that very number. An audit of the whole project found that
% NOTHING read the frozen hyper-parameters, so the grid changed no result at
% all: it was computed and then ignored. The width and the penalty are now read
% from the freeze, which is what makes running the grid worth the night it costs.
%
% Second, it returned the BEST of ncand draws. That is an optimistic selection:
% the retained score is better than the model is worth, because the luckiest
% draw was kept. The arm is now reported by its MEDIAN draw, the spread is kept
% beside it, and the draws are PAIRED across arms through one fixed seed list,
% so a comparison between representations cannot be decided by which arm drew a
% better hidden layer. On these data that is not a formality: over the fifty
% draws the whole-field arm's validation error spans 0.90 W/m2, which is the
% figure stored in its `spread` field.
%
% THE SIX-TIMES COMPARISON THAT USED TO STAND HERE HAD NO MEASUREMENT BEHIND
% IT. It came from a smoke run over forty days with twelve draws, not from the
% study's data: the real grid was written with NCAND = 0, so S.siso.cand and
% S.mimo.cand are empty and no per-cell draw spread was ever measured. The
% per-cell arm is fitted once, so bench.mat records its spread as 0.0000 rather
% than as a narrow one. Closing that comparison needs a paired multi-seed run
% of both families, which has not been made.
% THE INPUT WIDTH IS NOT size(X,2), AND READING IT THAT WAY CHOSE THE WRONG
% FAMILY. The whole-field arms pass a lazy struct from WIN, {n, d, get}, which
% streams rows on demand rather than materialising a 8713 x 17328 matrix.
% size(X,2) on that struct is 1, so the family test "1 <= 64" returned siso and
% the whole-field arms were fitted at the per-pixel width and penalty. The
% width has to come from the object that knows it.
if isstruct(X), din = X.d; else, din = size(X, 2); end
[Nh, lam] = frozen_hyper(din);
if nargin >= 7 && ~isempty(lams) && ~isequal(lams(:).', lam)
    warning('hms_forecast_bench:frozenLambda', ...
        ['The ridge penalty is frozen at %g; the %d values passed by the caller ' ...
         'are ignored, so that every arm is fitted identically.'], lam, numel(lams));
end

SEEDS = 1:ncand;                       % the same draws for every arm
sc = inf(1, ncand);
for c = 1:ncand
    A = accum(X, Y, Nh, SEEDS(c), act);
    Mc = solve(A, lam);
    sc(c) = score(elm_pred(Mc, Xva));
    if mod(c, 10) == 0
        fprintf('      %s: %d/%d draws, median so far %.2f\n', ...
                act, c, ncand, median(sc(1:c)));
    end
end

% THE REPRESENTATIVE DRAW IS REFITTED RATHER THAN STORED. Holding ncand models
% would hold ncand copies of a d_in x Nh weight matrix, hundreds of megabytes
% each for the whole-field arm. One extra accumulation out of ncand+1 is the
% cheaper way to return the median model rather than the luckiest one.
[~, ord] = sort(sc);
med_seed = SEEDS(ord(ceil(numel(ord)/2)));
A = accum(X, Y, Nh, med_seed, act);
M = solve(A, lam);

info = struct('scores', sc, 'median', median(sc), 'best', min(sc), ...
              'worst', max(sc), 'spread', max(sc) - min(sc), ...
              'Nh', Nh, 'lambda', lam, 'median_seed', med_seed);
end

function [Nh, lam] = frozen_hyper(d_in)
%FROZEN_HYPER The width and penalty chosen once, read rather than chosen again.
%
%   The family is inferred from the input width: a per-pixel model reads its own
%   twenty-four lags and a whole-field model reads tens of thousands of values,
%   with nothing in this project between the two. The rule is written here
%   rather than guessed at each call site.
f = fullfile(pwd, 'results', 'hyper_frozen.mat');
assert(isfile(f), 'hms_forecast_bench:noFreeze', ...
    ['No frozen hyper-parameters at %s. Run hms_gridsearch, then hms_freeze, ' ...
     'before any forecasting campaign. This function used to carry Nh = 1024 ' ...
     'in its own body, which silently ignored the grid; failing here is the ' ...
     'point.'], f);
H = load(f);
if d_in <= 64, fam = 'siso'; else, fam = 'mimo'; end
assert(isfield(H, 'H') && isfield(H.H, fam), 'hms_forecast_bench:noFamily', ...
    'The freeze at %s carries no entry for family %s.', f, fam);
Nh  = H.H.(fam).Nh;
lam = H.H.(fam).lambda;
fprintf('      frozen: family %s, Nh = %d, lambda = %g\n', fam, Nh, lam);
end

function A = accum(X, Y, Nh, seed, act)
%ACCUM The one expensive pass: standardisation, random layer, H'H and H'Y.
[N, din] = size2(X);  [~, dout] = size2(Y);
rng(seed);
W = randn(din, Nh)/sqrt(din);  b = randn(1, Nh)*0.1;
blk = max(1, floor(2e7/max(1, din+dout)));
mu = zeros(1,din); M2 = zeros(1,din); cnt = 0;
for i0 = 1:blk:N
    i1 = min(N,i0+blk-1); Xb = rows(X,i0,i1); nb = i1-i0+1;
    mb = mean(Xb,1); M2b = sum((Xb-mb).^2,1); dl = mb-mu; tt = cnt+nb;
    M2 = M2+M2b+dl.^2*(cnt*nb/tt); mu = mu+dl*(nb/tt); cnt = tt;
end
sd = sqrt(M2/max(1,cnt)); sd(sd<eps) = 1;
HtH = zeros(Nh); HtY = zeros(Nh,dout);
for i0 = 1:blk:N
    i1 = min(N,i0+blk-1);
    Hb = actf((rows(X,i0,i1)-mu)./sd*W + b, act);
    HtH = HtH + Hb.'*Hb;  HtY = HtY + Hb.'*rows(Y,i0,i1);
end
A = struct('HtH',HtH,'HtY',HtY,'W',W,'b',b,'mu',mu,'sd',sd, ...
           'Nh',Nh,'act',act,'d_in',din,'d_out',dout);
end

function M = solve(A, lam)
%SOLVE The cheap part: one ridge solve. lambda = 0 is allowed and is then the
% ordinary least squares solution, which is what makes it a meaningful point
% of the sweep rather than a special case.
beta = (A.HtH + lam*eye(A.Nh)) \ A.HtY;
M = struct('W',A.W,'b',A.b,'beta',beta,'mu',A.mu,'sd',A.sd,'Nh',A.Nh, ...
           'lambda',lam,'act',A.act,'d_in',A.d_in,'d_out',A.d_out, ...
           'n_stored',A.Nh*(A.d_in+A.d_out)+A.Nh+2*A.d_in);
end

function H = actf(Z, act)
if strcmp(act,'tanh'), H = tanh(Z); else, H = Z; end
end

function P = elm_pred(M, X)
N = size2(X);  blk = max(1, floor(2e7/max(1,M.d_in+M.d_out)));
P = zeros(N, M.d_out);
for i0 = 1:blk:N
    i1 = min(N,i0+blk-1);
    P(i0:i1,:) = actf((rows(X,i0,i1)-M.mu)./M.sd*M.W + M.b, M.act) * M.beta;
end
end

% ------------------------------------------------------------- plumbing
function [n,d] = size2(S)
if isstruct(S), n=S.n; d=S.d; else, n=size(S,1); d=size(S,2); end
end
function B = rows(S,i,j)
if isstruct(S), B=S.get(i,j); else, B=S(i:j,:); end
end
function X = win(Z,o,L)
X = struct('n',numel(o),'d',size(Z,1)*L,'get',@(i,j) getw(Z,o(i:j),L));
end
function B = getw(Z,oo,L)
p=size(Z,1); B=zeros(numel(oo),p*L);
for k=1:numel(oo), B(k,:)=reshape(Z(:,oo(k)-L+1:oo(k)),1,[]); end
end
function Y = tgt(Z,o,H)
p=size(Z,1); Y=zeros(numel(o),p*H);
for k=1:numel(o), Y(k,:)=reshape(Z(:,o(k)+1:o(k)+H),1,[]); end
end
function K = flags(f,o,H,d)
%FLAGS Which (origin, horizon, cell) triples enter a score.
%
%   f is d x N: the evaluation set per CELL and per step. An earlier version
%   took a 1 x N vector computed at the domain centre and replicated it across
%   every cell with repmat, which is where the centre-only mask entered the
%   forecasting side of this study.
K=false(numel(o),d*H);
for h=1:H, K(:,(h-1)*d+1:h*d)=f(:,o(:)+h).'; end
end

function t = epoch_of(days, cfg)
%EPOCH_OF UTC seconds at the middle of every hour of the given days.
%   The reduced calendar holds 365 days a year with the leap days removed, so a
%   day index is counted against that calendar and not against a real one.
d0 = datetime(cfg.absolute_start_date,'InputFormat','yyyy-MM-dd','TimeZone','UTC');
y0 = year(d0);  all_days = NaT(0,1,'TimeZone','UTC');
while numel(all_days) < cfg.n_days
    dd = (datetime(y0,1,1,'TimeZone','UTC'):datetime(y0,12,31,'TimeZone','UTC')).';
    dd = dd(~(month(dd)==2 & day(dd)==29));
    all_days = [all_days; dd]; %#ok<AGROW>
    y0 = y0 + 1;
end
sel = all_days(days(:));  H = cfg.hours_per_day;
t = posixtime(repelem(sel,1,H)) + repmat((0:H-1)*3600 + 1800, numel(sel), 1);
t = reshape(t.',1,[]);
end
function P = clip(P), P = max(P,0); end
function e = errm(P,Y,K)
% POOLED, the one convention of this study; see HMS_SCORE.
D = P - Y;
e = hms_score(sum((D.^2).*K, 2), sum(K, 2));
end
function e = errh(P,Y,K,d,H)
e=zeros(1,H);
for h=1:H
    c=(h-1)*d+1:h*d; D=P(:,c)-Y(:,c); k=K(:,c);
    if any(k(:)), e(h) = hms_score(sum((D.^2).*k,2), sum(k,2));
    else,         e(h) = NaN; end
end
end
