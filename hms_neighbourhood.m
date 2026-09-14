function R = hms_neighbourhood(ks, outfile, lam_force)
%HMS_NEIGHBOURHOOD Where does spatial information die? A sweep over patch radius.
%
%   R = HMS_NEIGHBOURHOOD(ks, outfile)
%
%   WHY THIS EXPERIMENT EXISTS. The rest of this study compares a model that
%   reads ONE cell against models that read ALL of them, and concludes that the
%   field does not pay for itself. That is a binary comparison, and it is not
%   the comparison the literature makes. Where spatial information has been
%   found to help, the support is small: sixteen neighbouring locations in
%   Mukhoty et al., and in Agoua et al. a Lasso that keeps four satellite pixels
%   at fifteen minutes and seven at three hours out of two thousand offered.
%   Nobody uses the whole field, so a study that only tests the whole field has
%   not tested the thing that works.
%
%   WHAT IT MEASURES. The same per-pixel forecaster, reading a k by k patch
%   centred on the cell it predicts, for k = 1, 3, 5, 7. k = 1 is the blind
%   model of the main comparison; larger k adds neighbours and adds storage.
%   The answer is a curve rather than a verdict, and there are three shapes it
%   can take, all of them informative:
%
%     - monotone decreasing : spatial information pays, and the main conclusion
%       was an artefact of reading the field densely rather than locally;
%     - a minimum at small k : there is a useful radius, and the study should
%       report it rather than a binary;
%     - monotone increasing : neighbours never pay at this resolution, which
%       makes the conclusion much harder to attack than it is now.
%
%   THE SCORED SET IS IDENTICAL FOR EVERY k, or the curve would compare
%   different problems. Every cell of the evaluation mask is predicted at every
%   k; patch cells that fall outside the grid are filled by replicating the
%   nearest in-grid cell, which is the standard edge convention and adds no
%   information. Patch cells inside the grid are used as they are, mask or no
%   mask: the mask says where a PREDICTION is scored, not where the product has
%   values to read.
%
%   THE ROWS ARE NEVER MATERIALISED. At k = 7 the design matrix would be
%   8713 origins times 722 cells by 1176 features, which is 59 GB in double.
%   HMS_ELM accepts a getter and asks for one block of rows at a time; the
%   arithmetic is identical and the ridge solve that follows is exact.

% LAM_FORCE EXISTS TO REMOVE ONE CONFOUND. The frozen tier picks its penalty
% from the input width, so k = 1 is fitted at 1e-8 and k >= 3 at 1e-6: the
% step from k = 1 to k = 3 changes the patch AND the penalty at once. Passing
% a penalty here holds it fixed across the sweep, so that what varies is the
% neighbourhood alone. The two runs are reported side by side rather than one
% of them being chosen after the fact.
if nargin < 1 || isempty(ks), ks = [1 3 5 7]; end
if nargin < 3, lam_force = []; end
if nargin < 2 || isempty(outfile)
    outfile = fullfile(pwd, 'results', 'neighbourhood.mat');
end
if ~isfolder(fileparts(outfile)), mkdir(fileparts(outfile)); end

n = 32;  cfg = hms_config('n', n);
L = 24;  H = 24;  Td = 24;  ELEV = 5;
dtr = 1:365;  dte = 366:730;
d = n*n;
% THE MACHINE IS CHOSEN BY THE SAME RULE AS EVERY OTHER ARM, which is the
% frozen decisive tier selecting its family from the input width. That means
% the machine GROWS with k: k = 1 has 24 inputs and takes the narrow family,
% k >= 3 has 216 or more and takes the wide one. This is deliberate and it
% favours the spatial arms rather than this study's conclusion -- a sweep that
% held the narrow machine fixed could be dismissed as starving the larger
% patches. The width is therefore resolved inside the loop, and k = 1 must
% reproduce the per-pixel arm of the main bench exactly, which is the check
% that this file is wired correctly.
HS = hms_horizons();

t0 = tic;
[Ftr, Mtr, info] = hms_build_field(dtr, cfg);
[Fte, ~, ~]      = hms_build_field(dte, cfg);
Vtr = reshape(Ftr, d, []);  Vte = reshape(Fte, d, []);
clear Ftr Fte
msk = mean(Mtr,3) > 0.5;  msk = msk(:);

[Lon, Lat] = meshgrid(info.lon_grid, info.lat_grid);
cal = struct('resolved', true, 'days_in_year', 365, ...
             'source', sprintf('HelioClim-3, origin %s', cfg.absolute_start_date));
day_te = reshape(hms_eval_mask(Lat, Lon, hms_epoch(dte, cfg), cal, ...
                              struct('eval_deg', ELEV)), d, []);
Kte_c = day_te & msk;

otr = hms_codex_origins([1 size(Vtr,2)], max(L,Td), H);
ote = hms_codex_origins([1 size(Vte,2)], max(L,Td), H);
ip  = find(msk).';
SC = hms_scale_ref();  mu = SC.mu;   % nRMSE divides by the MEAN of the
                                     % scored observations, not by their
                                     % spread; see HMS_SCALE_REF.

fprintf('\n=== where does spatial information die? patch sweep ===\n');
fprintf('%d cells scored, %d train origins, %d test origins\n', ...
        numel(ip), numel(otr), numel(ote));
fprintf('fields in %.0f s\n\n', toc(t0));

R = struct('k',{},'n_in',{},'rmse',{},'rmse_h',{},'n_par',{},'Nh',{}, ...
           'family',{},'secs',{},'met',{},'secs_met',{});
if isfile(outfile)
    q = load(outfile);
    if isfield(q,'R') && ~isempty(q.R), R = q.R; end
end

% THE NICE DENOMINATOR IS SIMPLE PERSISTENCE, as in every other table of this
% study, computed here on this arm's own test origins rather than borrowed.
% Using k = 1 as the denominator would have made this sweep incomparable with
% everything else while looking perfectly reasonable.
Pp = zeros(d, numel(ote), numel(HS), 'single');
for q = 1:d
    Pp(q,:,:) = repmat(single(max(Vte(q, ote), 0)).', 1, 1, numel(HS));
end
ctx0 = struct('Vte', Vte, 'Kte', Kte_c, 'ote', ote, 'n', n);
[~, LREF] = hms_metrics_h(@(ih) slice_h(Pp, ctx0, HS(ih), ih), mu, ...
    struct('horizons', HS, 'res', 3.4, 'dt', 60, 'isref', true));
clear Pp

for k = ks
    if ~isempty(R) && any([R.k] == k)
        fprintf('  k = %d already done\n', k);  continue
    end
    tk = tic;
    NB = nbr_index(k, n);          % (d x k^2) index of each cell's patch
    din = k*k*L;
    EC = hms_elm_config('decisive', din);
    Nh = EC.Nh;  LAM = EC.lambda;
    if ~isempty(lam_force), LAM = lam_force; end

    Xtr = struct('n', numel(otr)*numel(ip), 'd', din, ...
                 'get', @(i0,i1) rows_in(Vtr, otr, ip, NB, L, i0, i1));
    Ytr = struct('n', numel(otr)*numel(ip), 'd', H, ...
                 'get', @(i0,i1) rows_out(Vtr, otr, ip, H, i0, i1));

    M = hms_elm('fit', Xtr, Ytr, Nh, 0, 1);
    M.lambda = LAM * mean(diag(M.acc.HtH));
    M.beta = (M.acc.HtH + M.lambda*eye(Nh)) \ M.acc.HtY;

    % ---- predict cell by cell, and keep the reported horizons for the metrics
    P = zeros(d, numel(ote), numel(HS), 'single');
    se = 0;  nk = 0;  seh = zeros(1,H);  nkh = zeros(1,H);
    for q = ip
        A = patch_lags(Vte, ote, NB(q,:), L);
        Q = max(hms_elm('predict', M, A), 0);
        Y = zeros(numel(ote), H);  Kq = false(numel(ote), H);
        for h = 1:H
            Y(:,h)  = Vte(q, ote+h).';
            Kq(:,h) = Kte_c(q, ote+h).';
        end
        D = (Q - Y).^2 .* Kq;
        se = se + sum(D(:));   nk = nk + sum(Kq(:));
        seh = seh + sum(D,1);  nkh = nkh + sum(Kq,1);
        P(q,:,:) = reshape(single(Q(:,HS)), 1, numel(ote), numel(HS));
    end

    e  = sqrt(se / max(nk,1));
    eh = sqrt(seh ./ max(nkh,1));

    ctx = struct('Vte', Vte, 'Kte', Kte_c, 'ote', ote, 'n', n);
    [MET, ~, tmet] = hms_metrics_h(@(ih) slice_h(P, ctx, HS(ih), ih), mu, ...
        struct('horizons', HS, 'res', 3.4, 'dt', 60, 'niceref', LREF));

    % THE SAME COUNT AS HMS_ELM, which this line used not to be. It stopped at
    % the random layer, the fitted weights and the biases, and omitted the two
    % standardisation vectors, 2*din values, that a receiver must also hold. The
    % omission was invisible because every row of this sweep carried it, so the
    % column was internally consistent and disagreed only with the bench.
    % Found in the referee pass of 10 September 2026; see HMS_FIX_NPAR.
    npar = Nh*(din + H) + Nh + 2*din;
    R = hms_push_row(R, struct('k', k, 'n_in', din, 'rmse', e, 'rmse_h', eh, ...
        'n_par', npar, 'Nh', Nh, 'family', EC.family, ...
        'secs', toc(tk), 'met', MET, 'secs_met', tmet));
    fprintf(['  k = %d  %4d inputs  lambda %g (%s)  nRMSE %.4f  %d params  ' ...
             '| h=3 %.4f  GPR %.1f%%  [%.1f min]\n'], ...
            k, din, LAM, EC.family, e/mu, npar, MET.nrmse(HS==3), ...
            MET.gpr(HS==3), toc(tk)/60);
    save(outfile, 'R', 'ks', 'L', 'H', 'ELEV', 'HS', '-v7.3');
    clear P M Xtr Ytr
end

fprintf('\n%-6s %-8s %-8s %-10s %-14s\n', 'k', 'inputs', 'family', 'nRMSE', 'parameters');
for j = 1:numel(R)
    fprintf('%-6d %-8d %-8s %-10.4f %-14d\n', R(j).k, R(j).n_in, ...
            R(j).family, R(j).rmse/mu, R(j).n_par);
end
fprintf('\ntotal %.1f min\n', toc(t0)/60);
save(outfile, 'R', 'ks', 'L', 'H', 'ELEV', 'HS', '-v7.3');
fprintf('written %s\n', outfile);
end

% ---------------------------------------------------------------------------
function NB = nbr_index(k, n)
%NBR_INDEX For each cell, the linear indices of its k by k patch.
%   Cells outside the grid are replaced by the nearest in-grid cell, which is
%   the standard edge convention and introduces no information the interior
%   does not already have.
r = (k-1)/2;
[cc, rr] = meshgrid(1:n, 1:n);          % column-major, matching reshape(V,n,n)
NB = zeros(n*n, k*k);
c = 0;
for dy = -r:r
    for dx = -r:r
        c = c + 1;
        yy = min(max(rr + dy, 1), n);
        xx = min(max(cc + dx, 1), n);
        NB(:, c) = sub2ind([n n], yy(:), xx(:));
    end
end
end

function A = patch_lags(V, o, nb, L)
%PATCH_LAGS One cell's patch over L lags: (origins) by (k^2 * L).
%   Built one neighbour at a time with a single indexing operation rather than
%   one element at a time: at k = 7 the scalar version costs a million
%   assignments per cell and the fit never finishes.
idx = o(:) + (-L+1:0);              % (origins x L) absolute time indices
A = zeros(numel(o), numel(nb)*L);
for j = 1:numel(nb)
    row = V(nb(j), :);
    A(:, (j-1)*L + (1:L)) = row(idx);
end
end

function X = rows_in(V, o, ip, NB, L, i0, i1)
%ROWS_IN The block of design rows the fitter asked for, built on demand.
%   Rows are ordered CELL-MAJOR -- cell one's origins, then cell two's -- so a
%   requested block covers a contiguous run of cells, and within each cell a
%   contiguous run of origins. Both runs are built with array indexing; a
%   row-at-a-time version is three orders of magnitude slower and was the first
%   version of this function.
no = numel(o);
X  = zeros(i1-i0+1, size(NB,2)*L);
r  = i0;
while r <= i1
    ci = floor((r-1)/no) + 1;          % which cell this row belongs to
    j0 = mod(r-1, no) + 1;             % first origin of this cell in the block
    j1 = min(no, j0 + (i1 - r));       % last one
    A  = patch_lags(V, o(j0:j1), NB(ip(ci),:), L);
    X(r-i0 + (1:(j1-j0+1)), :) = A;
    r = r + (j1 - j0 + 1);
end
end

function Y = rows_out(V, o, ip, H, i0, i1)
no = numel(o);
Y  = zeros(i1-i0+1, H);
r  = i0;
while r <= i1
    ci = floor((r-1)/no) + 1;
    j0 = mod(r-1, no) + 1;
    j1 = min(no, j0 + (i1 - r));
    oo = o(j0:j1);
    row = V(ip(ci), :);
    Y(r-i0 + (1:(j1-j0+1)), :) = row(oo(:) + (1:H));
    r = r + (j1 - j0 + 1);
end
end

function [Pv, Yv, Kv] = slice_h(P, ctx, h, ih)
%SLICE_H The three volumes at one horizon, in the grid layout gamma needs.
n = ctx.n;
o = ctx.ote(:).' + h;
Pv = reshape(double(P(:,:,ih)), n, n, []);
Yv = reshape(ctx.Vte(:, o),     n, n, []);
Kv = reshape(ctx.Kte(:, o),     n, n, []);
end
