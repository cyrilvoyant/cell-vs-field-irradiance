function R = hms_unet_arm(outfile, bases)
%HMS_UNET_ARM The spatial reference, through a network that shares its weights.
%
%   R = HMS_UNET_ARM(outfile, bases)
%
%   WHY THIS ARM EXISTS AND WHY IT IS THE ONE THAT DECIDES. Two independent
%   readers of the manuscript raised the same objection, and it is the strongest
%   one against the paper: a study concluding that no spatial representation
%   beats a per-pixel model, while testing only models whose first layer is
%   DENSE, has measured its own architecture rather than the value of spatial
%   information. A dense layer costs a parameter per input, so a whole-field
%   model is forced to be enormous and loses on frugality before it starts. A
%   convolution shares its weights and its parameter count does not grow with
%   the grid at all. Until that architecture is run, the conclusion has to be
%   narrowed to dense models -- in the title, not in a footnote.
%
%   THE COMPARISON IS AT EQUAL STORAGE, NOT EQUAL CAPACITY. The per-pixel
%   machine this arm has to beat stores about a hundred thousand numbers, so the
%   base width is chosen to land near that and the sweep over bases is there to
%   show what the network does on either side of it. Giving the network ten
%   times the budget would answer a question nobody asked.
%
%   MATLAB REMAINS THE ONLY PLACE A SCORE IS COMPUTED. Python trains the network
%   and returns predictions; the mask, the night forcing, the clipping and every
%   metric are applied here, by the same code that scores every other arm. An
%   arm that computed its own error would not be comparable however carefully
%   the other file was written.
%
%   THE VOLUMES ARE PASSED IN SINGLE PRECISION. The training input is origins by
%   lags by rows by columns, which in double is 1.7 GB before the targets are
%   counted; the network trains in single anyway, so passing double would spend
%   two gigabytes of disk and memory to carry digits the optimiser discards.

if nargin < 1 || isempty(outfile)
    outfile = fullfile(pwd, 'results', 'unet_bench.mat');
end
if nargin < 2 || isempty(bases), bases = [8 16 24]; end
if ~isfolder(fileparts(outfile)), mkdir(fileparts(outfile)); end

PY = 'python';
SCRIPT = fullfile(pwd, 'python', 'unet_bench.py');
assert(isfile(SCRIPT), 'hms_unet_arm:noScript', 'Missing %s', SCRIPT);

n = 32;  cfg = hms_config('n', n);
L = 24;  H = 24;  Td = 24;  ELEV = 5;
dtr = 1:365;  dte = 366:730;
d = n*n;

t0 = tic;
[Ftr, Mtr, info] = hms_build_field(dtr, cfg);
[Fte, ~, ~]      = hms_build_field(dte, cfg);
Vtr = reshape(Ftr, d, []);  Vte = reshape(Fte, d, []);
clear Ftr Fte
msk = mean(Mtr,3) > 0.5;

[Lon, Lat] = meshgrid(info.lon_grid, info.lat_grid);
cal = struct('resolved', true, 'days_in_year', 365, ...
             'source', sprintf('HelioClim-3, origin %s', cfg.absolute_start_date));
[K5, NT] = hms_eval_mask(Lat, Lon, hms_epoch(dte, cfg), cal, ...
                        struct('eval_deg', ELEV));
day_te = reshape(K5, d, []) & msk(:);
sun_te = ~reshape(NT, d, []);

otr = hms_codex_origins([1 size(Vtr,2)], max(L,Td), H);
ote = hms_codex_origins([1 size(Vte,2)], max(L,Td), H);
hms_resid('origins', ote);   % for the paired test; no-op when capture is off
Yte = tgt(Vte, ote, H, d);
Kte = flg(day_te, ote, H, d);
NIGHT = ~flg(sun_te, ote, H, d);
SC = hms_scale_ref();  mu = SC.mu;   % nRMSE divides by the MEAN of the
                                     % scored observations, not by their
                                     % spread; see HMS_SCALE_REF.

% THE SIX METRICS, AND THE REFERENCE THEY NEED. This arm has to appear on the
% same horizon figure as the twenty-seven of the Corsica campaign, so it is
% scored by the same function, HMS_METRICS_H. NICE is a ratio against simple
% persistence, which is therefore computed here on this arm's own test set:
% borrowing the denominator from another campaign would divide by norms taken
% on different origins.
HSU = hms_horizons();
Pp  = repmat(Vte(:, ote).', 1, H);
Pp(NIGHT) = 0;  Pp = max(Pp, 0);
[~, LREF] = hms_metrics_h(@(ih) slice_h(Pp, Yte, Kte, HSU(ih), d, n), mu, ...
    struct('horizons', HSU, 'res', 3.4, 'dt', 60, 'isref', true));
clear Pp

fprintf('\n=== U-Net, the spatial reference ===\n');
fprintf('%d support cells, %d train origins, %d test origins, fields in %.0f s\n', ...
        nnz(msk), numel(otr), numel(ote), toc(t0));

R = struct('base',{},'rmse',{},'rmse_h',{},'n_par',{},'s_per_epoch',{}, ...
           'train_s',{},'total_s',{},'met',{},'secs_met',{});
if isfile(outfile)
    q = load(outfile);
    if isfield(q,'R') && ~isempty(q.R), R = q.R; end
end

for b = bases
    if any([R.base] == b)
        fprintf('  base %-3d already done\n', b);  continue
    end
    ta = tic;
    try
        % ---- the volumes the network reads, in single
        Xtr = cube(Vtr, otr, -L+1:0, n);
        Ytr = cube(Vtr, otr,  1:H,   n);
        Xte = cube(Vte, ote, -L+1:0, n);
        mask = single(msk); %#ok<NASGU>

        ex = fullfile(tempdir, sprintf('unet_in_%d.mat', b));
        ox = strrep(ex, 'unet_in_', 'unet_out_');
        % A HANDOVER FILE LEFT BY AN ABORTED RUN IS A TRUNCATED HDF5, and
        % SAVE refuses to overwrite one, reporting the file as corrupt. The
        % run that produced it was killed for reasons that have nothing to do
        % with this one, so the stale file is removed rather than diagnosed.
        if isfile(ex), delete(ex); end
        if isfile(ox), delete(ox); end
        save(ex, 'Xtr', 'Ytr', 'Xte', 'mask', '-v7.3');
        clear Xtr Ytr Xte

        cmd = sprintf('"%s" "%s" "%s" "%s" --base %d', PY, SCRIPT, ex, ox, b);
        fprintf('  base %-3d training ...\n', b);
        [st, so] = system(cmd);
        if st ~= 0
            fprintf('  base %-3d PYTHON FAILED\n%s\n', b, so);
            if isfile(ex), delete(ex); end
            continue
        end
        disp(so);

        r = load(ox);
        Pv = double(r.pred);                       % (n_test, H, n, n)
        inf1 = jsondecode(r.info);
        P = zeros(numel(ote), d*H);
        for h = 1:H
            sl = reshape(Pv(:,h,:,:), numel(ote), d);
            P(:, (h-1)*d + (1:d)) = sl;
        end
        P(NIGHT) = 0;  P = max(P, 0);

        [MET, ~, tmet] = hms_metrics_h( ...
            @(ih) slice_h(P, Yte, Kte, HSU(ih), d, n), mu, ...
            struct('horizons', HSU, 'res', 3.4, 'dt', 60, 'niceref', LREF));
        % per-origin errors, only when HMS_RESID is on; see HMS_RESID
        hms_resid('add', sprintf('UNet-base%d', b), P, Yte, Kte, d, H);

        R = hms_push_row(R, struct('base', b, 'rmse', errm(P, Yte, Kte), ...
            'rmse_h', errh(P, Yte, Kte, d, H), ...
            'n_par', inf1.params.total, ...
            's_per_epoch', inf1.seconds_per_epoch, ...
            'train_s', inf1.train_seconds, 'total_s', toc(ta), ...
            'met', MET, 'secs_met', tmet)); %#ok<AGROW>
        fprintf(['  base %-3d RMSE %7.2f W/m2  nRMSE %.4f  %d parameters  ' ...
                 '%.1f s/epoch  %.1f min\n'], b, R(end).rmse, R(end).rmse/mu, ...
                R(end).n_par, R(end).s_per_epoch, R(end).total_s/60);
        delete(ex);  delete(ox);
        save(outfile, 'R', 'L', 'H', 'ELEV', 'dtr', 'dte', 'bases', '-v7.3');
    catch ME
        fprintf('  base %-3d FAILED: %s\n', b, ME.message);
    end
end

fprintf('\n%-8s %-10s %-14s %-12s\n', 'base', 'nRMSE', 'parameters', 'minutes');
for k = 1:numel(R)
    fprintf('%-8d %-10.4f %-14d %-12.1f\n', R(k).base, R(k).rmse/mu, ...
            R(k).n_par, R(k).total_s/60);
end
fprintf('\ntotal %.1f min\n', toc(t0)/60);
save(outfile, 'R', 'L', 'H', 'ELEV', 'dtr', 'dte', 'bases', '-v7.3');
fprintf('written %s\n', outfile);
end

% ---------------------------------------------------------------------------
function C = cube(V, o, off, n)
%CUBE Origins by channels by rows by columns, in single, for the network.
C = zeros(numel(o), numel(off), n, n, 'single');
for k = 1:numel(off)
    C(:,k,:,:) = reshape(single(V(:, o + off(k)).'), numel(o), 1, n, n);
end
end

function [Pv, Yv, Kv] = slice_h(P, Y, K, h, d, nside)
c = (h-1)*d + (1:d);
Pv = reshape(P(:,c).', nside, nside, []);
Yv = reshape(Y(:,c).', nside, nside, []);
Kv = reshape(K(:,c).', nside, nside, []);
end

function Y = tgt(V, o, H, d)
Y = zeros(numel(o), d*H);
for h = 1:H, Y(:, (h-1)*d + (1:d)) = V(:, o+h).'; end
end

function K = flg(F, o, H, d)
K = false(numel(o), d*H);
for h = 1:H, K(:, (h-1)*d + (1:d)) = F(:, o(:)+h).'; end
end

function e = errm(P, Y, K)
% POOLED, the one convention of this study; see HMS_SCORE.
D = P - Y;
e = hms_score(sum((D.^2).*K, 2), sum(K, 2));
end

function e = errh(P, Y, K, d, H)
e = zeros(1,H);
for h = 1:H
    c = (h-1)*d + (1:d);
    D = P(:,c) - Y(:,c);  k = K(:,c);
    if any(k(:)), e(h) = hms_score(sum((D.^2).*k,2), sum(k,2));
    else,         e(h) = NaN; end
end
end
