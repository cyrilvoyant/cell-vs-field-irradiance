function R = hms_lstm_arm(reps, ps, outfile)
%HMS_LSTM_ARM The trained sequence model, on the raw field and on each latent.
%
%   R = HMS_LSTM_ARM(reps, ps, outfile)
%
%   WHY A TRAINED MODEL CHANGES THE QUESTION. An extreme learning machine draws
%   its first layer at random and never trains it, so fitting costs one pass to
%   accumulate H'H plus one solve, and neither depends on the input dimension.
%   In that architecture there is no computational reason to compress anything:
%   the compression argument is not wrong there, it is empty there. A model
%   trained by backpropagation pays for its input dimension four times over ---
%   in trained weights, in stored activations, in time per epoch and in the data
%   a larger parameter count needs. This arm is where compression can pay, and
%   the point of running it is the cost as much as the error.
%
%   ONE STANDARD CONFIGURATION, NOT TUNED. A single LSTM layer of 256 units, a
%   linear head, fifteen epochs of Adam at 1e-3, batch 64, seed fixed. Nothing
%   is searched, per representation or otherwise: tuning one arm and not the
%   others is the fault this study exists to avoid, and an untuned model that is
%   identical everywhere compares representations, which is the question.
%
%   MATLAB REMAINS THE ONLY SCORING AUTHORITY. Python receives arrays and
%   returns predictions and timings. It never computes an error, a mask or a
%   split; every number that reaches the paper is evaluated here, on the same
%   mask and the same targets as every other arm.
%
%   THE ANSWER IS CONDITIONAL AND SAYS SO. A convolutional recurrent network
%   shares weights across space, so its parameter count does not follow the
%   input size and this whole argument weakens. What is defended is narrow:
%   compression pays for models whose first layer is dense and learned.

if nargin < 1 || isempty(reps), reps = {'field','radon','dct','pca'}; end
if nargin < 2 || isempty(ps),   ps   = [49 196 392]; end
if nargin < 3 || isempty(outfile)
    outfile = fullfile(pwd, 'results', 'lstm_bench.mat');
end
if ~isfolder(fileparts(outfile)), mkdir(fileparts(outfile)); end

PY = 'python';
SCRIPT = fullfile(pwd, 'python', 'lstm_bench.py');
assert(isfile(SCRIPT), 'hms_lstm_arm:noScript', 'No %s.', SCRIPT);

n = 32;  cfg = hms_config('n', n);
L = 24;  H = 24;  Td = 24;  ELEV = 5;
dtr = 1:365;  dte = 366:730;

R = struct('rep',{},'p',{},'rmse',{},'rmse_h',{},'n_par',{},'n_in_block',{}, ...
           'met',{},'secs_met',{}, ...
           's_per_epoch',{},'train_s',{},'total_s',{});
if isfile(outfile)
    o = load(outfile);
    if isfield(o,'R') && ~isempty(o.R), R = o.R; end
end

fprintf('\n=== LSTM arm, one standard configuration ===\n');
t0 = tic;
[Ftr, Mtr, info] = hms_build_field(dtr, cfg);
[Fte, ~, ~]      = hms_build_field(dte, cfg);
Vtr = reshape(Ftr, n*n, []);  Vte = reshape(Fte, n*n, []);
clear Ftr Fte
msk = mean(Mtr,3) > 0.5;  msk = msk(:);  d = numel(msk);
fprintf('fields built in %.0f s, island %d of %d cells\n', toc(t0), nnz(msk), d);

[Lon, Lat] = meshgrid(info.lon_grid, info.lat_grid);
cal = struct('resolved', true, 'days_in_year', 365, ...
             'source', sprintf('HelioClim-3, origin %s', cfg.absolute_start_date));
em = @(dd) reshape(hms_eval_mask(Lat, Lon, epoch_of(dd, cfg), cal, ...
                                struct('eval_deg', ELEV)), n*n, []);
day_te = em(dte);

otr = hms_codex_origins([1 size(Vtr,2)], max(L,Td), H);
ote = hms_codex_origins([1 size(Vte,2)], max(L,Td), H);
hms_resid('origins', ote);   % for the paired test; no-op when capture is off
Yte = tgt(Vte, ote, H);
Kte = flg(day_te, ote, H, d) & repmat(msk, H, 1).';
SC = hms_scale_ref();  mu = SC.mu;   % nRMSE divides by the MEAN of the
                                     % scored observations, not by their
                                     % spread; see HMS_SCALE_REF.

% THE SIX METRICS, SCORED BY THE SAME FUNCTION AS EVERY OTHER ARM. NICE needs
% simple persistence as its denominator, computed here on this arm's own test
% origins rather than borrowed from another campaign. Night forcing is not
% applied: Kte already excludes every cell below the elevation threshold, so a
% night cell is never scored and forcing it would change no number.
HSU = hms_horizons();
Pp  = max(repmat(Vte(:, ote).', 1, H), 0);
[~, LREF] = hms_metrics_h(@(ih) slice_h(Pp, Yte, Kte, HSU(ih), d, n), mu, ...
    struct('horizons', HSU, 'res', 3.4, 'dt', 60, 'isref', true));
clear Pp
fprintf('origins  train %d | test %d\n', numel(otr), numel(ote));

for ir = 1:numel(reps)
    rep = reps{ir};
    plist = ps;
    if strcmp(rep, 'field'), plist = NaN; end     % the raw field has no p
    for ip = 1:numel(plist)
        p = plist(ip);
        tag = rep;
        if ~isnan(p), tag = sprintf('%s p=%d', rep, p); end
        if any(strcmp({R.rep}, rep) & isequaln([R.p], p)), continue; end
        ta = tic;
        try
            if strcmp(rep, 'field')
                Ztr = Vtr(msk,:);  Zte = Vte(msk,:);  dec = [];
            else
                E = hms_latent(rep, n, p, cfg, Vtr);
                Ztr = E.encode(Vtr);  Zte = E.encode(Vte);  dec = E.decode;
            end

            % the sequence tensors the script expects, observations in rows
            Xtr = seqs(Ztr, otr, L);
            Xte = seqs(Zte, ote, L);
            % THE TARGET IS ALWAYS THE SAME SPACE AS THE INPUT. For the raw
            % field the model predicts cells; for a latent it predicts latent
            % coefficients and the encoder's own inverse brings them back. One
            % line either way, so the branch that used to sit here was two
            % identical blocks pretending to be a choice.
            % the script reads Xtr, Ytr, Xte and steps by those names
            Ytr = tgt(Ztr, otr, H);  steps = L;                 %#ok<NASGU>
            ex = fullfile(tempdir, sprintf('lstm_in_%s.mat', matlab.lang.makeValidName(tag)));
            ox = strrep(ex, 'lstm_in_', 'lstm_out_');
            save(ex, 'Xtr', 'Ytr', 'Xte', 'steps', '-v7');
            cmd = sprintf('"%s" "%s" "%s" "%s"', PY, SCRIPT, ex, ox);
            fprintf('  %-16s running LSTM ...\n', tag);
            [st, so] = system(cmd);
            if st ~= 0
                fprintf('  %-16s PYTHON FAILED\n%s\n', tag, so);
                continue
            end
            disp(so);
            r = load(ox);
            Pz = double(r.pred);
            inf1 = jsondecode(r.info);

            if isempty(dec)
                P = zeros(numel(ote), d*H);
                for h = 1:H
                    P(:, (h-1)*d + find(msk).') = Pz(:, (h-1)*nnz(msk)+(1:nnz(msk)));
                end
            else
                P = decodeH(Pz, dec, size(Ztr,1), H, d);
            end

            Pc = max(P, 0);
            [MET, ~, tmet] = hms_metrics_h( ...
                @(ih) slice_h(Pc, Yte, Kte, HSU(ih), d, n), mu, ...
                struct('horizons', HSU, 'res', 3.4, 'dt', 60, 'niceref', LREF));
            % THE FIELD ORDER MATCHES THE DECLARATION ABOVE, and it has to:
            % assigning a struct whose fields are the same but ordered
            % differently into a struct array raises 'subscripted assignment
            % between dissimilar structures', four hours into a training run.
            % per-origin errors, only when HMS_RESID is on; see HMS_RESID
            hms_resid('add', sprintf('LSTM-%s-p%s', rep, num2str(p)), ...
                     Pc, Yte, Kte, d, H);

            R = hms_push_row(R, struct('rep', rep, 'p', p, ...
                'rmse', errm(Pc, Yte, Kte), ...
                'rmse_h', errh(Pc, Yte, Kte, d, H), ...
                'n_par', inf1.params.total, 'n_in_block', inf1.params.input_block, ...
                'met', MET, 'secs_met', tmet, ...
                's_per_epoch', inf1.seconds_per_epoch, ...
                'train_s', inf1.train_seconds, 'total_s', toc(ta))); %#ok<AGROW>
            fprintf('  %-16s RMSE %7.2f W/m2 | %d weights (%d in the input block) | %.1f s/epoch | %.1f min\n', ...
                    tag, R(end).rmse, R(end).n_par, R(end).n_in_block, ...
                    R(end).s_per_epoch, R(end).total_s/60);
            delete(ex); delete(ox);
            save(outfile, 'R', 'L', 'H', 'ELEV', 'dtr', 'dte', '-v7.3');
        catch ME
            fprintf('  %-16s FAILED: %s\n', tag, ME.message);
        end
    end
end
fprintf('\ntotal %.1f min\n', toc(t0)/60);
save(outfile, 'R', 'L', 'H', 'ELEV', 'dtr', 'dte', '-v7.3');
fprintf('written %s\n', outfile);
end

% ---------------------------------------------------------------------------
function X = seqs(Z, o, L)
%SEQS Observations in rows, each a window of L steps flattened in time order.
p = size(Z,1);  X = zeros(numel(o), L*p);
for k = 1:L, X(:, (k-1)*p+(1:p)) = Z(:, o+k-L).'; end
end

function [Pv, Yv, Kv] = slice_h(P, Y, K, h, d, nside)
c = (h-1)*d + (1:d);
Pv = reshape(P(:,c).', nside, nside, []);
Yv = reshape(Y(:,c).', nside, nside, []);
Kv = reshape(K(:,c).', nside, nside, []);
end

function Y = tgt(Z, o, H)
d = size(Z,1);  Y = zeros(numel(o), d*H);
for h = 1:H, Y(:, (h-1)*d+(1:d)) = Z(:, o+h).'; end
end

function P = decodeH(Pz, dec, p, H, d)
P = zeros(size(Pz,1), d*H);
for h = 1:H
    Zh = Pz(:, (h-1)*p+(1:p)).';
    P(:, (h-1)*d+(1:d)) = dec(Zh).';
end
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
