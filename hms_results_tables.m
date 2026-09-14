function hms_results_tables(outdir)
%HMS_RESULTS_TABLES The result tables of the paper, generated from the archives.
%
%   HMS_RESULTS_TABLES(outdir)
%
%   WHY THESE TABLES ARE GENERATED AND NOT TYPED. The storage column of the
%   route table was typed by hand once, and every one of its entries was wrong:
%   4 821 296 where the archive said 4 924 769, and so on down the column. The
%   error was invisible because the numbers were plausible and internally
%   consistent. Nothing typed from a result file stays correct across a
%   renormalisation, a rerun or a rename, so nothing is typed.
%
%   THE NORMALISER IS THE MEAN, not the spread. nRMSE is RMSE divided by the
%   mean of the observations over the scored set, which is the convention of the
%   field; HMS_SCALE_REF holds the constant and its provenance. Every arm here is
%   stored in W/m2 in the archives, so the division happens once, here.
%
%   NAMES COME FROM THE NOMENCLATURE and not from the campaign, so that a row in
%   a table reads without the text: ELM-pixel, ELM-field, Radon-ELM-field.
%
%   THE BEST VALUE IN EACH HORIZON COLUMN IS BOLD, which is what a reader scans
%   for and what a column of undifferentiated numbers refuses to give.

if nargin < 1 || isempty(outdir)
    outdir = fullfile(pwd, 'papier', 'tables');
end
if ~isfolder(outdir), mkdir(outdir); end

SC = hms_scale_ref();  mu = SC.mu;
HS = hms_horizons();

% ---------------------------------------------------------------- collect
A = collect(mu, HS);
fprintf('\n%d arms collected\n', numel(A));

% ------------------------------------------------------- the main comparison
f = fullfile(outdir, 'ladder.tex');
emit_ladder(f, A, HS);
fprintf('written %s\n', f);

% ------------------------------------------------------------- the routes
f = fullfile(outdir, 'routes.tex');
emit_routes(f, A, HS);
fprintf('written %s\n', f);
end

% ===========================================================================
function A = collect(mu, HS)
%COLLECT Every arm the paper reports, from whichever archive holds it.
R = fullfile(pwd, 'results');
A = struct('name',{},'reads',{},'e',{},'eh',{},'theta',{},'secs',{}, ...
           'gpr3',{},'fam',{},'nice3',{},'gam3',{});

q = tryload(fullfile(R,'bench.mat'));
if ~isempty(q)
    for k = 1:numel(q.R)
        [nm, rd, fam] = rename_bench(q.R(k).name);
        if isempty(nm), continue; end
        A(end+1) = one(nm, rd, q.R(k), mu, HS, fam); %#ok<AGROW>
    end
end

u = tryload(fullfile(R,'unet_bench.mat'));
if ~isempty(u)
    for k = 1:numel(u.R)
        A(end+1) = one(sprintf('U-Net-field (%d channels)', u.R(k).base), ...
                       'the field', u.R(k), mu, HS, 'deep'); %#ok<AGROW>
    end
end

l = tryload(fullfile(R,'lstm_bench.mat'));
if ~isempty(l)
    for k = 1:numel(l.R)
        if isnan(l.R(k).p), nm = 'LSTM-field';
        else, nm = sprintf('LSTM-%s-field ($p{=}%d$)', ...
                           code_name(l.R(k).rep), l.R(k).p); end
        A(end+1) = one(nm, 'the field', l.R(k), mu, HS, 'deep'); %#ok<AGROW>
    end
end

% BLEND READS ONE CELL, and putting it anywhere else would misread the table.
% The operator combines this cell's present value with this cell's value one
% period earlier; it never looks sideways. It belongs beside the two
% persistences it interpolates between, not with the arms that read space.
b = tryload(fullfile(R,'blend_bench.mat'));
if ~isempty(b)
    for k = 1:numel(b.R)
        A(end+1) = one(b.R(k).name, 'one cell', b.R(k), mu, HS, 'naive'); %#ok<AGROW>
    end
end

% THE REANALYSIS SITS IN THE SAME TABLE, in its own block and under its own
% normaliser. Cyril, comment #81: "era5 doit juste etre integre dans le bench".
% Three things make it a separate block and not another row in the ranking.
% It assimilates observations made at the hour it describes, so it is not a
% forecast and cannot be ranked against forecasts. It carries one error and
% not six, because a reanalysis does not degrade with lead time, exactly as
% cyclic persistence does not. And it is scored on 58 quarter-degree cells
% against the same product aggregated to them, not on the 722 fine cells, so
% its own out-of-sample mean is the only correct divisor: the archive already
% divided by it and the value is taken as it stands.
e = tryload(fullfile(R,'era5_field_corsica.mat'));
if ~isempty(e) && isfield(e,'S')
    S = e.S;
    assert(strcmp(S.scale_name, 'oos_mean_ghi'), ...
           'hms_results_tables:era5scale', ...
           'the reanalysis row is only comparable on the mean normaliser');
    A(end+1) = struct('name', '\texttt{ERA5}', ...
                      'reads', 'the atmosphere at that hour', ...
                      'e', S.nrmse, 'eh', repmat(S.nrmse, 1, numel(HS)), ...
                      'theta', NaN, 'secs', NaN, 'gpr3', S.gpr, ...
                      'fam', 'reference', 'nice3', NaN, 'gam3', NaN);
end
end

function s = one(nm, reads, r, mu, HS, fam)
eh = nan(1,numel(HS));  g = NaN;  n3 = NaN;  m3 = NaN;
if isfield(r,'met') && ~isempty(r.met)
    eh = r.met.nrmse * (mu_of(r) / mu);   % archives normalised by the old sig
    i3 = find(r.met.h == 3, 1);
    if ~isempty(i3)
        g = r.met.gpr(i3);
        % NICE AND THE MEAN GAMMA AT THREE HOURS, which the paper defined and
        % never showed (Cyril, 14 September 2026). Both are dimensionless.
        if isfield(r.met, 'nicesig'), n3 = r.met.nicesig(i3); end
        if isfield(r.met, 'gamma'),   m3 = r.met.gamma(i3);   end
    end
end
% THE TIME AN ARM COST TO FIT SITS BESIDE WHAT IT COSTS TO STORE. The two are
% not the same axis and the paper needs both: a route can be cheap to hold and
% expensive to build. Arms whose archive records no duration show a dash
% rather than a zero.
sec = NaN;
for f = {'secs', 'total_s'}
    if isfield(r, f{1}) && ~isempty(r.(f{1})) && isfinite(r.(f{1}))
        sec = r.(f{1});  break
    end
end
s = struct('name', nm, 'reads', reads, 'e', r.rmse / mu, 'eh', eh, ...
           'theta', r.n_par, 'secs', sec, 'gpr3', g, 'fam', fam, ...
           'nice3', n3, 'gam3', m3);
end

function v = mu_of(r)
%MU_OF What the archived met block was actually divided by.
%
%   THIS USED TO RETURN A HARD-CODED 267.1184 AND IGNORE ITS ARGUMENT, which
%   was correct for archives written before the convention was fixed and
%   becomes a BUG the moment any of them is migrated. A migrated row stores
%   met.nrmse already divided by the mean; multiplying it again by
%   267.1184/mu would correct a correction and make the table wrong by a
%   factor of 1.53, silently, from a table that is right today.
%
%   So the archive is asked instead of assumed. A row that declares
%   scale_name = 'oos_mean_ghi' is already on the study convention and needs no
%   rescaling; anything else is legacy and carries the old constant.
LEGACY = 267.1184;
% THE MET BLOCK STATES ITS OWN NORMALISER, and that is read first. A fresh
% archive from the current bench carries met.scale_value but no row-level
% scale_name, and fell through to the legacy constant: every horizon column
% came out 0.655 times too small (found 14 September 2026 on the GitHub copy).
if isstruct(r) && isfield(r, 'met') && isstruct(r.met) && ...
        isfield(r.met, 'scale_value') && isscalar(r.met.scale_value) && ...
        isfinite(r.met.scale_value)
    v = r.met.scale_value;
elseif isstruct(r) && isfield(r, 'scale_name') && ...
        strcmp(r.scale_name, 'oos_mean_ghi')
    SC = hms_scale_ref();
    v = SC.mu;               % factor becomes one: nothing to undo
    if isfield(r, 'scale_value') && isscalar(r.scale_value) && ...
            isfinite(r.scale_value)
        v = r.scale_value;   % the domain's own mean, when it carries one
    end
else
    v = LEGACY;
end
end

function [nm, reads, fam] = rename_bench(s)
%RENAME_BENCH Campaign label to nomenclature. Anything unrecognised is dropped
%   rather than shown under a name nobody defined.
nm = '';  reads = 'the field';  fam = 'latent';
if startsWith(s, 'ELM per pixel')
    nm = '\armpixel';  reads = 'one cell';  fam = 'blind';
elseif startsWith(s, 'ridge')
    nm = 'Ridge-pixel';  reads = 'one cell';  fam = 'blind';
elseif startsWith(s, 'tanh')
    nm = '\armfield';  fam = 'dense';
elseif startsWith(s, 'linear')
    nm = '\armfield{} (linear)';  fam = 'dense';
elseif strcmp(s, 'persistence')
    nm = 'Pers';  reads = 'one cell';  fam = 'naive';
elseif strcmp(s, 'cyclic')
    nm = 'Pers-24h';  reads = 'one cell';  fam = 'naive';
else
    t = regexp(s, '^(\w+) p=(\d+)', 'tokens', 'once');
    if ~isempty(t)
        nm = sprintf('\\armlat{%s} ($p{=}%s$)', code_name(t{1}), t{2});
    end
end
end

function c = code_name(s)
switch lower(s)
    case 'radon',       c = 'Radon';
    case 'dct',         c = 'DCT';
    case 'pca',         c = 'PCA';
    case 'wavelet',     c = 'Wav';
    case 'autoencoder', c = 'AE';
    case 'randproj',    c = 'RP';
    case 'subsample',   c = 'Sub';
    case 'field',       c = 'field';
    otherwise,          c = s;
end
end

% ===========================================================================
function emit_ladder(f, A, HS)
%EMIT_LADDER The arms that carry Q1, ordered by pooled error.
%
%   THE BEST VALUE IS BOLD IN EVERY COLUMN, not only in the horizon columns.
%   Cyril, comments #55 and #56: a reader scans for the winner and the pooled
%   column and the pass rate were the two that refused to give it. The pass
%   rate is a rate, so its best is the largest; every error column's best is
%   the smallest.
%
%   THE REANALYSIS IS BELOW THE RULE and is never bold. It has the lowest
%   error in the table and it is not competing: it is scored on a coarser grid
%   and it knows the hour it describes. Letting it take a bold cell would say
%   the opposite of what the paper says.
keep = ismember({A.fam}, {'blind','dense','deep','naive'});
S = A(keep);
[~, o] = sort([S.e]);  S = S(o);
M = reshape([S.eh], numel(HS), []).';
best = min(M, [], 1);
bpool = min([S.e]);
bgpr = max([S.gpr3]);
bnice = min([S.nice3]);
bgam = min([S.gam3]);
Rf = A(strcmp({A.fam}, 'reference'));
% A TIE IS DECIDED ON THE PRINTED DIGITS. Cyril, 14 September 2026: cyclic
% persistence prints the same pass rate as the per-cell route and was not bold,
% because the unrounded values differ beyond the printed decimal.
same = @(v, b, fmt) isfinite(v) && isfinite(b) && strcmp(sprintf(fmt, v), sprintf(fmt, b));

fid = fopen(f, 'w');
fprintf(fid, ['%% generated by HMS_RESULTS_TABLES -- do not edit\n' ...
              '\\begin{tabular}{@{}l r %s r r r r r@{}}\n\\toprule\n'], ...
        repmat('r', 1, numel(HS)));
fprintf(fid, 'arm & all');
for h = HS, fprintf(fid, ' & $%d$\\,h', h); end
fprintf(fid, [' & parameters & fit time (min) & \\NICE{}$^{\\Sigma}$ $3$\\,h' ...
              ' & mean $\\gamma$ $3$\\,h & \\texttt{GPR} $3$\\,h \\\\\n\\midrule\n']);
for k = 1:numel(S)
    fprintf(fid, '%s & %s', S(k).name, num(S(k).e, same(S(k).e, bpool, '%.3f')));
    for j = 1:numel(HS)
        fprintf(fid, ' & %s', num(M(k,j), same(M(k,j), best(j), '%.3f')));
    end
    fprintf(fid, ' & %s & %s & %s & %s & %s \\\\\n', kM(S(k).theta), ...
            mins(S(k).secs), num(S(k).nice3, same(S(k).nice3, bnice, '%.3f')), ...
            num(S(k).gam3, same(S(k).gam3, bgam, '%.3f')), ...
            pct(S(k).gpr3, same(S(k).gpr3, bgpr, '%.1f')));
end
for k = 1:numel(Rf)
    fprintf(fid, '\\midrule\n%s$^\\dag$ & %s', Rf(k).name, num(Rf(k).e, false));
    for j = 1:numel(HS), fprintf(fid, ' & %s', num(Rf(k).eh(j), false)); end
    fprintf(fid, ' & %s & %s & %s & %s & %s \\\\\n', kM(Rf(k).theta), ...
            mins(Rf(k).secs), num(Rf(k).nice3, false), num(Rf(k).gam3, false), ...
            pct(Rf(k).gpr3, false));
end
fprintf(fid, '\\bottomrule\n\\end{tabular}\n');
fclose(fid);
end

function emit_routes(f, A, HS)
%EMIT_ROUTES Every code at every payload, which is the material of Q2.
%
%   THE FITTING TIME SITS HERE TOO. Cyril, comment #66: "la difference avec les
%   precedents tableaux... donnez au moins les data ? le temps de calcul ?".
%   Without it the reader cannot tell a route that is cheap to hold from one
%   that is cheap to build, and the two are different axes.
S = A(strcmp({A.fam}, 'latent'));
[~, o] = sort([S.e]);  S = S(o);
ix = [1 3 5];  ix = ix(ix <= numel(HS));
be = min([S.e]);
fid = fopen(f, 'w');
fprintf(fid, '%% generated by HMS_RESULTS_TABLES -- do not edit\n');
fprintf(fid, '\\begin{tabular}{@{}l r %s r r@{}}\n\\toprule\n', ...
        repmat('r', 1, numel(ix)));
fprintf(fid, 'route & all');
for j = ix, fprintf(fid, ' & $%d$\\,h', HS(j)); end
fprintf(fid, ' & parameters & fit time (min) \\\\\n\\midrule\n');
for k = 1:numel(S)
    fprintf(fid, '%s & %s', S(k).name, num(S(k).e, strcmp(sprintf('%.3f', S(k).e), sprintf('%.3f', be))));
    for j = ix, fprintf(fid, ' & %s', num(S(k).eh(j), false)); end
    fprintf(fid, ' & %s & %s \\\\\n', kM(S(k).theta), mins(S(k).secs));
end
fprintf(fid, '\\bottomrule\n\\end{tabular}\n');
fclose(fid);
end

% ===========================================================================
function s = num(v, bold)
if ~isfinite(v), s = '--'; return; end
s = sprintf('%.3f', v);
if bold, s = ['\textbf{' s '}']; end
end

function s = pct(v, bold)
if nargin < 2, bold = false; end
if ~isfinite(v), s = '--'; return; end
s = sprintf('%.1f\\%%', v);
if bold, s = ['\textbf{' s '}']; end
end

function s = mins(v)
%MINS Fitting time in minutes, or a dash where the archive recorded none.
if isfinite(v), s = sprintf('%.0f', v/60); else, s = '--'; end
end

function s = thou(v)
if ~isfinite(v), s = '--'; return; end
s = regexprep(sprintf('%d', round(v)), '(\d)(?=(\d{3})+$)', '$1\\,');
end

function s = kM(v)
%KM A parameter count in thousands (k) or millions (M). Cyril, 14 September
%   2026: "pour le nombre de parametres ecrire k, M... plus simple".
if ~isfinite(v), s = '--'; return; end
if v >= 1e6,     s = sprintf('%.1fM', v/1e6);
elseif v >= 1e3, s = sprintf('%.1fk', v/1e3);
else,            s = sprintf('%d', round(v));
end
end

function q = tryload(f)
if isfile(f), q = load(f); else, q = []; end
end
