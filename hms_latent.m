function E = hms_latent(kind, n, p, cfg, train)
%HMS_LATENT Build one latent encoder, at a prescribed latent dimension.
%
%   E = HMS_LATENT(kind, n, p, cfg, train)
%
%   kind  : 'radon' | 'radon_best' | 'dct' | 'wavelet' | 'randproj' | 'pca'
%           | 'subsample' | 'autoencoder'
%   n     : grid side
%   p     : LATENT DIMENSION per map, identical for every kind
%   cfg   : configuration from HMS_CONFIG
%   train : n^2 x N_train matrix of training maps, used only by the kinds
%           that learn a basis; ignored by the others
%
%   E.encode  : function handle, (n^2 x T) field columns -> (p x T) latent
%   E.decode  : function handle, (p x T) latent -> (n^2 x T) reconstruction,
%               using the encoder's OWN inverse. Empty when the kind has no
%               natural inverse.
%   E.n_decode: scalars the receiver must store to run E.decode
%   E.p       : the latent dimension actually obtained
%   E.learned : true when the encoder was fitted on data
%   E.n_basis : scalars the receiver must store to use this encoder, which is
%               zero for the analytic kinds and the basis size for the others
%   E.fit_time: seconds spent fitting, zero for the analytic kinds
%
%   THE POINT OF THIS FILE. The question of the study is not which compressor
%   is best but which LATENT is the best state to forecast in. Every encoder
%   here therefore produces the same number of coefficients per map, so the
%   forecaster that consumes them is identical in size and the comparison is
%   about the representation alone. The sinogram is one candidate among
%   several; that it comes from an integral transform rather than a basis
%   expansion is precisely what is being tested.
%
%   Two of the encoders are fitted on data and two costs follow that the
%   analytic ones do not pay: the basis has to be stored by whoever decodes
%   or interprets it, and it has to be estimated in the first place. Both are
%   reported, and both are computed on the training split alone.

if nargin < 4 || isempty(cfg), cfg = hms_config('n', n); end
if nargin < 5, train = []; end
E = struct('kind', lower(kind), 'p', p, 'learned', false, ...
           'n_basis', 0, 'fit_time', 0, 'note', '', ...
           'decode', [], 'n_decode', 0);

switch lower(kind)

% ------------------------------------------------------------------ radon
case 'radon'
    ns = size(radon(zeros(n),0),1);
    Na = max(1, round(p / ns));
    op = hms_operator('radon', n, cfg, Na);
    E.p      = ns * Na;
    E.encode = @(X) op.R * X;
    E.param  = Na;
    th = (0:Na-1) * 180 / Na;
    E.decode = @(Z) rad_dec(Z, n, ns, Na, th);
    E.n_decode = 0;                          % filtered backprojection is analytic
    E.note   = sprintf('Na = %d directions, ns = %d detectors', Na, ns);

% ------------------------------------------------------------- radon_best
case 'radon_best'
    % THE SAME OPERATOR, WITH THE CHOICES IT ACTUALLY HAS MADE IN ITS FAVOUR.
    % The plain 'radon' kind inherits three things rather than choosing them:
    % the detector resolution MATLAB returns for this grid, an arbitrary
    % alignment of that detector, and filtered back projection as the inverse.
    % HMS_RADON_BEST measured all three and found each one costs the route
    % something, the first two heavily below p = 300. This kind reads that
    % measurement and uses the configuration it selected, decoding with the
    % least-squares inverse of the exact operator that encoded.
    %
    % IT IS NOT FREE AND THE ACCOUNTING SAYS SO. A least-squares inverse is a
    % stored n^2-by-p matrix, so E.n_decode is not zero here as it is for the
    % analytic route. That is the trade this kind exists to expose: the
    % configuration that makes the geometry competitive is the configuration
    % that costs it its one structural advantage over a fitted basis.
    f = fullfile(pwd, 'results', 'radon_best.mat');
    assert(isfile(f), 'hms_latent:noRadonBest', ...
        ['No configuration table at %s. Run HMS_RADON_BEST first: this kind ' ...
         'reads a measurement rather than inventing a configuration.'], f);
    Q = load(f);
    assert(isfield(Q,'protocol') && strcmp(Q.protocol,'oos-v1') && ...
           isfield(Q,'OOS'), 'hms_latent:legacyRadonBest', ...
          ['%s contains a test-selected legacy sweep. Run HMS_RADON_BEST to ' ...
           'create the oos-v1 configuration table before using this route.'], f);
    % Use the configuration frozen on the selection year for the largest
    % declared budget not exceeding p. No test-surface argmin exists in the
    % file, so this consumer cannot accidentally re-select on year 2.
    cand = find([Q.OOS.p] <= p);
    assert(~isempty(cand), 'hms_latent:budget', ...
        'No frozen sinogram configuration costs %d numbers or fewer.', p);
    [~,jc] = max([Q.OOS(cand).p]);
    ib = cand(jc);                         % independent of table ordering
    nr = Q.OOS(ib).nr;  Na = Q.OOS(ib).Na;  ph = Q.OOS(ib).phase;

    nsf = size(radon(zeros(n),0), 1);
    Dn  = hms_binops(nsf, nr, ph);
    op  = hms_operator('radon', n, cfg, Na);
    Aop = kron(speye(Na), sparse(Dn)) * op.R;      % (nr*Na) x n^2
    G   = Aop.' * Aop;
    G   = G + 1e-8 * mean(diag(G)) * speye(size(G,1));
    Dec = full(G \ Aop.');                          % n^2 x (nr*Na)

    E.p        = nr * Na;
    E.encode   = @(X) Aop * X;
    E.decode   = @(Z) Dec * Z;
    E.n_decode = numel(Dec);
    E.param    = Na;
    E.note     = sprintf(['n_rho = %d, N_a = %d, partition phase %.2f, ' ...
                          'least-squares inverse'], nr, Na, ph);

% -------------------------------------------------------------------- dct
case 'dct'
    keep = zigzag_order(n);  keep = keep(1:p);
    E.encode = @(X) dct_enc(X, n, keep);
    E.decode = @(Z) dct_dec(Z, n, keep);
    E.n_decode = 0;                          % the inverse DCT is analytic and free
    E.param  = p;
    E.note   = 'low-frequency-first ordering, positions are a convention';

% ---------------------------------------------------------------- wavelet
case 'wavelet'
    % A fixed retention map, chosen once on the training split by average
    % coefficient energy, then frozen. Keeping the largest coefficients of
    % each map instead would require transmitting their positions, which
    % doubles the payload and is the usual way an adaptive transform is made
    % to look better than it is.
    wname = 'bior4.4';                       % the CDF 9/7 pair of JPEG2000
    % PERIODIC EXTENSION IS NOT OPTIONAL HERE. wavedec2 follows the GLOBAL
    % dwtmode, 'sym' by default, and a symmetric extension returns MORE
    % coefficients than the image has pixels: the expansion is a redundant
    % frame, not a basis. Keeping p of those is then not the same budget as
    % keeping p of n^2, and the round trip is not exact even when everything
    % is retained -- measured at 2.75e-2 instead of machine precision. The
    % mode is set here and restored on exit, and the coefficient count is
    % asserted rather than trusted.
    wcl = wav_mode_per();                    %#ok<NASGU> restored on cleanup
    [c0, S0] = wavedec2(zeros(n), wmaxlev([n n], wname), wname);
    assert(numel(c0) == n*n, 'hms_latent:wavFrame', ...
        ['Wavelet expansion has %d coefficients for %d pixels: the extension ' ...
         'mode is not periodic, so this is a frame and not a basis.'], ...
        numel(c0), n*n);
    assert(numel(c0) >= p, 'hms_latent:wavTooSmall', ...
        'Wavelet expansion has %d coefficients, fewer than p = %d.', numel(c0), p);
    if isempty(train)
        keep = 1:p;                          % low-frequency first
        E.note = 'bior4.4, first p coefficients (no training set supplied)';
    else
        t0 = tic;
        acc = zeros(1, numel(c0));
        m = min(size(train,2), 2000);        % a subsample is enough for energy
        idx = round(linspace(1, size(train,2), m));
        for j = idx
            acc = acc + abs(wavedec2(reshape(train(:,j), n, n), ...
                            wmaxlev([n n], wname), wname)).^2;
        end
        [~, ord] = sort(acc, 'descend');
        keep = sort(ord(1:p));
        E.fit_time = toc(t0);
        E.learned  = true;
        E.n_basis  = p;                      % the retained positions
        E.note = sprintf('bior4.4, %d positions selected on train by energy', p);
    end
    E.encode = @(X) wav_enc(X, n, wname, keep);
    E.decode = @(Z) wav_dec(Z, n, wname, keep, numel(c0), S0);
    E.n_decode = 0;                          % waverec2 is analytic and free
    E.param  = p;

% --------------------------------------------------------------- randproj
case 'randproj'
    st = rng; rng(cfg.seed);
    R = randn(p, n*n) / sqrt(p);
    rng(st);
    E.encode = @(X) R * X;
    % The pseudo-inverse is the natural decoder. It is large, but like R it is
    % regenerated from the shared seed rather than transmitted, so it costs
    % nothing on the link and everything in local memory.
    Rp = pinv(R);
    E.decode = @(Z) Rp * Z;
    E.n_decode = 0;
    E.param  = p;
    E.note   = 'Gaussian, shared seed, so the matrix is not transmitted';

% -------------------------------------------------------------------- pca
case 'pca'
    assert(~isempty(train), 'hms_latent:pcaNeedsTrain', ...
        'The PCA latent must be fitted on a training split.');
    t0 = tic;
    mu = mean(train, 2);
    m  = min(size(train,2), 4000);
    idx = round(linspace(1, size(train,2), m));
    [U, ~, ~] = svd(train(:,idx) - mu, 'econ');
    k = min(p, size(U,2));
    Uk = U(:, 1:k);
    E.fit_time = toc(t0);
    E.learned  = true;
    E.n_basis  = numel(Uk) + numel(mu);      % the receiver must store this
    E.p        = k;
    E.encode   = @(X) Uk' * (X - mu);
    % The natural inverse RESTORES THE MEAN. Omitting mu here is exactly the
    % defect that made a centred latent reconstruct to zero mean and look
    % broken; the fault was never in the method.
    E.decode   = @(Z) Uk * Z + mu;
    E.n_decode = 0;                          % already counted in n_basis
    E.param    = k;
    if k < p
        E.note = sprintf('rank-limited: %d components available for p = %d', k, p);
    else
        E.note = sprintf('%d components fitted on the training split', k);
    end

% -------------------------------------------------------------- autoencoder
case 'autoencoder'
    % THE THIRD FAMILY, AND THE ONE THIS STUDY WAS MISSING. A learned nonlinear
    % code: one hidden layer of width p with a tanh, a linear decoder, trained
    % to reconstruct the field. It is fitted on the TRAINING SPLIT ONLY, like
    % the principal components, and it is the arm that should reconstruct best
    % and cost most -- both halves of that expectation are measured rather than
    % asserted.
    %
    % WHAT IT COSTS ITS RECEIVER, AND WHY THAT IS THE POINT. The decoder is
    % p n^2 + n^2 stored scalars, against n^2(k+1) for the principal
    % components, so a nonlinear code is not cheaper to ship merely because it
    % is cleverer. The accounting of Section sec:params applies to it
    % unchanged.
    %
    % THE FIT IS SEEDED AND THE SCHEDULE IS FIXED. A representation whose score
    % moves with its initialisation cannot be compared with an analytic one, so
    % the seed is set here and the number of epochs is not tuned per latent
    % dimension: tuning it would be tuning one arm and not the others.
    assert(~isempty(train), 'hms_latent:aeNeedsTrain', ...
        'The autoencoder latent must be fitted on a training split.');
    t0 = tic;
    rng(7, 'twister');
    mu = mean(train, 2);
    sd = std(train, 0, 2);  sd(sd < eps) = 1;
    Xn = ((train - mu) ./ sd).';                 % observations in rows
    m  = min(size(Xn,1), 6000);
    Xn = Xn(round(linspace(1, size(Xn,1), m)), :);

    din = size(train, 1);
    net = dlnetwork([
        featureInputLayer(din, 'Name', 'in')
        fullyConnectedLayer(p,   'Name', 'enc')
        tanhLayer(               'Name', 'code')
        fullyConnectedLayer(din, 'Name', 'dec')]);

    opts = trainingOptions('adam', ...
        'MaxEpochs', 120, ...
        'MiniBatchSize', 256, ...
        'InitialLearnRate', 1e-3, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', false, ...
        'Plots', 'none', ...
        'ExecutionEnvironment', 'cpu');
    net = trainnet(Xn, Xn, net, 'mse', opts);

    We = net.Layers(2).Weights;  be = net.Layers(2).Bias;
    Wd = net.Layers(4).Weights;  bd = net.Layers(4).Bias;

    E.fit_time = toc(t0);
    E.learned  = true;
    E.n_basis  = numel(We) + numel(be) + numel(mu) + numel(sd);
    E.p        = p;
    E.encode   = @(X) tanh(We * ((X - mu) ./ sd) + be);
    E.decode   = @(Z) (Wd * Z + bd) .* sd + mu;
    E.n_decode = numel(Wd) + numel(bd);
    E.param    = p;
    E.note     = sprintf('shallow autoencoder, %d units, %d epochs on the training split', ...
                         p, opts.MaxEpochs);

% -------------------------------------------------------------- subsample
case 'subsample'
    m = max(1, floor(sqrt(p)));
    E.p      = m*m;
    E.encode = @(X) sub_enc(X, n, m);
    E.decode = @(Z) sub_dec(Z, n, m);        % piecewise constant, the true inverse
    E.n_decode = 0;
    E.param  = m;
    E.note   = sprintf('block average to %dx%d, the trivial encoder', m, m);

otherwise
    error('hms_latent:unknownKind', 'Unknown latent "%s".', kind);
end
end

% =========================================================================
function Z = dct_enc(X, n, keep)
Z = zeros(numel(keep), size(X,2));
for t = 1:size(X,2)
    D = dct2(reshape(X(:,t), n, n));  Z(:,t) = D(keep);
end
end

function Z = wav_enc(X, n, wname, keep)
wcl = wav_mode_per();                        %#ok<NASGU>
lev = wmaxlev([n n], wname);
Z = zeros(numel(keep), size(X,2));
for t = 1:size(X,2)
    c = wavedec2(reshape(X(:,t), n, n), lev, wname);
    Z(:,t) = c(keep);
end
end

function Z = sub_enc(X, n, m)
W = box_weights(n, m);
Z = zeros(m*m, size(X,2));
for t = 1:size(X,2)
    B = W * reshape(X(:,t), n, n) * W.';
    Z(:,t) = B(:);
end
end

function W = box_weights(n, m)
%BOX_WEIGHTS Exact area-averaging matrix, m x n, one row per output cell.
% Output cell k covers the input interval [k*n/m, (k+1)*n/m) and a pixel that
% straddles a boundary contributes in proportion to the overlap.
%
% WHY THIS REPLACED imresize(..., 'box'). imresize aligns pixel CENTRES with a
% half-pixel offset of its own, which is a legitimate convention but not the
% block partition stated here, and not the one the Python implementation used.
% The two then disagreed by 4.5e-2 on the latent while each looked correct in
% isolation. Writing the operator out makes it identical on both sides and
% removes a silent dependency on a toolbox's internal alignment.
W = zeros(m, n);
e = (0:m) * (n / m);
for k = 1:m
    lo = e(k); hi = e(k+1);
    for i = floor(lo):min(ceil(hi), n) - 1
        W(k, i+1) = max(0, min(hi, i+1) - max(lo, i));
    end
end
W = W ./ sum(W, 2);
end

function X = dct_dec(Z, n, keep)
X = zeros(n*n, size(Z,2));
for t = 1:size(Z,2)
    D = zeros(n, n);  D(keep) = Z(:,t);
    X(:,t) = reshape(idct2(D), [], 1);
end
end

function X = wav_dec(Z, n, wname, keep, ncoef, S)
wcl = wav_mode_per();                        %#ok<NASGU>
lev = wmaxlev([n n], wname);  %#ok<NASGU>
X = zeros(n*n, size(Z,2));
for t = 1:size(Z,2)
    c = zeros(1, ncoef);  c(keep) = Z(:,t);
    R = waverec2(c, S, wname);
    X(:,t) = reshape(R(1:n, 1:n), [], 1);
end
end

function X = sub_dec(Z, n, m)
% Block averaging maps m x m onto n x n by replication: the reconstruction is
% piecewise constant on the same blocks the encoder averaged over. Anything
% smoother would be an interpolator doing work the encoder never did. The
% index map is written out for the same reason as BOX_WEIGHTS: so that it is
% the operator declared here and not a toolbox's alignment convention.
jj = min(floor((0:n-1) * m / n), m - 1) + 1;
X = zeros(n*n, size(Z,2));
for t = 1:size(Z,2)
    B = reshape(Z(:,t), m, m);
    X(:,t) = reshape(B(jj, jj), [], 1);
end
end

function X = rad_dec(Z, n, ns, Na, th)
% Filtered backprojection, the analytic inverse of the Radon transform. It is
% not the adjoint and it is not exact on a discrete grid; that approximation
% is a property of the transform and must be charged to it, not corrected by
% fitting a decoder on our own data.
X = zeros(n*n, size(Z,2));
for t = 1:size(Z,2)
    S = reshape(Z(:,t), ns, Na);
    R = iradon(S, th, 'linear', 'Ram-Lak', 1, n);
    X(:,t) = R(:);
end
end

function cl = wav_mode_per()
%WAV_MODE_PER Switch the global wavelet extension to periodic, and put it back.
% The Wavelet Toolbox keeps the extension mode in global state, so it cannot be
% passed as an argument. The previous mode is restored by onCleanup, which runs
% on a normal return and on an error alike, so no caller is left with a mode it
% did not set.
old = dwtmode('status', 'nodisp');
if strcmp(old, 'per')
    cl = onCleanup(@() []);
else
    dwtmode('per', 'nodisp');
    cl = onCleanup(@() dwtmode(old, 'nodisp'));
end
end

function ord = zigzag_order(n)
[I, J] = ndgrid(1:n, 1:n);
s = I(:) + J(:);  tb = I(:);
odd = mod(s,2) == 1;  tb(odd) = n + 1 - I(odd);
[~, ord] = sortrows([s, tb]);
end
