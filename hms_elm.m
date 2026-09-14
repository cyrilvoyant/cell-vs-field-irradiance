function out = hms_elm(mode, varargin)
%HMS_ELM Extreme learning machine, MIMO multi-horizon, memory-streamed.
%
%   M  = HMS_ELM('fit', X, Y, Nh, lambda, seed [, batchSize])
%   Yh = HMS_ELM('predict', M, X)
%
%   MINI-BATCH RIDGE, AND IT IS EXACT.
%   Training walks the samples in blocks and accumulates only
%
%       H'H   N_h x N_h        and        H'Y   N_h x d_out
%
%   then solves (H'H + lambda I) beta = H'Y once. Unlike a mini-batch gradient
%   method this is not an approximation: the accumulation of H'H and H'Y over
%   blocks is exactly the same matrix as a single pass over all samples, so
%   the closed-form ridge solution is recovered exactly whatever the block
%   size. Batching bounds memory, it does not trade accuracy for it, and the
%   result does not depend on the block order.
%
%   The default block is about 150 MB of design matrix; pass batchSize to
%   force a different one. The input standardisation is computed in the same
%   streamed way, from running sums, so no full pass over X is ever held.
%
%   X : N x d_in    one row per sample (the lag window, already flattened)
%   Y : N x d_out   one row per sample (the whole 24-step horizon at once)
%
%   A single hidden layer with random, fixed input weights and a ridge solve
%   for the output weights. Nothing is trained by gradient descent, which is
%   what makes this the frugal reference of this group's earlier work.
%
%   MULTI-HORIZON BY CONSTRUCTION, AND THE SAVING IS COMPUTATIONAL ONLY. Y holds
%   the complete 24-step volume, so one call returns every horizon. What that
%   buys is one accumulation of H'H and H'Y instead of twenty-four, which is the
%   whole of the factor 24: the pass over the samples is the cost, and it does
%   not depend on how many output columns are carried.
%
%   IT BUYS NO COHERENCE, AND AN EARLIER VERSION OF THIS COMMENT CLAIMED IT DID.
%   Under the squared loss the ridge solution separates column by column ---
%   beta = (H'H + lambda I) \ H'Y, and column j of beta depends only on column j
%   of Y --- so the twenty-four outputs are twenty-four independent regressions
%   that happen to share a hidden layer. Nothing constrains them to agree with
%   one another, and the horizons can contradict each other exactly as freely as
%   twenty-four separately fitted models could. Only a loss coupling the columns
%   would change that, and none is used here.
%
%   WHY IT IS STREAMED. The direct arm of the benchmark forecasts the image
%   volume itself: at n = 64 with 24 lags the input is 98304 wide and the
%   output just as wide, so the hidden matrix H would be N x N_h and the
%   design matrix N x 98304. Materialising them is what makes the direct
%   approach look impossible. Only three objects are ever held here:
%
%       W       d_in x N_h        the random projection
%       H'H     N_h x N_h         accumulated over sample blocks
%       H'Y     N_h x d_out       accumulated over sample blocks
%
%   so the memory cost is set by N_h and never by the number of samples. The
%   parameter COUNT is unchanged -- N_h*(d_in + d_out) scalars must still be
%   stored to run the model, and that is the number the frugality accounting
%   uses. Streaming makes the experiment runnable; it does not make the model
%   cheaper, and the two must not be confused.

switch lower(mode)

case 'fit'
    [X, Y, Nh, lambda, seed] = deal(varargin{1:5});
    if numel(varargin) >= 6 && ~isempty(varargin{6})
        blk = varargin{6};                        % explicit mini-batch size
    else
        blk = [];
    end
    [N, d_in]  = size2(X);
    [Ny, d_out] = size2(Y);
    assert(N == Ny, 'hms_elm:sampleMismatch', ...
        'X has %d samples but Y has %d.', N, Ny);

    % Standardisation in one streamed pass, by Chan's parallel update rather
    % than E[X^2]-E[X]^2. The naive form subtracts two large nearly equal
    % numbers and loses most of its significant digits when the mean is large
    % compared with the spread, which is exactly the case for irradiance.
    if isempty(blk)
        % THE HIDDEN BLOCK IS THE LARGE ONE, AND IT WAS NOT COUNTED. This rule
        % used to size the batch from d_in + d_out alone, which is the design
        % matrix and the targets. But the array actually held at once is
        % tanh(Xb*W + b), of size blk by N_h, and N_h is the one number this
        % function exists to make large. On the per-pixel family d_in + d_out
        % is 48, so the old rule chose four hundred thousand rows, and at
        % N_h = 4096 that hidden block is 13.6 GB -- on a machine with 15.6.
        % The result was not an error but something worse: the system paged,
        % committed 26 GB against 15.6 of physical memory, and nine arms of a
        % campaign failed with "out of memory" hours apart for a reason that
        % looked like bad luck. Counting N_h puts the block back where the
        % comment always said it was, at about 160 MB.
        blk = max(1, floor(2e7 / max(1, d_in + d_out + Nh)));
    end
    blk = max(1, min(blk, N));
    mu = zeros(1, d_in);  M2 = zeros(1, d_in);  cnt = 0;
    for i0 = 1:blk:N
        i1 = min(N, i0 + blk - 1);
        Xb = rows(X, i0, i1);
        nb = i1 - i0 + 1;
        mb = mean(Xb, 1);
        M2b = sum((Xb - mb).^2, 1);
        delta = mb - mu;
        tot = cnt + nb;
        M2 = M2 + M2b + delta.^2 * (cnt * nb / tot);
        mu = mu + delta * (nb / tot);
        cnt = tot;
    end
    sd = sqrt(M2 / max(1, cnt));            % population sd, N not N-1
    sd(sd < eps) = 1;

    % One random hidden layer makes a result a property of its seed. The
    % protocol of this group's earlier work draws a number of candidate
    % layers and keeps the best, so that what is reported is the behaviour of
    % the architecture rather than of one draw. Selection is made outside this
    % function, on held-out data: choosing the layer by training error would
    % simply select the one that overfits most comfortably.
    rng(seed);
    W = randn(d_in, Nh) / sqrt(d_in);
    b = randn(1, Nh) * 0.1;

    HtH = zeros(Nh, Nh);
    HtY = zeros(Nh, d_out);
    for i0 = 1:blk:N
        i1 = min(N, i0 + blk - 1);
        Xb = (rows(X, i0, i1) - mu) ./ sd;
        Hb = tanh(Xb * W + b);
        HtH = HtH + (Hb' * Hb);
        HtY = HtY + (Hb' * rows(Y, i0, i1));
    end

    beta = (HtH + lambda * eye(Nh)) \ HtY;

    % HtH AND HtY DO NOT DEPEND ON LAMBDA, and a caller sweeping penalties
    % should not pay for the pass over the samples once per penalty. They are
    % returned so that a grid can accumulate once per width and solve for every
    % penalty, which turns eight passes into one at no cost in memory and no
    % change whatsoever to the numbers.
    acc = struct('HtH', HtH, 'HtY', HtY, 'W', W, 'b', b, 'mu', mu, 'sd', sd);

    % Everything the receiver must hold to run the model, counted honestly:
    % the random input layer, its biases, the fitted output weights, and the
    % standardisation vectors without which the model cannot be applied.
    n_weights = Nh * (d_in + d_out);
    n_meta    = Nh + 2 * d_in;              % b, mu, sd
    out = struct('W', W, 'b', b, 'beta', beta, 'mu', mu, 'sd', sd, ...
        'Nh', Nh, 'lambda', lambda, 'd_in', d_in, 'd_out', d_out, ...
        'n_weights', n_weights, ...
        'n_meta',    n_meta, ...
        'n_stored',  n_weights + n_meta, ...
        'n_fitted',  Nh * d_out, ...
        'batch',     blk, ...
        'acc',       acc);

case 'predict'
    M = varargin{1};  X = varargin{2};
    if numel(varargin) >= 3 && ~isempty(varargin{3})
        sink = varargin{3};     % sink(i0, i1, block), nothing is accumulated
    else
        sink = [];
    end
    N   = size2(X);
    blk = max(1, floor(2e7 / max(1, M.d_in + M.d_out + M.Nh)));
    % Without a sink the full N x d_out answer is materialised, so memory does
    % depend on N here. That is unavoidable if the caller wants every
    % prediction at once; passing a sink lets it consume each block and keeps
    % the footprint bounded by the block, which is what a long test set needs.
    if isempty(sink), out = zeros(N, M.d_out); else, out = []; end
    for i0 = 1:blk:N
        i1 = min(N, i0 + blk - 1);
        Xb = (rows(X, i0, i1) - M.mu) ./ M.sd;
        Yb = tanh(Xb * M.W + M.b) * M.beta;
        if isempty(sink), out(i0:i1, :) = Yb; else, sink(i0, i1, Yb); end
    end

otherwise
    error('hms_elm:mode', 'mode must be ''fit'' or ''predict''.');
end
end

% =========================================================================
% A sample source is either a plain matrix or a struct with fields
%   .n    number of rows
%   .d    number of columns
%   .get  @(i0,i1) -> the rows i0..i1 as a dense block
% The lazy form exists because the sliding-window design matrix is never
% worth materialising: with hourly origins over 500 days, a 24-lag window on
% the raw image volume is 12000 x 24576, about 2.4 GB, and it is nothing but
% overlapping views of a much smaller array.

function [n, d] = size2(S)
if isstruct(S), n = S.n; d = S.d; else, n = size(S,1); d = size(S,2); end
end

function B = rows(S, i0, i1)
if isstruct(S), B = S.get(i0, i1); else, B = S(i0:i1, :); end
end
