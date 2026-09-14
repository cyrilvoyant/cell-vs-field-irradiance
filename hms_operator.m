function op = hms_operator(kind, n, cfg, param)
%HMS_OPERATOR Build a linear measurement operator and its EXACT adjoint.
%
%   op = HMS_OPERATOR(kind, n, cfg, param)
%
%   kind  : 'radon' or 'randproj'
%   n     : grid side
%   param : number of angles (radon) or number of measurements (randproj)
%
%   op.A   : function handle, n x n image -> measurements
%   op.At  : function handle, measurements -> n x n image, the true adjoint
%   op.m   : number of measurements per image
%
%   WHY THE MATRIX IS BUILT EXPLICITLY FOR THE RADON OPERATOR.
%   A previous version used MATLAB's unfiltered iradon as the adjoint, with a
%   single scalar calibrated once by the dot-product identity. The self-test
%   rejected it: the identity <Ax,y> = <x,At y> was off by 20 %, not by a
%   scale factor. Unfiltered backprojection is not proportional to the
%   adjoint of radon, for two independent reasons. iradon zeroes everything
%   outside the inscribed circle while radon integrates over the whole
%   square, corners included, so the two operators do not even act between
%   the same spaces; and the interpolation used when smearing a projection
%   back is not the transpose of the interpolation used when sampling it.
%   No scalar repairs either. Since the regularised inversion solves the
%   normal equations, an adjoint that is merely close turns it into the
%   solution of a different, unstated problem, and a sweep over Na would then
%   move the effective regularisation weight without anyone noticing.
%
%   The matrix is therefore assembled column by column, R(:,j) = radon(e_j),
%   so At is R' and the identity holds to machine precision by construction.
%   A single pixel projects onto about two detector bins per angle, so R is
%   very sparse: at n = 64, Na = 180 it holds roughly 1.5e6 non-zeros, some
%   tens of megabytes. This is affordable at the grid sizes used for the
%   regularised inversion and is cached across calls.
%
%   IT DOES NOT SCALE, AND THAT IS STATED RATHER THAN HIDDEN. Assembly costs
%   n^2 forward transforms, so large grids must use the matrix-free filtered
%   backprojection of HMS_DECODE instead, which is fast but is NOT the adjoint
%   and is never used as one.

persistent RCACHE

if nargin < 3 || isempty(cfg), cfg = hms_config('n', n); end

switch lower(kind)

case 'radon'
    Na    = param;
    theta = (0:Na-1) * (180/Na);
    ns    = size(radon(zeros(n), 0), 1);

    key = sprintf('radon|%d|%d', n, Na);
    if isempty(RCACHE) || ~isfield(RCACHE, 'key') || ~strcmp(RCACHE.key, key)
        lim = 4e8;
        assert(n*n <= 1e5, 'hms_operator:tooLarge', ...
            ['Explicit assembly needs %d forward transforms; use the ' ...
             'matrix-free FBP of hms_decode for grids this large.'], n*n);
        I = cell(n*n,1); J = cell(n*n,1); V = cell(n*n,1);
        e = zeros(n);
        for j = 1:n*n
            e(j) = 1;
            c    = radon(e, theta);
            e(j) = 0;
            nz   = find(abs(c) > 1e-12);
            I{j} = nz;
            J{j} = repmat(j, numel(nz), 1);
            V{j} = c(nz);
        end
        I = vertcat(I{:}); J = vertcat(J{:}); V = vertcat(V{:});
        assert(numel(V) < lim, 'hms_operator:tooManyNonzeros', ...
            'Sparse operator would hold %d non-zeros.', numel(V));
        RCACHE = struct('key', key, 'R', sparse(I, J, V, ns*Na, n*n));
    end
    R = RCACHE.R;

    op = struct( ...
        'name',  'radon', ...
        'param', Na, ...
        'theta', theta, ...
        'ns',    ns, ...
        'm',     ns*Na, ...
        'R',     R, ...
        'A',     @(X) reshape(R  * X(:), ns, Na), ...
        'At',    @(S) reshape(R' * S(:), n,  n));

case 'randproj'
    k  = param;
    st = rng;  rng(cfg.seed);
    R  = randn(k, n*n) / sqrt(k);
    rng(st);                              % never perturb the caller's stream
    op = struct( ...
        'name',  'randproj', ...
        'param', k, ...
        'theta', [], ...
        'ns',    k, ...
        'm',     k, ...
        'R',     R, ...
        'A',     @(X) R  * X(:), ...
        'At',    @(y) reshape(R' * y, n, n));

otherwise
    error('hms_operator:unknownKind', 'Unknown operator "%s".', kind);
end

op.n = n;
end
