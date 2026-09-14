function out = hms_resid(action, varargin)
%HMS_RESID Keep the per-origin errors the pooled score throws away.
%
%   hms_resid('on', dir)        start capturing, into this directory
%   hms_resid('off')            stop capturing
%   hms_resid('add', name, P, Y, K, d, H)   capture one arm
%   S = hms_resid('load', dir, name)        read one arm back
%
%   WHY THIS COSTS NOTHING TO COMPUTE. The pooled scorer already forms
%   sum((P-Y).^2 .* K, 2), which is one number PER ORIGIN, and then collapses
%   it to a single scalar in the same expression. The vector exists for a
%   microsecond and is discarded. This keeps it. No model is refitted, no
%   metric is recomputed, and the pooled number the campaign reports is the
%   pooled number it would have reported anyway.
%
%   WHY IT IS NEEDED AT ALL. The main comparison of this study, the per-cell
%   route against every arm that reads space, is reported as one number per arm
%   with no interval and no test. The method to test it was designed on 1
%   September -- pair on non-overlapping daily origins, n around 365 over the
%   test year -- and was applied only to the thirteen-domain question. It could
%   not be applied here because the archives kept summaries and the predictions
%   were deleted after scoring.
%
%   CAPTURE IS OFF BY DEFAULT AND CHANGES NOTHING WHEN OFF. A campaign that
%   does not switch it on runs the identical code path it ran before, which is
%   the property that lets the re-run be checked against the published numbers.

persistent DIR ORIG
out = [];

switch lower(action)
    case 'origins'
        % The test-origin vector, stored so each captured arm knows which
        % instant each of its rows belongs to. Without it the paired test
        % cannot space origins a day apart and would have to assume the rows
        % are consecutive hours, which is an assumption and not a fact.
        ORIG = varargin{1}(:).';

    case 'on'
        DIR = varargin{1};
        if ~isfolder(DIR), mkdir(DIR); end
        % A CAPTURE DIRECTORY THAT ALREADY HOLDS FILES CANNOT BE TRUSTED. The
        % campaign was once started and killed a minute later; whatever it had
        % already written would sit here looking exactly like a fresh result,
        % and a later run producing only three arms would silently inherit the
        % fourth from the aborted attempt. Refusing is cheap; a mixed capture is
        % undetectable afterwards.
        old = dir(fullfile(DIR, '*.mat'));
        assert(isempty(old), 'hms_resid:dirty', ...
            ['%s already holds %d captured file(s). Move or delete them: a ' ...
             'capture must not be able to mix two runs.'], DIR, numel(old));
        fprintf('hms_resid: capturing per-origin errors into %s\n', DIR);

    case 'off'
        DIR = [];

    case 'add'
        if isempty(DIR), return; end          % off: do nothing, cost nothing
        [name, P, Y, K, d, H] = varargin{1:6};
        no = size(P, 1);

        % ALL H HORIZONS, NOT THE SIX THAT GET REPORTED. The campaign's headline
        % number is pooled over every one of the H blocks, so a capture holding
        % only the reported six could never reproduce it: the check would
        % compare a six-block pool against a twenty-four-block published value
        % and fail for a reason that has nothing to do with the models. Codex
        % caught this before it cost four hours.
        sse_all = nan(no, H);
        nk_all  = nan(no, H);
        for h = 1:H
            c = (h-1)*d + 1 : h*d;            % this horizon's block of columns
            D = P(:, c) - Y(:, c);
            k = K(:, c);
            sse_all(:, h) = sum((D.^2) .* k, 2);   % ONE VALUE PER ORIGIN
            nk_all(:, h)  = sum(k, 2);
        end

        % the reported subset, kept as a convenience view of the same numbers
        HS  = hms_horizons();
        sse = sse_all(:, HS);
        nk  = nk_all(:, HS);
        assert(~isempty(ORIG) && numel(ORIG) == no, 'hms_resid:origins', ...
            ['Capture is on but the origin vector is missing or has %d entries ' ...
             'for %d rows. Call hms_resid(''origins'', ote) where ote is built.'], ...
            numel(ORIG), no);
        S = struct('name', name, 'horizons', HS, 'sse', sse, 'nk', nk, ...
                   'H', H, 'sse_all', sse_all, 'nk_all', nk_all, ...
                   'n_origin', no, 'origin', ORIG, ...
                   'note', ['per-origin summed squared error and scored ' ...
                            'count; pool with HMS_SCORE to recover the ' ...
                            'campaign number exactly'], ...
                   'computed', datestr(now, 'yyyy-mm-dd HH:MM'));
        f = fullfile(DIR, [matlab.lang.makeValidName(name) '.mat']);
        save(f, '-struct', 'S', '-v7.3');
        fprintf('    hms_resid: %d origins x %d horizons -> %s\n', ...
                no, numel(HS), f);

    case 'load'
        [dirn, name] = varargin{1:2};
        f = fullfile(dirn, [matlab.lang.makeValidName(name) '.mat']);
        assert(isfile(f), 'hms_resid:missing', 'No captured residuals at %s.', f);
        out = load(f);

    otherwise
        error('hms_resid:action', 'Unknown action ''%s''.', action);
end
end
