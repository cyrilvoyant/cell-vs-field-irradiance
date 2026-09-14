function c = hms_elm_config(tier, d_in)
%HMS_ELM_CONFIG The width and penalty of every extreme learning machine, once.
%
%   c = HMS_ELM_CONFIG(tier, d_in)
%
%   tier : 'decisive' or 'breadth'
%   d_in : the input width, which selects the family within the decisive tier
%
%   c.Nh, c.lambda, c.tier, c.why
%
%   WHY THIS FILE EXISTS. Five campaigns were carrying four different widths in
%   four different literals -- 2048, 512, 256 and one more -- and nobody had
%   decided that; it happened. The consequence was a positive control running a
%   quarter the width of the campaign it existed to interpret, which is exactly
%   the objection a referee raises and cannot be answered afterwards. One file,
%   one decision, and a campaign that wants a different width has to say which
%   tier it belongs to rather than write a number.
%
%   THERE ARE TWO TIERS AND THE SECOND IS NOT A COMPROMISE, IT IS AN ARITHMETIC
%   FACT. The decisive comparisons run at the width the grid search froze. The
%   breadth studies -- thirteen climates, three aggregation steps -- cannot: the
%   quarter-hourly arm already costs sixteen times its hourly counterpart, and
%   multiplying the width by eight on top of that is more than five days for one
%   study. So they run narrower, at ONE width held identical across every cell
%   of the study, because what varies there is the domain or the step and never
%   the machine.
%
%   AND THE RULE THAT MAKES THAT HONEST: no number from the breadth tier is ever
%   compared with a number from the decisive tier. The breadth studies answer
%   "does the ordering survive", which is a question asked inside one table.
%   Putting the two tiers on one axis was tried once, in the parameter figure,
%   and had to be undone: it compared representations and capacities at the same
%   time and invited exactly the reading the figure exists to prevent.
%
%   THE DECISIVE WIDTH IS AT THE EDGE OF ITS SWEEP AND THE PAPER SAYS SO. The
%   grid runs 16 to 2048 and the error falls monotonically throughout; 2048 is
%   the largest affordable width, not an optimum. The per-pixel family is close
%   to flat there -- doubling from 1024 buys 1.7% -- and the whole-field family
%   is not, buying 17%. That asymmetry matters when reading the comparison: the
%   arm that wins is the one nearer its ceiling.

if nargin < 1 || isempty(tier), tier = 'decisive'; end
if nargin < 2, d_in = []; end

switch lower(tier)
case 'decisive'
    f = fullfile(pwd, 'results', 'hyper_frozen.mat');
    assert(isfile(f), 'hms_elm_config:noFreeze', ...
        ['No frozen hyper-parameters at %s. Run HMS_GRIDSEARCH then HMS_FREEZE ' ...
         'before any decisive campaign: a width chosen at the call site is a ' ...
         'width nobody chose.'], f);
    q = load(f);  H = q.H;
    assert(~isempty(d_in), 'hms_elm_config:noWidth', ...
        'The decisive tier selects its family from the input width.');
    if d_in <= 64, fam = 'siso'; else, fam = 'mimo'; end
    c = struct('Nh', H.(fam).Nh, 'lambda', H.(fam).lambda, ...
               'tier', 'decisive', 'family', fam, ...
               'why', sprintf(['frozen by hms_freeze on the %s family; the ' ...
                               'sweep is monotone to its edge at Nh = %d'], ...
                              fam, H.(fam).Nh));

case 'breadth'
    % ONE WIDTH FOR EVERY BREADTH STUDY, so that the thirteen climates and the
    % three aggregation steps are read on the same machine even though they are
    % different tables. It is the largest width at which the quarter-hourly arm
    % fits in a weekend, and it was chosen for that and stated.
    c = struct('Nh', 256, 'lambda', 1e-8, 'tier', 'breadth', 'family', 'siso', ...
               'why', ['the widest machine at which the quarter-hourly arm of ' ...
                       'the time-step study fits in a weekend']);

otherwise
    error('hms_elm_config:tier', ...
        'Unknown tier ''%s''; it is ''decisive'' or ''breadth''.', tier);
end
end
