function origins = hms_codex_origins(bounds, L, H, stride)
%HMS_CODEX_ORIGINS Strict split-local histories and targets; no cross-split use.
% bounds=[first,last] are inclusive indices of one chronological split.
% Origin t uses history t-L+1:t and predicts t+1:t+H.
% This intentionally excludes history from preceding splits. If a persistence
% reference needs more than L steps, supply max(L,reference_history).
if nargin < 4, stride = 1; end
validateattributes(bounds, {'numeric'}, {'vector','numel',2,'integer','positive','finite'});
validateattributes(L, {'numeric'}, {'scalar','integer','positive','finite'});
validateattributes(H, {'numeric'}, {'scalar','integer','positive','finite'});
validateattributes(stride, {'numeric'}, {'scalar','integer','positive','finite'});
assert(bounds(2)>=bounds(1), 'hms_codex_origins:bounds','Split bounds must be ordered.');
origins = (bounds(1)+L-1):stride:(bounds(2)-H);
assert(all(origins-L+1>=bounds(1)) && all(origins+H<=bounds(2)), ...
    'hms_codex_origins:leak','A window crossed a declared split boundary.');
end
