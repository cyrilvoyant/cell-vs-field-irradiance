function e = hms_score(se, nk)
%HMS_SCORE The one error convention of this study: pooled over cells and time.
%
%   e = HMS_SCORE(se, nk)
%
%   se : summed squared error, a scalar or a per-origin vector
%   nk : the matching count of scored values
%
%   THE CONVENTION, STATED ONCE AND USED EVERYWHERE. Every scored value carries
%   the same weight, wherever and whenever it was measured:
%
%       e = sqrt( sum of squared errors / number of scored values )
%
%   The average is over the cells of every map, inside the evaluation mask,
%   across the whole period. Nothing is averaged twice.
%
%   WHY THIS FILE EXISTS. The study had TWO conventions and did not know it.
%   The reconstruction campaigns pooled; the forecasting campaigns averaged each
%   forecast origin's own root mean square and then averaged those. On cyclic
%   persistence at three hours, on identical data with an identical mask, the
%   two gave 0.6046 and 0.4263 -- forty-two per cent apart. A reader comparing a
%   reconstruction figure with a forecasting figure was comparing two different
%   quantities, and nothing in the text warned them.
%
%   The difference is Jensen's inequality wearing a disguise. A noon map has
%   three hundred lit cells and large errors; a dawn map has thirty and small
%   ones. Pooling lets noon dominate, which is what happens physically. Taking
%   each origin's own root first gives the dawn map the same weight as the noon
%   map, which answers a different question -- what a user seeing one forecast
%   per hour experiences -- and is a defensible convention that this study does
%   not use.
%
%   POOLED IS CHOSEN because it is the simpler statement, because it is what the
%   reconstruction half already did, and because one convention that needs no
%   explanation beats two that need a paragraph.

e = sqrt(sum(se(:)) / max(sum(nk(:)), 1));
end
