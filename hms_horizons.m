function h = hms_horizons()
%HMS_HORIZONS The horizons this paper reports, in one place.
%
%   h = HMS_HORIZONS()
%
%   Every table and every figure that breaks a score down by lead time uses
%   this list, so that a reader comparing two of them is comparing the same
%   columns. It is a subset of the twenty-four the models actually produce:
%   consecutive hours are strongly correlated, twenty-four columns do not fit
%   in a table, and a metric like the gamma index costs a minute per horizon
%   per arm, so reporting all of them would buy resolution nobody reads at a
%   price that is paid every time the campaign is rerun.
%
%   THE CHOICE IS GEOMETRIC, NOT DECORATIVE. One, two and three hours are the
%   range where a persistence-type reference is hardest to beat; six and twelve
%   cross the middle of a day; twenty-four returns to the same hour of the
%   clock, where the diurnal cycle makes the problem easy again for exactly the
%   wrong reason. A method has to be shown at all three regimes or the number
%   quoted is the regime that flattered it.

h = [1 2 3 6 12 24];
end
