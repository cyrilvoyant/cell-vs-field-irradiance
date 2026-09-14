function make_basic_2years(src, dst)
%MAKE_BASIC_2YEARS Keep the first two years of the HelioClim-3 archive.
%
%   make_basic_2years(src, dst)
%
%   src : the author's Basic folder, holding GHI_HC3.mat (8 years, 70080 hours)
%   dst : the Basic folder of this repository
%
%   The benchmark trains on days 1-365 and tests on days 366-730, so only the
%   first 730 days, 17520 hours, are ever read. The cell array is cut, not
%   changed: cell k of the copy is cell k of the original.

H = 730 * 24;
q = load(fullfile(src, 'GHI_HC3.mat'), 'GHI_HC3');
GHI_HC3 = q.GHI_HC3(1, 1:H);
save(fullfile(dst, 'GHI_HC3.mat'), 'GHI_HC3', '-v7');

r = load(fullfile(dst, 'GHI_HC3.mat'), 'GHI_HC3');
same = isequaln(r.GHI_HC3, q.GHI_HC3(1, 1:H));      % isequaln: missing values are NaN
fprintf('kept %d of %d hours, %d points per hour, class %s, identical: %d\n', ...
        numel(GHI_HC3), numel(q.GHI_HC3), numel(GHI_HC3{1}), class(GHI_HC3{1}), same);
end
