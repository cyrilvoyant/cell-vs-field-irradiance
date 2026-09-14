function R = hms_push_row(R, row)
%HMS_PUSH_ROW Append one result to a campaign, whatever order its fields came in.
%
%   R = HMS_PUSH_ROW(R, row)
%
%   WHY THIS EXISTS. Appending to a struct array with R(end+1) = struct(...)
%   requires the new struct's fields to match the array's not only in NAME but
%   in ORDER. When they do not, MATLAB raises "subscripted assignment between
%   dissimilar structures" -- and it raises it at the moment of the append,
%   which in this study is after the arm has been fitted. Adding a metric field
%   to a campaign therefore cost one aborted run of the Corsica bench and would
%   have cost four hours of LSTM training, in both cases for a mistake that is
%   invisible on reading: the declaration and the assignment sit two hundred
%   lines apart and both look right.
%
%   WHAT IT DOES. If the field SETS agree, the row is reordered to the array's
%   order and appended. If they differ, the error says WHICH fields are missing
%   and which are unexpected, immediately, instead of naming neither. An empty
%   array with declared fields fixes the order for everything that follows; an
%   array with no fields at all takes the first row's order as canonical.
%
%   This is a guard, not a convenience. A campaign that runs for eleven hours
%   should not be able to fail on the shape of its own bookkeeping.

if isempty(R) && isempty(fieldnames(R))
    R = row;  return
end

want = fieldnames(R);
got  = fieldnames(row);
if ~isequal(sort(want), sort(got))
    missing = setdiff(want, got);
    extra   = setdiff(got, want);
    error('hms_push_row:fields', ...
        ['The row does not carry the campaign''s fields.\n' ...
         '  missing from the row : %s\n' ...
         '  not in the campaign  : %s\n' ...
         'Add the field to the struct that declares R, or stop assigning it.'], ...
        strjoin(cellstr(missing), ', '), strjoin(cellstr(extra), ', '));
end

R(end+1) = orderfields(row, want);
end
