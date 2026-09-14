function t = hms_epoch(days, cfg)
%HMS_EPOCH UTC seconds at the middle of every hour of the given days.
%
%   t = HMS_EPOCH(days, cfg)
%
%   The reduced calendar holds 365 days a year with the leap days removed, so a
%   day index is counted against that calendar and not against a real one. Every
%   solar-geometry call in this study needs the same conversion, and six files
%   were carrying their own private copy of it under the name EPOCH_OF. Six
%   copies of a calendar convention is six chances to drift apart, so the
%   convention now lives in one file. The body is the one HMS_FORECAST_BENCH used
%   and the outputs are identical to the byte.

d0 = datetime(cfg.absolute_start_date, 'InputFormat', 'yyyy-MM-dd', ...
              'TimeZone', 'UTC');
y0 = year(d0);  all_days = NaT(0, 1, 'TimeZone', 'UTC');
while numel(all_days) < cfg.n_days
    dd = (datetime(y0,1,1,'TimeZone','UTC'):datetime(y0,12,31,'TimeZone','UTC')).';
    dd = dd(~(month(dd) == 2 & day(dd) == 29));
    all_days = [all_days; dd]; %#ok<AGROW>
    y0 = y0 + 1;
end
sel = all_days(days(:));  H = cfg.hours_per_day;
t = posixtime(repelem(sel, 1, H)) + repmat((0:H-1)*3600 + 1800, numel(sel), 1);
t = reshape(t.', 1, []);
end
