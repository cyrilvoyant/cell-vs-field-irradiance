function [Dn, Up] = hms_binops(ns, nr, phase)
%HMS_BINOPS A coarser detector, as a pair of matrices.
%
%   [Dn, Up] = HMS_BINOPS(ns, nr, phase)
%
%   Dn maps a profile of ns fine offset bins onto nr coarse ones by AVERAGING
%   the fine bins each coarse bin covers, weighting partial overlaps by their
%   length. That is what a detector of that width measures: it integrates over
%   its strip. Subsampling would model nothing physical and would throw away
%   information a real instrument keeps.
%
%   Up interpolates the coarse bin centres back onto the fine offset axis, so
%   whatever inverse follows -- filtered back projection or a least-squares
%   solve -- is unchanged and the comparison isolates the encoder.
%
%   PHASE slides the whole partition, in units of a coarse bin. It is not a
%   detail: moving it by half a bin changes the reconstruction error by as much
%   as changing the number of bins does, because the coarse detector aliases
%   against the fine offset axis. Any result quoted at one phase is quoting an
%   alignment as if it were a resolution.
%
%   At nr = ns and phase 0 both matrices are exactly the identity, which is the
%   assertion HMS_DETECTOR_CHECK makes before anything else.
%
%   This lived as a private copy inside two files that then disagreed about
%   what a coarse detector was. One definition, one file.

if nargin < 3 || isempty(phase), phase = 0; end
w = ns / nr;
edges = 0.5 + (-phase + (0:nr))*w;

Dn = zeros(nr, ns);
for k = 1:nr
    for j = 1:ns
        Dn(k,j) = max(0, min(edges(k+1), j+0.5) - max(edges(k), j-0.5));
    end
    if sum(Dn(k,:)) > 0, Dn(k,:) = Dn(k,:) / sum(Dn(k,:)); end
end

ctr = (edges(1:end-1) + edges(2:end)) / 2;
Up = zeros(ns, nr);
for j = 1:ns
    Up(j,:) = interp1(ctr, eye(nr), min(max(j, ctr(1)), ctr(end)), 'linear');
end
end
