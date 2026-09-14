function ok = hms_gamma_check()
%HMS_GAMMA_CHECK Verify the gamma index against cases with a closed-form answer.
%
%   ok = HMS_GAMMA_CHECK()
%
%   WHY THIS FILE EXISTS. The gamma index of this study was found to disagree
%   with the framework it cites in two ways at once: its intensity tolerance was
%   relative to the local observation where the framework defines an absolute
%   one in W/m2, and the absolute option in its own interface was read, stored,
%   reported, and never used -- so passing a tolerance in watts changed nothing
%   and two different conventions returned identical numbers. That coincidence
%   is what exposed it. A criterion with one free parameter and a neighbourhood
%   search is exactly the kind of code that can be wrong for weeks while
%   producing plausible output, so it is checked against answers that can be
%   worked out on paper.
%
%   THE FRAMEWORK, as published:
%
%       gamma(x,y,t) = min sqrt( (dx^2 + dy^2)/DTA^2
%                              + dt^2/TTA^2
%                              + (pred - obs)^2/IDT^2 )
%
%   with DTA in kilometres, TTA in minutes and IDT in W/m2, all absolute, and a
%   search extending to 3/2 of each tolerance. A cell passes when gamma <= 1.
%
%   EVERY CASE BELOW HAS AN ANSWER THAT FOLLOWS FROM THAT LINE ALONE. A uniform
%   field is used wherever a neighbourhood could otherwise rescue a point: on a
%   constant field no displacement improves the intensity term, so the minimum
%   is attained at zero displacement and gamma is the intensity ratio exactly.
%   That is what makes the expected values exact rather than approximate.

n = 16;  nt = 20;  M = true(n,n,nt);
DTA_PX = 1;  RES = 3.4;  DT = 60;  IDT = 50;
A = repmat(400*ones(n), 1, 1, nt);          % uniform: no neighbour helps
o = struct('res', RES, 'dt', DT, 'mask', M, 'idt', IDT);

G0 = hms_gamma(A, A, o);
fprintf('\n=== gamma against the published formula ===\n');
fprintf('DTA = %g km (1 pixel), TTA = %g min, IDT = %g W/m2, step %g min\n\n', ...
        RES*DTA_PX, G0.opt.tta, IDT, DT);

T = {};
T = chk(T, 'a perfect forecast', @() hms_gamma(A, A, o), 0, 100);
Z = zeros(size(A));
T = chk(T, 'a perfect zero forecast', @() hms_gamma(Z, Z, o), 0, 100);
T = chk(T, 'error equal to IDT', @() hms_gamma(A+IDT, A, o), 1, 100);
T = chk(T, 'error of twice IDT', @() hms_gamma(A+2*IDT, A, o), 2, 0);
T = chk(T, 'error of half IDT', @() hms_gamma(A+IDT/2, A, o), 0.5, 100);
T = chk(T, 'a negative error of IDT', @() hms_gamma(A-IDT, A, o), 1, 100);

% the option must act: the same error against a doubled tolerance halves gamma
T = chk(T, 'opt.idt is honoured', ...
        @() hms_gamma(A+IDT, A, setfield(o,'idt',2*IDT)), 0.5, 100); %#ok<SFLD>

% a displacement of exactly one tolerance, with the value matching there
B = A;  B(:,:,1:end) = A;                    % values identical everywhere
G = hms_gamma(circshift(B,[1 0 0]), A, o);
T = chk(T, 'a one-pixel shift of an identical field', @() G, 0, 100);

% a field that varies in one direction only, shifted by one pixel: the
% intensity term at zero displacement is the gradient, and the search may
% instead pay one DTA and match exactly. gamma is then min(grad/IDT, 1).
% THE WRAP ROW IS EXCLUDED, and that is the point of the case rather than a
% convenience. CIRCSHIFT carries the top of the ramp round to the bottom, so
% one row of the sixteen sees a jump of fifteen gradients instead of one; it
% is capped at gamma 1 by paying a full DTA to find a match, which is correct
% behaviour and would make the closed-form expectation for the OTHER fifteen
% rows unreadable. Masking it out leaves a case whose answer is exactly the
% gradient over the tolerance.
grad = 25;                                   % W/m2 per pixel, half of IDT
C = repmat(400 + grad*(1:n).', 1, n, nt);
Mw = M;  Mw(1,:,:) = false;                  % the row circshift wraps into
G = hms_gamma(circshift(C,[1 0 0]), C, setfield(o,'mask',Mw)); %#ok<SFLD>
T = chk(T, 'a one-pixel shift of a ramp of IDT/2 per pixel', @() G, ...
        grad/IDT, 100);

% THE CASE THAT MOTIVATED REMOVING THE TEMPORAL SEARCH, kept so that nobody
% reintroduces it without seeing what it does. Simple persistence one step
% ahead predicts the current field for the next instant. If the criterion is
% allowed to search one step in time it finds that field exactly, at the cost
% of the temporal term alone, and a forecast wrong by a full tolerance is
% scored as if it were perfect. The field below moves by exactly IDT per step,
% so persistence is wrong by exactly one tolerance everywhere: gamma must be 1
% and the pass rate must NOT be a hundred per cent for the wrong reason.
nt2 = 12;
E = zeros(n, n, nt2);
% TWO tolerances per step, not one: at one tolerance the case passes either
% way and proves nothing. At two, persistence is wrong by 2*IDT, so gamma is
% 2 without a temporal search and 1 with one -- the difference the removal
% of the temporal term is meant to make.
for k = 1:nt2, E(:,:,k) = 400 + (k-1)*2*IDT; end
persist = E(:,:,1:nt2-1);                          % the previous frame
truth   = E(:,:,2:nt2);                            % what actually happens
Mp = true(n, n, nt2-1);
op = struct('res', RES, 'dt', DT, 'mask', Mp, 'idt', IDT);
T = chk(T, 'persistence, no temporal search', ...
        @() hms_gamma(persist, truth, op), 2, 0);
Gt = hms_gamma(persist, truth, setfield(op, 'tta_steps', 1)); %#ok<SFLD>
fprintf(['  %-46s gamma %7.3f  GPR %5.1f%%  <- what a one-step temporal\n' ...
         '  %-46s              tolerance would have given\n'], ...
        'the same, with a one-step temporal tolerance', Gt.mean_local, Gt.gpr, '');

% the search reaches 3/2 of a tolerance and no further
D = repmat(400*ones(n), 1, 1, nt);
D2 = D;  D2(:) = 400 + 10*IDT;               % far in value everywhere
G = hms_gamma(D2, D, o);
fprintf('  %-46s gamma %7.3f  (a value ten tolerances away cannot be rescued)\n', ...
        'a value far outside every tolerance', G.mean_local);

ok = all(cellfun(@(t) t.pass, T));
fprintf('\n%d/%d closed-form cases agree with the formula\n', ...
        nnz(cellfun(@(t) t.pass, T)), numel(T));
if ~ok
    fprintf('AT LEAST ONE CASE DISAGREES. The criterion is not the published one.\n');
end
end

% ---------------------------------------------------------------------------
function T = chk(T, name, fn, g_exp, gpr_exp)
G = fn();
dg = abs(G.mean_local - g_exp);
dp = abs(G.gpr - gpr_exp);
pass = dg < 1e-6 && dp < 1e-6;
T{end+1} = struct('name', name, 'pass', pass); %#ok<AGROW>
fprintf('  [%s] %-42s gamma %7.4f (%.4f)   GPR %6.1f%% (%.1f)\n', ...
        tern(pass,'PASS','FAIL'), name, G.mean_local, g_exp, G.gpr, gpr_exp);
end

function v = tern(c, a, b)
if c, v = a; else, v = b; end
end
