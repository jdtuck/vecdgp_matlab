function A = vdgp_reapprox(A, X, ord)
%VDGP_REAPPROX  Rebuild ordering and conditioning sets in the current space.
%
%   A = VDGP_REAPPROX(A, X)      re-draws a random ordering and recomputes the
%                                nearest-neighbour conditioning sets for the
%                                (warped) design X.
%   A = VDGP_REAPPROX(A, X, ORD) uses the supplied ordering.
%
%   This implements the "updating the conditioning sets in the warped space"
%   strategy of Section 4 of Sauer, Cooper & Gramacy (2023), i.e.
%   deepgp's continue(..., re_approx = TRUE).

if ~A.vecchia
    A.X_ord = X(A.ord, :);
    return
end
if nargin < 3 || isempty(ord), ord = 'random'; end
A = vdgp_create_approx(X, A.m, ord, true);
end
