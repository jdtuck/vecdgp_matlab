function A = vdgp_create_approx(X, m, ord, vecchia)
%VDGP_CREATE_APPROX  Build (or fake) the Vecchia approximation object.
%
%   A = VDGP_CREATE_APPROX(X, M, ORD, VECCHIA)
%
%   X        n-by-d design
%   M        conditioning-set size
%   ORD      [] for a random ordering, a string accepted by VDGP_ORDER, or an
%            explicit permutation of 1:n
%   VECCHIA  true  -> sparse Vecchia object (ordering + nearest neighbours)
%            false -> "exact" object; the same interface is used downstream so
%                     the fitting code never branches
%
%   Fields
%     .vecchia  logical
%     .n .d .m
%     .ord      1-by-n permutation; the ordered design is X(ord,:)
%     .rev      inverse permutation, so v(rev) puts an ordered vector back into
%               the original ordering
%     .X_ord    the ordered design (updated in place during MCMC when the
%               layer is latent -- see VDGP_UPDATE_APPROX)
%     .NNarray  n-by-(m+1) conditioning sets, NaN padded (Vecchia only)
%
%   Note: as in deepgp, ORD and NNarray are computed once from the layer's
%   *initial* inputs and then held fixed while the latent inputs move; call
%   VDGP_REAPPROX to rebuild them in the current warped space.

if nargin < 4 || isempty(vecchia), vecchia = true; end
if nargin < 3, ord = []; end

if ~ismatrix(X), error('vdgp_create_approx:X', 'X must be a matrix.'); end
[n, d] = size(X);

A = struct();
A.vecchia = logical(vecchia);
A.n = n;
A.d = d;

if isempty(ord), ord = 'random'; end
A.ord = vdgp_order(X, ord);
[~, A.rev] = sort(A.ord);
A.X_ord = X(A.ord, :);

if ~A.vecchia
    A.m = n - 1;
    A.NNarray = [];
    return
end

if isempty(m), m = min(25, n - 1); end
m = min(m, n - 1);
A.m = m;
A.NNarray = vdgp_ordered_nn(A.X_ord, m);
end
