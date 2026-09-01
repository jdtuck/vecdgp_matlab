function E = vdgp_U_entries(A, theta, g, v, cov_type)
%VDGP_U_ENTRIES  Non-zero entries of the Vecchia inverse-Cholesky factor U.
%
%   E = VDGP_U_ENTRIES(A, THETA, G, V, COV_TYPE) returns a struct with
%
%      E.I, E.J, E.V  triplets of the sparse upper-triangular factor U
%      E.diagU        n-vector of diagonal entries
%      E.n            dimension
%
%   such that  Q = U * U'  approximates inv(Sigma) at unit scale, in the
%   ordering carried by A.
%
%   For observation i with conditioning set c(i),
%
%        b_i = Sigma(c(i),c(i))^{-1} Sigma(c(i), i)
%        d_i = Sigma(i,i) - Sigma(i,c(i)) b_i
%        U(i,i) = 1/sqrt(d_i),   U(c(i),i) = -b_i / sqrt(d_i)
%
%   evaluated in the stable form "reverse the set so the target is last,
%   Cholesky, take the last row of the inverse factor" (VDGP_BATCH_LAST_ROW_INV).
%
%   Two structural shortcuts keep this fast:
%     * the leading (m+1)-by-(m+1) block is exact -- observations 1..m+1
%       condition on ALL their predecessors -- so it is obtained from a single
%       dense Cholesky rather than m separate small ones;
%     * every remaining row has exactly m+1 entries, so all of them are
%       processed by one batched, fully vectorised Cholesky.
%
%   Cost: O(n m^3) flops, O(n m) memory.

persistent HAVE_MEX
if isempty(HAVE_MEX)
    HAVE_MEX = (exist('vdgp_U_entries_mex', 'file') == 3);
end

if nargin < 5 || isempty(cov_type), cov_type = 'matern'; end
if nargin < 4 || isempty(v), v = 2.5; end

n = A.n;
if isscalar(g), gv = repmat(g, n, 1); else, gv = g(:); end

if ~A.vecchia
    K = vdgp_cov(A.X_ord, [], theta, gv, 1, v, cov_type);
    L = chol(K, 'lower');
    U = inv(L).';
    [I, J, V] = find(triu(U));
    E = struct('I', I, 'J', J, 'V', V, 'diagU', diag(U), 'n', n);
    return
end

% ---------------- optional MEX fast path ---------------------------------
% Identical arithmetic, but each row's (m+1)-by-(m+1) block stays in cache and
% the loop over rows is OpenMP-parallel.  Build it with VDGP_BUILD_MEX.
if HAVE_MEX
    if isscalar(theta)
        th = repmat(theta, 1, A.d);
    else
        th = reshape(theta, 1, []);
    end
    if strcmpi(cov_type, 'exp2'), vc = 999; else, vc = v; end
    [I, J, V, diagU] = vdgp_U_entries_mex(A.X_ord, A.NNarray, gv, th, vc);
    E = struct('I', I, 'J', J, 'V', V, 'diagU', diagU, 'n', n);
    return
end

m   = A.m;
nb  = min(n, m + 1);                 % size of the exact leading block

% ---------------- leading exact block ------------------------------------
K0 = vdgp_cov(A.X_ord(1:nb, :), [], theta, gv(1:nb), 1, v, cov_type);
[L0, pf] = chol(K0, 'lower');
if pf ~= 0
    K0 = K0 + (1e-10 * mean(diag(K0))) * eye(nb);
    L0 = chol(K0, 'lower');
end
U0 = inv(L0).';
U0 = triu(U0);
[I0, J0, V0] = find(U0);
d0 = diag(U0);

if n <= m + 1
    E = struct('I', I0, 'J', J0, 'V', V0, 'diagU', d0, 'n', n);
    return
end

% ---------------- remaining rows, one batched group ----------------------
rows = (m + 2):n;
R    = numel(rows);
k    = m + 1;
NN   = A.NNarray(rows, 1:k);
ridx = fliplr(NN);                    % target LAST

I = zeros(R * k, 1);
J = zeros(R * k, 1);
V = zeros(R * k, 1);
dg = zeros(R, 1);
ptr = 0;

chunk = max(1, floor(1.2e7 / (k * k)));
for a = 1:chunk:R
    b  = min(R, a + chunk - 1);
    Rb = b - a + 1;
    rid = ridx(a:b, :);

    P = reshape(A.X_ord(rid(:), :), Rb, k, A.d);
    P = permute(P, [2 3 1]);

    K = vdgp_batch_cov(P, theta, v, cov_type);
    for j = 1:k
        K(j, j, :) = K(j, j, :) + reshape(gv(rid(:, j)), 1, 1, Rb);
    end

    L  = vdgp_batch_chol(K);
    av = vdgp_batch_last_row_inv(L);
    A2 = reshape(av, k, Rb);

    np = Rb * k;
    I(ptr+1:ptr+np) = rid(:);
    J(ptr+1:ptr+np) = repmat(rows(a:b).', k, 1);
    V(ptr+1:ptr+np) = reshape(A2.', np, 1);
    ptr = ptr + np;
    dg(a:b) = A2(k, :).';
end

diagU = zeros(n, 1);
diagU(1:nb) = d0;
diagU(rows)  = dg;

E = struct('I', [I0; I(1:ptr)], 'J', [J0; J(1:ptr)], 'V', [V0; V(1:ptr)], ...
           'diagU', diagU, 'n', n);
end
