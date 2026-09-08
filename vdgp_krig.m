function out = vdgp_krig(y, X, Xnew, theta, g, tau2, opts, mode, m, NN, nsamp, Ajoint)
%VDGP_KRIG  Gaussian process prediction, with or without Vecchia.
%
%   OUT = VDGP_KRIG(Y, X, XNEW, THETA, G, TAU2, OPTS, MODE, M, NN, NSAMP)
%
%   MODE
%     'mean'   posterior mean only (used for the latent layers, which are
%              propagated through the network by their kriging mean, exactly
%              as in deepgp)
%     'lite'   posterior mean and pointwise variance
%     'joint'  posterior mean and full n_new-by-n_new covariance
%     'sample' posterior mean and NSAMP joint draws, WITHOUT ever forming the
%              covariance.  Under Vecchia a draw is
%
%                  f = mu + sqrt(tau2) * U22^{-T} z ,   z ~ N(0, I)
%
%              because Cov = tau2 * U22^{-T} U22^{-1}; that is one sparse
%              triangular solve, O(n_new m), per batch of draws.  'joint' also
%              returns draws when NSAMP > 0.
%
%   With OPTS.vecchia = true the prediction is the Vecchia predictor of
%   Sauer, Cooper & Gramacy (2023):
%
%     * 'mean'/'lite' condition each new location on its M nearest observed
%       locations only -- the "independent" scheme of Section 3.3.  All
%       n_new small solves are batched, so the cost is O(n_new m^3).
%
%     * 'joint' appends the new locations AFTER the observed ones in the
%       Vecchia ordering, builds the sparse factor U of the joint precision
%       and reads off the conditional distribution from its blocks,
%
%           U = [ U11  U12 ;  0  U22 ]  (upper triangular)
%           E[Y2|Y1] = -U22^{-T} U12' Y1 ,     Cov[Y2|Y1] = tau2 * M M' ,
%           M = U22^{-T}
%
%       which is the "joint" scheme of Section 3.3 and only ever touches
%       sparse triangular solves.
%
%   NN (optional) supplies precomputed nearest-neighbour indices for the
%   'mean'/'lite' modes so that repeated calls inside the MCMC loop -- where X
%   changes but the neighbour structure is reused -- do not repeat the search.

persistent HAVE_KRIG_MEX
if isempty(HAVE_KRIG_MEX)
    HAVE_KRIG_MEX = (exist('vdgp_krig_mex', 'file') == 3);
end

if nargin < 8  || isempty(mode), mode = 'lite'; end
if nargin < 9  || isempty(m),    m = opts.m;    end
if nargin < 10, NN = []; end
if nargin < 11 || isempty(nsamp), nsamp = 0; end
if nargin < 12, Ajoint = []; end

y = y(:);
n    = size(X, 1);
nnew = size(Xnew, 1);
v    = opts.v;
ct   = opts.cov;
sampling = strcmpi(mode, 'sample');
joint   = strcmpi(mode, 'joint') || (sampling && nsamp > 0);
want_s2 = ~strcmpi(mode, 'mean') && ~sampling;
if sampling && nsamp < 1, nsamp = 1; end

% nugget carried by the PREDICTIVE locations
if isfield(opts, 'pred_noise') && ~opts.pred_noise
    gp = 0;
else
    gp = g;
end

if ~opts.vecchia
    % ---------------- exact --------------------------------------------
    K  = vdgp_cov(X, [], theta, g, 1, v, ct);
    L  = chol(K, 'lower');
    kx = vdgp_cov(X, Xnew, theta, 0, 1, v, ct);      % n-by-nnew
    Ly = L \ y;
    Lk = L \ kx;
    mu = (Lk.' * Ly);
    out.mean = mu;
    if joint
        Kn = vdgp_cov(Xnew, [], theta, gp, 1, v, ct);
        S  = tau2 * (Kn - Lk.' * Lk);
        S  = (S + S.') / 2;
        if ~sampling
            out.Sigma = S;
            out.s2 = diag(S);
        end
        if nsamp > 0
            Lc = local_safe_chol(S);
            out.draws = mu + Lc * randn(nnew, nsamp);
        end
    elseif want_s2
        out.s2 = tau2 * (1 + gp - sum(Lk.^2, 1).');
    end
    return
end

% ---------------- Vecchia ----------------------------------------------
if joint
    if isempty(Ajoint)
        ordn = randperm(nnew);
        A    = vdgp_create_approx([X; Xnew(ordn, :)], m, 'none', true);
    else
        % Reuse an ordering and conditioning sets built once by the caller;
        % only the coordinates change from one MCMC draw to the next.
        ordn = Ajoint.ordn;
        A    = vdgp_update_approx(Ajoint.A, [X; Xnew(ordn, :)]);
    end
    gv   = [repmat(g, n, 1); repmat(max(gp, opts.eps), nnew, 1)];
    U    = vdgp_create_U(A, theta, gv, v, ct);

    U12 = U(1:n, n+1:end);
    U22 = U(n+1:end, n+1:end);

    rhs = U12.' * y;
    muo = -(U22.' \ rhs);

    mu = zeros(nnew, 1);  mu(ordn) = muo;
    out.mean = mu;

    if nsamp > 0
        % Cov = tau2 * U22^{-T} U22^{-1}, so sqrt(tau2) * (U22' \ z) is a draw
        % from the centred joint predictive -- no covariance is ever formed.
        Do = sqrt(tau2) * (U22.' \ randn(nnew, nsamp));
        D = zeros(nnew, nsamp);
        D(ordn, :) = full(Do);
        out.draws = mu + D;
    end

    if ~sampling
        M  = U22.' \ speye(nnew);         % lower triangular
        S  = tau2 * (M * M.');
        S  = full((S + S.') / 2);
        Sg = zeros(nnew);  Sg(ordn, ordn) = S;
        out.Sigma = Sg;
        out.s2    = diag(Sg);
    end
    return
end

m = min(m, n);
if isempty(NN)
    NN = vdgp_knn(X, Xnew, m);
end
k = size(NN, 2);

% ---------------- optional MEX fast path ---------------------------------
if HAVE_KRIG_MEX
    dd = size(X, 2);
    if isscalar(theta), th = repmat(theta, 1, dd); else, th = reshape(theta, 1, []); end
    if strcmpi(ct, 'exp2'), vc = 999; else, vc = v; end
    [mu, qfv] = vdgp_krig_mex(X, Xnew, NN, y, th, g, vc, double(want_s2));
    out.mean = mu;
    if want_s2
        out.s2 = max(tau2 * (1 + gp - qfv), 0);
    end
    return
end

mu = zeros(nnew, 1);
if want_s2, s2 = zeros(nnew, 1); end

chunk = max(1, floor(1.2e7 / max(k*k, 1)));
for a = 1:chunk:nnew
    b  = min(nnew, a + chunk - 1);
    R  = b - a + 1;
    id = NN(a:b, :);

    P = reshape(X(id(:), :), R, k, size(X, 2));
    P = permute(P, [2 3 1]);                     % k-by-d-by-R

    K = vdgp_batch_cov(P, theta, v, ct);
    for j = 1:k
        K(j, j, :) = K(j, j, :) + g;
    end
    L = vdgp_batch_chol(K);

    % cross covariance between each new point and its neighbours
    Q  = permute(reshape(Xnew(a:b, :), R, 1, size(X, 2)), [2 3 1]);   % 1-by-d-by-R
    Kx = local_cross(P, Q, theta, v, ct);                             % k-by-1-by-R

    Yn = reshape(y(id).', k, 1, R);
    RHS = cat(2, Yn, Kx);                        % k-by-2-by-R
    S   = vdgp_batch_fsolve(L, RHS);
    Ly  = S(:, 1, :);
    Lk  = S(:, 2, :);

    mu(a:b) = reshape(sum(Ly .* Lk, 1), R, 1);
    if want_s2
        s2(a:b) = tau2 * (1 + gp - reshape(sum(Lk.^2, 1), R, 1));
    end
end

out.mean = mu;
if want_s2
    out.s2 = max(s2, 0);
end
end

% -------------------------------------------------------------------------
function L = local_safe_chol(S)
% Lower Cholesky factor of a predictive covariance, with jitter if the matrix
% is only numerically semi-definite (common when test points nearly coincide).
n = size(S, 1);
sc = mean(abs(diag(S)));
if sc == 0, sc = 1; end
jit = 0;
for a = 0:6
    if jit > 0, S = S + jit * eye(n); end
    [L, p] = chol(S, 'lower');
    if p == 0, return; end
    jit = max(jit * 10, 1e-12 * sc);
end
% last resort: symmetric square root with the negative spectrum clipped
[V, D] = eig((S + S.') / 2);
L = V * diag(sqrt(max(diag(D), 0)));
end

% -------------------------------------------------------------------------
function Kx = local_cross(P, Q, theta, v, ct)
% P: k-by-d-by-R neighbour coordinates, Q: 1-by-d-by-R query coordinates
[k, d, R] = size(P);
exp2 = strcmpi(ct, 'exp2');
pw = 1; if ~exp2, pw = 2; end
if isscalar(theta), s = repmat(theta, 1, d); else, s = reshape(theta, 1, []); end
s = s .^ pw;

D2 = zeros(k, 1, R);
for j = 1:d
    D2 = D2 + ((P(:, j, :) - Q(1, j, :)).^2) / s(j);
end
D2 = max(D2, 0);

if exp2
    Kx = exp(-D2);
else
    r = sqrt(D2);
    switch v
        case 0.5, Kx = exp(-r);
        case 1.5, Kx = (1 + sqrt(3)*r) .* exp(-sqrt(3)*r);
        case 2.5, Kx = (1 + sqrt(5)*r + (5/3)*r.^2) .* exp(-sqrt(5)*r);
        otherwise, error('vdgp_krig:v', 'v must be 0.5, 1.5 or 2.5.');
    end
end
end
