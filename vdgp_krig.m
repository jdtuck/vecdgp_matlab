function out = vdgp_krig(y, X, Xnew, theta, g, tau2, opts, mode, m, NN)
%VDGP_KRIG  Gaussian process prediction, with or without Vecchia.
%
%   OUT = VDGP_KRIG(Y, X, XNEW, THETA, G, TAU2, OPTS, MODE, M, NN)
%
%   MODE
%     'mean'   posterior mean only (used for the latent layers, which are
%              propagated through the network by their kriging mean, exactly
%              as in deepgp)
%     'lite'   posterior mean and pointwise variance
%     'joint'  posterior mean and full n_new-by-n_new covariance
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

if nargin < 8 || isempty(mode), mode = 'lite'; end
if nargin < 9 || isempty(m),    m = opts.m;    end
if nargin < 10, NN = []; end

y = y(:);
n    = size(X, 1);
nnew = size(Xnew, 1);
v    = opts.v;
ct   = opts.cov;
joint = strcmpi(mode, 'joint');
want_s2 = ~strcmpi(mode, 'mean');

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
        out.Sigma = (S + S.') / 2;
        out.s2 = diag(out.Sigma);
    elseif want_s2
        out.s2 = tau2 * (1 + gp - sum(Lk.^2, 1).');
    end
    return
end

% ---------------- Vecchia ----------------------------------------------
if joint
    ordn = randperm(nnew);
    Xall = [X; Xnew(ordn, :)];
    A    = vdgp_create_approx(Xall, m, 'none', true);
    gv   = [repmat(g, n, 1); repmat(max(gp, opts.eps), nnew, 1)];
    U    = vdgp_create_U(A, theta, gv, v, ct);

    U12 = U(1:n, n+1:end);
    U22 = U(n+1:end, n+1:end);

    rhs   = U12.' * y;
    muo   = -(U22.' \ rhs);
    M     = U22.' \ speye(nnew);          % lower triangular, dense-ish
    S     = tau2 * (M * M.');
    S     = full((S + S.') / 2);

    mu = zeros(nnew, 1);  mu(ordn) = muo;
    Sg = zeros(nnew);     Sg(ordn, ordn) = S;

    out.mean  = mu;
    out.Sigma = Sg;
    out.s2    = diag(Sg);
    return
end

m = min(m, n);
if isempty(NN)
    NN = vdgp_knn(X, Xnew, m);
end
k = size(NN, 2);

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
