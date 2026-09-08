function out = vdgp_logl(y, A, theta, g, v, cov_type, outer)
%VDGP_LOGL  (Vecchia) multivariate normal log-likelihood.
%
%   OUT = VDGP_LOGL(Y, A, THETA, G, V, COV_TYPE, OUTER)
%
%   Evaluates  log N( Y ; 0, tau2 * Sigma )  under the Vecchia approximation
%   carried by A.  With Q = U*U' ~= inv(Sigma) at unit scale and Yo the
%   response in the Vecchia ordering,
%
%       quad = || U' Yo ||^2 ,      logdet(Q) = 2 * sum(log(diag(U)))
%
%   OUTER = true   tau2 is profiled out under the reference prior
%                  p(tau2) ∝ 1/tau2, giving tau2_hat = quad / n and
%                     ll = -n/2 log(2 pi) - n/2 log(tau2_hat)
%                          + 1/2 logdet(Q) - n/2
%   OUTER = false  tau2 is fixed at 1 (latent layers, for identifiability)
%                     ll = -n/2 log(2 pi) + 1/2 logdet(Q) - 1/2 quad
%
%   The sparse factor is never assembled: the quadratic form is accumulated
%   straight from the triplets, which is where most of the MCMC time is spent.
%
%   OUT is a struct with fields .ll, .tau2, .quad, .logdet.

persistent HAVE_LOGL_MEX
if isempty(HAVE_LOGL_MEX)
    HAVE_LOGL_MEX = (exist('vdgp_logl_mex', 'file') == 3);
end

if nargin < 7 || isempty(outer),    outer = true;        end
if nargin < 6 || isempty(cov_type), cov_type = 'matern'; end
if nargin < 5 || isempty(v),        v = 2.5;             end

n  = numel(y);
yo = y(A.ord);
yo = yo(:);

if HAVE_LOGL_MEX && A.vecchia
    % The whole likelihood in one call: U is never materialised, so nothing
    % of size n*(m+1) is allocated or returned.
    if isscalar(g), gv = repmat(g, n, 1); else, gv = g(:); end
    if isscalar(theta)
        th = repmat(theta, 1, A.d);
    else
        th = reshape(theta, 1, []);
    end
    if strcmpi(cov_type, 'exp2'), vc = 999; else, vc = v; end
    [quad, logdet] = vdgp_logl_mex(A.X_ord, A.NNarray, gv, th, vc, yo);
else
    E = vdgp_U_entries(A, theta, g, v, cov_type);
    % (U' * yo)_j = sum_i U(i,j) yo(i)
    Uty  = accumarray(E.J, E.V .* yo(E.I), [n, 1]);
    quad = sum(Uty.^2);
    logdet = 2 * sum(log(E.diagU));
end

if outer
    tau2 = quad / n;
    ll = -0.5 * n * log(2*pi) - 0.5 * n * log(tau2) + 0.5 * logdet - 0.5 * n;
else
    tau2 = 1;
    ll = -0.5 * n * log(2*pi) + 0.5 * logdet - 0.5 * quad;
end

if ~isfinite(ll), ll = -Inf; end

out = struct('ll', ll, 'tau2', tau2, 'quad', quad, 'logdet', logdet);
end
