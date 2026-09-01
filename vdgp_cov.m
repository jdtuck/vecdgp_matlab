function K = vdgp_cov(X1, X2, theta, g, tau2, v, cov_type)
%VDGP_COV  Covariance matrix used throughout the package.
%
%   K = VDGP_COV(X1, X2, THETA, G, TAU2, V, COV_TYPE)
%
%   X1        n1-by-d matrix of inputs
%   X2        n2-by-d matrix of inputs (pass [] for X2 = X1)
%   THETA     scalar (isotropic) or 1-by-d vector (separable / ARD) lengthscale
%   G         nugget, added to the diagonal only when X2 is empty.  May be a
%             scalar or an n1-vector (heterogeneous nugget).
%   TAU2      scale
%   V         Matern smoothness (0.5, 1.5, 2.5); ignored for 'exp2'
%   COV_TYPE  'matern' or 'exp2'
%
%   Parameterisation follows the deepgp R package:
%
%      exp2  :  k(x,x') = exp( -sum_k (x_k - x'_k)^2 / theta_k )
%      matern:  r = sqrt( sum_k (x_k - x'_k)^2 / theta_k^2 )
%               v=0.5 : exp(-r)
%               v=1.5 : (1 + sqrt(3) r) exp(-sqrt(3) r)
%               v=2.5 : (1 + sqrt(5) r + 5 r^2/3) exp(-sqrt(5) r)
%
%   so THETA is a squared-distance scale for 'exp2' and an ordinary
%   lengthscale for 'matern'.

if nargin < 7 || isempty(cov_type), cov_type = 'matern'; end
if nargin < 6 || isempty(v),        v = 2.5;             end
if nargin < 5 || isempty(tau2),     tau2 = 1;            end
if nargin < 4 || isempty(g),        g = 0;               end

symmetric = isempty(X2);
if symmetric, X2 = X1; end

exp2 = strcmpi(cov_type, 'exp2');
pw   = 1; if ~exp2, pw = 2; end

D2 = vdgp_sqdist(X1, X2, theta, pw);

if exp2
    K = exp(-D2);
else
    r = sqrt(max(D2, 0));
    switch v
        case 0.5
            K = exp(-r);
        case 1.5
            K = (1 + sqrt(3) * r) .* exp(-sqrt(3) * r);
        case 2.5
            K = (1 + sqrt(5) * r + (5/3) * r.^2) .* exp(-sqrt(5) * r);
        otherwise
            error('vdgp_cov:v', 'v must be 0.5, 1.5 or 2.5 (got %g).', v);
    end
end

if symmetric && any(g(:) ~= 0)
    n = size(X1, 1);
    if isscalar(g)
        K(1:(n+1):end) = K(1:(n+1):end) + g;
    else
        K(1:(n+1):end) = K(1:(n+1):end) + reshape(g, 1, n);
    end
end

K = tau2 * K;
end
