function fit = fit_one_layer(x, y, varargin)
%FIT_ONE_LAYER  MCMC for a (Vecchia-approximated) one-layer GP.
%
%   FIT = FIT_ONE_LAYER(X, Y)                       uses the defaults
%   FIT = FIT_ONE_LAYER(X, Y, 'name', value, ...)   overrides options
%   FIT = FIT_ONE_LAYER(X, Y, OPTS)                 with OPTS from VDGP_OPTIONS
%
%   Model
%       Y ~ N_n( 0, tau2 * ( K(X) + g I ) )
%   with tau2 marginalised under the reference prior, Gamma priors on the
%   lengthscale theta and the nugget g, and Metropolis-Hastings updates using
%   uniform sliding-window proposals.  With OPTS.vecchia = true every
%   likelihood is evaluated through the sparse Vecchia factor U of
%   Sauer, Cooper & Gramacy (2023).
%
%   The returned struct is consumed by DGP_TRIM and DGP_PREDICT.
%
%   See also FIT_TWO_LAYER, FIT_THREE_LAYER, DGP_PREDICT, VDGP_OPTIONS.

opts = local_opts(varargin{:});
[xs, ys, sc, opts] = vdgp_prep(x, y, opts);
n = size(xs, 1);

t0 = tic;
A = vdgp_create_approx(xs, opts.m, opts.ordering, opts.vecchia);

nmcmc = opts.nmcmc;
theta = zeros(nmcmc, 1);
gsamp = zeros(nmcmc, 1);
tau2  = zeros(nmcmc, 1);
llv   = zeros(nmcmc, 1);

theta(1) = opts.theta_y_0;
if isempty(opts.true_g)
    gsamp(1) = opts.g_0;
else
    gsamp(1) = opts.true_g;
end

out = vdgp_logl(ys, A, theta(1), gsamp(1), opts.v, opts.cov, true);
llv(1)  = out.ll;
tau2(1) = out.tau2;
ll = out.ll;

for t = 2:nmcmc
    % ---- nugget -------------------------------------------------------
    if isempty(opts.true_g)
        [gnew, ll, tt] = vdgp_sample_g(ys, A, theta(t-1), gsamp(t-1), ll, ...
                                       opts.alpha.g, opts.beta.g, opts);
        gsamp(t) = gnew;
        if ~isempty(tt), tau2(t-1) = tt; end
    else
        gsamp(t) = opts.true_g;
    end

    % ---- lengthscale --------------------------------------------------
    [th, ll, tt] = vdgp_sample_theta(ys, A, theta(t-1), 1, gsamp(t), ll, ...
                                     opts.alpha.theta, opts.beta.theta, opts, true);
    theta(t) = th;
    if isempty(tt)
        tau2(t) = tau2(t-1);
    else
        tau2(t) = tt;
    end
    llv(t) = ll;

    if opts.verb > 0 && mod(t, opts.verb) == 0
        fprintf('fit_one_layer: iteration %d / %d   ll = %.3f\n', t, nmcmc, ll);
    end
end

fit = struct();
fit.class   = 'gp';
fit.layers  = 1;
fit.x = xs; fit.y = ys;
fit.x_raw = x; fit.y_raw = y;
fit.scaling = sc;
fit.n = n;
fit.nmcmc = nmcmc;
fit.theta_y = theta;
fit.g = gsamp;
fit.tau2 = tau2;
fit.ll = llv;
fit.x_approx = A;
fit.opts = opts;
fit.time = toc(t0);
end

% -------------------------------------------------------------------------
function opts = local_opts(varargin)
if nargin == 1 && isstruct(varargin{1})
    opts = vdgp_options(varargin{1});
else
    opts = vdgp_options(varargin{:});
end
end
