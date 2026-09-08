function fit = fit_three_layer(x, y, varargin)
%FIT_THREE_LAYER  MCMC for a Vecchia-approximated three-layer deep GP.
%
%   FIT = FIT_THREE_LAYER(X, Y, ...)
%
%   Model
%        Y | W        ~  N_n( 0, tau2 * ( K_y(W) + g I ) )
%        W(:,k) | Z   ~  N_n( 0, K_k(Z) )
%        Z(:,k) | X   ~  N_n( 0, K_k(X) )
%
%   Sampling follows the same Gibbs scheme as FIT_TWO_LAYER with an extra
%   elliptical-slice block for Z (whose likelihood is the middle-layer density
%   of W given Z) and an extra set of Metropolis-Hastings lengthscales
%   theta_z.  All Gaussian evaluations go through the Vecchia factor when
%   OPTS.vecchia is true.
%
%   See also FIT_TWO_LAYER, DGP_TRIM, DGP_PREDICT, VDGP_OPTIONS.

opts = local_opts(varargin{:});
[xs, ys, sc, opts] = vdgp_prep(x, y, opts);
[n, d] = size(xs);
D = opts.D;

t0 = tic;

if isempty(opts.z_init), z = local_init(xs, D); else, z = opts.z_init; end
if isempty(opts.w_init), w = z;                 else, w = opts.w_init; end
D = size(w, 2); opts.D = D;

if isempty(opts.ord_init)
    x_approx = vdgp_create_approx(xs, opts.m, opts.ordering, opts.vecchia);
else
    x_approx = vdgp_create_approx(xs, opts.m, opts.ord_init, opts.vecchia);
end
% reuse x's conditioning sets for any layer that starts at the identity
if isequal(z, xs)
    z_approx = x_approx;
else
    z_approx = vdgp_create_approx(z, opts.m, x_approx.ord, opts.vecchia);
end
if isequal(w, z)
    w_approx = z_approx;
elseif isequal(w, xs)
    w_approx = x_approx;
else
    w_approx = vdgp_create_approx(w, opts.m, x_approx.ord, opts.vecchia);
end

nmcmc = opts.nmcmc;
theta_y = zeros(nmcmc, 1);
theta_w = zeros(nmcmc, D);
theta_z = zeros(nmcmc, D);
gsamp   = zeros(nmcmc, 1);
tau2    = zeros(nmcmc, 1);
llv     = zeros(nmcmc, 1);

theta_y(1)   = opts.theta_y_0;
theta_w(1,:) = opts.theta_w_0;
theta_z(1,:) = opts.theta_z_0;
if isempty(opts.true_g), gsamp(1) = opts.g_0; else, gsamp(1) = opts.true_g; end

keep = 1:opts.store_every:nmcmc;
if keep(end) ~= nmcmc, keep = [keep, nmcmc]; end
W = zeros(n, D, numel(keep));  W(:, :, 1) = w;
Z = zeros(n, D, numel(keep));  Z(:, :, 1) = z;
kptr = 1;

out = vdgp_logl(ys, w_approx, theta_y(1), gsamp(1), opts.v, opts.cov, true);
ll = out.ll; llv(1) = ll; tau2(1) = out.tau2;

for t = 2:nmcmc

    % ---- nugget --------------------------------------------------------
    if isempty(opts.true_g)
        [gnew, ll, tt] = vdgp_sample_g(ys, w_approx, theta_y(t-1), gsamp(t-1), ...
                                       ll, opts.alpha.g, opts.beta.g, opts);
        gsamp(t) = gnew;
        if ~isempty(tt), tau2(t-1) = tt; end
    else
        gsamp(t) = opts.true_g;
    end

    % ---- outer lengthscale ---------------------------------------------
    [th, ll, tt] = vdgp_sample_theta(ys, w_approx, theta_y(t-1), 1, gsamp(t), ...
                                     ll, opts.alpha.theta_y, opts.beta.theta_y, ...
                                     opts, true);
    theta_y(t) = th;
    if isempty(tt), tau2(t) = tau2(t-1); else, tau2(t) = tt; end

    % ---- middle lengthscales (W | Z) ------------------------------------
    for k = 1:D
        o = vdgp_logl(w(:, k), z_approx, theta_w(t-1, k), opts.eps, opts.v, opts.cov, false);
        theta_w(t, k) = vdgp_sample_theta(w(:, k), z_approx, theta_w(t-1, k), 1, ...
                                          opts.eps, o.ll, opts.alpha.theta_w, ...
                                          opts.beta.theta_w, opts, false);
    end

    % ---- inner lengthscales (Z | X) -------------------------------------
    for k = 1:D
        o = vdgp_logl(z(:, k), x_approx, theta_z(t-1, k), opts.eps, opts.v, opts.cov, false);
        theta_z(t, k) = vdgp_sample_theta(z(:, k), x_approx, theta_z(t-1, k), 1, ...
                                          opts.eps, o.ll, opts.alpha.theta_z, ...
                                          opts.beta.theta_z, opts, false);
    end

    % ---- Z layer (ESS against the middle-layer likelihood) --------------
    ll_mid = 0;
    for k = 1:D
        o = vdgp_logl(w(:, k), z_approx, theta_w(t, k), opts.eps, opts.v, opts.cov, false);
        ll_mid = ll_mid + o.ll;
    end
    [z, z_approx] = vdgp_sample_z(w, z, x_approx, z_approx, theta_w(t, :), ...
                                  theta_z(t, :), ll_mid, opts);

    % ---- W layer (ESS against the outer likelihood) ---------------------
    [w, w_approx, ll, tt] = vdgp_sample_w(ys, w, z_approx, w_approx, ...
                                          theta_y(t), theta_w(t, :), gsamp(t), ...
                                          ll, opts);
    if ~isempty(tt), tau2(t) = tt; end
    llv(t) = ll;

    % ---- optional re-approximation --------------------------------------
    if opts.reapprox > 0 && mod(t, opts.reapprox) == 0 && opts.vecchia
        z_approx = vdgp_reapprox(z_approx, z);
        w_approx = vdgp_reapprox(w_approx, w);
        out = vdgp_logl(ys, w_approx, theta_y(t), gsamp(t), opts.v, opts.cov, true);
        ll = out.ll;
    end

    if kptr < numel(keep) && t == keep(kptr + 1)
        kptr = kptr + 1;
        W(:, :, kptr) = w;
        Z(:, :, kptr) = z;
    end

    if opts.verb > 0 && mod(t, opts.verb) == 0
        fprintf('fit_three_layer: iteration %d / %d   ll = %.3f\n', t, nmcmc, ll);
    end
end

fit = struct();
fit.class  = 'dgp3';
fit.layers = 3;
fit.x = xs; fit.y = ys;
fit.x_raw = x; fit.y_raw = y;
fit.scaling = sc;
fit.n = n; fit.d = d; fit.D = D;
fit.nmcmc = nmcmc;
fit.theta_y = theta_y;
fit.theta_w = theta_w;
fit.theta_z = theta_z;
fit.g = gsamp;
fit.tau2 = tau2;
fit.ll = llv;
fit.w = W;
fit.z = Z;
fit.w_iters = keep(1:kptr);
fit.x_approx = x_approx;
fit.z_approx = z_approx;
fit.w_approx = w_approx;
fit.opts = opts;
fit.time = toc(t0);
end

% -------------------------------------------------------------------------
function w = local_init(x, D)
d = size(x, 2);
if D == d
    w = x;
elseif D < d
    xc = x - mean(x, 1);
    [~, ~, V] = svd(xc, 0);
    w = xc * V(:, 1:D);
    w = (w - min(w, [], 1)) ./ max(max(w, [], 1) - min(w, [], 1), eps);
else
    w = [x, repmat(x(:, end), 1, D - d)];
end
end

% -------------------------------------------------------------------------
function opts = local_opts(varargin)
if nargin == 1 && isstruct(varargin{1})
    opts = vdgp_options(varargin{1});
else
    opts = vdgp_options(varargin{:});
end
end
