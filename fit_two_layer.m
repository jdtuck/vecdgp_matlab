function fit = fit_two_layer(x, y, varargin)
%FIT_TWO_LAYER  MCMC for a Vecchia-approximated two-layer deep GP.
%
%   FIT = FIT_TWO_LAYER(X, Y)
%   FIT = FIT_TWO_LAYER(X, Y, 'name', value, ...)
%   FIT = FIT_TWO_LAYER(X, Y, OPTS)
%
%   Model (Sauer, Cooper & Gramacy 2023, arXiv:2204.02904)
%
%        Y | W        ~  N_n( 0, tau2 * ( K_y(W) + g I ) )
%        W(:,k) | X   ~  N_n( 0, K_k(X) ) ,      k = 1..D
%
%   with unit scale and zero nugget on the latent layer for identifiability.
%   tau2 is marginalised out under a reference prior; theta_y, theta_w and g
%   get Gamma priors and Metropolis-Hastings updates with uniform
%   sliding-window proposals; W is updated by elliptical slice sampling.
%
%   When OPTS.vecchia is true (the default) every Gaussian likelihood and
%   every prior draw is computed from the sparse inverse-Cholesky factor U of
%   the Vecchia approximation, so one Gibbs sweep costs O(n m^3) rather than
%   O(n^3).  The ordering and the conditioning sets are built once from X (and
%   from the initial W) and then held fixed while W moves; set
%   OPTS.reapprox = k > 0 to rebuild them in the warped space every k
%   iterations (Section 4 of the paper / re_approx = TRUE in deepgp).
%
%   Key options (see VDGP_OPTIONS): nmcmc, m, vecchia, cov, v, true_g, D,
%   ordering, reapprox, store_every, scale, verb.
%
%   See also FIT_ONE_LAYER, FIT_THREE_LAYER, DGP_TRIM, DGP_PREDICT.

opts = local_opts(varargin{:});
[xs, ys, sc, opts] = vdgp_prep(x, y, opts);
[n, d] = size(xs);
D = opts.D;

t0 = tic;

% --- initialise the latent layer at the identity mapping ------------------
if isempty(opts.w_init)
    w = local_init_w(xs, D);
else
    w = opts.w_init;
    if size(w, 1) ~= n
        error('fit_two_layer:w_init', 'w_init must have %d rows.', n);
    end
    D = size(w, 2); opts.D = D;
end

if isempty(opts.ord_init)
    x_approx = vdgp_create_approx(xs, opts.m, opts.ordering, opts.vecchia);
else
    x_approx = vdgp_create_approx(xs, opts.m, opts.ord_init, opts.vecchia);
end
% the w layer reuses the SAME ordering, as in deepgp
w_approx = vdgp_create_approx(w, opts.m, x_approx.ord, opts.vecchia);

nmcmc = opts.nmcmc;
theta_y = zeros(nmcmc, 1);
theta_w = zeros(nmcmc, D);
gsamp   = zeros(nmcmc, 1);
tau2    = zeros(nmcmc, 1);
llv     = zeros(nmcmc, 1);

theta_y(1)   = opts.theta_y_0;
theta_w(1,:) = opts.theta_w_0;
if isempty(opts.true_g), gsamp(1) = opts.g_0; else, gsamp(1) = opts.true_g; end

keep = 1:opts.store_every:nmcmc;
if keep(end) ~= nmcmc, keep = [keep, nmcmc]; end
W = zeros(n, D, numel(keep));
W(:, :, 1) = w;
kptr = 1;

out = vdgp_logl(ys, w_approx, theta_y(1), gsamp(1), opts.v, opts.cov, true);
ll = out.ll; llv(1) = ll; tau2(1) = out.tau2;

for t = 2:nmcmc

    % ---- nugget -------------------------------------------------------
    if isempty(opts.true_g)
        [gnew, ll, tt] = vdgp_sample_g(ys, w_approx, theta_y(t-1), gsamp(t-1), ...
                                       ll, opts.alpha.g, opts.beta.g, opts);
        gsamp(t) = gnew;
        if ~isempty(tt), tau2(t-1) = tt; end
    else
        gsamp(t) = opts.true_g;
    end

    % ---- outer lengthscale --------------------------------------------
    [th, ll, tt] = vdgp_sample_theta(ys, w_approx, theta_y(t-1), 1, gsamp(t), ...
                                     ll, opts.alpha.theta_y, opts.beta.theta_y, ...
                                     opts, true);
    theta_y(t) = th;
    if isempty(tt), tau2(t) = tau2(t-1); else, tau2(t) = tt; end

    % ---- inner lengthscales (one per hidden node) ----------------------
    for k = 1:D
        oin = vdgp_logl(w(:, k), x_approx, theta_w(t-1, k), opts.eps, ...
                        opts.v, opts.cov, false);
        thw = vdgp_sample_theta(w(:, k), x_approx, theta_w(t-1, k), 1, opts.eps, ...
                                oin.ll, opts.alpha.theta_w, opts.beta.theta_w, ...
                                opts, false);
        theta_w(t, k) = thw;
    end

    % ---- latent layer (elliptical slice sampling) ----------------------
    [w, w_approx, ll, tt] = vdgp_sample_w(ys, w, x_approx, w_approx, ...
                                          theta_y(t), theta_w(t, :), gsamp(t), ...
                                          ll, opts);
    if ~isempty(tt), tau2(t) = tt; end
    llv(t) = ll;

    % ---- optional re-approximation in the warped space -----------------
    if opts.reapprox > 0 && mod(t, opts.reapprox) == 0 && opts.vecchia
        w_approx = vdgp_reapprox(w_approx, w);
        out = vdgp_logl(ys, w_approx, theta_y(t), gsamp(t), opts.v, opts.cov, true);
        ll = out.ll;
    end

    % ---- storage -------------------------------------------------------
    if kptr < numel(keep) && t == keep(kptr + 1)
        kptr = kptr + 1;
        W(:, :, kptr) = w;
    end

    if opts.verb > 0 && mod(t, opts.verb) == 0
        fprintf('fit_two_layer: iteration %d / %d   ll = %.3f\n', t, nmcmc, ll);
    end
end

fit = struct();
fit.class  = 'dgp2';
fit.layers = 2;
fit.x = xs; fit.y = ys;
fit.x_raw = x; fit.y_raw = y;
fit.scaling = sc;
fit.n = n; fit.d = d; fit.D = D;
fit.nmcmc = nmcmc;
fit.theta_y = theta_y;
fit.theta_w = theta_w;
fit.g = gsamp;
fit.tau2 = tau2;
fit.ll = llv;
fit.w = W;
fit.w_iters = keep(1:kptr);
fit.x_approx = x_approx;
fit.w_approx = w_approx;
fit.opts = opts;
fit.time = toc(t0);
end

% -------------------------------------------------------------------------
function w = local_init_w(x, D)
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
