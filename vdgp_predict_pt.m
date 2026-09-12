function [mu, s2, out] = vdgp_predict_pt(P, x_new, varargin)
%VDGP_PREDICT_PT  Posterior prediction at a few points, batched over MCMC draws.
%
%   [MU, S2] = VDGP_PREDICT_PT(P, X_NEW)
%   [MU, S2, OUT] = VDGP_PREDICT_PT(P, X_NEW, 'name', value)
%
%   P       from VDGP_PREDICTOR
%   X_NEW   1-by-d, or a small n_new-by-d block
%
%   Returns the posterior predictive mean and variance on the ORIGINAL y
%   scale, identical to DGP_PREDICT(..., 'lite', true) up to round-off.
%
%   Options
%     'per_draw'  also return the per-draw mu_t, s2_t in OUT (default false)
%     'm'         conditioning-set size for THIS call, overriding the
%                 predictor's.  Prediction cost grows like m^3, and a smaller
%                 m at prediction time than at fit time is usually harmless --
%                 demo_scaling quantifies the trade-off.
%     'path'      which orientation to use: 'draws' (batch over MCMC draws,
%                 loop over points -- best for few points and many draws),
%                 'points' (batch over points, loop over draws -- best for
%                 many points), or 'auto' (default).  Same answers either way.
%     'thin'      use every k-th retained draw (default 1).  The posterior
%                 predictive is a mixture over draws; its mean and variance
%                 converge in T long before the chain does, so thinning here
%                 is the cheapest available speed-up.
%
%   Intended for calibration and other outer MCMC loops, where prediction is
%   called many times with one proposed input.  Two things make it fast:
%
%     * the conditioning set is found by a direct scan over the n training
%       inputs, which for a single point is O(n d) and negligible -- the
%       whole-design neighbour search DGP_PREDICT does up front is what costs
%       seconds, and it is pointless for one point;
%
%     * every retained MCMC draw is evaluated in ONE batched Cholesky and
%       triangular solve rather than one call per draw.  For a fixed test
%       point the conditioning set is the same for all draws, so the m-by-m
%       blocks differ only through the draw's lengthscale and nugget, and
%       assembling them as m-by-m-by-T pages turns T interpreted calls into
%       a handful of array operations.
%
%   Accuracy is unchanged: this evaluates exactly the same Vecchia predictor
%   as DGP_PREDICT, with the same conditioning sets.
%
%   See also VDGP_PREDICTOR, DGP_PREDICT.

o = struct('per_draw', false, 'm', [], 'thin', 1, 'path', 'auto');
for i = 1:2:numel(varargin)
    if ~isfield(o, varargin{i})
        error('vdgp_predict_pt:opt', 'Unknown option "%s".', varargin{i});
    end
    o.(varargin{i}) = varargin{i+1};
end

if ~isempty(o.m), P.m = min(o.m, P.n); end
if o.thin > 1
    keep = 1:round(o.thin):P.T;
    P.theta_y = P.theta_y(keep); P.g = P.g(keep); P.tau2 = P.tau2(keep);
    if isfield(P, 'w'), P.w = P.w(:, :, keep); P.theta_w = P.theta_w(keep, :); end
    if isfield(P, 'z'), P.z = P.z(:, :, keep); P.theta_z = P.theta_z(keep, :); end
    P.T = numel(keep);
end

if isvector(x_new) && P.d > 1 && numel(x_new) == P.d
    x_new = x_new(:).';
elseif isvector(x_new) && P.d == 1
    x_new = x_new(:);
end

% ---- scale the inputs exactly as the fit did ---------------------------
if P.scaling.scaled
    xs = (x_new - P.scaling.xmin) ./ P.scaling.xrange;
else
    xs = x_new;
end
nnew = size(xs, 1);

% ---- orientation --------------------------------------------------------
switch lower(o.path)
    case 'auto',   use_points = (nnew >= 4) || (P.T <= 4);
    case 'points', use_points = true;
    case 'draws',  use_points = false;
    otherwise
        error('vdgp_predict_pt:path', ...
              'path must be ''auto'', ''points'' or ''draws''.');
end

if use_points
    % Batch over test points, loop over draws, in blocks so that the
    % n_new-by-T moment arrays stay bounded.
    mu_all = zeros(nnew, 1);
    ms_all = zeros(nnew, 1);
    if o.per_draw
        out.mu_t = zeros(nnew, P.T);
        out.s2_t = zeros(nnew, P.T);
    end
    for a = 1:P.chunk:P.T
        b  = min(P.T, a + P.chunk - 1);
        tt = a:b;
        [mt, st] = vdgp_moments_draws(P, xs, tt);     % original y scale
        mu_all = mu_all + sum(mt, 2);
        ms_all = ms_all + sum(st + mt.^2, 2);
        if o.per_draw
            out.mu_t(:, tt) = mt;
            out.s2_t(:, tt) = st;
        end
    end
    mu = mu_all / P.T;
    s2 = max(ms_all / P.T - mu.^2, 0);
    out.mean = mu;
    out.s2   = s2;
    out.sd   = sqrt(s2);
    return
end

mu = zeros(nnew, 1);
s2 = zeros(nnew, 1);
if o.per_draw
    out.mu_t = zeros(nnew, P.T);
    out.s2_t = zeros(nnew, P.T);
else
    out = struct();
end

gp_all = P.g;
if isfield(P.opts, 'pred_noise') && ~P.opts.pred_noise
    gp_all = zeros(size(P.g));
end

for i = 1:nnew
    x0 = xs(i, :);

    % ---- conditioning set: one scan over the training inputs ------------
    d2 = sum((P.x - x0).^2, 2);
    idx = local_mink(d2, P.m);

    Xn   = P.x(idx, :);                       % m-by-d, fixed across draws
    D2nn = local_pairwise(Xn);                % m-by-m, fixed across draws
    d2x  = sum((Xn - x0).^2, 2);              % m-by-1, fixed across draws

    mu_t = zeros(P.T, 1);
    s2_t = zeros(P.T, 1);

    for a = 1:P.chunk:P.T
        b  = min(P.T, a + P.chunk - 1);
        tt = a:b;

        switch P.layers
            case 1
                Pin  = [];                      % outer design is X itself
                Qin  = [];
                yin  = P.y(idx);
            case 2
                wstar = local_latent_fixed(D2nn, d2x, P.w(idx, :, tt), ...
                                           P.theta_w(tt, :), P);
                Pin = permute(P.w(idx, :, tt), [1 2 3]);   % m-by-D-by-nt
                Qin = reshape(wstar.', 1, P.D, numel(tt)); % 1-by-D-by-nt
                yin = P.y(idx);
            case 3
                zstar = local_latent_fixed(D2nn, d2x, P.z(idx, :, tt), ...
                                           P.theta_z(tt, :), P);
                % middle layer: design is z(idx,:) per draw, target is zstar
                Zp = P.z(idx, :, tt);
                Qz = reshape(zstar.', 1, size(Zp, 2), numel(tt));
                wstar = local_latent_moving(Zp, Qz, P.w(idx, :, tt), ...
                                            P.theta_w(tt, :), P);
                Pin = P.w(idx, :, tt);
                Qin = reshape(wstar.', 1, P.D, numel(tt));
                yin = P.y(idx);
        end

        % ---- outer layer ------------------------------------------------
        if P.layers == 1
            Kp  = local_pages_fixed(D2nn, P.theta_y(tt), P);
            kxp = local_pages_fixed_cross(d2x, P.theta_y(tt), P);
            Yp  = repmat(yin, 1, 1, numel(tt));
        else
            D2p = local_d2_pages(Pin, Pin);
            d2c = local_d2_cross(Pin, Qin);
            Kp  = local_kern(D2p, P.theta_y(tt), P);
            kxp = local_kern(d2c, P.theta_y(tt), P);
            Yp  = repmat(yin, 1, 1, numel(tt));
        end

        gt = reshape(P.g(tt), 1, 1, numel(tt));
        for j = 1:P.m
            Kp(j, j, :) = Kp(j, j, :) + gt;
        end

        L  = vdgp_batch_chol(Kp);
        S  = vdgp_batch_fsolve(L, cat(2, Yp, kxp));
        La = S(:, 1, :);
        Lk = S(:, 2, :);

        mu_t(tt) = reshape(sum(La .* Lk, 1), [], 1);
        qf       = reshape(sum(Lk .^ 2, 1), [], 1);
        s2_t(tt) = max(P.tau2(tt) .* (1 + gp_all(tt) - qf), 0);
    end

    m1 = mean(mu_t);
    v1 = max(mean(s2_t + mu_t.^2) - m1^2, 0);

    mu(i) = m1 * P.scaling.ysd + P.scaling.ymean;
    s2(i) = v1 * P.scaling.ysd^2;
    if o.per_draw
        out.mu_t(i, :) = mu_t.' * P.scaling.ysd + P.scaling.ymean;
        out.s2_t(i, :) = s2_t.' * P.scaling.ysd^2;
    end
end

out.mean = mu;
out.s2   = s2;
out.sd   = sqrt(s2);
end

% =========================================================================
function wstar = local_latent_fixed(D2nn, d2x, Wn, Th, P)
%LOCAL_LATENT_FIXED  Latent layer whose design is X (fixed across draws).
%   Wn  m-by-D-by-nt responses, Th nt-by-D lengthscales.
nt = size(Wn, 3);
D  = size(Wn, 2);
wstar = zeros(nt, D);
for k = 1:D
    Kp  = local_pages_fixed(D2nn, Th(:, k), P);
    kxp = local_pages_fixed_cross(d2x, Th(:, k), P);
    for j = 1:P.m
        Kp(j, j, :) = Kp(j, j, :) + P.opts.eps;
    end
    L = vdgp_batch_chol(Kp);
    Y = reshape(Wn(:, k, :), P.m, 1, nt);
    S = vdgp_batch_fsolve(L, cat(2, Y, kxp));
    wstar(:, k) = reshape(sum(S(:, 1, :) .* S(:, 2, :), 1), [], 1);
end
end

% -------------------------------------------------------------------------
function wstar = local_latent_moving(Zp, Qz, Wn, Th, P)
%LOCAL_LATENT_MOVING  Latent layer whose design moves with the draw.
nt = size(Wn, 3);
D  = size(Wn, 2);
wstar = zeros(nt, D);
D2p = local_d2_pages(Zp, Zp);
d2c = local_d2_cross(Zp, Qz);
for k = 1:D
    Kp  = local_kern(D2p, Th(:, k), P);
    kxp = local_kern(d2c, Th(:, k), P);
    for j = 1:P.m
        Kp(j, j, :) = Kp(j, j, :) + P.opts.eps;
    end
    L = vdgp_batch_chol(Kp);
    Y = reshape(Wn(:, k, :), P.m, 1, nt);
    S = vdgp_batch_fsolve(L, cat(2, Y, kxp));
    wstar(:, k) = reshape(sum(S(:, 1, :) .* S(:, 2, :), 1), [], 1);
end
end

% -------------------------------------------------------------------------
function K = local_pages_fixed(D2, theta, P)
%LOCAL_PAGES_FIXED  Kernel pages from ONE distance matrix and per-page theta.
sc = reshape(theta(:) .^ P.pw, 1, 1, []);
K  = local_corr(D2 ./ sc, P.vc);
end

function K = local_pages_fixed_cross(d2, theta, P)
sc = reshape(theta(:) .^ P.pw, 1, 1, []);
K  = local_corr(d2 ./ sc, P.vc);
end

function K = local_kern(D2pages, theta, P)
%LOCAL_KERN  Kernel pages from per-page distances and per-page theta.
sc = reshape(theta(:) .^ P.pw, 1, 1, []);
K  = local_corr(D2pages ./ sc, P.vc);
end

% -------------------------------------------------------------------------
function D2 = local_pairwise(Xn)
D2 = sum(Xn.^2, 2) + sum(Xn.^2, 2).' - 2 * (Xn * Xn.');
D2 = max(D2, 0);
D2(1:(size(Xn,1)+1):end) = 0;
end

function D2 = local_d2_pages(A, B)
%LOCAL_D2_PAGES  Pairwise squared distances within each page (unscaled).
[k, d, N] = size(A);
D2 = zeros(k, k, N);
for j = 1:d
    a = reshape(A(:, j, :), k, 1, N);
    b = reshape(B(:, j, :), 1, k, N);
    D2 = D2 + (a - b).^2;
end
D2 = max(D2, 0);
end

function D2 = local_d2_cross(A, q)
%LOCAL_D2_CROSS  Squared distances from each page's rows to that page's query.
[k, d, N] = size(A);
D2 = zeros(k, 1, N);
for j = 1:d
    D2 = D2 + (A(:, j, :) - q(1, j, :)).^2;
end
D2 = max(D2, 0);
end

% -------------------------------------------------------------------------
function K = local_corr(D2, vc)
if vc > 900
    K = exp(-D2);
    return
end
r = sqrt(max(D2, 0));
switch vc
    case 0.5, K = exp(-r);
    case 1.5, K = (1 + sqrt(3)*r) .* exp(-sqrt(3)*r);
    case 2.5, K = (1 + sqrt(5)*r + (5/3)*r.^2) .* exp(-sqrt(5)*r);
    otherwise, error('vdgp_predict_pt:v', 'v must be 0.5, 1.5 or 2.5.');
end
end

% -------------------------------------------------------------------------
function idx = local_mink(v, k)
if exist('mink', 'builtin') == 5 || exist('mink', 'file') == 2
    [~, idx] = mink(v, k);
else
    [~, o] = sort(v);
    idx = o(1:k);
end
idx = idx(:);
end
