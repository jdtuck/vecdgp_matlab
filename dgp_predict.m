function pred = dgp_predict(fit, x_new, varargin)
%DGP_PREDICT  Posterior predictive distribution of a fitted (deep) GP.
%
%   PRED = DGP_PREDICT(FIT, X_NEW)
%   PRED = DGP_PREDICT(FIT, X_NEW, 'name', value, ...)
%
%   Options
%     'lite'      true (default) -> pointwise variances only
%                 false          -> full predictive covariance (memory n_new^2)
%     'm'         conditioning-set size for prediction (default: the fit's m)
%     'nsamp'     number of joint posterior predictive SAMPLE PATHS to return
%                 (default 0).  See below.
%     'sample_latent'  when drawing sample paths, also draw the latent layers
%                 from their own predictive instead of propagating their
%                 kriging means (default false, which matches deepgp)
%     'per_draw'  return the per-iteration means/variances mu_t, s2_t
%                 (default false)
%     'nn_warped' false (default): the conditioning sets for the test points
%                 are found ONCE in x space and reused for every MCMC draw --
%                 the same choice the sampler makes, where the ordering and
%                 neighbour sets are built from x and held fixed while the
%                 latent layers move (Section 4 of the paper found updating
%                 them in warped space gives only marginal gains).
%                 true: redo the neighbour search in the warped space of every
%                 draw.  Costs roughly 20x more and rarely changes anything.
%     'cores'     >0 requests parfor over MCMC draws (default fit.opts.cores).
%                 With the Parallel Computing Toolbox this is close to linear
%                 in the number of workers; it is the main lever once the
%                 neighbour search is out of the inner loop.
%
%   The posterior predictive is a mixture over the retained MCMC draws.  For
%   each draw t the latent layers are propagated forward by their kriging
%   means (as in deepgp) and the outer layer supplies mu_t and s2_t; the
%   mixture is then summarised by the laws of total expectation and variance
%
%        mu  = mean_t mu_t
%        s2  = mean_t ( s2_t + mu_t^2 ) - mu^2
%
%   SAMPLE PATHS.  With 'nsamp', K the function also returns K draws from the
%   posterior predictive, in PRED.f (n_new-by-K).  Each draw is a JOINT draw
%   from N(mu_t, Sigma_t) for one retained MCMC iteration t -- not a draw from
%   the summarised N(mu, Sigma), which would collapse a possibly multi-modal
%   mixture to a single Gaussian.  Requested draws are spread evenly over the
%   retained iterations, and PRED.f_iter records which iteration produced each
%   column, so PRED.f is a valid sample from the mixture and its sample mean
%   and covariance converge to PRED.mean and PRED.Sigma.
%
%   Under the Vecchia approximation a joint draw costs one sparse triangular
%   solve, sqrt(tau2) * (U22' \ z), and no n_new-by-n_new covariance is ever
%   formed -- so sample paths are affordable at test sizes where 'lite', false
%   would not be.
%
%   Draws include the nugget unless OPTS.pred_noise is false, in which case
%   they are draws of the latent function.
%
%   PRED has fields .mean, .s2, .sd, .x_new, plus .Sigma (when 'lite' is
%   false), .f and .f_iter (when 'nsamp' > 0), and .mu_t/.s2_t (when
%   'per_draw' is true).  Everything is on the ORIGINAL y scale.
%
%   See also FIT_TWO_LAYER, DGP_TRIM.

p = local_parse(varargin, struct('lite', true, 'm', [], 'per_draw', false, ...
                                 'nsamp', 0, 'sample_latent', false, ...
                                 'nn_warped', false, 'cores', []));
opts = fit.opts;
if isempty(p.m), p.m = opts.m; end
if isempty(p.cores), p.cores = opts.cores; end
p.nsamp = max(0, round(p.nsamp));

if isvector(x_new), x_new = x_new(:); end
if fit.scaling.scaled
    xs_new = (x_new - fit.scaling.xmin) ./ fit.scaling.xrange;
else
    xs_new = x_new;
end
nnew = size(xs_new, 1);

T  = fit.nmcmc;
mu_t = zeros(nnew, T);
s2_t = zeros(nnew, T);
if ~p.lite
    Sig = zeros(nnew, nnew);
end

% Spread the requested sample paths evenly over the retained iterations, so
% that PRED.f is a draw from the mixture rather than from one part of it.
if p.nsamp > 0
    t_assign = 1 + floor((0:(p.nsamp - 1)) * T / p.nsamp);
    ndraw_t  = accumarray(t_assign(:), 1, [T, 1]);
else
    ndraw_t  = zeros(T, 1);
end
Fc = cell(T, 1);

% Neighbour structure for the latent layers: x is fixed, so this is computed
% once and reused for every MCMC draw (see 'nn_warped').
NNx = [];
if opts.vecchia
    NNx = vdgp_knn(fit.x, xs_new, min(p.m, fit.n));
end

% Likewise for the joint construction used by 'lite', false and by sampling:
% build the combined ordering and conditioning sets once, then only refresh
% the coordinates per draw.
Ajoint = [];
if opts.vecchia && ~p.nn_warped && (~p.lite || p.nsamp > 0)
    ordn = randperm(nnew);
    Ajoint = struct('ordn', ordn, ...
                    'A', vdgp_create_approx([fit.x; xs_new(ordn, :)], p.m, 'none', true));
end

% The MCMC draws are independent given the fit, so this loop is the natural
% place for parallelism (deepgp parallelises prediction the same way).  With
% p.cores = 0 MATLAB runs it serially; Octave ignores the worker count.
if p.lite
    parfor (t = 1:T, p.cores)
        [mt, st, ~, dr] = local_one_draw(fit, xs_new, t, opts, p, NNx, true, ndraw_t(t), Ajoint);
        mu_t(:, t) = mt;
        s2_t(:, t) = st;
        Fc{t} = dr;
    end
else
    parfor (t = 1:T, p.cores)
        [mt, st, Sg, dr] = local_one_draw(fit, xs_new, t, opts, p, NNx, false, ndraw_t(t), Ajoint);
        mu_t(:, t) = mt;
        s2_t(:, t) = st;
        Sig = Sig + (Sg + mt * mt.') / T;
        Fc{t} = dr;
    end
end

mu = mean(mu_t, 2);
s2 = mean(s2_t + mu_t.^2, 2) - mu.^2;
s2 = max(s2, 0);

% ---- back to the original response scale --------------------------------
ysd = fit.scaling.ysd;  ym = fit.scaling.ymean;
pred = struct();
pred.x_new = x_new;
pred.mean  = mu * ysd + ym;
pred.s2    = s2 * ysd^2;
pred.sd    = sqrt(pred.s2);
if ~p.lite
    Sig = Sig - mu * mu.';
    Sig = (Sig + Sig.') / 2;
    pred.Sigma = Sig * ysd^2;
end
if p.per_draw
    pred.mu_t = mu_t * ysd + ym;
    pred.s2_t = s2_t * ysd^2;
end
if p.nsamp > 0
    F = [Fc{:}];
    pred.f = F * ysd + ym;
    f_iter = zeros(1, 0);
    for t = 1:T
        f_iter = [f_iter, repmat(t, 1, ndraw_t(t))]; %#ok<AGROW>
    end
    pred.f_iter = f_iter;
end
end

% -------------------------------------------------------------------------
function [mu, s2, Sg, Dr] = local_one_draw(fit, xs_new, t, opts, p, NNx, lite, ndraw, Ajoint)
% Propagate one posterior draw of the latent layers forward, condition the
% outer layer on it, and optionally take NDRAW joint sample paths.
Sg = [];
Dr = [];
nnew = size(xs_new, 1);

% latent layers propagated by their kriging means (as deepgp does); this is
% what the reported mean and variance are always based on
[Xin, Xout, NNo] = local_forward(fit, xs_new, t, opts, p, NNx, false, Ajoint);

want_here = (ndraw > 0) && ~p.sample_latent;

if lite
    o = vdgp_krig(fit.y, Xin, Xout, fit.theta_y(t), fit.g(t), fit.tau2(t), ...
                  opts, 'lite', p.m, NNo);
    mu = o.mean; s2 = o.s2;
    if want_here
        os = vdgp_krig(fit.y, Xin, Xout, fit.theta_y(t), fit.g(t), fit.tau2(t), ...
                       opts, 'sample', p.m, [], ndraw, Ajoint);
        Dr = os.draws;
    end
else
    ns = 0; if want_here, ns = ndraw; end
    o = vdgp_krig(fit.y, Xin, Xout, fit.theta_y(t), fit.g(t), fit.tau2(t), ...
                  opts, 'joint', p.m, [], ns, Ajoint);
    mu = o.mean; s2 = o.s2; Sg = o.Sigma;
    if want_here, Dr = o.draws; end
end

% fully-Bayesian variant: redraw the warping for every sample path, so the
% draws carry the latent layers' own predictive uncertainty too
if (ndraw > 0) && p.sample_latent
    Dr = zeros(nnew, ndraw);
    for s = 1:ndraw
        [Xin2, Xout2] = local_forward(fit, xs_new, t, opts, p, NNx, true, Ajoint);
        os = vdgp_krig(fit.y, Xin2, Xout2, fit.theta_y(t), fit.g(t), fit.tau2(t), ...
                       opts, 'sample', p.m, [], 1, Ajoint);
        Dr(:, s) = os.draws;
    end
end
end

% -------------------------------------------------------------------------
function [Xin, Xout, NNo] = local_forward(fit, xs_new, t, opts, p, NNx, draw, Ajoint)
% Push the test locations through the latent layers of MCMC draw t.  DRAW
% false uses each layer's kriging mean; DRAW true takes a joint draw from each
% layer's own predictive distribution.
nnew = size(xs_new, 1);

% Conditioning sets for the test points.  Row i of NNx indexes the training
% rows nearest to test point i in x space, and the latent layers are just
% re-coordinatisations of those same training rows, so NNx is a valid (and
% fixed) conditioning set at every layer.  Re-deriving it in warped space for
% each MCMC draw is what 'nn_warped' asks for, and it costs about 20x more.
if p.nn_warped
    NNlat = [];
else
    NNlat = NNx;
end

switch fit.layers
    case 1
        Xin = fit.x; Xout = xs_new; NNo = NNx;
    case 2
        wt = fit.w(:, :, t);
        wn = zeros(nnew, size(wt, 2));
        for k = 1:size(wt, 2)
            wn(:, k) = local_layer(wt(:, k), fit.x, xs_new, fit.theta_w(t, k), ...
                                   NNx, opts, p, draw, Ajoint);
        end
        Xin = wt; Xout = wn; NNo = NNlat;
    case 3
        zt = fit.z(:, :, t);
        wt = fit.w(:, :, t);
        zn = zeros(nnew, size(zt, 2));
        for k = 1:size(zt, 2)
            zn(:, k) = local_layer(zt(:, k), fit.x, xs_new, fit.theta_z(t, k), ...
                                   NNx, opts, p, draw, Ajoint);
        end
        wn = zeros(nnew, size(wt, 2));
        for k = 1:size(wt, 2)
            wn(:, k) = local_layer(wt(:, k), zt, zn, fit.theta_w(t, k), ...
                                   NNlat, opts, p, draw, Ajoint);
        end
        Xin = wt; Xout = wn; NNo = NNlat;
    otherwise
        error('dgp_predict:layers', 'Unsupported number of layers.');
end
end

% -------------------------------------------------------------------------
function vals = local_layer(yy, Xtr, Xte, th, NNuse, opts, p, draw, Ajoint)
% One hidden node pushed forward: kriging mean, or a joint draw.
if draw
    o = vdgp_krig(yy, Xtr, Xte, th, opts.eps, 1, opts, 'sample', p.m, NNuse, 1, Ajoint);
    vals = o.draws;
else
    o = vdgp_krig(yy, Xtr, Xte, th, opts.eps, 1, opts, 'mean', p.m, NNuse);
    vals = o.mean;
end
end

% -------------------------------------------------------------------------
function p = local_parse(args, p)
if numel(args) == 1 && isstruct(args{1})
    fn = fieldnames(args{1});
    for i = 1:numel(fn), p.(fn{i}) = args{1}.(fn{i}); end
    return
end
for i = 1:2:numel(args)
    if ~isfield(p, args{i})
        error('dgp_predict:opt', 'Unknown prediction option "%s".', args{i});
    end
    p.(args{i}) = args{i+1};
end
end
