function pred = dgp_predict(fit, x_new, varargin)
%DGP_PREDICT  Posterior predictive distribution of a fitted (deep) GP.
%
%   PRED = DGP_PREDICT(FIT, X_NEW)
%   PRED = DGP_PREDICT(FIT, X_NEW, 'name', value, ...)
%
%   Options
%     'lite'     true (default) -> pointwise variances only
%                false          -> full predictive covariance (memory n_new^2)
%     'm'        conditioning-set size for prediction (default: the fit's m)
%     'samples'  return the per-iteration means/variances (default false)
%     'cores'    >0 requests parfor over MCMC draws (default fit.opts.cores)
%
%   The posterior predictive is a mixture over the retained MCMC draws.  For
%   each draw t the latent layers are propagated forward by their kriging
%   means (as in deepgp) and the outer layer supplies mu_t and s2_t; the
%   mixture is then summarised by the laws of total expectation and variance
%
%        mu  = mean_t mu_t
%        s2  = mean_t ( s2_t + mu_t^2 ) - mu^2
%
%   PRED has fields .mean, .s2, .sd, .x_new, and (when 'lite' is false)
%   .Sigma.  Everything is returned on the ORIGINAL y scale.
%
%   See also FIT_TWO_LAYER, DGP_TRIM.

p = local_parse(varargin, struct('lite', true, 'm', [], 'samples', false, ...
                                 'cores', []));
opts = fit.opts;
if isempty(p.m), p.m = opts.m; end
if isempty(p.cores), p.cores = opts.cores; end

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

% neighbour structure for the inner layer never changes (x is fixed)
NNx = [];
if opts.vecchia
    NNx = vdgp_knn(fit.x, xs_new, min(p.m, fit.n));
end

% The MCMC draws are independent given the fit, so this loop is the natural
% place for parallelism (deepgp parallelises prediction the same way).  With
% p.cores = 0 MATLAB runs it serially; Octave ignores the worker count.
if p.lite
    parfor (t = 1:T, p.cores)
        [mt, st] = local_one_draw(fit, xs_new, t, opts, p, NNx, true);
        mu_t(:, t) = mt;
        s2_t(:, t) = st;
    end
else
    parfor (t = 1:T, p.cores)
        [mt, st, Sg] = local_one_draw(fit, xs_new, t, opts, p, NNx, false);
        mu_t(:, t) = mt;
        s2_t(:, t) = st;
        Sig = Sig + (Sg + mt * mt.') / T;
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
if p.samples
    pred.mu_t = mu_t * ysd + ym;
    pred.s2_t = s2_t * ysd^2;
end
end

% -------------------------------------------------------------------------
function [mu, s2, Sg] = local_one_draw(fit, xs_new, t, opts, p, NNx, lite)
% Propagate one posterior draw of the latent layers forward and condition the
% outer layer on it.
nnew = size(xs_new, 1);
Sg = [];

switch fit.layers
    case 1
        Xin = fit.x; Xout = xs_new; NNo = NNx;
    case 2
        wt = fit.w(:, :, t);
        wn = zeros(nnew, size(wt, 2));
        for k = 1:size(wt, 2)
            o = vdgp_krig(wt(:, k), fit.x, xs_new, fit.theta_w(t, k), ...
                          opts.eps, 1, opts, 'mean', p.m, NNx);
            wn(:, k) = o.mean;
        end
        Xin = wt; Xout = wn; NNo = [];
    case 3
        zt = fit.z(:, :, t);
        wt = fit.w(:, :, t);
        zn = zeros(nnew, size(zt, 2));
        for k = 1:size(zt, 2)
            o = vdgp_krig(zt(:, k), fit.x, xs_new, fit.theta_z(t, k), ...
                          opts.eps, 1, opts, 'mean', p.m, NNx);
            zn(:, k) = o.mean;
        end
        wn = zeros(nnew, size(wt, 2));
        for k = 1:size(wt, 2)
            o = vdgp_krig(wt(:, k), zt, zn, fit.theta_w(t, k), ...
                          opts.eps, 1, opts, 'mean', p.m);
            wn(:, k) = o.mean;
        end
        Xin = wt; Xout = wn; NNo = [];
    otherwise
        error('dgp_predict:layers', 'Unsupported number of layers.');
end

if lite
    o = vdgp_krig(fit.y, Xin, Xout, fit.theta_y(t), fit.g(t), fit.tau2(t), ...
                  opts, 'lite', p.m, NNo);
    mu = o.mean; s2 = o.s2;
else
    o = vdgp_krig(fit.y, Xin, Xout, fit.theta_y(t), fit.g(t), fit.tau2(t), ...
                  opts, 'joint', p.m);
    mu = o.mean; s2 = o.s2; Sg = o.Sigma;
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
