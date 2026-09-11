function [f, info] = vdgp_draw_pt(P, x_new, varargin)
%VDGP_DRAW_PT  Draw from the posterior predictive -- the calibration entry point.
%
%   F = VDGP_DRAW_PT(P, X_NEW)                 one draw
%   F = VDGP_DRAW_PT(P, X_NEW, 'nsamp', K)     K draws
%   [F, INFO] = VDGP_DRAW_PT(...)              also the draws' mu_t, s2_t, index
%
%   P       from VDGP_PREDICTOR
%   X_NEW   1-by-d (or a small n_new-by-d block -- see the caveat below)
%   F       n_new-by-K draws, on the ORIGINAL y scale
%
%   Use this, not VDGP_PREDICT_PT, when the calibration likelihood needs an
%   emulator REALISATION rather than a mean and a variance.
%
%   Why it is so much cheaper.  The posterior predictive is the equal-weight
%   mixture (1/T) sum_t N(mu_t, s2_t) over retained MCMC draws.  A draw from
%   that mixture is: pick t uniformly, then draw N(mu_t, s2_t).  So a single
%   realisation needs mu_t and s2_t for the ONE selected t -- not for all T.
%   VDGP_PREDICT_PT evaluates every draw because it has to report the mixture
%   mean and variance; sampling does not, which makes this roughly T times
%   faster.  It is exact, not an approximation: see the mixture check in
%   test_vdgp.
%
%   Options
%     'nsamp'  number of draws (default 1)
%     'idx'    use these MCMC iteration indices instead of drawing them
%              uniformly.  Pass a fixed index to hold the emulator realisation
%              constant -- see the note on noisy likelihoods below.
%     'm'      conditioning-set size for this call (default: the predictor's)
%
%   WHICH t TO USE, AND WHEN.  Redrawing t at every calibration step gives a
%   noisy likelihood: the chain then targets the calibration posterior only
%   approximately, because the emulator realisation changes underneath it.
%   The usual alternatives are
%
%     * hold one emulator draw fixed for a whole calibration chain
%       ('idx', t0) and repeat the calibration over several t0, pooling the
%       results -- the modularised / multiple-imputation approach; or
%     * average the likelihood over K draws at each step ('nsamp', K), which
%       reduces but does not remove the noise.
%
%   This function supports all three; which is appropriate is a modelling
%   decision, not a software one.
%
%   CAVEAT for n_new > 1.  Draws at different points here share the selected
%   MCMC iteration but are otherwise drawn from their own marginals, so they
%   do NOT carry the emulator's correlation across those points.  If your
%   likelihood compares several inputs at once and they are close relative to
%   the lengthscale, use DGP_PREDICT(..., 'lite', false, 'nsamp', K), which
%   produces properly correlated joint draws.
%
%   See also VDGP_PREDICTOR, VDGP_PREDICT_PT, DGP_PREDICT.

o = struct('nsamp', 1, 'idx', [], 'm', []);
for i = 1:2:numel(varargin)
    if ~isfield(o, varargin{i})
        error('vdgp_draw_pt:opt', 'Unknown option "%s".', varargin{i});
    end
    o.(varargin{i}) = varargin{i+1};
end

K = max(1, round(o.nsamp));
if isempty(o.idx)
    tsel = randi(P.T, 1, K);
else
    tsel = o.idx(:).';
    if any(tsel < 1 | tsel > P.T)
        error('vdgp_draw_pt:idx', 'idx must lie in 1..%d.', P.T);
    end
    if numel(tsel) == 1 && K > 1
        tsel = repmat(tsel, 1, K);
    end
    K = numel(tsel);
end

% Restrict the predictor to just the selected draws, then reuse the ordinary
% per-draw machinery -- with T = K this is a handful of small solves.
%
% Build a fresh struct rather than copying P and overwriting fields: P.w is
% n-by-D-by-T and "Q = P; Q.w = ..." can duplicate the whole array on every
% call, which would cost more than the prediction itself.
Q = struct('layers', P.layers, 'x', P.x, 'y', P.y, 'n', P.n, 'd', P.d, ...
           'm', P.m, 'D', P.D, 'opts', P.opts, 'scaling', P.scaling, ...
           'chunk', P.chunk, 'vc', P.vc, 'pw', P.pw, 'T', K, ...
           'theta_y', P.theta_y(tsel), 'g', P.g(tsel), 'tau2', P.tau2(tsel));
if isfield(P, 'w')
    Q.w = P.w(:, :, tsel);  Q.theta_w = P.theta_w(tsel, :);
end
if isfield(P, 'z')
    Q.z = P.z(:, :, tsel);  Q.theta_z = P.theta_z(tsel, :);
end

args = {'per_draw', true};
if ~isempty(o.m), args = [args, {'m', o.m}]; end
[~, ~, out] = vdgp_predict_pt(Q, x_new, args{:});

mu_t = out.mu_t;                       % n_new-by-K, original y scale
s2_t = out.s2_t;
f    = mu_t + sqrt(max(s2_t, 0)) .* randn(size(mu_t));

if nargout > 1
    info = struct('idx', tsel, 'mu_t', mu_t, 's2_t', s2_t);
end
end
