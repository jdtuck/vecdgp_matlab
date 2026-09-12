function [mu_t, s2_t] = vdgp_moments_draws(P, xs, tsel)
%VDGP_MOMENTS_DRAWS  Per-draw predictive moments, batched over TEST POINTS.
%
%   [MU_T, S2_T] = VDGP_MOMENTS_DRAWS(P, XS, TSEL)
%
%   P     predictor from VDGP_PREDICTOR
%   XS    n_new-by-d test inputs, already on the SCALED x scale
%   TSEL  indices of the retained MCMC draws to evaluate
%
%   MU_T, S2_T are n_new-by-numel(TSEL), on the ORIGINAL y scale.
%
%   This is the second of the package's two prediction orientations:
%
%     * batch over DRAWS, loop over points -- VDGP_PREDICT_PT's own code.
%       Wins when there are few points and many retained draws, which is the
%       one-point-at-a-time calibration pattern.
%
%     * batch over POINTS, loop over draws -- this function.  Wins when there
%       are many points and few draws, because each draw becomes a single
%       VDGP_KRIG call over the whole test block, and VDGP_KRIG is the
%       MEX-accelerated path.
%
%   Both evaluate the same Vecchia predictor with the same conditioning sets,
%   so they agree to round-off; see the check in test_vdgp.
%
%   See also VDGP_PREDICT_PT, VDGP_DRAW_PT, VDGP_KRIG.

nnew = size(xs, 1);
K    = numel(tsel);

opts = P.opts;
opts.m       = P.m;
opts.vecchia = true;

% conditioning sets for the test block, in x space, computed once
NNx = vdgp_knn(P.x, xs, min(P.m, P.n));

mu_t = zeros(nnew, K);
s2_t = zeros(nnew, K);

for j = 1:K
    t = tsel(j);

    switch P.layers
        case 1
            Xin = P.x;  Xout = xs;
        case 2
            wt = P.w(:, :, t);
            wn = zeros(nnew, P.D);
            for k = 1:P.D
                o = vdgp_krig(wt(:, k), P.x, xs, P.theta_w(t, k), ...
                              opts.eps, 1, opts, 'mean', P.m, NNx);
                wn(:, k) = o.mean;
            end
            Xin = wt;  Xout = wn;
        case 3
            zt = P.z(:, :, t);
            zn = zeros(nnew, size(zt, 2));
            for k = 1:size(zt, 2)
                o = vdgp_krig(zt(:, k), P.x, xs, P.theta_z(t, k), ...
                              opts.eps, 1, opts, 'mean', P.m, NNx);
                zn(:, k) = o.mean;
            end
            wt = P.w(:, :, t);
            wn = zeros(nnew, P.D);
            for k = 1:P.D
                o = vdgp_krig(wt(:, k), zt, zn, P.theta_w(t, k), ...
                              opts.eps, 1, opts, 'mean', P.m, NNx);
                wn(:, k) = o.mean;
            end
            Xin = wt;  Xout = wn;
        otherwise
            error('vdgp_moments_draws:layers', 'Unsupported number of layers.');
    end

    o = vdgp_krig(P.y, Xin, Xout, P.theta_y(t), P.g(t), P.tau2(t), ...
                  opts, 'lite', P.m, NNx);
    mu_t(:, j) = o.mean;
    s2_t(:, j) = o.s2;
end

mu_t = mu_t * P.scaling.ysd + P.scaling.ymean;
s2_t = s2_t * P.scaling.ysd^2;
end
