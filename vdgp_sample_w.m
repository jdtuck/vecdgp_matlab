function [w, w_obj, ll, tau2, nprop] = vdgp_sample_w(y, w, prior_obj, w_obj, ...
                                                     theta_y, theta_w, g, ll_prev, opts)
%VDGP_SAMPLE_W  Elliptical slice sampling of the layer that feeds the response.
%
%   [W, W_OBJ, LL, TAU2] = VDGP_SAMPLE_W(Y, W, PRIOR_OBJ, W_OBJ, THETA_Y,
%                                        THETA_W, G, LL_PREV, OPTS)
%
%   W          n-by-D current latent layer (original ordering)
%   PRIOR_OBJ  approximation object for the layer BELOW w (x for a two-layer
%              DGP, z for a three-layer DGP); supplies the MVN prior draws
%   W_OBJ      approximation object whose inputs are w itself; supplies the
%              outer-layer likelihood of Y | W
%   THETA_W    1-by-D lengthscales of the w layer (one per hidden node)
%
%   Each column is updated in turn with Murray, Adams & MacKay's (2010)
%   elliptical slice sampler, which is tuning-free and always accepts:
%
%       nu ~ N(0, Sigma_w)                       (Vecchia prior draw, O(n m))
%       threshold = ll_prev + log(Unif(0,1))
%       a ~ Unif(0, 2 pi),  bracket [a - 2 pi, a]
%       repeat:  w' = w cos(a) + nu sin(a)
%                accept if loglik(Y | w') > threshold, else shrink the bracket
%
%   The prior mean is zero, matching Sauer, Cooper & Gramacy (2023).

D = size(w, 2);
tau2 = [];
nprop = 0;

for i = 1:D
    w_prior = vdgp_rand_mvn(prior_obj, theta_w(i), opts.eps, opts.v, opts.cov, 1);

    a    = 2*pi*rand;
    amin = a - 2*pi;
    amax = a;
    ll_threshold = ll_prev + log(rand);

    w_prev = w(:, i);
    accept = false;
    count  = 0;
    while ~accept
        count = count + 1;
        nprop = nprop + 1;
        w(:, i) = w_prev * cos(a) + w_prior * sin(a);
        w_obj   = vdgp_update_approx(w_obj, w);
        out     = vdgp_logl(y, w_obj, theta_y, g, opts.v, opts.cov, true);
        if out.ll > ll_threshold
            ll_prev = out.ll;
            tau2    = out.tau2;
            accept  = true;
        else
            if a < 0, amin = a; else, amax = a; end
            a = amin + (amax - amin) * rand;
            if count > 100
                % Numerically stuck: fall back to the previous state.
                w(:, i) = w_prev;
                w_obj   = vdgp_update_approx(w_obj, w);
                out     = vdgp_logl(y, w_obj, theta_y, g, opts.v, opts.cov, true);
                ll_prev = out.ll;
                tau2    = out.tau2;
                accept  = true;
                warning('vdgp_sample_w:ess', 'ESS reached 100 proposals; keeping current state.');
            end
        end
    end
end

ll = ll_prev;
if isempty(tau2)
    out  = vdgp_logl(y, w_obj, theta_y, g, opts.v, opts.cov, true);
    tau2 = out.tau2;
end
end
