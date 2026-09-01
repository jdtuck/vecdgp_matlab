function [z, z_obj, ll] = vdgp_sample_z(w, z, x_obj, z_obj, theta_w, theta_z, ll_prev, opts)
%VDGP_SAMPLE_Z  Elliptical slice sampling of the inner layer of a 3-layer DGP.
%
%   [Z, Z_OBJ, LL] = VDGP_SAMPLE_Z(W, Z, X_OBJ, Z_OBJ, THETA_W, THETA_Z,
%                                  LL_PREV, OPTS)
%
%   The prior draws come from the x-layer (lengthscales THETA_Z) and the
%   likelihood is the middle-layer density of W given Z, i.e. the sum over the
%   hidden nodes of  log N( W(:,k) ; 0, Sigma_k(Z) )  with unit scale and a
%   negligible nugget -- exactly deepgp's sample_z.

D = size(z, 2);
Dw = size(w, 2);

for i = 1:D
    z_prior = vdgp_rand_mvn(x_obj, theta_z(i), opts.eps, opts.v, opts.cov, 1);

    a    = 2*pi*rand;
    amin = a - 2*pi;
    amax = a;
    ll_threshold = ll_prev + log(rand);

    z_prev = z(:, i);
    accept = false;
    count  = 0;
    while ~accept
        count = count + 1;
        z(:, i) = z_prev * cos(a) + z_prior * sin(a);
        z_obj   = vdgp_update_approx(z_obj, z);

        ll_new = 0;
        for k = 1:Dw
            o = vdgp_logl(w(:, k), z_obj, theta_w(k), opts.eps, opts.v, opts.cov, false);
            ll_new = ll_new + o.ll;
        end

        if ll_new > ll_threshold
            ll_prev = ll_new;
            accept  = true;
        else
            if a < 0, amin = a; else, amax = a; end
            a = amin + (amax - amin) * rand;
            if count > 100
                z(:, i) = z_prev;
                z_obj   = vdgp_update_approx(z_obj, z);
                ll_prev = local_mid_ll(w, z_obj, theta_w, opts);
                accept  = true;
                warning('vdgp_sample_z:ess', 'ESS reached 100 proposals; keeping current state.');
            end
        end
    end
end

ll = ll_prev;
end

% -------------------------------------------------------------------------
function ll = local_mid_ll(w, z_obj, theta_w, opts)
ll = 0;
for k = 1:size(w, 2)
    o = vdgp_logl(w(:, k), z_obj, theta_w(k), opts.eps, opts.v, opts.cov, false);
    ll = ll + o.ll;
end
end
