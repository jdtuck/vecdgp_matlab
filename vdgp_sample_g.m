function [g, ll, tau2, acc] = vdgp_sample_g(y, A, theta, g, ll_prev, alpha, beta, opts)
%VDGP_SAMPLE_G  Metropolis-Hastings update of the nugget.
%
%   Identical machinery to VDGP_SAMPLE_THETA (uniform sliding-window proposal
%   with a Gamma(ALPHA, BETA) prior on g - eps); the nugget only ever appears
%   in the outer/observation layer, so the scale is always profiled out.

l  = opts.l;
u  = opts.u;
ep = opts.eps;

lo = l * g / u;
hi = u * g / l;
g_new = lo + (hi - lo) * rand;

out = vdgp_logl(y, A, theta, g_new, opts.v, opts.cov, true);

lp_new = vdgp_lgamma(g_new - ep, alpha, beta);
lp_old = vdgp_lgamma(g     - ep, alpha, beta);

lacc = (out.ll + lp_new) - (ll_prev + lp_old) + log(g) - log(g_new);

if log(rand) < lacc
    g    = g_new;
    ll   = out.ll;
    tau2 = out.tau2;
    acc  = true;
else
    ll   = ll_prev;
    tau2 = [];
    acc  = false;
end
end
