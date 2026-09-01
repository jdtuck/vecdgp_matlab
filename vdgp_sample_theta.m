function [theta, ll, tau2, acc] = vdgp_sample_theta(y, A, theta, idx, g, ll_prev, ...
                                                    alpha, beta, opts, outer)
%VDGP_SAMPLE_THETA  Metropolis-Hastings update of one lengthscale.
%
%   [THETA, LL, TAU2, ACC] = VDGP_SAMPLE_THETA(Y, A, THETA, IDX, G, LL_PREV,
%                                              ALPHA, BETA, OPTS, OUTER)
%
%   Uses the uniform sliding-window proposal of the deepgp package,
%
%        theta* ~ Unif( l * theta / u , u * theta / l )                l=1, u=2
%
%   whose density is proportional to 1/theta, so the Hastings ratio
%   q(theta|theta*) / q(theta*|theta) = theta / theta* contributes
%   log(theta) - log(theta*) to the acceptance ratio.  The prior is
%   Gamma(ALPHA, BETA) on theta - eps (rate parameterisation), matching
%   dgamma(theta - eps, alpha, beta) in deepgp.
%
%   Only element IDX of the (possibly vector) THETA is updated.

if nargin < 10 || isempty(outer), outer = true; end

l  = opts.l;
u  = opts.u;
ep = opts.eps;

th_old = theta(idx);
lo = l * th_old / u;
hi = u * th_old / l;
th_new = lo + (hi - lo) * rand;

theta_star = theta;
theta_star(idx) = th_new;

out = vdgp_logl(y, A, theta_star, g, opts.v, opts.cov, outer);

lp_new = vdgp_lgamma(th_new - ep, alpha, beta);
lp_old = vdgp_lgamma(th_old - ep, alpha, beta);

lacc = (out.ll + lp_new) - (ll_prev + lp_old) + log(th_old) - log(th_new);

if log(rand) < lacc
    theta = theta_star;
    ll    = out.ll;
    tau2  = out.tau2;
    acc   = true;
else
    ll   = ll_prev;
    tau2 = [];
    acc  = false;
end
end
