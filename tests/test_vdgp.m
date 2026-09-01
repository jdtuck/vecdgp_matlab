function test_vdgp()
%TEST_VDGP  Verification suite for the Vecchia deep GP package.
%
%   Run with   test_vdgp
%
%   Each check compares an implementation against an independent, exact
%   reference (a dense Cholesky, a closed-form conditional, or a known target
%   distribution).  All of them must pass before the sampler output is worth
%   believing.

here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, '..'));
if exist('rng', 'file'), rng(11); else, rand('seed', 11); randn('seed', 11); end

res = false(0, 1);          % one entry per check

fprintf('\n=== Vecchia DGP verification ===\n\n');
if exist('vdgp_U_entries_mex', 'file') == 3
    fprintf('  (MEX acceleration active)\n\n');
end

n = 45; d = 2;
X = rand(n, d);
y = randn(n, 1);
theta = 0.35; g = 1e-3; v = 2.5; ct = 'matern';

% -- 1. U reproduces the exact precision when the conditioning set is full --
A = vdgp_create_approx(X, n-1, 'random', true);
[U, dU] = vdgp_create_U(A, theta, g, v, ct);
K = vdgp_cov(A.X_ord, [], theta, g, 1, v, ct);
err = max(max(abs(full(U*U.') - inv(K))));
res(end+1) = report('U*U'' equals inv(Sigma) at m = n-1', err < 1e-8, ...
                    sprintf('max err %.2e', err));

err = abs(2*sum(log(dU)) + log(det(K)));
res(end+1) = report('log|Q| from diag(U)', err < 1e-8, sprintf('err %.2e', err));

% -- 2. likelihood matches the dense reference, and converges in m ----------
Kf = vdgp_cov(X, [], theta, g, 1, v, ct);
Lf = chol(Kf, 'lower');
ll_exact = -0.5*n*log(2*pi) - sum(log(diag(Lf))) - 0.5*sum((Lf\y).^2);
o = vdgp_logl(y, A, theta, g, v, ct, false);
res(end+1) = report('Vecchia log-likelihood at m = n-1', abs(o.ll - ll_exact) < 1e-8, ...
                    sprintf('%.10f vs %.10f', o.ll, ll_exact));

lls = zeros(1, 4); ms = [5 10 20 44];
for i = 1:4
    Am = vdgp_create_approx(X, ms(i), 'random', true);
    oo = vdgp_logl(y, Am, theta, g, v, ct, false);
    lls(i) = oo.ll;
end
mono = all(diff(abs(lls - ll_exact)) < 1e-8);
res(end+1) = report('log-likelihood error decreases with m', mono, ...
                    sprintf('|err| = [%s]', sprintf('%.3f ', abs(lls - ll_exact))));

% -- 3. profiled (outer) likelihood ---------------------------------------
o = vdgp_logl(y, A, theta, g, v, ct, true);
tau2ref = (y' * (Kf \ y)) / n;
llref = -0.5*n*log(2*pi) - 0.5*n*log(tau2ref) - sum(log(diag(Lf))) - 0.5*n;
res(end+1) = report('profiled outer log-likelihood and tau2', ...
                    abs(o.ll - llref) < 1e-7 && abs(o.tau2 - tau2ref) < 1e-9, ...
                    sprintf('tau2 %.6f vs %.6f', o.tau2, tau2ref));

% -- 4. prior sampling has the right covariance ----------------------------
nS = 20000; ns = 12;
Xs = rand(ns, 1);
As = vdgp_create_approx(Xs, ns-1, 'random', true);
S = zeros(ns, nS);
for i = 1:nS
    S(:, i) = vdgp_rand_mvn(As, 0.3, 1e-6, v, ct, 1);
end
Kref = vdgp_cov(Xs, [], 0.3, 1e-6, 1, v, ct);
Cerr = max(max(abs(cov(S.') - Kref)));
res(end+1) = report('rand_mvn empirical covariance', Cerr < 0.06, ...
                    sprintf('max err %.3f', Cerr));

% -- 5. Vecchia prediction equals the exact GP conditional at full m -------
nnew = 20;
Xn = rand(nnew, d);
optsE = vdgp_options('vecchia', false, 'cov', ct, 'v', v);
optsV = vdgp_options('vecchia', true,  'cov', ct, 'v', v, 'm', n);
pe = vdgp_krig(y, X, Xn, theta, g, 2.0, optsE, 'lite');
pv = vdgp_krig(y, X, Xn, theta, g, 2.0, optsV, 'lite', n);
e1 = max(abs(pe.mean - pv.mean)); e2 = max(abs(pe.s2 - pv.s2));
res(end+1) = report('lite prediction, m = n, matches exact', e1 < 1e-9 && e2 < 1e-9, ...
                    sprintf('mean %.2e, s2 %.2e', e1, e2));

optsJ = vdgp_options('vecchia', true, 'cov', ct, 'v', v, 'm', n + nnew);
pj  = vdgp_krig(y, X, Xn, theta, g, 2.0, optsJ, 'joint', n + nnew);
pej = vdgp_krig(y, X, Xn, theta, g, 2.0, optsE, 'joint');
e1 = max(abs(pej.mean - pj.mean));
e2 = max(max(abs(pej.Sigma - pj.Sigma)));
res(end+1) = report('joint prediction, full m, matches exact', e1 < 1e-7 && e2 < 1e-6, ...
                    sprintf('mean %.2e, Sigma %.2e', e1, e2));

% -- 5b. joint sample paths have the covariance the joint mode reports -----
% 'sample' never forms Sigma (it applies U22^{-T} to white noise), so this
% checks that shortcut against the explicitly assembled joint covariance.
os = vdgp_krig(y, X, Xn, theta, g, 2.0, optsJ, 'sample', n + nnew, [], 30000);
e1 = max(abs(mean(os.draws, 2) - pj.mean)) / sqrt(max(diag(pj.Sigma)));
e2 = max(max(abs(cov(os.draws.') - pj.Sigma))) / max(abs(pj.Sigma(:)));
res(end+1) = report('joint draws match the joint covariance', ...
                    e1 < 0.05 && e2 < 0.08, ...
                    sprintf('mean %.3f sd, cov %.3f rel', e1, e2));

% -- 6. the Metropolis-Hastings kernel targets its prior -------------------
% With n = 1 there are no distances, so the likelihood does not depend on
% theta and the sampler must reproduce the Gamma(alpha, beta) prior exactly.
A1 = vdgp_create_approx(0.5, 0, 'none', true);
opts1 = vdgp_options('vecchia', true, 'cov', ct, 'v', v);
alpha = 1.5; beta = 3.9/4;
th = 0.5; T = 40000; keepT = zeros(T, 1);
oo = vdgp_logl(1.0, A1, th, 1e-4, v, ct, false); ll = oo.ll;
for t = 1:T
    [th, ll] = vdgp_sample_theta(1.0, A1, th, 1, 1e-4, ll, alpha, beta, opts1, false);
    keepT(t) = th;
end
mref = alpha/beta; sref = sqrt(alpha)/beta;
em = abs(mean(keepT) - mref)/mref; es = abs(std(keepT) - sref)/sref;
res(end+1) = report('MH kernel recovers the Gamma prior', em < 0.05 && es < 0.08, ...
                    sprintf('mean %.3f (%.3f), sd %.3f (%.3f)', ...
                            mean(keepT), mref, std(keepT), sref));

% -- 7. elliptical slice sampling leaves the prior invariant ---------------
% Same trick: with n = 1 the outer likelihood is free of w, so ESS must
% return draws from the N(0, 1 + eps) prior.
Aw = vdgp_create_approx(0.5, 0, 'none', true);
optsw = vdgp_options('vecchia', true, 'cov', ct, 'v', v);
w = 0; T = 20000; keepW = zeros(T, 1);
wobj = vdgp_create_approx(0, 0, 'none', true);
oo = vdgp_logl(1.0, wobj, 0.1, 1e-4, v, ct, true); ll = oo.ll;
for t = 1:T
    [w, wobj, ll] = vdgp_sample_w(1.0, w, Aw, wobj, 0.1, 0.1, 1e-4, ll, optsw);
    keepW(t) = w;
end
em = abs(mean(keepW)); es = abs(std(keepW) - 1);
res(end+1) = report('ESS leaves the MVN prior invariant', em < 0.05 && es < 0.05, ...
                    sprintf('mean %.3f (0), sd %.3f (1)', mean(keepW), std(keepW)));

% -- 8. conditioning sets are valid ---------------------------------------
Ac = vdgp_create_approx(rand(200, 2), 15, 'random', true);
NN = Ac.NNarray;
ok = true;
for i = 2:200
    c = NN(i, 2:end); c = c(~isnan(c));
    if numel(c) ~= min(15, i-1) || any(c >= i) || numel(unique(c)) ~= numel(c)
        ok = false; break
    end
    dall = sum((Ac.X_ord(1:i-1, :) - Ac.X_ord(i, :)).^2, 2);
    [~, oidx] = sort(dall);
    if ~isequal(sort(c(:)), sort(oidx(1:numel(c))))
        ok = false; break
    end
end
res(end+1) = report('ordered nearest-neighbour sets are exact', ok, '');

% -- 9. maxmin ordering is a permutation ----------------------------------
Xm = rand(300, 3);
om = vdgp_order(Xm, 'maxmin');
res(end+1) = report('maxmin ordering is a valid permutation', ...
                    isequal(sort(om), 1:300), '');

% -- 10. end-to-end: the deepgp 1-D example -------------------------------
f = @(x) (x <= 0.58) .* (sin(pi*x*6) + cos(pi*x*12)) + (x > 0.58) .* (5*x - 4.9);
xtr = linspace(0, 1, 25)'; ytr = f(xtr);
xte = linspace(0, 1, 200)'; yte = f(xte);
fit = fit_two_layer(xtr, ytr, 'nmcmc', 1200, 'vecchia', true, 'm', 10, ...
                    'true_g', 1e-6, 'verb', 0);
fit = dgp_trim(fit, 600, 2);
p = dgp_predict(fit, xte);
rmse  = sqrt(mean((p.mean - yte).^2));
cov95 = mean(abs(p.mean - yte) <= 1.96 * p.sd);
res(end+1) = report('two-layer Vecchia DGP on the deepgp example', ...
                    rmse < 0.08 && cov95 > 0.85, ...
                    sprintf('RMSE %.4f, coverage %.2f', rmse, cov95));

% -- 11. posterior predictive sample paths reproduce the mixture -----------
% pred.f pools joint draws across MCMC iterations, so its sample mean and
% covariance must converge to the reported mixture mean and covariance.
xs = xte(1:5:end);
ps = dgp_predict(fit, xs, 'lite', false, 'nsamp', 6000);
e1 = max(abs(mean(ps.f, 2) - ps.mean)) / max(ps.sd);
e2 = max(max(abs(cov(ps.f.') - ps.Sigma))) / max(abs(ps.Sigma(:)));
res(end+1) = report('sample paths reproduce the predictive mixture', ...
                    e1 < 0.08 && e2 < 0.15 && size(ps.f, 2) == 6000, ...
                    sprintf('mean %.3f sd, cov %.3f rel', e1, e2));

% each column must be traceable to the MCMC iteration that produced it
okit = numel(ps.f_iter) == 6000 && all(ps.f_iter >= 1) && ...
       all(ps.f_iter <= fit.nmcmc) && numel(unique(ps.f_iter)) == fit.nmcmc;
res(end+1) = report('sample paths are spread over all retained draws', okit, ...
                    sprintf('%d iterations used', numel(unique(ps.f_iter))));

np = sum(res);
nf = numel(res) - np;
fprintf('\n%d passed, %d failed\n\n', np, nf);
if nf > 0
    error('test_vdgp:fail', '%d checks failed.', nf);
end
end

% -------------------------------------------------------------------------
function ok = report(name, ok, detail)
%REPORT  Print one check and pass its verdict back to the caller.
%   A plain local function (not a nested one): MATLAB and Octave disagree
%   about workspace sharing for nested functions defined mid-body, so the
%   pass/fail tally is accumulated by the caller instead.
ok = logical(ok);
if ok
    fprintf('  PASS  %-52s %s\n', name, detail);
else
    fprintf('  FAIL  %-52s %s\n', name, detail);
end
end
