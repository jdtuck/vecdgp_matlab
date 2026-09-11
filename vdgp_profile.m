function info = vdgp_profile(n, d, m, D)
%VDGP_PROFILE  Where the time goes, and whether the fast paths are active.
%
%   vdgp_profile              synthetic 2-D problem at n = 5000, m = 25
%   vdgp_profile(n, d, m, D)  choose the size, input dimension, conditioning
%                             set size and hidden width
%   INFO = vdgp_profile(...)  also returns the numbers as a struct
%
%   Prints the per-call cost of every kernel the sampler uses, the projected
%   cost of a full run, and -- most importantly -- whether the MEX
%   accelerations and a parallel pool are available.  Run this first whenever
%   the sampler feels slow: the usual answer is that VDGP_BUILD_MEX has not
%   been run.

if nargin < 1 || isempty(n), n = 5000; end
if nargin < 2 || isempty(d), d = 2;    end
if nargin < 3 || isempty(m), m = 25;   end
if nargin < 4 || isempty(D), D = d;    end

have = struct('U',    exist('vdgp_U_entries_mex', 'file') == 3, ...
              'logl', exist('vdgp_logl_mex', 'file')     == 3, ...
              'krig', exist('vdgp_krig_mex', 'file')     == 3, ...
              'knn',  exist('knnsearch', 'file') == 2 || exist('knnsearch', 'builtin') == 5, ...
              'mink', exist('mink', 'file') == 2 || exist('mink', 'builtin') == 5);

fprintf('\n=== vdgp_profile:  n = %d, d = %d, m = %d, D = %d ===\n\n', n, d, m, D);
fprintf('  MEX vdgp_logl_mex      : %s\n', local_yn(have.logl));
fprintf('  MEX vdgp_U_entries_mex : %s\n', local_yn(have.U));
fprintf('  MEX vdgp_krig_mex      : %s\n', local_yn(have.krig));
fprintf('  knnsearch (Stats tbx)  : %s\n', local_yn(have.knn));
if ~(have.logl && have.U && have.krig)
    fprintf('\n  >> Run vdgp_build_mex to compile the C kernels. Expect roughly a\n');
    fprintf('     10x speed-up on everything below.\n');
end

X = rand(n, d);
y = randn(n, 1);

t = tic; A = vdgp_create_approx(X, m, 'random', true); t_nn = toc(t);

R = max(3, min(20, round(2e5 / n)));
t = tic; for r = 1:R, o = vdgp_logl(y, A, 0.1, 1e-4, 2.5, 'matern', true); end
t_ll = toc(t) / R;
t = tic; for r = 1:R, s = vdgp_rand_mvn(A, 0.1, 1e-8, 2.5, 'matern', 1); end
t_rm = toc(t) / R;

nnew = min(1000, n);
Xt = rand(nnew, d);
NNx = vdgp_knn(X, Xt, m);
opts = vdgp_options('m', m);
t = tic; for r = 1:R, k = vdgp_krig(y, X, Xt, 0.1, 1e-4, 1, opts, 'lite', m, NNx); end
t_kr = toc(t) / R;

% A Gibbs sweep is ~2 + 2D likelihood calls for the hyperparameters plus the
% elliptical slice sampler, which needs of order 10 likelihood evaluations per
% hidden node once n is in the thousands.
nll_sweep = 2 + 2*D + 10*D;
t_sweep   = nll_sweep * t_ll + D * t_rm;
t_pred    = (D + 1) * t_kr;

fprintf('\n  one-off setup (ordering + neighbours) : %8.3f s\n', t_nn);
fprintf('  log-likelihood, one call              : %8.4f s\n', t_ll);
fprintf('  prior draw (rand_mvn), one call       : %8.4f s\n', t_rm);
fprintf('  kriging %d test points, one call     : %8.4f s\n', nnew, t_kr);
fprintf('\n  projected Gibbs sweep (~%d likelihoods): %7.3f s\n', nll_sweep, t_sweep);
fprintf('  projected 10000-iteration fit          : %7.1f min\n', 1e4 * t_sweep / 60);
fprintf('  projected prediction, 2500 draws       : %7.1f min\n', 2500 * t_pred / 60);

fprintf('\n  Levers, in order of effect:\n');
fprintf('   1. vdgp_build_mex (if any line above says NO)\n');
fprintf('   2. threads: the C kernels are OpenMP-parallel over observations;\n');
fprintf('      set OMP_NUM_THREADS before starting MATLAB\n');
fprintf('   3. m: sampler cost grows with m, though more slowly than m^3 --\n');
fprintf('      measured ~2.9x from m = 15 to m = 25 at n = 2000. Run\n');
fprintf('      demos/demo_scaling for the cost AND the accuracy it costs you.\n');
fprintf('   4. prediction: for one point at a time (calibration), use\n');
fprintf('      vdgp_predictor + vdgp_predict_pt, and thin the retained draws --\n');
fprintf('      thinning is close to free. dgp_predict(..., ''cores'', N)\n');
fprintf('      parallelises the many-test-point case over MCMC draws.\n\n');

info = struct('have', have, 't_setup', t_nn, 't_logl', t_ll, 't_randmvn', t_rm, ...
              't_krig', t_kr, 't_sweep', t_sweep, 'n', n, 'd', d, 'm', m, 'D', D);
end

% -------------------------------------------------------------------------
function s = local_yn(tf)
if tf, s = 'yes'; else, s = 'NO'; end
end
