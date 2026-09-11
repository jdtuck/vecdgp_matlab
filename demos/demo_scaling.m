%DEMO_SCALING  Cost and accuracy benchmark for the Vecchia DGP.
%
% Four questions, each answered with measurements rather than assertion:
%
%   1. how does the sampler cost grow with n?          (expect O(n))
%   2. how does it grow with the conditioning size m?  (expect O(m^3))
%   3. what does prediction cost, for a batch of test points and for the
%      one-point-at-a-time pattern a calibration MCMC uses?
%   4. how much accuracy do you give up by shrinking m or thinning the
%      retained draws at prediction time -- the two cheap speed-ups?
%
% Results are returned in the struct BENCH and plotted.  Defaults are sized
% to finish in a few minutes; raise NS / NMCMC for a serious run.
%
% Run from the repo root (or with the repo on the path).

clear; close all;
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, '..'));

if exist('rng', 'file'), rng(4); else, rand('seed', 4); randn('seed', 4); end

% ---- knobs ---------------------------------------------------------------
NS        = [500 1000 2000 4000];   % training sizes for the n-scaling study
MS        = [5 10 15 25 35];        % conditioning sizes for the m study
NMCMC     = 30;                     % sweeps timed per configuration
M_FIX     = 25;                     % m held fixed in the n study
N_FIX     = 2000;                   % n held fixed in the m study
EXACT_MAX = 1000;                   % skip the exact sampler above this n
NTEST     = 500;                    % test points for the batch-prediction study
TRUE_G    = 1e-4;

f = @(X) atan(20 * (X(:,1) - 0.5)) .* sin(3*pi*X(:,2)) + 0.5 * X(:,2);

bench = struct();

% =========================================================================
% 1. cost vs n
% =========================================================================
fprintf('\n=== 1. sampler cost vs n  (m = %d) ===\n', M_FIX);
fprintf('%8s %12s %12s %10s\n', 'n', 'vecchia s/it', 'exact s/it', 'ratio');
res_n = [];
for n = NS
    X = rand(n, 2);  y = f(X) + 0.01*randn(n, 1);

    A = vdgp_create_approx(X, M_FIX, 'random', true);      %#ok<NASGU>
    t = tic; fv = fit_two_layer(X, y, 'nmcmc', NMCMC+1, 'm', M_FIX, ...
                                'verb', 0, 'true_g', TRUE_G); tv = toc(t);
    % subtract the one-off setup so the number is a steady-state sweep cost
    t = tic; vdgp_create_approx(X, M_FIX, 'random', true); tset = toc(t);
    sv = (tv - tset) / NMCMC;

    se = NaN;
    if n <= EXACT_MAX
        t = tic; fe = fit_two_layer(X, y, 'nmcmc', NMCMC+1, 'vecchia', false, ...
                                    'verb', 0, 'true_g', TRUE_G); se = toc(t)/NMCMC;
    end
    if isnan(se)
        fprintf('%8d %12.4f %12s %10s\n', n, sv, '-', '-');
    else
        fprintf('%8d %12.4f %12.4f %10.1f\n', n, sv, se, se/sv);
    end
    res_n(end+1, :) = [n, sv, se, tset]; %#ok<SAGROW>
end
bench.n = struct('cols', {{'n','vecchia_s_per_sweep','exact_s_per_sweep','setup_s'}}, ...
                 'data', res_n);

% =========================================================================
% 2. cost vs m
% =========================================================================
fprintf('\n=== 2. sampler cost vs m  (n = %d) ===\n', N_FIX);
fprintf('%8s %12s %14s\n', 'm', 's/sweep', 'vs m^3 pred.');
X = rand(N_FIX, 2);  y = f(X) + 0.01*randn(N_FIX, 1);
res_m = [];
for mm = MS
    t = tic; fv = fit_two_layer(X, y, 'nmcmc', NMCMC+1, 'm', mm, ...
                                'verb', 0, 'true_g', TRUE_G); tv = toc(t);
    t = tic; vdgp_create_approx(X, mm, 'random', true); tset = toc(t);
    res_m(end+1, :) = [mm, (tv - tset)/NMCMC]; %#ok<SAGROW>
end
base = res_m(1,2) / (MS(1)+1)^3;
for i = 1:size(res_m,1)
    fprintf('%8d %12.4f %14.4f\n', res_m(i,1), res_m(i,2), base*(res_m(i,1)+1)^3);
end
bench.m = struct('cols', {{'m','s_per_sweep'}}, 'data', res_m);

% =========================================================================
% 3. prediction cost: batch vs one-at-a-time
% =========================================================================
fprintf('\n=== 3. prediction cost ===\n');
n = 2000;
X = rand(n, 2);  y = f(X) + 0.01*randn(n, 1);
fit = fit_two_layer(X, y, 'nmcmc', 400, 'm', M_FIX, 'verb', 0, 'true_g', TRUE_G);
fit = dgp_trim(fit, 200, 1);
Xt  = rand(NTEST, 2);  yt = f(Xt);
x0  = rand(1, 2);

t = tic; pB = dgp_predict(fit, Xt); tB = toc(t);
t = tic; P = vdgp_predictor(fit); tP = toc(t);
t = tic; for i = 1:10, vdgp_predict_pt(P, x0); end; tS = toc(t)/10;
t = tic; for i = 1:3,  dgp_predict(fit, x0);   end; tS0 = toc(t)/3;

fprintf('  T = %d retained draws\n', fit.nmcmc);
fprintf('  batch, %d test points   : dgp_predict      %8.3f s  (%.4f s/point)\n', ...
        NTEST, tB, tB/NTEST);
fprintf('  ONE point (calibration) : dgp_predict      %8.4f s\n', tS0);
fprintf('                          : vdgp_predict_pt  %8.4f s   => %.0fx\n', tS, tS0/tS);
fprintf('  predictor build (once)  : %8.4f s\n', tP);
bench.pred = struct('T', fit.nmcmc, 'batch_s', tB, 'ntest', NTEST, ...
                    'one_dgp_predict_s', tS0, 'one_predict_pt_s', tS, 'build_s', tP);

% =========================================================================
% 4. accuracy vs the two cheap speed-ups
% =========================================================================
fprintf('\n=== 4. what shrinking m / thinning costs you ===\n');
[muRef, s2Ref] = vdgp_predict_pt(P, Xt);          % full m, full T = reference
rmseRef = sqrt(mean((muRef - yt).^2));
fprintf('  reference (m = %d, all %d draws): RMSE %.4f, mean sd %.4f\n', ...
        P.m, P.T, rmseRef, mean(sqrt(s2Ref)));
fprintf('\n%6s %6s %10s %10s %12s %12s\n', 'm', 'thin', 's/call', 'RMSE', ...
        'd(mean)/sd', 'd(sd)/sd');
res_a = [];
for mm = [5 10 15 25]
    for th = [1 5 20]
        t = tic; for i = 1:5, vdgp_predict_pt(P, x0, 'm', mm, 'thin', th); end
        tc = toc(t)/5;
        [mu2, s22] = vdgp_predict_pt(P, Xt, 'm', mm, 'thin', th);
        rm  = sqrt(mean((mu2 - yt).^2));
        dm  = mean(abs(mu2 - muRef) ./ sqrt(s2Ref));
        ds  = mean(abs(sqrt(s22) - sqrt(s2Ref)) ./ sqrt(s2Ref));
        fprintf('%6d %6d %10.4f %10.4f %12.3f %12.3f\n', mm, th, tc, rm, dm, ds);
        res_a(end+1, :) = [mm, th, tc, rm, dm, ds]; %#ok<SAGROW>
    end
end
bench.acc = struct('cols', {{'m','thin','s_per_call','rmse','dmean_over_sd','dsd_over_sd'}}, ...
                   'data', res_a);
fprintf(['\n  d(mean)/sd is the shift in the predictive mean as a fraction of\n' ...
         '  the predictive sd -- under about 0.1 the change is invisible next\n' ...
         '  to the uncertainty you are already reporting.\n']);

% =========================================================================
% plots
% =========================================================================
figure('Position', [100 100 1100 700]);

subplot(2,2,1);
loglog(res_n(:,1), res_n(:,2), 'o-', 'LineWidth', 1.5); hold on;
ok = ~isnan(res_n(:,3));
if any(ok), loglog(res_n(ok,1), res_n(ok,3), 's-', 'LineWidth', 1.5); end
loglog(res_n(:,1), res_n(1,2)*res_n(:,1)/res_n(1,1), 'k:', 'LineWidth', 1);
xlabel('n'); ylabel('seconds / Gibbs sweep'); grid on;
legend({'Vecchia', 'exact', 'O(n)'}, 'Location', 'northwest');
title(sprintf('cost vs n  (m = %d)', M_FIX));

subplot(2,2,2);
loglog(res_m(:,1)+1, res_m(:,2), 'o-', 'LineWidth', 1.5); hold on;
loglog(res_m(:,1)+1, base*(res_m(:,1)+1).^3, 'k:', 'LineWidth', 1);
xlabel('m + 1'); ylabel('seconds / Gibbs sweep'); grid on;
legend({'measured', 'O(m^3)'}, 'Location', 'northwest');
title(sprintf('cost vs m  (n = %d)', N_FIX));

subplot(2,2,3);
mv = unique(res_a(:,1));
for i = 1:numel(mv)
    sel = res_a(:,1) == mv(i);
    semilogy(res_a(sel,2), res_a(sel,3), 'o-', 'LineWidth', 1.3); hold on;
end
xlabel('thinning of retained draws'); ylabel('seconds / single-point call');
grid on; legend(arrayfun(@(v) sprintf('m = %d', v), mv, 'UniformOutput', false));
title('calibration-path cost');

subplot(2,2,4);
for i = 1:numel(mv)
    sel = res_a(:,1) == mv(i);
    semilogx(res_a(sel,3), res_a(sel,5), 'o-', 'LineWidth', 1.3); hold on;
end
xlabel('seconds / call'); ylabel('|\Deltamean| / sd'); grid on;
legend(arrayfun(@(v) sprintf('m = %d', v), mv, 'UniformOutput', false));
title('accuracy bought per second');

drawnow;
if ~isempty(getenv('VDGP_SAVEFIG'))
    print(gcf, fullfile(here, 'demo_scaling.png'), '-dpng', '-r120');
end

fprintf('\nResults are in the struct "bench".\n');
