%DEMO_1D  The deepgp 1-D example, run with the Vecchia-approximated DGP.
%
% Reproduces the test problem from ?fit_two_layer in the deepgp R package:
%
%     f(x) = sin(6 pi x) + cos(12 pi x)   for x <= 0.58
%            5x - 4.9                     otherwise
%
% a function with a sharp regime change that a stationary GP handles badly and
% a two-layer DGP handles well.  The R code this mirrors is
%
%     fit <- fit_two_layer(x, y, nmcmc = 2000, vecchia = TRUE, m = 10)
%     fit <- trim(fit, 1000, 2)
%     fit <- predict(fit, xx, cores = 1)
%
% Run from the package root (or with the package on the path).

clear; close all;
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, '..'));

rng_seed = 1;
if exist('rng', 'file'), rng(rng_seed); else, rand('seed', rng_seed); randn('seed', rng_seed); end

% ---- data ---------------------------------------------------------------
f = @(x) (x <= 0.58) .* (sin(pi*x*6) + cos(pi*x*12)) + (x > 0.58) .* (5*x - 4.9);
x  = linspace(0, 1, 25)';
y  = f(x);
xx = linspace(0, 1, 200)';
yy = f(xx);

nmcmc = 2000;
burn  = 1000;
thin  = 2;

% ---- two-layer DGP, Vecchia (m = 10) ------------------------------------
fprintf('--- two-layer DGP, vecchia = true, m = 10 ---\n');
fit2 = fit_two_layer(x, y, 'nmcmc', nmcmc, 'vecchia', true, 'm', 10, ...
                     'true_g', 1e-6, 'verb', 500);
fit2 = dgp_trim(fit2, burn, thin);
p2   = dgp_predict(fit2, xx);

% ---- one-layer GP for comparison ----------------------------------------
fprintf('--- one-layer GP, vecchia = true, m = 10 ---\n');
fit1 = fit_one_layer(x, y, 'nmcmc', nmcmc, 'vecchia', true, 'm', 10, ...
                     'true_g', 1e-6, 'verb', 0);
fit1 = dgp_trim(fit1, burn, thin);
p1   = dgp_predict(fit1, xx);

% ---- three-layer DGP -----------------------------------------------------
fprintf('--- three-layer DGP, vecchia = true, m = 10 ---\n');
fit3 = fit_three_layer(x, y, 'nmcmc', nmcmc, 'vecchia', true, 'm', 10, ...
                       'true_g', 1e-6, 'verb', 0);
fit3 = dgp_trim(fit3, burn, thin);
p3   = dgp_predict(fit3, xx);

% ---- scores --------------------------------------------------------------
score = @(p) [sqrt(mean((p.mean - yy).^2)), ...
              mean(abs(p.mean - yy) <= 1.96 * p.sd), ...
              mean(vdgp_crps(yy, p.mean, p.sd))];
fprintf('\n%-16s %10s %10s %10s %8s\n', 'model', 'RMSE', 'cover95', 'CRPS', 'sec');
fprintf('%-16s %10.4f %10.3f %10.4f %8.1f\n', 'one layer',   score(p1), fit1.time);
fprintf('%-16s %10.4f %10.3f %10.4f %8.1f\n', 'two layer',   score(p2), fit2.time);
fprintf('%-16s %10.4f %10.3f %10.4f %8.1f\n', 'three layer', score(p3), fit3.time);

% ---- posterior predictive sample paths -----------------------------------
% Each column of ps.f is a JOINT draw at all 200 test locations, taken from
% one retained MCMC iteration -- so the paths are smooth and show what the
% posterior actually believes, which a mean +/- 2 sd band cannot.
ps = dgp_predict(fit2, xx, 'nsamp', 30);

% ---- plots ---------------------------------------------------------------
figure('Position', [100 100 1200 700]);

subplot(2,3,1);
vdgp_band(xx, p1.mean, p1.sd); hold on;
plot(xx, yy, 'k--', 'LineWidth', 1);
plot(x, y, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 4);
title(sprintf('one-layer GP  (RMSE %.3f)', sqrt(mean((p1.mean-yy).^2))));
xlabel('x'); ylabel('y'); grid on;

subplot(2,3,2);
vdgp_band(xx, p2.mean, p2.sd); hold on;
plot(xx, yy, 'k--', 'LineWidth', 1);
plot(x, y, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 4);
title(sprintf('two-layer Vecchia DGP  (RMSE %.3f)', sqrt(mean((p2.mean-yy).^2))));
xlabel('x'); ylabel('y'); grid on;

subplot(2,3,3);
plot(xx, ps.f, 'Color', [0.45 0.60 0.85], 'LineWidth', 0.5); hold on;
plot(xx, ps.mean, 'Color', [0.10 0.30 0.70], 'LineWidth', 2);
plot(x, y, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 4);
title('30 posterior predictive sample paths');
xlabel('x'); ylabel('y'); grid on;

subplot(2,3,4);
W   = squeeze(fit2.w(:, 1, :));      % n-by-T posterior draws of the warping
sel = round(linspace(1, size(W, 2), min(20, size(W, 2))));
plot(x, W(:, sel), 'Color', [0.6 0.7 0.9]); hold on;
plot(x, W(:, end), 'Color', [0.10 0.30 0.70], 'LineWidth', 2);
title('posterior draws of the latent warping w(x)');
xlabel('x'); ylabel('w'); grid on;

subplot(2,3,5);
plot(fit2.ll, 'LineWidth', 1); grid on;
title('outer log-likelihood after burn-in'); xlabel('retained iteration');

subplot(2,3,6);
semilogy(fit2.theta_y, 'LineWidth', 1); hold on;
semilogy(fit2.theta_w, 'LineWidth', 1);
semilogy(fit2.tau2, 'LineWidth', 1); grid on;
legend({'\theta_y', '\theta_w', '\tau^2'}, 'Location', 'best');
title('hyperparameter traces'); xlabel('retained iteration');

drawnow;
if ~isempty(getenv('VDGP_SAVEFIG'))
    print(gcf, fullfile(here, 'demo_1d.png'), '-dpng', '-r120');
end
