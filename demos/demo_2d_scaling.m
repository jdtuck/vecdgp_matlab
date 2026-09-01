%DEMO_2D_SCALING  Cost and accuracy of the Vecchia DGP as n grows.
%
% Fits a two-layer DGP to a 2-D non-stationary test function on a sequence of
% training sizes, with and without the Vecchia approximation, and reports
%
%   * seconds per MCMC iteration        -> the O(n m^3) vs O(n^3) claim
%   * out-of-sample RMSE / CRPS / 95% coverage
%
% The exact sampler is only run while it is affordable (n <= EXACT_MAX).
%
% This is the MATLAB analogue of the scaling study in Section 5 of
% Sauer, Cooper & Gramacy (2023), on a much smaller scale so that it finishes
% in minutes rather than hours.  Increase NS and NMCMC to push it further.

clear; close all;
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, '..'));

if exist('rng', 'file'), rng(2); else, rand('seed', 2); randn('seed', 2); end

% Non-stationary 2-D test function: a steep front in x1 modulated by x2.
f = @(X) atan(20 * (X(:, 1) - 0.5)) .* sin(3*pi*X(:, 2)) + 0.5 * X(:, 2);

NS        = [200 500 1000 2000];   % training sizes
NMCMC     = 400;                   % keep small so the demo finishes quickly
BURN      = 200;
THIN      = 2;
M         = 25;                    % Vecchia conditioning-set size
EXACT_MAX = 500;                   % above this the exact sampler is skipped
NTEST     = 500;

Xt = rand(NTEST, 2);
yt = f(Xt);

res = [];
fprintf('%6s %9s %10s %10s %10s %10s\n', 'n', 'method', 's/iter', 'RMSE', 'CRPS', 'cover95');
for n = NS
    X = rand(n, 2);
    y = f(X) + 1e-3 * randn(n, 1);

    % ---- Vecchia ---------------------------------------------------------
    fv = fit_two_layer(X, y, 'nmcmc', NMCMC, 'vecchia', true, 'm', M, ...
                       'true_g', 1e-4, 'verb', 0);
    tv = fv.time / NMCMC;
    fv = dgp_trim(fv, BURN, THIN);
    pv = dgp_predict(fv, Xt);
    sv = [sqrt(mean((pv.mean - yt).^2)), mean(vdgp_crps(yt, pv.mean, pv.sd)), ...
          mean(abs(pv.mean - yt) <= 1.96 * pv.sd)];
    fprintf('%6d %9s %10.4f %10.4f %10.4f %10.3f\n', n, 'vecchia', tv, sv);
    res(end+1, :) = [n, 1, tv, sv]; %#ok<SAGROW>

    % ---- exact -----------------------------------------------------------
    if n <= EXACT_MAX
        fe = fit_two_layer(X, y, 'nmcmc', NMCMC, 'vecchia', false, ...
                           'true_g', 1e-4, 'verb', 0);
        te = fe.time / NMCMC;
        fe = dgp_trim(fe, BURN, THIN);
        pe = dgp_predict(fe, Xt);
        se = [sqrt(mean((pe.mean - yt).^2)), mean(vdgp_crps(yt, pe.mean, pe.sd)), ...
              mean(abs(pe.mean - yt) <= 1.96 * pe.sd)];
        fprintf('%6d %9s %10.4f %10.4f %10.4f %10.3f\n', n, 'exact', te, se);
        res(end+1, :) = [n, 0, te, se]; %#ok<SAGROW>
    end
end

% ---- plots ---------------------------------------------------------------
v = res(res(:,2) == 1, :);
e = res(res(:,2) == 0, :);

figure('Position', [100 100 900 350]);
subplot(1,2,1);
loglog(v(:,1), v(:,3), 'o-', 'LineWidth', 1.5); hold on;
if ~isempty(e), loglog(e(:,1), e(:,3), 's-', 'LineWidth', 1.5); end
ref = v(1,3) * (v(:,1) / v(1,1));
loglog(v(:,1), ref, 'k:', 'LineWidth', 1);
xlabel('n'); ylabel('seconds per MCMC iteration'); grid on;
legend({'Vecchia', 'exact', 'O(n) reference'}, 'Location', 'northwest');
title('cost');

subplot(1,2,2);
semilogx(v(:,1), v(:,4), 'o-', 'LineWidth', 1.5); hold on;
if ~isempty(e), semilogx(e(:,1), e(:,4), 's-', 'LineWidth', 1.5); end
xlabel('n'); ylabel('out-of-sample RMSE'); grid on;
legend({'Vecchia', 'exact'}); title('accuracy');

drawnow;
if ~isempty(getenv('VDGP_SAVEFIG'))
    print(gcf, fullfile(here, 'demo_2d_scaling.png'), '-dpng', '-r120');
end
