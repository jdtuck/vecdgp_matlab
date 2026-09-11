function B = vdgp_bench_samples(varargin)
%VDGP_BENCH_SAMPLES  How cost grows with n, the number of training samples.
%
%   B = VDGP_BENCH_SAMPLES()                  default sweep
%   B = VDGP_BENCH_SAMPLES('ns', [...], ...)  choose the sweep and the sizes
%
%   Sweeps the training-set size n and reports, at each n:
%
%     fit s/sweep     what one Gibbs sweep of the sampler costs
%     build s         one-off cost of VDGP_PREDICTOR
%     draw_pt s       ONE emulator realisation -- the calibration inner loop
%     predict_pt s    mean and variance over all retained draws
%     scan %          share of draw_pt spent finding the conditioning set
%     RMSE            held-out accuracy, so cost can be read against fit
%
%   The retained-draw count T is held fixed across n (see 'T'), so the only
%   thing moving is the training size.
%
%   The "scan %" column is the one to watch. Per call, VDGP_DRAW_PT does an
%   O(n d) scan over the training inputs to find the m nearest, then a fixed
%   amount of O(m^3) linear algebra. The linear algebra does not grow with n,
%   so past some n the scan is the whole cost and the per-call time becomes
%   linear in n. If that column is heading for 100% on your problem, a
%   prebuilt neighbour-search structure is the next thing worth adding.
%
%   Options
%     'ns'      training sizes (default [250 500 1000 2000])
%     'd'       input dimension (default 2)
%     'm'       conditioning-set size (default 25)
%     'nmcmc'   sweeps per fit (default 600) and 'burn' (default 200).  These
%               size the chain, not the timings; the RMSE column is
%               indicative rather than a converged result at the default.
%     'T'       retained draws used for prediction, held fixed across n
%               (default 200)
%     'reps'    timing repetitions for the per-call numbers (default 100)
%     'ntest'   held-out test points (default 300)
%     'slow'    also time DGP_PREDICT at a single point (default false; it is
%               orders of magnitude slower and would dominate the runtime)
%     'plot'    draw the figure (default true)
%
%   Returns B with fields .data, .cols and .setup.
%
%   See also VDGP_DRAW_PT, VDGP_PREDICT_PT, VDGP_PREDICTOR, DEMO_SCALING.

o = struct('ns', [250 500 1000 2000], 'd', 2, 'm', 25, 'nmcmc', 600, ...
           'burn', 200, 'T', 200, 'reps', 100, 'ntest', 300, ...
           'slow', false, 'plot', true);
for i = 1:2:numel(varargin)
    if ~isfield(o, varargin{i})
        error('vdgp_bench_samples:opt', 'Unknown option "%s".', varargin{i});
    end
    o.(varargin{i}) = varargin{i+1};
end

if o.d == 1
    ftrue = @(X) (X(:,1) <= 0.58) .* (sin(pi*X(:,1)*6) + cos(pi*X(:,1)*12)) + ...
                 (X(:,1) > 0.58) .* (5*X(:,1) - 4.9);
else
    ftrue = @(X) atan(20*(X(:,1) - 0.5)) .* sin(3*pi*X(:,2)) + 0.5*X(:,2);
end

Xt = rand(o.ntest, o.d);
yt = ftrue(Xt);
x0 = Xt(1, :);

haveMex = exist('vdgp_logl_mex', 'file') == 3 && ...
          exist('vdgp_krig_mex', 'file') == 3;
fprintf('\n=== cost vs n (training samples):  d = %d, m = %d, T = %d ===\n', ...
        o.d, o.m, o.T);
if haveMex
    fprintf('    MEX kernels: active\n');
else
    fprintf(['    MEX kernels: NOT BUILT -- every timing below is the\n' ...
             '    pure-MATLAB path and is roughly 10x slower than it needs\n' ...
             '    to be. Run vdgp_build_mex first.\n']);
end
if o.slow
    fprintf('%7s %11s %9s %11s %12s %8s %10s %12s\n', 'n', 'fit s/it', ...
            'build s', 'draw_pt s', 'predict_pt s', 'scan %', 'RMSE', 'dgp_pred s');
else
    fprintf('%7s %11s %9s %11s %12s %8s %10s\n', 'n', 'fit s/it', ...
            'build s', 'draw_pt s', 'predict_pt s', 'scan %', 'RMSE');
end

res = [];
for n = o.ns
    X = rand(n, o.d);
    y = ftrue(X) + 0.01*randn(n, 1);

    t = tic;
    fit = fit_two_layer(X, y, 'nmcmc', o.nmcmc, 'm', o.m, 'verb', 0, 'true_g', 1e-4);
    tfit = toc(t);
    t = tic; vdgp_create_approx(X, o.m, 'random', true); tset = toc(t);
    s_sweep = max(tfit - tset, 0) / max(o.nmcmc - 1, 1);

    % keep a fixed number of retained draws so only n is varying
    fit = dgp_trim(fit, o.burn, 1);
    sel = unique(round(linspace(1, fit.nmcmc, min(o.T, fit.nmcmc))));
    fit = local_keep(fit, sel);

    t = tic; P = vdgp_predictor(fit); tbuild = toc(t);

    t = tic; for i = 1:o.reps, vdgp_draw_pt(P, x0); end
    tdraw = toc(t) / o.reps;

    r2 = max(3, round(o.reps / 10));
    t = tic; for i = 1:r2, vdgp_predict_pt(P, x0); end
    tpred = toc(t) / r2;

    % how much of a draw is just finding the conditioning set
    xs0 = x0;
    if fit.scaling.scaled
        xs0 = (x0 - fit.scaling.xmin) ./ fit.scaling.xrange;
    end
    Xs = fit.x;
    t = tic;
    for i = 1:o.reps
        d2 = sum((Xs - xs0).^2, 2);
        if exist('mink', 'builtin') == 5 || exist('mink', 'file') == 2
            [~, ~] = mink(d2, o.m);
        else
            [~, oo] = sort(d2); oo = oo(1:o.m); %#ok<NASGU>
        end
    end
    tscan = toc(t) / o.reps;

    mu = vdgp_predict_pt(P, Xt);
    rmse = sqrt(mean((mu - yt).^2));

    tg = NaN;
    if o.slow
        t = tic; dgp_predict(fit, x0); tg = toc(t);
    end

    if o.slow
        fprintf('%7d %11.4f %9.4f %11.5f %12.5f %7.0f%% %10.4f %12.4f\n', ...
                n, s_sweep, tbuild, tdraw, tpred, 100*tscan/tdraw, rmse, tg);
    else
        fprintf('%7d %11.4f %9.4f %11.5f %12.5f %7.0f%% %10.4f\n', ...
                n, s_sweep, tbuild, tdraw, tpred, 100*tscan/tdraw, rmse);
    end
    res(end+1, :) = [n, s_sweep, tbuild, tdraw, tpred, tscan, rmse, tg]; %#ok<SAGROW>
end

B = struct();
B.cols  = {'n','fit_s_per_sweep','build_s','draw_pt_s','predict_pt_s', ...
           'scan_s','rmse','dgp_predict_s'};
B.data  = res;
B.setup = struct('d', o.d, 'm', o.m, 'T', o.T, 'nmcmc', o.nmcmc, ...
                 'burn', o.burn, 'ntest', o.ntest, 'mex', haveMex);

fprintf(['\n  A calibration run of S steps costs about S x (draw_pt s), plus\n' ...
         '  one predictor build. The fit is a separate, one-off cost.\n']);
if ~isempty(res)
    fprintf('  At n = %d: 10,000 calibration steps ~ %.0f s.\n', ...
            res(end,1), 1e4 * res(end,4));
end

if o.plot
    figure('Position', [100 100 1050 400]);

    subplot(1,3,1);
    loglog(res(:,1), res(:,4), 'o-', 'LineWidth', 1.5); hold on;
    loglog(res(:,1), res(:,5), 's-', 'LineWidth', 1.5);
    loglog(res(:,1), res(1,4)*res(:,1)/res(1,1), 'k:', 'LineWidth', 1);
    xlabel('n (training samples)'); ylabel('seconds / call'); grid on;
    legend({'vdgp_draw_pt', 'vdgp_predict_pt', 'O(n)'}, ...
           'Location', 'northwest', 'Interpreter', 'none');
    title('prediction cost vs n');

    subplot(1,3,2);
    semilogx(res(:,1), 100*res(:,6)./res(:,4), 'o-', 'LineWidth', 1.5);
    xlabel('n (training samples)'); ylabel('% of a draw spent scanning');
    ylim([0 100]); grid on;
    title('where the n-dependence lives');

    subplot(1,3,3);
    loglog(res(:,1), res(:,2), 'o-', 'LineWidth', 1.5); hold on;
    loglog(res(:,1), res(:,7), 's-', 'LineWidth', 1.5);
    xlabel('n (training samples)'); grid on;
    legend({'fit, s / sweep', 'held-out RMSE'}, 'Location', 'best');
    title('fit cost and accuracy');

    drawnow;
    if ~isempty(getenv('VDGP_SAVEFIG'))
        print(gcf, 'vdgp_bench_samples.png', '-dpng', '-r120');
    end
end
end

% =========================================================================
function fT = local_keep(fit, sel)
fT = fit;
fT.theta_y = fit.theta_y(sel);
fT.g       = fit.g(sel);
fT.tau2    = fit.tau2(sel);
fT.ll      = fit.ll(sel);
if isfield(fit, 'theta_w') && ~isempty(fit.theta_w), fT.theta_w = fit.theta_w(sel, :); end
if isfield(fit, 'theta_z') && ~isempty(fit.theta_z), fT.theta_z = fit.theta_z(sel, :); end
if isfield(fit, 'w') && ~isempty(fit.w), fT.w = fit.w(:, :, sel); end
if isfield(fit, 'z') && ~isempty(fit.z), fT.z = fit.z(:, :, sel); end
fT.nmcmc   = numel(sel);
fT.w_iters = 1:numel(sel);
end
