function fit = dgp_continue(fit, nmcmc, re_approx)
%DGP_CONTINUE  Run more MCMC iterations from the current state.
%
%   FIT = DGP_CONTINUE(FIT, NMCMC)
%   FIT = DGP_CONTINUE(FIT, NMCMC, RE_APPROX)
%
%   Restarts the sampler at the last stored draw and appends the new chain,
%   mirroring deepgp's continue().  With RE_APPROX = true the Vecchia
%   ordering and conditioning sets of the latent layers are rebuilt in the
%   current warped space before continuing -- the strategy discussed in
%   Section 4 of Sauer, Cooper & Gramacy (2023).
%
%   Note: continuing after DGP_TRIM is allowed but the chains are then
%   concatenated onto the trimmed history.

if nargin < 3 || isempty(re_approx), re_approx = false; end

opts = fit.opts;
opts.nmcmc = nmcmc;
opts.scale = false;                     % fit.x / fit.y are already scaled
opts.g_0 = fit.g(end);
opts.theta_y_0 = fit.theta_y(end);
if re_approx
    opts.ord_init = [];                 % draw a fresh ordering
else
    opts.ord_init = fit.x_approx.ord;
end

switch fit.layers
    case 1
        new = fit_one_layer(fit.x, fit.y, opts);
    case 2
        opts.theta_w_0 = fit.theta_w(end, :);
        opts.w_init    = fit.w(:, :, end);
        new = fit_two_layer(fit.x, fit.y, opts);
    case 3
        opts.theta_w_0 = fit.theta_w(end, :);
        opts.theta_z_0 = fit.theta_z(end, :);
        opts.w_init    = fit.w(:, :, end);
        opts.z_init    = fit.z(:, :, end);
        new = fit_three_layer(fit.x, fit.y, opts);
end

% the scaling of the original fit is what predictions must undo
new.scaling = fit.scaling;
new.x_raw = fit.x_raw; new.y_raw = fit.y_raw;

fit = local_cat(fit, new);
end

% -------------------------------------------------------------------------
function fit = local_cat(a, b)
fit = b;
fit.theta_y = [a.theta_y; b.theta_y];
fit.g       = [a.g;       b.g];
fit.tau2    = [a.tau2;    b.tau2];
fit.ll      = [a.ll;      b.ll];
if isfield(a, 'theta_w'), fit.theta_w = [a.theta_w; b.theta_w]; end
if isfield(a, 'theta_z'), fit.theta_z = [a.theta_z; b.theta_z]; end
if isfield(a, 'w') && ~isempty(a.w), fit.w = cat(3, a.w, b.w); end
if isfield(a, 'z') && ~isempty(a.z), fit.z = cat(3, a.z, b.z); end
fit.nmcmc   = numel(fit.theta_y);
if isfield(a, 'w') && ~isempty(a.w)
    fit.w_iters = 1:size(fit.w, 3);
    if size(fit.w, 3) ~= fit.nmcmc
        % store_every > 1: keep the parameter chains aligned to stored draws
        fit.w_iters = round(linspace(1, fit.nmcmc, size(fit.w, 3)));
    end
else
    fit.w_iters = 1:fit.nmcmc;
end
fit.time = a.time + b.time;
end
