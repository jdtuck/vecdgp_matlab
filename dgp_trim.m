function fit = dgp_trim(fit, burn, thin)
%DGP_TRIM  Discard burn-in and thin the MCMC chains.
%
%   FIT = DGP_TRIM(FIT, BURN)         removes the first BURN iterations
%   FIT = DGP_TRIM(FIT, BURN, THIN)   also keeps every THIN-th remaining draw
%
%   Mirrors deepgp's trim().  All parameter chains and every stored latent
%   layer are subset consistently.

if nargin < 3 || isempty(thin), thin = 1; end

if isfield(fit, 'w_iters') && ~isempty(fit.w_iters)
    it = fit.w_iters;
else
    it = 1:fit.nmcmc;
end

sel = find(it > burn);
if isempty(sel)
    error('dgp_trim:empty', 'Burn-in removes every stored sample.');
end
sel = sel(1:thin:end);
its = it(sel);

fit.theta_y = fit.theta_y(its);
fit.g       = fit.g(its);
fit.tau2    = fit.tau2(its);
fit.ll      = fit.ll(its);
if isfield(fit, 'theta_w') && ~isempty(fit.theta_w)
    fit.theta_w = fit.theta_w(its, :);
end
if isfield(fit, 'theta_z') && ~isempty(fit.theta_z)
    fit.theta_z = fit.theta_z(its, :);
end
if isfield(fit, 'w') && ~isempty(fit.w)
    fit.w = fit.w(:, :, sel);
end
if isfield(fit, 'z') && ~isempty(fit.z)
    fit.z = fit.z(:, :, sel);
end

fit.nmcmc   = numel(its);
fit.w_iters = 1:numel(its);
end
