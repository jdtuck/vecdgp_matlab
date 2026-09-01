function [x, y, sc, opts] = vdgp_prep(x, y, opts)
%VDGP_PREP  Input checking, optional scaling and option defaults.
%
%   [X, Y, SC, OPTS] = VDGP_PREP(X, Y, OPTS)
%
%   When OPTS.scale is true, X is mapped column-wise to [0,1] and Y is
%   centred and scaled to unit variance -- the regime the default Gamma
%   priors are designed for.  SC records the transformation so that
%   predictions can be returned on the original scale.

if isvector(x), x = x(:); end
y = y(:);
if size(x, 1) ~= numel(y)
    error('vdgp_prep:size', 'x and y must have the same number of rows.');
end
if any(~isfinite(x(:))) || any(~isfinite(y))
    error('vdgp_prep:finite', 'x and y must be finite.');
end

sc = struct('scaled', false, 'xmin', [], 'xrange', [], 'ymean', 0, 'ysd', 1);

if opts.scale
    xmin = min(x, [], 1);
    xrng = max(x, [], 1) - xmin;
    xrng(xrng == 0) = 1;
    x = (x - xmin) ./ xrng;
    ym = mean(y);
    ys = std(y);
    if ys == 0, ys = 1; end
    y = (y - ym) / ys;
    sc = struct('scaled', true, 'xmin', xmin, 'xrange', xrng, 'ymean', ym, 'ysd', ys);
end

n = size(x, 1);
if isempty(opts.m), opts.m = min(25, n - 1); end
opts.m = min(opts.m, n - 1);
if isempty(opts.D), opts.D = size(x, 2); end

if ~any(strcmpi(opts.cov, {'matern', 'exp2'}))
    error('vdgp_prep:cov', 'cov must be ''matern'' or ''exp2''.');
end
if strcmpi(opts.cov, 'exp2'), opts.v = 999; end
end
