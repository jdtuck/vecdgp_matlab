function lp = vdgp_lgamma(x, shape, rate)
%VDGP_LGAMMA  Log density of Gamma(shape, rate) -- R's dgamma(..., log = TRUE).
%
%   Uses the rate parameterisation, p(x) = rate^shape x^(shape-1) e^(-rate x)
%   / Gamma(shape).  Returns -Inf for x <= 0.  Only GAMMALN (base MATLAB) is
%   required, so no Statistics Toolbox dependency.

lp = -Inf(size(x));
ok = x > 0;
lp(ok) = shape .* log(rate) - gammaln(shape) + (shape - 1) .* log(x(ok)) ...
         - rate .* x(ok);
end
