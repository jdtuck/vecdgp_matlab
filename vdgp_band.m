function h = vdgp_band(x, mu, sd, col)
%VDGP_BAND  Shade a mean +/- 1.96 sd predictive band.
%
%   VDGP_BAND(X, MU, SD) draws the 95% band and the mean on the current axes.
%   VDGP_BAND(X, MU, SD, COL) sets the fill colour.

if nargin < 4 || isempty(col), col = [0.85 0.90 0.97]; end
x = x(:); mu = mu(:); sd = sd(:);
lo = mu - 1.96*sd;
hi = mu + 1.96*sd;
h = fill([x; flipud(x)], [lo; flipud(hi)], col, 'EdgeColor', 'none');
hold on;
plot(x, mu, '-', 'Color', [0.10 0.30 0.70], 'LineWidth', 1.5);
end
