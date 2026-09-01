function s = vdgp_crps(y, mu, sd)
%VDGP_CRPS  Continuous ranked probability score for a Gaussian forecast.
%
%   S = VDGP_CRPS(Y, MU, SD) returns the elementwise CRPS
%
%       crps = sd * ( z (2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi) ),
%       z = (y - mu) / sd
%
%   Lower is better.  This is the proper score used to compare uncertainty
%   quantification in Sauer, Cooper & Gramacy (2023).  Implemented with ERF so
%   that no Statistics Toolbox function is needed.

z   = (y(:) - mu(:)) ./ sd(:);
Phi = 0.5 * (1 + erf(z / sqrt(2)));
phi = exp(-0.5 * z.^2) / sqrt(2*pi);
s   = sd(:) .* (z .* (2*Phi - 1) + 2*phi - 1/sqrt(pi));
end
