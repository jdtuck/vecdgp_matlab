function D2 = vdgp_sqdist(X1, X2, theta, pw)
%VDGP_SQDIST  Scaled squared distances.
%
%   D2 = VDGP_SQDIST(X1, X2, THETA, PW) returns the n1-by-n2 matrix with
%
%       D2(i,j) = sum_k ( X1(i,k) - X2(j,k) )^2 / THETA(k)^PW
%
%   THETA may be scalar or 1-by-d.  PW is 1 (squared-exponential
%   parameterisation) or 2 (Matern parameterisation).

if nargin < 4 || isempty(pw), pw = 1; end
if isempty(X2), X2 = X1; end

d = size(X1, 2);
if isscalar(theta)
    s = repmat(theta, 1, d);
else
    s = reshape(theta, 1, []);
    if numel(s) ~= d
        error('vdgp_sqdist:theta', 'theta must be scalar or of length %d.', d);
    end
end
s = s .^ pw;

A = X1 ./ sqrt(s);
B = X2 ./ sqrt(s);

D2 = sum(A.^2, 2) + sum(B.^2, 2).' - 2 * (A * B.');
D2 = max(D2, 0);
end
