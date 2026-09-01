function K = vdgp_batch_cov(P, theta, v, cov_type)
%VDGP_BATCH_COV  Covariance matrices for a stack of small point sets.
%
%   K = VDGP_BATCH_COV(P, THETA, V, COV_TYPE) where P is k-by-d-by-N returns
%   the k-by-k-by-N array of covariance matrices, one per page, with unit
%   scale and no nugget.  Parameterisation matches VDGP_COV.

[k, d, N] = size(P);

exp2 = strcmpi(cov_type, 'exp2');
pw   = 1; if ~exp2, pw = 2; end

if isscalar(theta)
    s = repmat(theta, 1, d);
else
    s = reshape(theta, 1, []);
end
s = s .^ pw;

D2 = zeros(k, k, N);
for j = 1:d
    a  = reshape(P(:, j, :), k, 1, N) / sqrt(s(j));
    b  = reshape(P(:, j, :), 1, k, N) / sqrt(s(j));
    D2 = D2 + (a - b).^2;
end
D2 = max(D2, 0);

if exp2
    K = exp(-D2);
else
    r = sqrt(D2);
    switch v
        case 0.5
            K = exp(-r);
        case 1.5
            K = (1 + sqrt(3) * r) .* exp(-sqrt(3) * r);
        case 2.5
            K = (1 + sqrt(5) * r + (5/3) * r.^2) .* exp(-sqrt(5) * r);
        otherwise
            error('vdgp_batch_cov:v', 'v must be 0.5, 1.5 or 2.5.');
    end
end
end
