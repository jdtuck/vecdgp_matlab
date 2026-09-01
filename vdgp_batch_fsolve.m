function X = vdgp_batch_fsolve(L, B)
%VDGP_BATCH_FSOLVE  Forward substitution L*X = B for every page.
%
%   X = VDGP_BATCH_FSOLVE(L, B) with L k-by-k-by-N lower triangular and
%   B k-by-q-by-N returns X of size k-by-q-by-N.

[k, ~, N] = size(L);
q = size(B, 2);
X = zeros(k, q, N, class(B));
for i = 1:k
    s = B(i, :, :);
    if i > 1
        Lp = reshape(L(i, 1:i-1, :), i-1, 1, N);
        s  = s - sum(Lp .* X(1:i-1, :, :), 1);
    end
    X(i, :, :) = s ./ L(i, i, :);
end
end
