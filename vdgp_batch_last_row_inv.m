function a = vdgp_batch_last_row_inv(L)
%VDGP_BATCH_LAST_ROW_INV  Last row of inv(L) for every page.
%
%   A = VDGP_BATCH_LAST_ROW_INV(L) with L k-by-k-by-N lower triangular returns
%   A of size k-by-1-by-N holding, for each page, the vector a with
%
%        a' = e_k' * inv(L)         equivalently     L' * a = e_k .
%
%   This is exactly the vector of non-zero entries of one column of the
%   Vecchia sparse inverse-Cholesky factor U: if the k points of the page are
%   ordered so that the "target" observation comes LAST, then a(k) = 1/L(k,k)
%   is the diagonal entry of U and a(1:k-1) are the off-diagonal entries.

[k, ~, N] = size(L);
a = zeros(k, 1, N, class(L));
a(k, 1, :) = 1 ./ L(k, k, :);
for j = k-1:-1:1
    s = sum(L(j+1:k, j, :) .* a(j+1:k, 1, :), 1);
    a(j, 1, :) = -s ./ L(j, j, :);
end
end
