function NN = vdgp_ordered_nn(X, m)
%VDGP_ORDERED_NN  Vecchia conditioning sets for an already-ordered design.
%
%   NN = VDGP_ORDERED_NN(X, M) returns an n-by-(M+1) matrix in which row i is
%
%        [ i , c(i) ]     with   |c(i)| = min(M, i-1)
%
%   and c(i) indexing the min(M, i-1) points among 1,...,i-1 that are closest
%   to X(i,:) in Euclidean distance.  Unused slots are NaN.  This is the
%   MATLAB counterpart of GpGp::find_ordered_nn used by the deepgp R package.

n = size(X, 1);
m = min(m, n - 1);
NN = nan(n, m + 1);
NN(:, 1) = (1:n).';

if n == 1, return; end

% --- the first m+1 rows are exact and cheap ------------------------------
top = min(n, m + 1);
for i = 2:top
    d = sum((X(1:i-1, :) - X(i, :)).^2, 2);
    [~, o] = sort(d);
    NN(i, 2:i) = o(1:i-1).';
end
if n <= m + 1, return; end

% --- remaining rows, processed in blocks ---------------------------------
% For a block of rows [a,b] every candidate has index < b, so we query the
% prefix X(1:b-1,:) for m + (b-a+1) neighbours and then discard those with an
% index >= the query row.  That is always enough because at most (b-a) of the
% returned neighbours can be "too late" for any row in the block.
blk = max(1, min(2048, round(4e6 / max(n, 1))));
blk = max(blk, 64);
a = m + 2;
while a <= n
    b = min(n, a + blk - 1);
    kq = min(b - 1, m + (b - a + 1));
    cand = vdgp_knn(X(1:b-1, :), X(a:b, :), kq);
    for r = 1:(b - a + 1)
        i  = a + r - 1;
        cr = cand(r, :);
        cr = cr(cr < i);
        if numel(cr) >= m
            NN(i, 2:m+1) = cr(1:m);
        else                                  % pathological ties: brute force
            d = sum((X(1:i-1, :) - X(i, :)).^2, 2);
            [~, o] = sort(d);
            NN(i, 2:m+1) = o(1:m).';
        end
    end
    a = b + 1;
end
end
