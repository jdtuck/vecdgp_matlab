function [L, ok] = vdgp_batch_chol(A, jitter)
%VDGP_BATCH_CHOL  Lower Cholesky factor of every page of a 3-D array.
%
%   [L, OK] = VDGP_BATCH_CHOL(A) where A is k-by-k-by-N returns the k-by-k-by-N
%   array of lower-triangular Cholesky factors.  The standard column algorithm
%   is used, fully vectorised across pages: O(N k^3) flops in only O(k^2)
%   vectorised passes, so there is no per-page interpreter overhead.
%
%   If some page fails to factorise, an increasing jitter is added to the
%   diagonal of every page and the factorisation is retried (up to 6 times).
%   OK is a 1-by-N logical row flagging pages that factorised cleanly.

if nargin < 2 || isempty(jitter), jitter = 0; end

[k, k2, N] = size(A);
if k ~= k2
    error('vdgp_batch_chol:square', 'Pages must be square.');
end
dgi = 1:(k+1):(k*k);                 % linear index of the diagonal in a page

attempt = 0;
maxatt  = 6;
Adg     = reshape(A, k*k, N);
scaleA  = mean(abs(reshape(Adg(dgi, :), [], 1)));
clear Adg
while true
    if jitter > 0
        Ar = reshape(A, k*k, N);
        Ar(dgi, :) = Ar(dgi, :) + jitter;
        A = reshape(Ar, k, k, N);
    end

    L = zeros(k, k, N);
    for j = 1:k
        dj = A(j, j, :);
        if j > 1
            dj = dj - sum(L(j, 1:j-1, :).^2, 2);
        end
        L(j, j, :) = sqrt(dj);
        if j < k
            s = A(j+1:k, j, :);
            if j > 1
                s = s - sum(L(j+1:k, 1:j-1, :) .* L(j, 1:j-1, :), 2);
            end
            L(j+1:k, j, :) = s ./ L(j, j, :);
        end
    end

    diagL = reshape(L, k*k, N);
    diagL = diagL(dgi, :);
    ok    = all(isfinite(diagL) & diagL > 0, 1);

    if all(ok) || attempt >= maxatt
        break
    end
    attempt = attempt + 1;
    jitter  = max(jitter * 10, 1e-10 * max(scaleA, 1));
end
end
