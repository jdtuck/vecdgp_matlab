function [idx, dist] = vdgp_knn(Xref, Xq, k)
%VDGP_KNN  k nearest neighbours of each row of Xq among the rows of Xref.
%
%   [IDX, DIST] = VDGP_KNN(XREF, XQ, K) returns nq-by-K index and distance
%   matrices, sorted by increasing distance.  Uses KNNSEARCH from the
%   Statistics and Machine Learning Toolbox when available and otherwise falls
%   back on a chunked brute-force search (Euclidean distance in both cases).

k = min(k, size(Xref, 1));

if exist('knnsearch', 'file') == 2 || exist('knnsearch', 'builtin') == 5
    [idx, dist] = knnsearch(Xref, Xq, 'K', k);
    return
end

nq = size(Xq, 1);
nr = size(Xref, 1);
idx  = zeros(nq, k);
dist = zeros(nq, k);

% chunk so that the distance block stays below ~64 MB
chunk = max(1, floor(8e6 / max(nr, 1)));
sref  = sum(Xref.^2, 2).';
for a = 1:chunk:nq
    b  = min(nq, a + chunk - 1);
    Xb = Xq(a:b, :);
    D2 = sum(Xb.^2, 2) + sref - 2 * (Xb * Xref.');
    D2 = max(D2, 0);
    if exist('mink', 'builtin') == 5 || exist('mink', 'file') == 2
        % partial selection: much cheaper than a full sort when k << nr
        [sd, si] = mink(D2, k, 2);
    else
        [sd, si] = sort(D2, 2);
        sd = sd(:, 1:k);  si = si(:, 1:k);
    end
    idx(a:b, :)  = si;
    dist(a:b, :) = sqrt(sd);
end
end
