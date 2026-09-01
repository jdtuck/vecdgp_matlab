function ord = vdgp_order(X, type)
%VDGP_ORDER  Orderings for the Vecchia approximation.
%
%   ORD = VDGP_ORDER(X, 'random')  random permutation (the default used in
%                                  Sauer, Cooper & Gramacy, 2023)
%   ORD = VDGP_ORDER(X, 'maxmin')  greedy max-min distance ordering: start at
%                                  the point closest to the centroid and
%                                  repeatedly add the point farthest (in the
%                                  min-distance sense) from those already
%                                  chosen.  O(n^2 d); use only for moderate n.
%   ORD = VDGP_ORDER(X, 'none')    identity
%
%   ORD is a permutation of 1:n; the ordered design is X(ORD,:).

n = size(X, 1);
if nargin < 2 || isempty(type), type = 'random'; end
if isnumeric(type)
    ord = type(:).';
    if ~isequal(sort(ord), 1:n)
        error('vdgp_order:perm', 'Supplied ordering is not a permutation of 1:n.');
    end
    return
end

switch lower(type)
    case 'random'
        ord = randperm(n);
    case 'none'
        ord = 1:n;
    case 'maxmin'
        if n > 20000
            warning('vdgp_order:big', ...
                'maxmin ordering is O(n^2); n = %d may be slow.', n);
        end
        ord = zeros(1, n);
        ctr = mean(X, 1);
        [~, first] = min(sum((X - ctr).^2, 2));
        ord(1) = first;
        dmin = sum((X - X(first, :)).^2, 2);
        dmin(first) = -Inf;
        for i = 2:n
            [~, nxt] = max(dmin);
            ord(i) = nxt;
            dmin = min(dmin, sum((X - X(nxt, :)).^2, 2));
            dmin(nxt) = -Inf;
        end
    otherwise
        error('vdgp_order:type', 'Unknown ordering "%s".', type);
end
end
