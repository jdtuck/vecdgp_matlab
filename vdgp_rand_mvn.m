function s = vdgp_rand_mvn(A, theta, g, v, cov_type, tau2)
%VDGP_RAND_MVN  Prior draw from N(0, tau2 * Sigma) under the Vecchia approx.
%
%   S = VDGP_RAND_MVN(A, THETA, G, V, COV_TYPE, TAU2)
%
%   With Q = U U' ~= inv(Sigma), a draw is obtained from a single sparse
%   triangular solve,
%
%        z ~ N(0, I) ,    U' s = z   =>   Cov(s) = inv(U U') = Sigma ,
%
%   which costs O(n m) once U is available.  The result is returned in the
%   ORIGINAL (unordered) indexing.

if nargin < 6 || isempty(tau2), tau2 = 1; end
if nargin < 5 || isempty(cov_type), cov_type = 'matern'; end
if nargin < 4 || isempty(v), v = 2.5; end

U = vdgp_create_U(A, theta, g, v, cov_type);
z = randn(A.n, 1);
s = U.' \ z;                 % lower-triangular sparse solve
s = s(A.rev);                % back to the original ordering
if tau2 ~= 1
    s = sqrt(tau2) * s;
end
end
