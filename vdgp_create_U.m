function [U, diagU] = vdgp_create_U(A, theta, g, v, cov_type)
%VDGP_CREATE_U  Sparse inverse Cholesky factor of the Vecchia precision.
%
%   [U, DIAGU] = VDGP_CREATE_U(A, THETA, G, V, COV_TYPE) returns the n-by-n
%   sparse upper-triangular U with  Q = U*U' ~= inv(Sigma)  at unit scale, in
%   the ordering carried by the approximation object A.
%
%   G may be a scalar or an n-vector (in the ordered indexing) so that, for
%   joint prediction, observed locations can carry the estimated nugget while
%   predictive locations carry only jitter.
%
%   See VDGP_U_ENTRIES for the construction; this wrapper just assembles the
%   sparse matrix.  Routines that need only a likelihood should call
%   VDGP_U_ENTRIES directly and avoid the assembly.

if nargin < 5, cov_type = 'matern'; end
if nargin < 4, v = 2.5; end

E = vdgp_U_entries(A, theta, g, v, cov_type);
U = sparse(E.I, E.J, E.V, E.n, E.n);
diagU = E.diagU;
end
