function A = vdgp_update_approx(A, X)
%VDGP_UPDATE_APPROX  Refresh the input locations of an approximation object.
%
%   A = VDGP_UPDATE_APPROX(A, X) replaces the stored (ordered) inputs with the
%   new values X, keeping the ordering and the conditioning sets fixed.  This
%   is the cheap operation performed every time the latent layer moves inside
%   the elliptical slice sampler: the Vecchia structure is unchanged, only the
%   coordinates that enter the covariance function are new.

A.X_ord = X(A.ord, :);
end
