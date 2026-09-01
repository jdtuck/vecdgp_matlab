function opts = vdgp_options(varargin)
%VDGP_OPTIONS  Default option/settings structure for the Vecchia-DGP package.
%
%   OPTS = VDGP_OPTIONS() returns the defaults.
%   OPTS = VDGP_OPTIONS('name',value,...) overrides individual fields.
%   OPTS = VDGP_OPTIONS(S) merges the fields of struct S into the defaults.
%
%   Defaults mirror the R package deepgp (Sauer, Cooper & Gramacy, 2023),
%   which accompanies
%
%     A. Sauer, A. Cooper and R. B. Gramacy (2023),
%     "Vecchia-approximated Deep Gaussian Processes for Computer Experiments",
%     Journal of Computational and Graphical Statistics, arXiv:2204.02904.
%
%   FIELDS
%     cov        'matern' (default) | 'exp2'
%     v          Matern smoothness: 0.5, 1.5 or 2.5 (default 2.5).  Ignored
%                for 'exp2'.
%     vecchia    true (default) -> Vecchia approximation; false -> exact O(n^3)
%     m          conditioning-set size (default min(25, n-1))
%     ordering   'random' (default, as in the paper) | 'maxmin' | numeric perm
%     nmcmc      number of MCMC iterations (default 10000)
%     true_g     [] to estimate the nugget, or a fixed value (e.g. 1e-6)
%     g_0        initial nugget (default 1e-3)
%     theta_y_0  initial outer lengthscale (default 0.1)
%     theta_w_0  initial middle lengthscale (default 0.1)
%     theta_z_0  initial inner lengthscale (default 0.1)
%     D          hidden-layer width (default = ncol(x))
%     l, u       uniform sliding-window proposal bounds (defaults 1 and 2):
%                   theta* ~ Unif(l*theta/u, u*theta/l)
%     alpha      struct of Gamma prior SHAPES  (.g .theta .theta_y .theta_w .theta_z)
%     beta       struct of Gamma prior RATES
%     scale      true (default): x is mapped to [0,1]^d and y is centred/scaled
%                internally; predictions are returned on the original y scale
%     reapprox   iteration count after which the Vecchia conditioning sets are
%                rebuilt in the *warped* space (0 = never, the default).  This
%                is the "updating conditioning sets" idea of Section 4 of the
%                paper / `continue(..., re_approx = TRUE)` in deepgp.
%     verb       progress printing interval in iterations (0 = silent, default 500)
%     cores      number of workers requested for parfor (0 = serial)
%     store_every  keep every k-th latent-layer draw (default 1)
%     eps        jitter (default sqrt(eps))
%
%   See also FIT_ONE_LAYER, FIT_TWO_LAYER, FIT_THREE_LAYER, DGP_PREDICT.

opts = struct();
opts.cov        = 'matern';
opts.v          = 2.5;
opts.vecchia    = true;
opts.m          = [];              % filled in by the fit routines
opts.ordering   = 'random';
opts.nmcmc      = 10000;
opts.true_g     = [];
opts.g_0        = 1e-3;
opts.theta_y_0  = 0.1;
opts.theta_w_0  = 0.1;
opts.theta_z_0  = 0.1;
opts.D          = [];
opts.l          = 1;
opts.u          = 2;
opts.scale      = true;
opts.reapprox   = 0;
opts.verb       = 500;
opts.cores      = 0;
opts.store_every= 1;
opts.eps        = sqrt(eps);
opts.sep        = false;           % reserved: separable (ARD) outer lengthscale
opts.pred_noise = true;            % include the nugget in predictive variances
                                   % (true reproduces deepgp; false gives the
                                   %  noise-free latent-function predictive)
opts.w_init     = [];              % warm start for the middle layer
opts.z_init     = [];              % warm start for the inner layer (3 layers)
opts.ord_init   = [];              % reuse an existing Vecchia ordering

% Gamma(shape = alpha, rate = beta) priors.  These are the deepgp defaults and
% assume x has been scaled to [0,1]^d and y to zero mean / unit variance.
opts.alpha = struct('g', 1.5, 'theta', 1.5, 'theta_y', 1.5, ...
                    'theta_w', 1.5, 'theta_z', 1.5);
opts.beta  = struct('g', 3.9,  'theta', 3.9/1.5, 'theta_y', 3.9/6, ...
                    'theta_w', 3.9/4, 'theta_z', 3.9/4);

if nargin == 0, return; end

% --- merge user input ----------------------------------------------------
if nargin == 1 && isstruct(varargin{1})
    user = varargin{1};
    fn = fieldnames(user);
    for i = 1:numel(fn)
        opts = setsub(opts, fn{i}, user.(fn{i}));
    end
else
    if mod(numel(varargin), 2) ~= 0
        error('vdgp_options:pairs', 'Options must be name/value pairs.');
    end
    for i = 1:2:numel(varargin)
        opts = setsub(opts, varargin{i}, varargin{i+1});
    end
end
end

% -------------------------------------------------------------------------
function opts = setsub(opts, name, value)
if ~ischar(name)
    error('vdgp_options:name', 'Option names must be strings.');
end
if any(strcmp(name, {'alpha', 'beta'})) && isstruct(value)
    fn = fieldnames(value);
    for i = 1:numel(fn)
        opts.(name).(fn{i}) = value.(fn{i});
    end
elseif isfield(opts, name)
    opts.(name) = value;
else
    error('vdgp_options:unknown', 'Unknown option "%s".', name);
end
end
