function P = vdgp_predictor(fit, varargin)
%VDGP_PREDICTOR  Pack a fitted DGP for fast repeated prediction at few points.
%
%   P = VDGP_PREDICTOR(FIT)                  after DGP_TRIM
%   P = VDGP_PREDICTOR(FIT, 'name', value)
%
%   Build this ONCE, outside a calibration loop, then call VDGP_PREDICT_PT
%   for each proposed input.  The object holds the retained MCMC draws in a
%   layout that lets every draw be evaluated in one batched linear-algebra
%   call rather than one call per draw.
%
%   Options
%     'm'          conditioning-set size (default: the fit's m)
%     'chunk'      max MCMC draws handled per batch (default 1024); lower it
%                  if memory is tight, raise it if it is not
%
%   Why a separate entry point.  DGP_PREDICT is organised for many test
%   points at once: it loops over MCMC draws and, inside each, krigs all the
%   test points together.  Calibration inverts that -- one (or a handful of)
%   test points, every retained draw -- so the loop is over the small
%   dimension and the batch is over the large one.  Reorganising it that way
%   costs nothing in accuracy and removes both the per-draw interpreter
%   overhead and the whole-design neighbour search.
%
%   See also VDGP_PREDICT_PT, DGP_PREDICT, DGP_TRIM.

opt = struct('m', [], 'chunk', 1024);
if numel(varargin) == 1 && isstruct(varargin{1})
    fn = fieldnames(varargin{1});
    for i = 1:numel(fn), opt.(fn{i}) = varargin{1}.(fn{i}); end
else
    for i = 1:2:numel(varargin)
        if ~isfield(opt, varargin{i})
            error('vdgp_predictor:opt', 'Unknown option "%s".', varargin{i});
        end
        opt.(varargin{i}) = varargin{i+1};
    end
end

if ~isfield(fit, 'layers')
    error('vdgp_predictor:fit', 'Expected a fit from fit_one/two/three_layer.');
end

P = struct();
P.layers  = fit.layers;
P.x       = fit.x;
P.y       = fit.y(:);
P.n       = size(fit.x, 1);
P.d       = size(fit.x, 2);
P.T       = fit.nmcmc;
P.opts    = fit.opts;
P.scaling = fit.scaling;
P.theta_y = fit.theta_y(:);
P.g       = fit.g(:);
P.tau2    = fit.tau2(:);
P.chunk   = max(1, round(opt.chunk));

if isempty(opt.m), P.m = fit.opts.m; else, P.m = opt.m; end
P.m = min(P.m, P.n);

switch fit.layers
    case 1
        P.D = 0;
    case 2
        P.w = fit.w;  P.theta_w = fit.theta_w;  P.D = size(fit.w, 2);
    case 3
        P.w = fit.w;  P.theta_w = fit.theta_w;
        P.z = fit.z;  P.theta_z = fit.theta_z;
        P.D = size(fit.w, 2);
    otherwise
        error('vdgp_predictor:layers', 'Unsupported number of layers.');
end

if size(P.theta_y, 1) ~= P.T
    error('vdgp_predictor:trim', ...
          ['Chain length (%d) does not match the stored draws. Run ' ...
           'DGP_TRIM before VDGP_PREDICTOR.'], P.T);
end

% kernel code, resolved once
if strcmpi(P.opts.cov, 'exp2'), P.vc = 999; P.pw = 1;
else,                          P.vc = P.opts.v; P.pw = 2;
end
end
