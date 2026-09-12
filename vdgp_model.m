classdef vdgp_model
    %VDGP_MODEL  Surrogate wrapper for calibration (mvbayes / impala).
    %
    %   obj  = vdgp_model(fit)            fit must already be DGP_TRIMmed
    %   f    = obj.predict(x)             ONE draw, random emulator index
    %   f    = obj.predict(x, idxSamples=k)   draw from emulator index k
    %   f    = obj.predict(x, nsamp=K)    K draws, random indices
    %   f    = obj.predict(X)             X may be n_new-by-d: one column of
    %                                     the result per input row
    %   [m,v]= obj.moments(x)             plug-in mean and variance, no sampling
    %
    %   PREDICT returns an nsamp-by-n_new matrix: one row per emulator draw,
    %   one column per input row.  It is the entry point to use inside a
    %   calibration MCMC when the likelihood should carry surrogate
    %   uncertainty.  One input row or many are both fine -- two internal
    %   orientations are available and the faster one is chosen by size; see
    %   'path' below.
    %
    %   Draws at different input rows share the selected emulator draw but
    %   come from their own marginals, so they do not carry the surrogate's
    %   correlation ACROSS rows.  If the likelihood compares several inputs
    %   that are close relative to the lengthscale, and that correlation
    %   matters, use dgp_predict(fit, X, 'lite', false, 'nsamp', K) instead.
    %
    %   idxSamples selects WHICH retained emulator draw is used, and is
    %   wrapped into range as mod(idxSamples-1, nmcmc)+1, so a calibration
    %   iteration counter can be passed straight in and will cycle through
    %   the emulator posterior.  Leave it unset for an independently drawn
    %   index at every call.
    %
    %   Which of those you want is a modelling decision:
    %
    %     * a fresh index each call gives a noisy likelihood -- the chain then
    %       targets the calibration posterior only approximately, because the
    %       emulator realisation moves underneath it;
    %     * a FIXED index for a whole chain, repeated over several indices, is
    %       the modularised / multiple-imputation approach;
    %     * cycling the index with the iteration counter sits between the two.
    %
    %   Note that a fixed idxSamples pins the emulator's mean and variance for
    %   that draw, but PREDICT still adds fresh predictive noise, so repeated
    %   calls differ. For a bitwise-reproducible realisation, fix the RNG
    %   stream around the call, or use MOMENTS and add noise yourself.
    %
    %   See also VDGP_FIT, VDGP_DRAW_PT, VDGP_PREDICT_PT, VDGP_PREDICTOR.

    properties
        model       % the trimmed fit
        P           % predictor built from it (VDGP_PREDICTOR)
        samples     % retained hyperparameter chains
        nmcmc       % number of retained emulator draws
    end

    methods
        function obj = vdgp_model(fit)
            arguments
                fit struct
            end
            if ~isfield(fit, 'nmcmc') || ~isfield(fit, 'tau2') || ~isfield(fit, 'x')
                error('vdgp_model:fit', ...
                      'Expected a fit from vdgp_fit / fit_*_layer.');
            end

            obj.model = fit;
            obj.nmcmc = fit.nmcmc;

            s = struct('tau2', fit.tau2, 'g', fit.g, 'theta_y', fit.theta_y);
            if isfield(fit, 'theta_w') && ~isempty(fit.theta_w)
                s.theta_w = fit.theta_w;
            end
            if isfield(fit, 'theta_z') && ~isempty(fit.theta_z)
                s.theta_z = fit.theta_z;
            end
            obj.samples = s;

            obj.P = vdgp_predictor(fit);
        end

        function pred = predict(obj, x_new, options)
            %PREDICT  Draw from the posterior predictive.
            %   Returns nsamp-by-n_new: rows are emulator draws.
            arguments
                obj
                x_new double
                options.idxSamples double = []
                options.nsamp (1,1) double {mustBeInteger, mustBePositive} = 1
                options.path (1,:) char {mustBeMember(options.path, ...
                    {'auto','points','draws'})} = 'auto'
            end

            idx = options.idxSamples;
            idx = idx(~isnan(idx));                 % NaN means "unspecified"

            if isempty(idx)
                f = vdgp_draw_pt(obj.P, x_new, 'nsamp', options.nsamp, ...
                                 'path', options.path);
            else
                % wrap into 1..nmcmc; mod(k,T) would give 0 on multiples of T
                idx = mod(idx(:).' - 1, obj.nmcmc) + 1;
                f = vdgp_draw_pt(obj.P, x_new, 'idx', idx, 'path', options.path);
            end

            pred = f.';                             % nsamp-by-n_new
        end

        function [mu, s2] = moments(obj, x_new, options)
            %MOMENTS  Posterior predictive mean and variance over all draws.
            %   No sampling: use this for a plug-in likelihood, or as the
            %   reference the draws from PREDICT scatter around.
            arguments
                obj
                x_new double
                options.path (1,:) char {mustBeMember(options.path, ...
                    {'auto','points','draws'})} = 'auto'
            end
            [m1, v1] = vdgp_predict_pt(obj.P, x_new, 'path', options.path);
            mu = m1.';
            s2 = v1.';
        end
    end
end
