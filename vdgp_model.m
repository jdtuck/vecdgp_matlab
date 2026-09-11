classdef vdgp_model
    %VDGP_MODEL VDGP model definition

    properties
        model
        P
        samples
    end

    methods
        function obj = vdgp_model(fit)
            arguments
                fit
            end
            
            obj.model = fit;
            samples.tau2 = fit.tau2;
            samples.g = fit.g;
            obj.P = vdgp_predictor(fit);   
        end

        function pred = predict(obj, x_new, options)
            arguments
                obj
                x_new
                options.idxSamples = nan;
            end
            idxSamples = options.idxSamples;
            if isnan(idxSamples) 
                idxSamples = 1:obj.model.nmcmc;
                nSamps = obj.model.nmcmc;
            else
                idxSamples = mod(idxSamples, obj.model.nmcmc);
                
                nSamps = length(idxSamples);
                idxSamples = 1:nSamps;
            end

            out = vdgp_draw_pt(obj.P, x_new, 'nsamp', nSamps);
            pred = out(:, idxSamples)';

        end
    end
end