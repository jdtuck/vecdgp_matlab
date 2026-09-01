classdef vdgp_model
    %VDGP_MODEL VDGP model definition

    properties
        model
        samples
    end

    methods
        function obj = vdgp_model(x, y, nmcmc, burn, thin, m, nugget)
            arguments
                x
                y
                nmcmc = 10000
                burn  = 5000
                thin = 2
                m = 25
                nugget = []
            end
            model = fit_two_layer(x, y, 'nmcmc', nmcmc, 'm', m, ...
                'true_g', nugget, 'verb', 500);
            model = dgp_trim(model, burn, thin);
            obj.model = model;
        end

        function pred = predict(obj, x_new, options)
            arguments
                obj
                x_new
                options.idxSamples = nan;
                options.B = 500;
            end
            idxSamples = options.idxSamples;
            if isnan(idxSamples) 
                idxSamples = 1:options.B;
            end

            out = dgp_predict(obj.model, x_new, 'nsamp', options.B);
            pred = out.f(:, idxSamples)';

        end
    end
end