classdef vdgp_model
    %VDGP_MODEL VDGP model definition

    properties
        model
        samples
    end

    methods
        function obj = vdgp_model(model)
            arguments
                model
            end
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