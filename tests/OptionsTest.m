classdef OptionsTest < matlab.unittest.TestCase
%OPTIONSTEST  Defaults, overrides, merging and validation of vdgp_options.

    methods (TestClassSetup)
        function addPackageToPath(testCase)
            import matlab.unittest.fixtures.PathFixture
            here = fileparts(mfilename('fullpath'));
            testCase.applyFixture(PathFixture(fullfile(here, '..')));
        end
    end

    methods (Test)

        function defaultsAreCorrect(testCase)
            o = vdgp_options();
            testCase.verifyEqual(o.cov, 'matern');
            testCase.verifyEqual(o.v, 2.5);
            testCase.verifyTrue(o.vecchia);
            testCase.verifyEmpty(o.m);
            testCase.verifyEqual(o.ordering, 'random');
            testCase.verifyEqual(o.nmcmc, 10000);
            testCase.verifyEqual(o.g_0, 1e-3);
            testCase.verifyEqual(o.theta_y_0, 0.1);
            testCase.verifyTrue(o.scale);
            testCase.verifyEqual(o.reapprox, 0);
            testCase.verifyEqual(o.verb, 500);
            testCase.verifyEqual(o.eps, sqrt(eps));
            testCase.verifyTrue(o.pred_noise);
        end

        function priorStructDefaults(testCase)
            o = vdgp_options();
            testCase.verifyEqual(o.alpha.g, 1.5);
            testCase.verifyEqual(o.alpha.theta, 1.5);
            testCase.verifyEqual(o.beta.g, 3.9);
            testCase.verifyEqual(o.beta.theta, 3.9/1.5, 'AbsTol', 1e-14);
            testCase.verifyEqual(o.beta.theta_w, 3.9/4, 'AbsTol', 1e-14);
        end

        function nameValueOverride(testCase)
            o = vdgp_options('cov', 'exp2', 'v', 1.5, 'm', 12, 'vecchia', false);
            testCase.verifyEqual(o.cov, 'exp2');
            testCase.verifyEqual(o.v, 1.5);
            testCase.verifyEqual(o.m, 12);
            testCase.verifyFalse(o.vecchia);
            % untouched fields keep defaults
            testCase.verifyEqual(o.nmcmc, 10000);
        end

        function structMerge(testCase)
            s = struct('cov', 'exp2', 'nmcmc', 42);
            o = vdgp_options(s);
            testCase.verifyEqual(o.cov, 'exp2');
            testCase.verifyEqual(o.nmcmc, 42);
            testCase.verifyEqual(o.v, 2.5);   % default preserved
        end

        function nestedPriorPartialMerge(testCase)
            % supplying alpha with only .theta must keep the other sub-fields
            o = vdgp_options('alpha', struct('theta', 9.0));
            testCase.verifyEqual(o.alpha.theta, 9.0);
            testCase.verifyEqual(o.alpha.g, 1.5);       % untouched default
            testCase.verifyEqual(o.alpha.theta_w, 1.5); % untouched default
        end

        function unknownOptionErrors(testCase)
            testCase.verifyError(@() vdgp_options('nonsense', 1), ...
                                 'vdgp_options:unknown');
        end

        function oddPairsError(testCase)
            testCase.verifyError(@() vdgp_options('cov'), 'vdgp_options:pairs');
        end

        function nonStringNameError(testCase)
            testCase.verifyError(@() vdgp_options(5, 1), 'vdgp_options:name');
        end
    end
end
