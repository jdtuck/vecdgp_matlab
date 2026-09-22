classdef KrigTest < matlab.unittest.TestCase
%KRIGTEST  Vecchia prediction against the exact GP conditional.
%
%   Migrated from tests/test_vdgp.m checks 5 and 5b.  At a full conditioning
%   set the Vecchia predictor is exact (lite and joint); the covariance-free
%   'sample' path must reproduce the joint covariance it never forms.

    properties
        X; y; Xn; theta = 0.35; g = 1e-3; v = 2.5; ct = 'matern';
        n = 45; nnew = 20;
    end

    methods (TestClassSetup)
        function setup(testCase)
            import matlab.unittest.fixtures.PathFixture
            here = fileparts(mfilename('fullpath'));
            testCase.applyFixture(PathFixture(fullfile(here, '..')));
            rng(11);
            testCase.X  = rand(testCase.n, 2);
            testCase.y  = randn(testCase.n, 1);
            testCase.Xn = rand(testCase.nnew, 2);
        end
    end

    methods (Test)

        function liteMatchesExactAtFullM(testCase)
            optsE = vdgp_options('vecchia', false, 'cov', testCase.ct, 'v', testCase.v);
            optsV = vdgp_options('vecchia', true,  'cov', testCase.ct, 'v', testCase.v, 'm', testCase.n);
            pe = vdgp_krig(testCase.y, testCase.X, testCase.Xn, testCase.theta, testCase.g, 2.0, optsE, 'lite');
            pv = vdgp_krig(testCase.y, testCase.X, testCase.Xn, testCase.theta, testCase.g, 2.0, optsV, 'lite', testCase.n);
            testCase.verifyLessThan(max(abs(pe.mean - pv.mean)), 1e-9);
            testCase.verifyLessThan(max(abs(pe.s2 - pv.s2)), 1e-9);
        end

        function jointMatchesExactAtFullM(testCase)
            optsE = vdgp_options('vecchia', false, 'cov', testCase.ct, 'v', testCase.v);
            mtot  = testCase.n + testCase.nnew;
            optsJ = vdgp_options('vecchia', true, 'cov', testCase.ct, 'v', testCase.v, 'm', mtot);
            pj  = vdgp_krig(testCase.y, testCase.X, testCase.Xn, testCase.theta, testCase.g, 2.0, optsJ, 'joint', mtot);
            pej = vdgp_krig(testCase.y, testCase.X, testCase.Xn, testCase.theta, testCase.g, 2.0, optsE, 'joint');
            testCase.verifyLessThan(max(abs(pej.mean - pj.mean)), 1e-7);
            testCase.verifyLessThan(max(max(abs(pej.Sigma - pj.Sigma))), 1e-6);
        end

        function sampleDrawsMatchJointCovariance(testCase)
            mtot  = testCase.n + testCase.nnew;
            optsJ = vdgp_options('vecchia', true, 'cov', testCase.ct, 'v', testCase.v, 'm', mtot);
            pj = vdgp_krig(testCase.y, testCase.X, testCase.Xn, testCase.theta, testCase.g, 2.0, optsJ, 'joint', mtot);
            os = vdgp_krig(testCase.y, testCase.X, testCase.Xn, testCase.theta, testCase.g, 2.0, optsJ, 'sample', mtot, [], 30000);
            e1 = max(abs(mean(os.draws, 2) - pj.mean)) / sqrt(max(diag(pj.Sigma)));
            e2 = max(max(abs(cov(os.draws.') - pj.Sigma))) / max(abs(pj.Sigma(:)));
            testCase.verifyLessThan(e1, 0.05, sprintf('mean %.3f sd', e1));
            testCase.verifyLessThan(e2, 0.08, sprintf('cov %.3f rel', e2));
        end
    end
end
