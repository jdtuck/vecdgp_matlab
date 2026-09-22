classdef MexParityTest < matlab.unittest.TestCase
%MEXPARITYTEST  MEX fast paths agree with the exact dense reference.
%
%   vdgp_U_entries / vdgp_logl / vdgp_krig dispatch to a compiled MEX when it is
%   present (persistent HAVE_*_MEX flag, set from exist(...)==3 on first call).
%   The MEX and pure-MATLAB paths implement identical arithmetic, so when the
%   MEX is built the dispatched result must still match an independent dense
%   Cholesky reference to round-off.  Each test ASSUMES the relevant MEX exists
%   and is therefore SKIPPED (not failed) on runners without a compiled binary
%   (e.g. the Linux CI runner, where the bundled .mexmaca64 files do not load).

    properties
        X; y; Xn; theta = 0.4; g = 1e-3; v = 2.5; ct = 'matern';
        n = 60; nnew = 15; m = 12;
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

        function uEntriesMexMatchesDense(testCase)
            testCase.assumeEqual(exist('vdgp_U_entries_mex', 'file'), 3, ...
                'vdgp_U_entries_mex not built -- skipping parity check.');
            % Full conditioning set so U*U' is the exact precision.
            A = vdgp_create_approx(testCase.X, testCase.n-1, 'random', true);
            [U, ~] = vdgp_create_U(A, testCase.theta, testCase.g, testCase.v, testCase.ct);
            K = vdgp_cov(A.X_ord, [], testCase.theta, testCase.g, 1, testCase.v, testCase.ct);
            err = max(max(abs(full(U*U.') - inv(K))));
            testCase.verifyLessThan(err, 1e-8, sprintf('U*U'' vs inv(K): %.2e', err));
        end

        function loglMexMatchesDense(testCase)
            testCase.assumeEqual(exist('vdgp_logl_mex', 'file'), 3, ...
                'vdgp_logl_mex not built -- skipping parity check.');
            A = vdgp_create_approx(testCase.X, testCase.n-1, 'random', true);
            o = vdgp_logl(testCase.y, A, testCase.theta, testCase.g, testCase.v, testCase.ct, false);
            Kf = vdgp_cov(testCase.X, [], testCase.theta, testCase.g, 1, testCase.v, testCase.ct);
            Lf = chol(Kf, 'lower');
            n = testCase.n;
            ll_exact = -0.5*n*log(2*pi) - sum(log(diag(Lf))) - 0.5*sum((Lf\testCase.y).^2);
            testCase.verifyEqual(o.ll, ll_exact, 'AbsTol', 1e-8);
        end

        function krigMexMatchesExact(testCase)
            testCase.assumeEqual(exist('vdgp_krig_mex', 'file'), 3, ...
                'vdgp_krig_mex not built -- skipping parity check.');
            % lite mode, full conditioning set -> exact GP conditional.
            optsE = vdgp_options('vecchia', false, 'cov', testCase.ct, 'v', testCase.v);
            optsV = vdgp_options('vecchia', true,  'cov', testCase.ct, 'v', testCase.v, 'm', testCase.n);
            pe = vdgp_krig(testCase.y, testCase.X, testCase.Xn, testCase.theta, testCase.g, 2.0, optsE, 'lite');
            pv = vdgp_krig(testCase.y, testCase.X, testCase.Xn, testCase.theta, testCase.g, 2.0, optsV, 'lite', testCase.n);
            testCase.verifyLessThan(max(abs(pe.mean - pv.mean)), 1e-9);
            testCase.verifyLessThan(max(abs(pe.s2 - pv.s2)), 1e-9);
        end
    end
end
