classdef FactorTest < matlab.unittest.TestCase
%FACTORTEST  Sparse inverse-Cholesky factor U and the Vecchia log-likelihood.
%
%   Migrated from tests/test_vdgp.m checks 1-3.  With a full conditioning set
%   (m = n-1) the Vecchia factor and likelihood are exact, so they are checked
%   against a dense Cholesky reference; the error must also decrease with m.

    properties
        X; y; theta = 0.35; g = 1e-3; v = 2.5; ct = 'matern';
    end

    methods (TestClassSetup)
        function setup(testCase)
            import matlab.unittest.fixtures.PathFixture
            here = fileparts(mfilename('fullpath'));
            testCase.applyFixture(PathFixture(fullfile(here, '..')));
            rng(11);
            testCase.X = rand(45, 2);
            testCase.y = randn(45, 1);
        end
    end

    methods (Test)

        function UfactorEqualsInversePrecision(testCase)
            n = size(testCase.X, 1);
            A = vdgp_create_approx(testCase.X, n-1, 'random', true);
            [U, dU] = vdgp_create_U(A, testCase.theta, testCase.g, testCase.v, testCase.ct);
            K = vdgp_cov(A.X_ord, [], testCase.theta, testCase.g, 1, testCase.v, testCase.ct);
            err = max(max(abs(full(U*U.') - inv(K))));
            testCase.verifyLessThan(err, 1e-8);
            % log|Q| from diag(U)
            errld = abs(2*sum(log(dU)) + log(det(K)));
            testCase.verifyLessThan(errld, 1e-8);
        end

        function likelihoodMatchesDense(testCase)
            n = size(testCase.X, 1);
            A = vdgp_create_approx(testCase.X, n-1, 'random', true);
            Kf = vdgp_cov(testCase.X, [], testCase.theta, testCase.g, 1, testCase.v, testCase.ct);
            Lf = chol(Kf, 'lower');
            ll_exact = -0.5*n*log(2*pi) - sum(log(diag(Lf))) - 0.5*sum((Lf\testCase.y).^2);
            o = vdgp_logl(testCase.y, A, testCase.theta, testCase.g, testCase.v, testCase.ct, false);
            testCase.verifyEqual(o.ll, ll_exact, 'AbsTol', 1e-8);
        end

        function likelihoodErrorDecreasesWithM(testCase)
            n = size(testCase.X, 1);
            Kf = vdgp_cov(testCase.X, [], testCase.theta, testCase.g, 1, testCase.v, testCase.ct);
            Lf = chol(Kf, 'lower');
            ll_exact = -0.5*n*log(2*pi) - sum(log(diag(Lf))) - 0.5*sum((Lf\testCase.y).^2);
            ms = [5 10 20 44]; lls = zeros(1, 4);
            for i = 1:4
                Am = vdgp_create_approx(testCase.X, ms(i), 'random', true);
                oo = vdgp_logl(testCase.y, Am, testCase.theta, testCase.g, testCase.v, testCase.ct, false);
                lls(i) = oo.ll;
            end
            testCase.verifyTrue(all(diff(abs(lls - ll_exact)) < 1e-8), ...
                sprintf('|err| = [%s]', sprintf('%.4f ', abs(lls - ll_exact))));
        end

        function profiledOuterLikelihood(testCase)
            n = size(testCase.X, 1);
            A = vdgp_create_approx(testCase.X, n-1, 'random', true);
            Kf = vdgp_cov(testCase.X, [], testCase.theta, testCase.g, 1, testCase.v, testCase.ct);
            Lf = chol(Kf, 'lower');
            o = vdgp_logl(testCase.y, A, testCase.theta, testCase.g, testCase.v, testCase.ct, true);
            tau2ref = (testCase.y' * (Kf \ testCase.y)) / n;
            llref = -0.5*n*log(2*pi) - 0.5*n*log(tau2ref) - sum(log(diag(Lf))) - 0.5*n;
            testCase.verifyEqual(o.ll, llref, 'AbsTol', 1e-7);
            testCase.verifyEqual(o.tau2, tau2ref, 'AbsTol', 1e-9);
        end

        function exactObjectMatchesDense(testCase)
            % vecchia = false path of vdgp_U_entries / vdgp_logl
            n = size(testCase.X, 1);
            A = vdgp_create_approx(testCase.X, [], 'none', false);
            o = vdgp_logl(testCase.y, A, testCase.theta, testCase.g, testCase.v, testCase.ct, false);
            Kf = vdgp_cov(testCase.X, [], testCase.theta, testCase.g, 1, testCase.v, testCase.ct);
            Lf = chol(Kf, 'lower');
            ll_exact = -0.5*n*log(2*pi) - sum(log(diag(Lf))) - 0.5*sum((Lf\testCase.y).^2);
            testCase.verifyEqual(o.ll, ll_exact, 'AbsTol', 1e-8);
        end
    end
end
