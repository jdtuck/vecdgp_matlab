classdef PriorSamplingTest < matlab.unittest.TestCase
%PRIORSAMPLINGTEST  Stochastic prior / MCMC-kernel correctness.
%
%   These use statistical tolerances against known target distributions and a
%   fixed RNG seed (rng(11), as in the original tests/test_vdgp.m).  Migrates
%   checks 4, 6 and 7 and adds a prior-recovery test for vdgp_sample_g.
%
%   The n=1 trick: with a single point there are no distances, so the
%   likelihood is free of the parameter and the Metropolis-Hastings / ESS
%   kernels must reproduce their prior exactly.

    properties
        v = 2.5; ct = 'matern';
    end

    methods (TestMethodSetup)
        function seed(~)
            rng(11);
        end
    end

    methods (TestClassSetup)
        function addPackageToPath(testCase)
            import matlab.unittest.fixtures.PathFixture
            here = fileparts(mfilename('fullpath'));
            testCase.applyFixture(PathFixture(fullfile(here, '..')));
        end
    end

    methods (Test)

        function randMvnEmpiricalCovariance(testCase)
            nS = 20000; ns = 12;
            Xs = rand(ns, 1);
            As = vdgp_create_approx(Xs, ns-1, 'random', true);
            S = zeros(ns, nS);
            for i = 1:nS
                S(:, i) = vdgp_rand_mvn(As, 0.3, 1e-6, testCase.v, testCase.ct, 1);
            end
            Kref = vdgp_cov(Xs, [], 0.3, 1e-6, 1, testCase.v, testCase.ct);
            Cerr = max(max(abs(cov(S.') - Kref)));
            testCase.verifyLessThan(Cerr, 0.06, sprintf('max cov err %.3f', Cerr));
        end

        function mhThetaRecoversGammaPrior(testCase)
            A1 = vdgp_create_approx(0.5, 0, 'none', true);
            opts1 = vdgp_options('vecchia', true, 'cov', testCase.ct, 'v', testCase.v);
            alpha = 1.5; beta = 3.9/4;
            th = 0.5; T = 40000; keepT = zeros(T, 1);
            oo = vdgp_logl(1.0, A1, th, 1e-4, testCase.v, testCase.ct, false); ll = oo.ll;
            for t = 1:T
                [th, ll] = vdgp_sample_theta(1.0, A1, th, 1, 1e-4, ll, alpha, beta, opts1, false);
                keepT(t) = th;
            end
            mref = alpha/beta; sref = sqrt(alpha)/beta;
            em = abs(mean(keepT) - mref)/mref;
            es = abs(std(keepT) - sref)/sref;
            testCase.verifyLessThan(em, 0.05, sprintf('mean %.3f (%.3f)', mean(keepT), mref));
            testCase.verifyLessThan(es, 0.08, sprintf('sd %.3f (%.3f)', std(keepT), sref));
        end

        function mhNuggetRecoversGammaPrior(testCase)
            % Same n=1 trick for vdgp_sample_g (always profiled/outer).
            A1 = vdgp_create_approx(0.5, 0, 'none', true);
            opts1 = vdgp_options('vecchia', true, 'cov', testCase.ct, 'v', testCase.v);
            alpha = 1.5; beta = 3.9;
            g = 0.4; T = 40000; keepG = zeros(T, 1);
            oo = vdgp_logl(1.0, A1, 0.5, g, testCase.v, testCase.ct, true); ll = oo.ll;
            for t = 1:T
                [g, ll] = vdgp_sample_g(1.0, A1, 0.5, g, ll, alpha, beta, opts1);
                keepG(t) = g;
            end
            mref = alpha/beta; sref = sqrt(alpha)/beta;
            em = abs(mean(keepG) - mref)/mref;
            es = abs(std(keepG) - sref)/sref;
            testCase.verifyLessThan(em, 0.06, sprintf('mean %.3f (%.3f)', mean(keepG), mref));
            testCase.verifyLessThan(es, 0.10, sprintf('sd %.3f (%.3f)', std(keepG), sref));
        end

        function essLeavesPriorInvariant(testCase)
            Aw = vdgp_create_approx(0.5, 0, 'none', true);
            optsw = vdgp_options('vecchia', true, 'cov', testCase.ct, 'v', testCase.v);
            w = 0; T = 20000; keepW = zeros(T, 1);
            wobj = vdgp_create_approx(0, 0, 'none', true);
            oo = vdgp_logl(1.0, wobj, 0.1, 1e-4, testCase.v, testCase.ct, true); ll = oo.ll;
            for t = 1:T
                [w, wobj, ll] = vdgp_sample_w(1.0, w, Aw, wobj, 0.1, 0.1, 1e-4, ll, optsw);
                keepW(t) = w;
            end
            testCase.verifyLessThan(abs(mean(keepW)), 0.05, ...
                sprintf('mean %.3f (0)', mean(keepW)));
            testCase.verifyLessThan(abs(std(keepW) - 1), 0.05, ...
                sprintf('sd %.3f (1)', std(keepW)));
        end
    end
end
