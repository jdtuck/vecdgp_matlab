classdef EndToEndTest < matlab.unittest.TestCase
%ENDTOENDTEST  Full fit -> predict -> calibration pipeline.
%
%   Migrated from tests/test_vdgp.m checks 10-14.  A two-layer Vecchia DGP is
%   fitted once (shared across the tests via TestClassSetup) on the deepgp 1-D
%   example, then predicted through dgp_predict and the calibration fast paths
%   vdgp_predictor / vdgp_predict_pt / vdgp_draw_pt, which must all agree.

    properties
        fit; xte; yte;
    end

    methods (TestClassSetup)
        function fitModel(testCase)
            import matlab.unittest.fixtures.PathFixture
            here = fileparts(mfilename('fullpath'));
            testCase.applyFixture(PathFixture(fullfile(here, '..')));
            rng(11);
            f = @(x) (x <= 0.58) .* (sin(pi*x*6) + cos(pi*x*12)) + (x > 0.58) .* (5*x - 4.9);
            xtr = linspace(0, 1, 25)'; ytr = f(xtr);
            testCase.xte = linspace(0, 1, 200)';
            testCase.yte = f(testCase.xte);
            fit = fit_two_layer(xtr, ytr, 'nmcmc', 1200, 'vecchia', true, 'm', 10, ...
                                'true_g', 1e-6, 'verb', 0);
            testCase.fit = dgp_trim(fit, 600, 2);
        end
    end

    methods (Test)

        function twoLayerAccuracyAndCoverage(testCase)
            p = dgp_predict(testCase.fit, testCase.xte);
            rmse  = sqrt(mean((p.mean - testCase.yte).^2));
            cov95 = mean(abs(p.mean - testCase.yte) <= 1.96 * p.sd);
            testCase.verifyLessThan(rmse, 0.08, sprintf('RMSE %.4f', rmse));
            testCase.verifyGreaterThan(cov95, 0.85, sprintf('coverage %.2f', cov95));
        end

        function samplePathsReproduceMixture(testCase)
            xs = testCase.xte(1:5:end);
            ps = dgp_predict(testCase.fit, xs, 'lite', false, 'nsamp', 6000);
            e1 = max(abs(mean(ps.f, 2) - ps.mean)) / max(ps.sd);
            e2 = max(max(abs(cov(ps.f.') - ps.Sigma))) / max(abs(ps.Sigma(:)));
            testCase.verifyLessThan(e1, 0.08, sprintf('mean %.3f sd', e1));
            testCase.verifyLessThan(e2, 0.15, sprintf('cov %.3f rel', e2));
            testCase.verifyEqual(size(ps.f, 2), 6000);
            % each column traceable to the MCMC iteration that produced it
            okit = numel(ps.f_iter) == 6000 && all(ps.f_iter >= 1) && ...
                   all(ps.f_iter <= testCase.fit.nmcmc) && ...
                   numel(unique(ps.f_iter)) == testCase.fit.nmcmc;
            testCase.verifyTrue(okit, ...
                sprintf('%d iterations used', numel(unique(ps.f_iter))));
        end

        function calibrationPathMatchesDgpPredict(testCase)
            Pp = vdgp_predictor(testCase.fit);
            xsel = testCase.xte(1:7:end);
            [muP, s2P] = vdgp_predict_pt(Pp, xsel);
            pRef = dgp_predict(testCase.fit, xsel);
            testCase.verifyLessThan(max(abs(muP - pRef.mean)), 1e-9);
            testCase.verifyLessThan(max(abs(s2P - pRef.s2)), 1e-9);
            % one point at a time equals the block
            mu1 = zeros(numel(xsel), 1);
            for i = 1:numel(xsel)
                mu1(i) = vdgp_predict_pt(Pp, xsel(i));
            end
            testCase.verifyLessThan(max(abs(mu1 - muP)), 1e-8);
        end

        function drawsReproduceMixture(testCase)
            Pp = vdgp_predictor(testCase.fit);
            x1 = testCase.xte(40);
            [muM, s2M] = vdgp_predict_pt(Pp, x1);
            ND = 120000; Fd = zeros(ND, 1);
            for bi = 1:120
                Fd((bi-1)*1000 + (1:1000)) = vdgp_draw_pt(Pp, x1, 'nsamp', 1000);
            end
            e1 = abs(mean(Fd) - muM) / sqrt(s2M);
            e2 = abs(var(Fd) - s2M) / s2M;
            testCase.verifyLessThan(e1, 0.05, sprintf('mean %.3f sd', e1));
            testCase.verifyLessThan(e2, 0.05, sprintf('var %.3f rel', e2));
        end

        function fixedDrawIndexIsReproducible(testCase)
            Pp = vdgp_predictor(testCase.fit);
            x1 = testCase.xte(40);
            [~, i1] = vdgp_draw_pt(Pp, x1, 'idx', 3);
            [~, i2] = vdgp_draw_pt(Pp, x1, 'idx', 3);
            testCase.verifyLessThan(abs(i1.mu_t - i2.mu_t), 1e-14);
            testCase.verifyEqual(i1.idx, 3);
        end

        function predictionOrientationsAgree(testCase)
            Pp = vdgp_predictor(testCase.fit);
            Xblk = testCase.xte(5:9);
            pA = dgp_predict(testCase.fit, Xblk);
            [mD, sD]  = vdgp_predict_pt(Pp, Xblk, 'path', 'draws');
            [mP, sPv] = vdgp_predict_pt(Pp, Xblk, 'path', 'points');
            testCase.verifyLessThan(max(abs(mD - mP)), 1e-8);
            testCase.verifyLessThan(max(abs(sD - sPv)), 1e-8);
            testCase.verifyLessThan(max(abs(pA.mean - mP)), 1e-8);
            % draw path, fixed emulator index
            [~, iD] = vdgp_draw_pt(Pp, Xblk, 'idx', 4, 'path', 'draws');
            [~, iP] = vdgp_draw_pt(Pp, Xblk, 'idx', 4, 'path', 'points');
            testCase.verifyLessThan(max(abs(iD.mu_t - iP.mu_t)), 1e-8);
            testCase.verifyLessThan(max(abs(iD.s2_t - iP.s2_t)), 1e-8);
            % block equals one at a time
            mu1 = zeros(numel(Xblk), 1);
            for i = 1:numel(Xblk)
                mu1(i) = vdgp_predict_pt(Pp, Xblk(i));
            end
            testCase.verifyLessThan(max(abs(mu1 - mP)), 1e-8);
        end
    end
end
