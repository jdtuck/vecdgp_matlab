classdef PrimitivesTest < matlab.unittest.TestCase
%PRIMITIVESTEST  Pure, deterministic building blocks of the Vecchia-DGP package.
%
%   Each covariance / distance / score primitive is checked against an
%   independent hand-written reference.  These functions have no randomness, so
%   the tolerances are at round-off.

    methods (TestClassSetup)
        function addPackageToPath(testCase)
            import matlab.unittest.fixtures.PathFixture
            here = fileparts(mfilename('fullpath'));
            testCase.applyFixture(PathFixture(fullfile(here, '..')));
        end
    end

    methods (Test)

        % ----------------------------------------------------------------- cov
        function maternMatchesFormula(testCase)
            rng(1);
            X1 = rand(6, 2); X2 = rand(4, 2);
            for v = [0.5 1.5 2.5]
                theta = [0.3 0.7];
                K = vdgp_cov(X1, X2, theta, 0, 1, v, 'matern');
                % reference: r = sqrt(sum_k (dx_k)^2 / theta_k^2)
                Kref = zeros(6, 4);
                for i = 1:6
                    for j = 1:4
                        r = sqrt(sum((X1(i,:) - X2(j,:)).^2 ./ theta.^2));
                        switch v
                            case 0.5, Kref(i,j) = exp(-r);
                            case 1.5, Kref(i,j) = (1 + sqrt(3)*r)*exp(-sqrt(3)*r);
                            case 2.5, Kref(i,j) = (1 + sqrt(5)*r + 5*r^2/3)*exp(-sqrt(5)*r);
                        end
                    end
                end
                testCase.verifyLessThan(max(abs(K(:) - Kref(:))), 1e-12, ...
                    sprintf('matern v=%g', v));
            end
        end

        function exp2MatchesFormula(testCase)
            rng(2);
            X1 = rand(5, 3); X2 = rand(5, 3);
            theta = [0.2 0.5 0.9];
            K = vdgp_cov(X1, X2, theta, 0, 1, [], 'exp2');
            Kref = zeros(5);
            for i = 1:5
                for j = 1:5
                    Kref(i,j) = exp(-sum((X1(i,:) - X2(j,:)).^2 ./ theta));
                end
            end
            testCase.verifyLessThan(max(abs(K(:) - Kref(:))), 1e-12);
        end

        function nuggetOnlyOnSymmetricDiagonal(testCase)
            rng(3);
            X = rand(7, 2); g = 0.05;
            % X2 empty -> nugget added to diagonal
            Ksym = vdgp_cov(X, [], 0.4, g, 1, 2.5, 'matern');
            Knog = vdgp_cov(X, [], 0.4, 0, 1, 2.5, 'matern');
            testCase.verifyEqual(diag(Ksym), diag(Knog) + g, 'AbsTol', 1e-12);
            testCase.verifyEqual(Ksym - diag(diag(Ksym)), ...
                                 Knog - diag(diag(Knog)), 'AbsTol', 1e-12);
            % X2 supplied -> no nugget even if X2 == X
            Kcross = vdgp_cov(X, X, 0.4, g, 1, 2.5, 'matern');
            testCase.verifyEqual(Kcross, Knog, 'AbsTol', 1e-12);
        end

        function vectorNuggetAndTau2(testCase)
            rng(4);
            X = rand(5, 2);
            gv = (1:5)' * 0.01;
            tau2 = 3.5;
            K = vdgp_cov(X, [], 0.4, gv, tau2, 2.5, 'matern');
            K0 = vdgp_cov(X, [], 0.4, 0, 1, 2.5, 'matern');
            Kref = tau2 * (K0 + diag(gv));
            testCase.verifyEqual(K, Kref, 'AbsTol', 1e-12);
        end

        function scalarThetaEqualsRepeatedVector(testCase)
            rng(5);
            X = rand(6, 3);
            Ks = vdgp_cov(X, [], 0.4, 0, 1, 2.5, 'matern');
            Kv = vdgp_cov(X, [], [0.4 0.4 0.4], 0, 1, 2.5, 'matern');
            testCase.verifyEqual(Ks, Kv, 'AbsTol', 1e-13);
        end

        function covRejectsBadV(testCase)
            X = rand(3, 1);
            testCase.verifyError(@() vdgp_cov(X, [], 0.4, 0, 1, 1.0, 'matern'), ...
                                 'vdgp_cov:v');
        end

        % -------------------------------------------------------------- sqdist
        function sqdistMatchesReference(testCase)
            rng(6);
            X1 = rand(5, 2); X2 = rand(4, 2); theta = [0.3 0.8];
            for pw = [1 2]
                D2 = vdgp_sqdist(X1, X2, theta, pw);
                Dref = zeros(5, 4);
                for i = 1:5
                    for j = 1:4
                        Dref(i,j) = sum((X1(i,:) - X2(j,:)).^2 ./ theta.^pw);
                    end
                end
                testCase.verifyEqual(D2, Dref, 'AbsTol', 1e-12, ...
                    sprintf('pw=%d', pw));
            end
        end

        function sqdistNonNegativeAndZeroDiag(testCase)
            rng(7);
            X = rand(8, 2);
            D2 = vdgp_sqdist(X, X, 0.5, 2);
            testCase.verifyGreaterThanOrEqual(min(D2(:)), 0);
            testCase.verifyLessThan(max(abs(diag(D2))), 1e-12);
        end

        function sqdistRejectsThetaLengthMismatch(testCase)
            X = rand(4, 3);
            testCase.verifyError(@() vdgp_sqdist(X, X, [0.1 0.2], 2), ...
                                 'vdgp_sqdist:theta');
        end

        % ---------------------------------------------------------------- crps
        function crpsMatchesClosedForm(testCase)
            y = [0.0; 1.0; -2.0]; mu = [0.0; 0.5; 0.0]; sd = [1.0; 2.0; 0.7];
            s = vdgp_crps(y, mu, sd);
            z = (y - mu) ./ sd;
            Phi = 0.5 * (1 + erf(z / sqrt(2)));
            phi = exp(-0.5 * z.^2) / sqrt(2*pi);
            sref = sd .* (z .* (2*Phi - 1) + 2*phi - 1/sqrt(pi));
            testCase.verifyEqual(s, sref, 'AbsTol', 1e-13);
        end

        function crpsPerfectForecastValue(testCase)
            % y == mu: crps = sd * (2 phi(0) - 1/sqrt(pi)) = sd*(2/sqrt(2pi) - 1/sqrt(pi))
            sd = 1.3;
            s = vdgp_crps(0, 0, sd);
            ref = sd * (2/sqrt(2*pi) - 1/sqrt(pi));
            testCase.verifyEqual(s, ref, 'AbsTol', 1e-13);
        end

        function crpsIncreasesWithError(testCase)
            sd = 1.0;
            s0 = vdgp_crps(0, 0, sd);
            s1 = vdgp_crps(2, 0, sd);
            s2 = vdgp_crps(5, 0, sd);
            testCase.verifyLessThan(s0, s1);
            testCase.verifyLessThan(s1, s2);
        end

        % -------------------------------------------------------------- lgamma
        function lgammaMatchesFormula(testCase)
            x = [0.1; 0.5; 1.0; 3.0]; shape = 1.5; rate = 2.6;
            lp = vdgp_lgamma(x, shape, rate);
            ref = shape*log(rate) - gammaln(shape) + (shape-1)*log(x) - rate*x;
            testCase.verifyEqual(lp, ref, 'AbsTol', 1e-12);
        end

        function lgammaNegInfForNonPositive(testCase)
            x = [-1; 0; 2];
            lp = vdgp_lgamma(x, 2.0, 1.0);
            testCase.verifyEqual(lp(1), -Inf);
            testCase.verifyEqual(lp(2), -Inf);
            testCase.verifyTrue(isfinite(lp(3)));
        end

        function lgammaShapePreserved(testCase)
            x = rand(3, 4) + 0.1;
            lp = vdgp_lgamma(x, 1.2, 0.9);
            testCase.verifyEqual(size(lp), size(x));
        end

        % ----------------------------------------------------------------- knn
        function knnMatchesBruteForce(testCase)
            rng(8);
            Xref = rand(40, 3); Xq = rand(10, 3); k = 5;
            [idx, dist] = vdgp_knn(Xref, Xq, k);
            testCase.verifyEqual(size(idx), [10 k]);
            for q = 1:10
                d = sqrt(sum((Xref - Xq(q,:)).^2, 2));
                [sd, so] = sort(d);
                testCase.verifyEqual(sort(idx(q,:)), sort(so(1:k)'), ...
                    'AbsTol', 0);   % same set of neighbours
                testCase.verifyEqual(sort(dist(q,:)), sd(1:k)', 'AbsTol', 1e-12);
            end
        end

        function knnClampsKToRefCount(testCase)
            rng(9);
            Xref = rand(4, 2); Xq = rand(3, 2);
            idx = vdgp_knn(Xref, Xq, 10);
            testCase.verifyEqual(size(idx, 2), 4);   % clamped to size(Xref,1)
        end
    end
end
