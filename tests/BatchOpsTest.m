classdef BatchOpsTest < matlab.unittest.TestCase
%BATCHOPSTEST  Batched (3-D page-wise) linear algebra primitives.
%
%   vdgp_batch_chol / _cov / _fsolve / _last_row_inv operate on k-by-k-by-N
%   stacks.  Each is checked page-by-page against the corresponding dense
%   MATLAB operation.

    methods (TestClassSetup)
        function addPackageToPath(testCase)
            import matlab.unittest.fixtures.PathFixture
            here = fileparts(mfilename('fullpath'));
            testCase.applyFixture(PathFixture(fullfile(here, '..')));
        end
    end

    methods (Static)
        function A = spdStack(k, N, seed)
            rng(seed);
            A = zeros(k, k, N);
            for p = 1:N
                M = randn(k);
                A(:, :, p) = M*M.' + k*eye(k);   % well-conditioned SPD
            end
        end
    end

    methods (Test)

        % --------------------------------------------------------- batch_chol
        function batchCholMatchesPerPage(testCase)
            k = 5; N = 8;
            A = BatchOpsTest.spdStack(k, N, 10);
            [L, ok] = vdgp_batch_chol(A);
            testCase.verifyTrue(all(ok));
            for p = 1:N
                Lref = chol(A(:, :, p), 'lower');
                testCase.verifyEqual(L(:, :, p), Lref, 'AbsTol', 1e-10, ...
                    sprintf('page %d', p));
            end
        end

        function batchCholReconstructs(testCase)
            k = 4; N = 6;
            A = BatchOpsTest.spdStack(k, N, 11);
            L = vdgp_batch_chol(A);
            for p = 1:N
                testCase.verifyEqual(L(:,:,p)*L(:,:,p).', A(:,:,p), ...
                    'AbsTol', 1e-9, sprintf('page %d', p));
            end
        end

        function batchCholRejectsNonSquare(testCase)
            A = zeros(3, 4, 2);
            testCase.verifyError(@() vdgp_batch_chol(A), 'vdgp_batch_chol:square');
        end

        function batchCholJitterRecoversSingularPage(testCase)
            % A page that is only positive-semidefinite forces the jitter retry.
            k = 3; N = 2;
            A = BatchOpsTest.spdStack(k, N, 12);
            A(:, :, 2) = ones(k);        % rank-1, not PD
            [L, ok] = vdgp_batch_chol(A);
            % after jitter, factor should be finite; page 1 stays clean
            testCase.verifyTrue(ok(1));
            testCase.verifyTrue(all(isfinite(L(:))));
        end

        % ---------------------------------------------------------- batch_cov
        function batchCovMatchesVdgpCov(testCase)
            rng(13);
            k = 6; d = 2; N = 5;
            P = rand(k, d, N);
            theta = [0.3 0.7]; v = 2.5;
            K = vdgp_batch_cov(P, theta, v, 'matern');
            for p = 1:N
                Kref = vdgp_cov(P(:, :, p), [], theta, 0, 1, v, 'matern');
                testCase.verifyEqual(K(:, :, p), Kref, 'AbsTol', 1e-12, ...
                    sprintf('page %d', p));
            end
        end

        function batchCovExp2(testCase)
            rng(14);
            k = 4; d = 3; N = 3;
            P = rand(k, d, N);
            theta = [0.2 0.5 0.9];
            K = vdgp_batch_cov(P, theta, [], 'exp2');
            for p = 1:N
                Kref = vdgp_cov(P(:, :, p), [], theta, 0, 1, [], 'exp2');
                testCase.verifyEqual(K(:, :, p), Kref, 'AbsTol', 1e-12);
            end
        end

        % ------------------------------------------------------- batch_fsolve
        function batchFsolveMatchesBackslash(testCase)
            k = 5; N = 4; q = 2;
            A = BatchOpsTest.spdStack(k, N, 15);
            L = vdgp_batch_chol(A);
            rng(16);
            B = randn(k, q, N);
            Xs = vdgp_batch_fsolve(L, B);
            for p = 1:N
                testCase.verifyEqual(Xs(:, :, p), L(:, :, p) \ B(:, :, p), ...
                    'AbsTol', 1e-10, sprintf('page %d', p));
            end
        end

        % ------------------------------------------------ batch_last_row_inv
        function batchLastRowInvMatchesInverse(testCase)
            k = 5; N = 4;
            A = BatchOpsTest.spdStack(k, N, 17);
            L = vdgp_batch_chol(A);
            a = vdgp_batch_last_row_inv(L);
            ek = zeros(k, 1); ek(k) = 1;
            for p = 1:N
                aref = (L(:, :, p).') \ ek;    % L' a = e_k
                testCase.verifyEqual(a(:, 1, p), aref, 'AbsTol', 1e-10, ...
                    sprintf('page %d', p));
            end
        end
    end
end
