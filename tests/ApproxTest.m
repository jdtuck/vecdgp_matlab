classdef ApproxTest < matlab.unittest.TestCase
%APPROXTEST  Orderings and Vecchia conditioning sets.
%
%   Covers vdgp_order, vdgp_create_approx (struct contract) and the exactness
%   of the ordered nearest-neighbour sets built by vdgp_ordered_nn.

    methods (TestClassSetup)
        function addPackageToPath(testCase)
            import matlab.unittest.fixtures.PathFixture
            here = fileparts(mfilename('fullpath'));
            testCase.applyFixture(PathFixture(fullfile(here, '..')));
        end
    end

    methods (Test)

        % --------------------------------------------------------- vdgp_order
        function maxminIsPermutation(testCase)
            rng(20);
            Xm = rand(300, 3);
            om = vdgp_order(Xm, 'maxmin');
            testCase.verifyEqual(sort(om), 1:300);
        end

        function randomIsPermutation(testCase)
            rng(21);
            om = vdgp_order(rand(50, 2), 'random');
            testCase.verifyEqual(sort(om), 1:50);
        end

        function noneIsIdentity(testCase)
            om = vdgp_order(rand(10, 2), 'none');
            testCase.verifyEqual(om, 1:10);
        end

        function suppliedPermIsValidated(testCase)
            p = [3 1 2 5 4];
            testCase.verifyEqual(vdgp_order(rand(5, 1), p), p);
            testCase.verifyError(@() vdgp_order(rand(5,1), [1 1 2 3 4]), ...
                                 'vdgp_order:perm');
        end

        function unknownOrderTypeErrors(testCase)
            testCase.verifyError(@() vdgp_order(rand(4,1), 'zigzag'), ...
                                 'vdgp_order:type');
        end

        % -------------------------------------------------- create_approx contract
        function vecchiaApproxFields(testCase)
            rng(22);
            X = rand(30, 2);
            A = vdgp_create_approx(X, 8, 'none', true);
            testCase.verifyTrue(A.vecchia);
            testCase.verifyEqual(A.n, 30);
            testCase.verifyEqual(A.d, 2);
            testCase.verifyEqual(A.m, 8);
            testCase.verifyEqual(A.ord, 1:30);            % 'none'
            testCase.verifyEqual(A.rev, 1:30);
            testCase.verifyEqual(A.X_ord, X(A.ord, :), 'AbsTol', 0);
            testCase.verifyEqual(size(A.NNarray), [30 9]);
        end

        function revIsInversePermutation(testCase)
            rng(23);
            X = rand(40, 2);
            A = vdgp_create_approx(X, 10, 'random', true);
            testCase.verifyEqual(A.ord(A.rev), 1:40);
            testCase.verifyEqual(A.X_ord(A.rev, :), X, 'AbsTol', 1e-14);
        end

        function mIsClampedToNMinus1(testCase)
            rng(24);
            A = vdgp_create_approx(rand(6, 2), 100, 'none', true);
            testCase.verifyEqual(A.m, 5);
        end

        function exactApproxHasNoNNarray(testCase)
            rng(25);
            A = vdgp_create_approx(rand(12, 2), [], 'none', false);
            testCase.verifyFalse(A.vecchia);
            testCase.verifyEqual(A.m, 11);
            testCase.verifyEmpty(A.NNarray);
        end

        % --------------------------------------------- ordered-NN exactness
        function orderedNNSetsAreExact(testCase)
            rng(26);
            Ac = vdgp_create_approx(rand(200, 2), 15, 'random', true);
            NN = Ac.NNarray;
            for i = 2:200
                c = NN(i, 2:end); c = c(~isnan(c));
                testCase.verifyEqual(numel(c), min(15, i-1), ...
                    sprintf('row %d cardinality', i));
                testCase.verifyTrue(all(c < i), sprintf('row %d predecessors', i));
                testCase.verifyEqual(numel(unique(c)), numel(c), ...
                    sprintf('row %d distinct', i));
                dall = sum((Ac.X_ord(1:i-1, :) - Ac.X_ord(i, :)).^2, 2);
                [~, oidx] = sort(dall);
                testCase.verifyEqual(sort(c(:)), sort(oidx(1:numel(c))), ...
                    sprintf('row %d nearest set', i));
            end
        end

        function firstColumnIsSelfIndex(testCase)
            rng(27);
            A = vdgp_create_approx(rand(50, 2), 10, 'random', true);
            testCase.verifyEqual(A.NNarray(:, 1), (1:50).');
        end
    end
end
