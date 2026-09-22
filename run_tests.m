% RUN_TESTS  Discover and run the matlab.unittest suite under tests/.
here = fileparts(mfilename('fullpath'));
suite = testsuite(fullfile(here, 'tests'), 'IncludeSubfolders', true);
runner = testrunner;
results = run(runner, suite)
