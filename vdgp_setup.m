function vdgp_setup()
%VDGP_SETUP  Add the Vecchia deep GP package to the MATLAB path.
%
%   Run once per session from the package root:
%
%       vdgp_setup
%
%   Adds the package folder plus demos/ and tests/.

here = fileparts(mfilename('fullpath'));
addpath(here);
addpath(fullfile(here, 'demos'));
addpath(fullfile(here, 'tests'));
fprintf('vecchia_dgp added to the path (%s)\n', here);
fprintf('Try:  test_vdgp   |   demo_1d   |   demo_2d_scaling\n');
end
