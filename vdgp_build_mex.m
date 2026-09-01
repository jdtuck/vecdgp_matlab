function vdgp_build_mex(force)
%VDGP_BUILD_MEX  Compile the optional MEX acceleration of the U factor.
%
%   vdgp_build_mex        compile if not already present
%   vdgp_build_mex(true)  force a rebuild
%
%   Builds mex/vdgp_U_entries_mex.c into the package root.  The pure-MATLAB
%   path is always available; the MEX is only ever a speed-up, and
%   VDGP_U_ENTRIES picks it up automatically once it exists.
%
%   OpenMP is enabled when the compiler supports it, which parallelises the
%   loop over observations (each row of U is independent) -- the same
%   parallelisation deepgp uses in C++.

if nargin < 1 || isempty(force), force = false; end

here = fileparts(mfilename('fullpath'));
src  = fullfile(here, 'mex', 'vdgp_U_entries_mex.c');
if exist(src, 'file') ~= 2
    error('vdgp_build_mex:src', 'Cannot find %s', src);
end

if ~force && exist('vdgp_U_entries_mex', 'file') == 3
    fprintf('vdgp_U_entries_mex is already built. Use vdgp_build_mex(true) to rebuild.\n');
    return
end

old = cd(here);
cleanupObj = onCleanup(@() cd(old)); %#ok<NASGU>  restores cwd on any exit

isOctave = exist('OCTAVE_VERSION', 'builtin') == 5;

if isOctave
    try
        mex('-O', '-fopenmp', src);
    catch
        warning('vdgp_build_mex:omp', 'OpenMP build failed; building serial.');
        mex('-O', src);
    end
else
    try
        if ispc
            mex('-O', 'COMPFLAGS=$COMPFLAGS /openmp', src);
        else
            mex('-O', 'CFLAGS=$CFLAGS -fopenmp -O3', 'LDFLAGS=$LDFLAGS -fopenmp', src);
        end
    catch err
        warning('vdgp_build_mex:omp', ...
                'OpenMP build failed (%s); building serial.', err.message);
        mex('-O', src);
    end
end

rehash;                                    % let the path see the new binary
clear vdgp_U_entries                       % drop the cached availability flag
if exist('vdgp_U_entries_mex', 'file') == 3
    fprintf('Built vdgp_U_entries_mex. VDGP_U_ENTRIES will use it automatically.\n');
else
    warning('vdgp_build_mex:missing', 'Build reported success but the MEX is not on the path.');
end
end
