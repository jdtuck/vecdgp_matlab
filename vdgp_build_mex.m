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
names = {'vdgp_U_entries_mex', 'vdgp_logl_mex', 'vdgp_krig_mex'};
for i = 1:numel(names)
    if exist(fullfile(here, 'mex', [names{i} '.c']), 'file') ~= 2
        error('vdgp_build_mex:src', 'Cannot find mex/%s.c', names{i});
    end
end

if ~force && all(cellfun(@(nm) exist(nm, 'file') == 3, names))
    fprintf('MEX files are already built. Use vdgp_build_mex(true) to rebuild.\n');
    return
end

old = cd(here);
cleanupObj = onCleanup(@() cd(old)); %#ok<NASGU>  restores cwd on any exit

isOctave = exist('OCTAVE_VERSION', 'builtin') == 5;

for i = 1:numel(names)
    src = fullfile(here, 'mex', [names{i} '.c']);
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
end

rehash;                                    % let the path see the new binaries
clear vdgp_U_entries vdgp_logl vdgp_krig   % drop the cached availability flags
built = cellfun(@(nm) exist(nm, 'file') == 3, names);
if all(built)
    fprintf(['Built %s.\nThey are picked up automatically by VDGP_LOGL, ' ...
             'VDGP_U_ENTRIES and VDGP_KRIG.\n'], strjoin(names, ', '));
else
    warning('vdgp_build_mex:missing', 'Not all MEX files landed on the path: %s', ...
            strjoin(names(~built), ', '));
end
end
