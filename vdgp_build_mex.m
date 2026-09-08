function ok = vdgp_build_mex(force, mode)
%VDGP_BUILD_MEX  Compile the optional C kernels, with OpenMP where possible.
%
%   vdgp_build_mex               compile if not already present
%   vdgp_build_mex(true)         force a rebuild
%   vdgp_build_mex(true, MODE)   force a particular toolchain strategy
%   OK = vdgp_build_mex(...)     true if every kernel built
%
%   MODE is one of
%     'auto'   (default) probe the available strategies in order
%     'mac'    Apple clang + a separate OpenMP runtime (see below)
%     'gcc'    plain -fopenmp (gcc, or clang built with OpenMP)
%     'msvc'   Microsoft compiler, /openmp
%     'serial' skip OpenMP entirely
%
%   Builds, from mex/:
%     vdgp_logl_mex       the likelihood -- the sampler's hot path
%     vdgp_U_entries_mex  the sparse factor U
%     vdgp_krig_mex       pointwise prediction
%     vdgp_omp_threads    reports the thread count (also used as the probe)
%
%   The pure-MATLAB paths always remain available; the MEX files are only ever
%   a speed-up, and are picked up automatically once they exist.
%
%   APPLE SILICON / macOS.  The clang that ships with Xcode rejects -fopenmp
%   because Apple does not bundle the OpenMP runtime, which is why a plain
%   build reports "unsupported option '-fopenmp'".  The fix is to install the
%   runtime once,
%
%       brew install libomp
%
%   after which this function finds it automatically.  It prefers to compile
%   against libomp's headers but LINK against the libiomp5 that ships inside
%   MATLAB, so only one OpenMP runtime is ever loaded into the process -- that
%   avoids the "OMP: Error #15 ... libiomp5.dylib already initialized" abort
%   that linking Homebrew's libomp alongside MATLAB's can cause.
%
%   Without OpenMP the kernels still build and run, single-threaded.  That is
%   not a disaster: the serial MEX is already an order of magnitude faster
%   than the pure-MATLAB path, and OpenMP adds the core count on top.

if nargin < 1 || isempty(force), force = false; end
if nargin < 2 || isempty(mode),  mode  = 'auto'; end

here  = fileparts(mfilename('fullpath'));
probe = 'vdgp_omp_threads';
names = {'vdgp_logl_mex', 'vdgp_U_entries_mex', 'vdgp_krig_mex'};
all_names = [{probe}, names];

for i = 1:numel(all_names)
    if exist(fullfile(here, 'mex', [all_names{i} '.c']), 'file') ~= 2
        error('vdgp_build_mex:src', 'Cannot find mex/%s.c', all_names{i});
    end
end

if ~force && all(cellfun(@(nm) exist(nm, 'file') == 3, all_names))
    fprintf('MEX files are already built. Use vdgp_build_mex(true) to rebuild.\n');
    ok = true;
    return
end

old = cd(here);
cleanupObj = onCleanup(@() cd(old)); %#ok<NASGU>  restores cwd on any exit

% If two OpenMP runtimes do end up loaded, this turns a hard abort into a
% warning.  Belt and braces: the link strategy above is meant to avoid it.
setenv('KMP_DUPLICATE_LIB_OK', 'TRUE');

isOctave = exist('OCTAVE_VERSION', 'builtin') == 5;
cands = local_candidates(mode, isOctave);

% ---- probe -------------------------------------------------------------
% Build the tiny thread reporter with each candidate and keep the first one
% that actually yields more than one thread.  A compiler that merely ignores
% an unknown flag would otherwise be accepted as a success.
chosen = [];
fallback = [];
for i = 1:numel(cands)
    c = cands{i};
    if ~local_try_build(fullfile(here, 'mex', [probe '.c']), c)
        continue
    end
    rehash;
    try
        nthreads = feval(probe);
    catch
        continue                                % built, but will not run
    end
    c.threads = nthreads;
    if nthreads > 1
        chosen = c;
        break
    elseif isempty(fallback)
        fallback = c;
    end
end
if isempty(chosen), chosen = fallback; end

if isempty(chosen)
    % Not fatal: every kernel has a pure-MATLAB fallback, so a machine with no
    % working compiler still runs the package (and still passes the tests).
    warning('vdgp_build_mex:probe', ...
            ['Could not build even a trivial MEX file -- continuing without ' ...
             'the C kernels. Run "mex -setup C" to configure a compiler.']);
    ok = false;
    return
end

fprintf('Toolchain: %s', chosen.name);
if chosen.threads > 1
    fprintf('  (OpenMP, %d threads)\n', chosen.threads);
else
    fprintf('  (single-threaded)\n');
end

% ---- build the real kernels with the winning flags ----------------------
built = false(1, numel(names));
for i = 1:numel(names)
    built(i) = local_try_build(fullfile(here, 'mex', [names{i} '.c']), chosen);
end

rehash;
clear vdgp_U_entries vdgp_logl vdgp_krig   % drop the cached availability flags

ok = all(built);
if ok
    fprintf('Built %s.\n', strjoin(names, ', '));
    fprintf('They are picked up automatically by VDGP_LOGL, VDGP_U_ENTRIES and VDGP_KRIG.\n');
else
    warning('vdgp_build_mex:missing', 'These did not build: %s', ...
            strjoin(names(~built), ', '));
end

if chosen.threads <= 1 && ~strcmpi(mode, 'serial')
    fprintf('\nOpenMP is off, so the kernels run on one core.\n');
    if ismac
        fprintf('On macOS:  brew install libomp   then rerun vdgp_build_mex(true).\n');
    end
    fprintf('The serial MEX is still ~10x faster than the pure-MATLAB path.\n');
end
end

% -------------------------------------------------------------------------
function tf = local_try_build(src, c)
tf = false;
try
    args = [{'-O'}, c.flags, {src}];
    evalc('mex(args{:})');           % keep a failed attempt quiet
    tf = true;
catch
    tf = false;
end
end

% -------------------------------------------------------------------------
function cands = local_candidates(mode, isOctave)
%LOCAL_CANDIDATES  Ordered list of compiler-flag strategies to try.
cands = {};

if isOctave
    % Octave's mkoctfile takes bare compiler flags, not MATLAB's NAME=VALUE
    % form, and is often already configured with OpenMP.
    if ~strcmpi(mode, 'serial')
        cands{end+1} = struct('name', 'mkoctfile -fopenmp', 'flags', {{'-fopenmp'}});
    end
    cands{end+1} = struct('name', 'mkoctfile defaults', 'flags', {{}});
    return
end

wantMac = ismac    && any(strcmpi(mode, {'auto', 'mac'}));
wantGcc = any(strcmpi(mode, {'auto', 'gcc'})) && ~ispc;
wantPc  = ispc     && any(strcmpi(mode, {'auto', 'msvc'}));

if strcmpi(mode, 'mac'),  wantGcc = false; end
if strcmpi(mode, 'gcc'),  wantMac = false; end

if wantPc
    cands{end+1} = struct('name', 'MSVC /openmp', ...
                          'flags', {{'COMPFLAGS=$COMPFLAGS /openmp'}});
end

if wantMac
    inc = local_find_omp_include();
    if ~isempty(inc)
        cf = ['CFLAGS=$CFLAGS -Xpreprocessor -fopenmp -O3 -I' inc];

        % 1. link against the libiomp5 MATLAB already loads -> one runtime
        mlib = fullfile(matlabroot, 'sys', 'os', computer('arch'));
        if exist(mlib, 'dir') == 7
            cands{end+1} = struct( ...
                'name', 'Apple clang + libomp headers, linked to MATLAB''s libiomp5', ...
                'flags', {{cf, ['LDFLAGS=$LDFLAGS -L' mlib ' -liomp5 -Wl,-rpath,' mlib]}});
        end

        % 2. otherwise link Homebrew/MacPorts libomp itself
        lib = local_omp_libdir(inc);
        if ~isempty(lib)
            cands{end+1} = struct( ...
                'name', 'Apple clang + libomp', ...
                'flags', {{cf, ['LDFLAGS=$LDFLAGS -L' lib ' -lomp -Wl,-rpath,' lib]}});
        end
    end
end

if wantGcc
    cands{end+1} = struct('name', 'gcc/clang -fopenmp', ...
                          'flags', {{'CFLAGS=$CFLAGS -fopenmp -O3', ...
                                     'LDFLAGS=$LDFLAGS -fopenmp'}});
end

% Always keep a plain fallback last.  Note some toolchains enable OpenMP in
% their own default flags, so this can still come back multi-threaded.
cands{end+1} = struct('name', 'compiler defaults (no OpenMP flag)', 'flags', {{}});
end

% -------------------------------------------------------------------------
function inc = local_find_omp_include()
%LOCAL_FIND_OMP_INCLUDE  Locate omp.h from Homebrew, MacPorts or Xcode.
inc = '';
prefixes = {};

% MATLAB launched from the Dock has a minimal PATH, so brew may not resolve;
% the fixed locations below cover the standard installs either way.
[st, out] = system('brew --prefix libomp 2>/dev/null');
if st == 0
    out = strtrim(out);
    if ~isempty(out), prefixes{end+1} = out; end
end
prefixes = [prefixes, {'/opt/homebrew/opt/libomp', '/usr/local/opt/libomp', ...
                       '/opt/homebrew', '/usr/local', '/opt/local'}];

for i = 1:numel(prefixes)
    p = prefixes{i};
    if exist(fullfile(p, 'include', 'omp.h'), 'file') == 2
        inc = fullfile(p, 'include');  return
    end
    if exist(fullfile(p, 'include', 'libomp', 'omp.h'), 'file') == 2
        inc = fullfile(p, 'include', 'libomp');  return
    end
end
end

% -------------------------------------------------------------------------
function lib = local_omp_libdir(inc)
lib = '';
p = fileparts(inc);
if strcmp(local_basename(inc), 'libomp'), p = fileparts(p); end
cand = fullfile(p, 'lib');
if exist(fullfile(cand, 'libomp.dylib'), 'file') == 2 || ...
   exist(fullfile(cand, 'libomp.so'), 'file') == 2
    lib = cand;
end
end

% -------------------------------------------------------------------------
function b = local_basename(p)
[~, b] = fileparts(p);
end
