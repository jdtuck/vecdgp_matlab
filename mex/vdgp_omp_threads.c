/*
 * vdgp_omp_threads.c
 *
 *   t = vdgp_omp_threads()
 *
 * Returns the number of OpenMP threads the compiled kernels will use, or 1 if
 * they were built without OpenMP.  VDGP_BUILD_MEX builds this first and runs
 * it as a probe: if it compiles and returns a sensible number, the same flags
 * are used for the real kernels.  VDGP_PROFILE reports it.
 *
 * Build:  vdgp_build_mex
 */

#include "mex.h"

#ifdef _OPENMP
#include <omp.h>
#endif

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    double t = 1.0;
    (void) nrhs; (void) prhs; (void) nlhs;
#ifdef _OPENMP
    t = (double) omp_get_max_threads();
#endif
    plhs[0] = mxCreateDoubleScalar(t);
}
