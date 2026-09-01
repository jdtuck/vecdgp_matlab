/*
 * vdgp_U_entries_mex.c
 *
 * Non-zero entries of the Vecchia sparse inverse-Cholesky factor U.
 *
 *   [I, J, V, diagU] = vdgp_U_entries_mex(X_ord, NNarray, gv, theta, vcode)
 *
 *   X_ord    n-by-d  ordered design (double)
 *   NNarray  n-by-(m+1) conditioning sets, 1-based, NaN padded (double)
 *   gv       n-by-1  nugget per ordered observation (double)
 *   theta    1-by-d  lengthscales, already expanded to full length (double)
 *   vcode    0.5 | 1.5 | 2.5 for Matern, 999 for squared exponential
 *
 *   I,J,V    nnz-by-1 triplets of the upper-triangular factor U (unit scale)
 *   diagU    n-by-1 diagonal of U
 *
 * For observation i with conditioning set c(i) this computes, in the stable
 * "reverse so the target is last, factorise, take the last row of the inverse
 * factor" form,
 *
 *      U(i,i) = 1/sqrt(d_i),   U(c(i),i) = -b_i/sqrt(d_i)
 *
 * Each row's work stays inside one (m+1)^2 block that fits in L1/L2 cache,
 * which is the whole point of doing this in C rather than as batched array
 * operations: the flop count is identical, the memory traffic is not.
 *
 * Rows are independent, so the loop is parallelised with OpenMP exactly as
 * the deepgp R package does.
 *
 * Build:  vdgp_build_mex     (from the package root)
 */

#include "mex.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define SQRT3 1.7320508075688772
#define SQRT5 2.2360679774997896

/* correlation as a function of the scaled squared distance */
static double corr_from_d2(double d2, double vcode)
{
    double r;
    if (d2 < 0.0) d2 = 0.0;
    if (vcode > 900.0) {                 /* exp2: theta scales d^2   */
        return exp(-d2);
    }
    r = sqrt(d2);                        /* matern: theta is a lengthscale */
    if (vcode < 1.0)  return exp(-r);
    if (vcode < 2.0)  return (1.0 + SQRT3 * r) * exp(-SQRT3 * r);
    return (1.0 + SQRT5 * r + (5.0 / 3.0) * r * r) * exp(-SQRT5 * r);
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    const double *X, *NN, *gv, *theta;
    double vcode, pw;
    mwSize n, d, kmax, i, j;
    mwSize nnz = 0;
    mwSize *krow, *offs;
    double *I, *J, *V, *diagU;

    int a;

    if (nrhs != 5)
        mexErrMsgIdAndTxt("vdgp:nrhs", "Five inputs required.");
    if (nlhs != 4)
        mexErrMsgIdAndTxt("vdgp:nlhs", "Four outputs required.");
    for (a = 0; a < 5; a++) {
        if (!mxIsDouble(prhs[a]) || mxIsComplex(prhs[a]) || mxIsSparse(prhs[a]))
            mexErrMsgIdAndTxt("vdgp:type",
                              "All inputs must be real, full, double arrays.");
    }

    X     = mxGetPr(prhs[0]);
    NN    = mxGetPr(prhs[1]);
    gv    = mxGetPr(prhs[2]);
    theta = mxGetPr(prhs[3]);
    vcode = mxGetScalar(prhs[4]);

    n    = mxGetM(prhs[0]);
    d    = mxGetN(prhs[0]);
    kmax = mxGetN(prhs[1]);

    if (mxGetM(prhs[1]) != n)
        mexErrMsgIdAndTxt("vdgp:dims", "NNarray must have n rows.");
    if (mxGetNumberOfElements(prhs[2]) != n)
        mexErrMsgIdAndTxt("vdgp:dims", "gv must have n elements.");
    if (mxGetNumberOfElements(prhs[3]) != d)
        mexErrMsgIdAndTxt("vdgp:dims", "theta must have d elements.");

    pw = (vcode > 900.0) ? 1.0 : 2.0;    /* theta^pw divides the squared distance */

    /* --- row sizes and output offsets ---------------------------------- */
    krow = (mwSize *) mxMalloc(n * sizeof(mwSize));
    offs = (mwSize *) mxMalloc((n + 1) * sizeof(mwSize));
    for (i = 0; i < n; i++) {
        mwSize k = 0;
        for (j = 0; j < kmax; j++) {
            double t = NN[i + j * n];
            if (mxIsNaN(t)) break;
            k++;
        }
        krow[i] = k;
        offs[i] = nnz;
        nnz += k;
    }
    offs[n] = nnz;

    plhs[0] = mxCreateDoubleMatrix(nnz, 1, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(nnz, 1, mxREAL);
    plhs[2] = mxCreateDoubleMatrix(nnz, 1, mxREAL);
    plhs[3] = mxCreateDoubleMatrix(n, 1, mxREAL);
    I     = mxGetPr(plhs[0]);
    J     = mxGetPr(plhs[1]);
    V     = mxGetPr(plhs[2]);
    diagU = mxGetPr(plhs[3]);

    /* --- one small dense problem per row ------------------------------- */
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        double *K   = (double *) malloc(kmax * kmax * sizeof(double));
        double *a   = (double *) malloc(kmax * sizeof(double));
        mwSize *rid = (mwSize *) malloc(kmax * sizeof(mwSize));
        long long ii;

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (ii = 0; ii < (long long) n; ii++) {
            mwSize k = krow[ii], p, q, l, base = offs[ii];
            double jitter = 0.0;
            int attempt, ok;

            /* reverse the conditioning set so the target sits last */
            for (p = 0; p < k; p++)
                rid[p] = (mwSize) NN[(mwSize) ii + (k - 1 - p) * n] - 1;

            /* covariance of the k points, column major */
            for (q = 0; q < k; q++) {
                for (p = q; p < k; p++) {
                    double d2 = 0.0, c;
                    if (p != q) {
                        for (l = 0; l < d; l++) {
                            double dl = X[rid[p] + l * n] - X[rid[q] + l * n];
                            double sc = (pw == 1.0) ? theta[l] : theta[l] * theta[l];
                            d2 += dl * dl / sc;
                        }
                        c = corr_from_d2(d2, vcode);
                    } else {
                        c = 1.0 + gv[rid[p]];
                    }
                    K[p + q * k] = c;
                    K[q + p * k] = c;
                }
            }

            /* Cholesky (lower), retried with jitter if it breaks down */
            for (attempt = 0; attempt < 6; attempt++) {
                ok = 1;
                if (attempt > 0) {
                    jitter = (jitter == 0.0) ? 1e-10 : jitter * 10.0;
                    for (p = 0; p < k; p++) K[p + p * k] += jitter;
                }
                for (q = 0; q < k; q++) {
                    double s = K[q + q * k];
                    for (l = 0; l < q; l++) s -= K[q + l * k] * K[q + l * k];
                    if (!(s > 0.0)) { ok = 0; break; }
                    K[q + q * k] = sqrt(s);
                    for (p = q + 1; p < k; p++) {
                        double t = K[p + q * k];
                        for (l = 0; l < q; l++) t -= K[p + l * k] * K[q + l * k];
                        K[p + q * k] = t / K[q + q * k];
                    }
                }
                if (ok) break;
                /* rebuild the block before retrying */
                for (q = 0; q < k; q++) {
                    for (p = q; p < k; p++) {
                        double d2 = 0.0, c;
                        if (p != q) {
                            for (l = 0; l < d; l++) {
                                double dl = X[rid[p] + l * n] - X[rid[q] + l * n];
                                double sc = (pw == 1.0) ? theta[l] : theta[l] * theta[l];
                                d2 += dl * dl / sc;
                            }
                            c = corr_from_d2(d2, vcode);
                        } else {
                            c = 1.0 + gv[rid[p]] + jitter;
                        }
                        K[p + q * k] = c;
                        K[q + p * k] = c;
                    }
                }
            }

            /* a = L' \ e_k  : the non-zero entries of column ii of U */
            a[k - 1] = 1.0 / K[(k - 1) + (k - 1) * k];
            for (p = k - 1; p-- > 0; ) {
                double s = 0.0;
                for (q = p + 1; q < k; q++) s += K[q + p * k] * a[q];
                a[p] = -s / K[p + p * k];
            }

            for (p = 0; p < k; p++) {
                I[base + p] = (double) (rid[p] + 1);
                J[base + p] = (double) (ii + 1);
                V[base + p] = a[p];
            }
            diagU[ii] = a[k - 1];
        }

        free(K); free(a); free(rid);
    }

    mxFree(krow);
    mxFree(offs);
}
