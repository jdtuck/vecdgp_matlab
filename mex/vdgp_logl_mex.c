/*
 * vdgp_logl_mex.c
 *
 * Vecchia Gaussian log-likelihood ingredients, without ever materialising U.
 *
 *   [quad, logdet] = vdgp_logl_mex(X_ord, NNarray, gv, theta, vcode, y_ord)
 *
 *   quad   = || U' y ||^2
 *   logdet = log |Q| = 2 * sum(log(diag(U)))
 *
 * This is the hot path of the sampler: a Gibbs sweep evaluates it ~30 times
 * and needs nothing else from U.  Computing it here avoids allocating and
 * returning the n*(m+1) triplets (about 4 MB at n = 6000, m = 25) on every
 * call, and avoids the ACCUMARRAY that would otherwise scatter them.
 *
 * Column ii of U has its non-zeros in rows c(ii) and ii, so
 *
 *      (U' y)_ii = sum_p a_p * y_ord[rid_p]
 *
 * is a pure per-row reduction: no races, no atomics, one OpenMP loop.
 *
 * Build:  vdgp_build_mex
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
    const double *X, *NN, *gv, *theta, *yord;
    double *Uty;
    double vcode, pw;
    mwSize n, d, kmax, i, j;
    mwSize nnz = 0;
    mwSize *krow, *offs;
    double *diagU;

    int a;

    if (nrhs != 6)
        mexErrMsgIdAndTxt("vdgp:nrhs", "Six inputs required.");
    if (nlhs != 2)
        mexErrMsgIdAndTxt("vdgp:nlhs", "Two outputs required.");
    for (a = 0; a < 6; a++) {
        if (!mxIsDouble(prhs[a]) || mxIsComplex(prhs[a]) || mxIsSparse(prhs[a]))
            mexErrMsgIdAndTxt("vdgp:type",
                              "All inputs must be real, full, double arrays.");
    }

    X     = mxGetPr(prhs[0]);
    NN    = mxGetPr(prhs[1]);
    gv    = mxGetPr(prhs[2]);
    theta = mxGetPr(prhs[3]);
    vcode = mxGetScalar(prhs[4]);
    yord  = mxGetPr(prhs[5]);

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

    plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
    Uty   = (double *) mxMalloc(n * sizeof(double));
    diagU = (double *) mxMalloc(n * sizeof(double));

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
            mwSize k = krow[ii], p, q, l;
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

            {
                double s = 0.0;
                for (p = 0; p < k; p++) s += a[p] * yord[rid[p]];
                Uty[ii]   = s;
                diagU[ii] = a[k - 1];
            }
        }

        free(K); free(a); free(rid);
    }

    {
        double quad = 0.0, logdet = 0.0;
        mwSize t;
        for (t = 0; t < n; t++) {
            quad   += Uty[t] * Uty[t];
            logdet += log(diagU[t]);
        }
        *mxGetPr(plhs[0]) = quad;
        *mxGetPr(plhs[1]) = 2.0 * logdet;
    }

    mxFree(Uty);
    mxFree(diagU);
    mxFree(krow);
    mxFree(offs);
}
