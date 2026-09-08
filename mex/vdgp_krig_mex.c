/*
 * vdgp_krig_mex.c
 *
 * Pointwise ("lite") Vecchia prediction: each test location conditions on its
 * m nearest training locations.
 *
 *   [mu, qf] = vdgp_krig_mex(X, Xnew, NN, y, theta, g, vcode, want_qf)
 *
 *   X        n-by-d     training inputs (double)
 *   Xnew     nnew-by-d  test inputs (double)
 *   NN       nnew-by-k  1-based indices of each test point's neighbours
 *   y        n-by-1     training responses
 *   theta    1-by-d     lengthscales, already expanded
 *   g        scalar     nugget on the training diagonal
 *   vcode    0.5 | 1.5 | 2.5 for Matern, 999 for squared exponential
 *   want_qf  nonzero to also return qf
 *
 *   mu       nnew-by-1  kriging mean
 *   qf       nnew-by-1  k(x)' K^{-1} k(x); the caller forms the variance as
 *                       tau2 * (1 + g_pred - qf)
 *
 * Same shape as vdgp_U_entries_mex: one small dense problem per row, kept in
 * cache, OpenMP over rows.
 *
 * Build:  vdgp_build_mex
 */

#include "mex.h"
#include <math.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define SQRT3 1.7320508075688772
#define SQRT5 2.2360679774997896

static double corr_from_d2(double d2, double vcode)
{
    double r;
    if (d2 < 0.0) d2 = 0.0;
    if (vcode > 900.0) return exp(-d2);
    r = sqrt(d2);
    if (vcode < 1.0)  return exp(-r);
    if (vcode < 2.0)  return (1.0 + SQRT3 * r) * exp(-SQRT3 * r);
    return (1.0 + SQRT5 * r + (5.0 / 3.0) * r * r) * exp(-SQRT5 * r);
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    const double *X, *Xn, *NN, *y, *theta;
    double g, vcode, pw;
    int want_qf, a;
    mwSize n, d, nnew, k;
    double *mu, *qf;

    if (nrhs != 8)
        mexErrMsgIdAndTxt("vdgp:nrhs", "Eight inputs required.");
    for (a = 0; a < 8; a++) {
        if (!mxIsDouble(prhs[a]) || mxIsComplex(prhs[a]) || mxIsSparse(prhs[a]))
            mexErrMsgIdAndTxt("vdgp:type",
                              "All inputs must be real, full, double arrays.");
    }

    X     = mxGetPr(prhs[0]);
    Xn    = mxGetPr(prhs[1]);
    NN    = mxGetPr(prhs[2]);
    y     = mxGetPr(prhs[3]);
    theta = mxGetPr(prhs[4]);
    g     = mxGetScalar(prhs[5]);
    vcode = mxGetScalar(prhs[6]);
    want_qf = (int) mxGetScalar(prhs[7]);

    n    = mxGetM(prhs[0]);
    d    = mxGetN(prhs[0]);
    nnew = mxGetM(prhs[1]);
    k    = mxGetN(prhs[2]);

    if (mxGetN(prhs[1]) != d)
        mexErrMsgIdAndTxt("vdgp:dims", "X and Xnew must have the same width.");
    if (mxGetM(prhs[2]) != nnew)
        mexErrMsgIdAndTxt("vdgp:dims", "NN must have one row per test point.");
    if (mxGetNumberOfElements(prhs[3]) != n)
        mexErrMsgIdAndTxt("vdgp:dims", "y must have n elements.");
    if (mxGetNumberOfElements(prhs[4]) != d)
        mexErrMsgIdAndTxt("vdgp:dims", "theta must have d elements.");

    pw = (vcode > 900.0) ? 1.0 : 2.0;

    plhs[0] = mxCreateDoubleMatrix(nnew, 1, mxREAL);
    mu = mxGetPr(plhs[0]);
    if (nlhs > 1) {
        plhs[1] = mxCreateDoubleMatrix(want_qf ? nnew : 0, 1, mxREAL);
        qf = want_qf ? mxGetPr(plhs[1]) : NULL;   /* never write into an empty array */
    } else {
        qf = NULL;
    }

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        double *K  = (double *) malloc(k * k * sizeof(double));
        double *kx = (double *) malloc(k * sizeof(double));
        double *yy = (double *) malloc(k * sizeof(double));
        mwSize *id = (mwSize *) malloc(k * sizeof(mwSize));
        long long ii;

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (ii = 0; ii < (long long) nnew; ii++) {
            mwSize p, q, l;
            double m1 = 0.0, q1 = 0.0;

            for (p = 0; p < k; p++) {
                id[p] = (mwSize) NN[(mwSize) ii + p * nnew] - 1;
                yy[p] = y[id[p]];
            }

            /* neighbour covariance, plus the cross covariance to the target */
            for (q = 0; q < k; q++) {
                for (p = q; p < k; p++) {
                    double d2 = 0.0, c;
                    if (p != q) {
                        for (l = 0; l < d; l++) {
                            double dl = X[id[p] + l * n] - X[id[q] + l * n];
                            double sc = (pw == 1.0) ? theta[l] : theta[l] * theta[l];
                            d2 += dl * dl / sc;
                        }
                        c = corr_from_d2(d2, vcode);
                    } else {
                        c = 1.0 + g;
                    }
                    K[p + q * k] = c;
                    K[q + p * k] = c;
                }
                {
                    double d2 = 0.0;
                    for (l = 0; l < d; l++) {
                        double dl = X[id[q] + l * n] - Xn[(mwSize) ii + l * nnew];
                        double sc = (pw == 1.0) ? theta[l] : theta[l] * theta[l];
                        d2 += dl * dl / sc;
                    }
                    kx[q] = corr_from_d2(d2, vcode);
                }
            }

            /* Cholesky (lower), in place */
            for (q = 0; q < k; q++) {
                double s = K[q + q * k];
                for (l = 0; l < q; l++) s -= K[q + l * k] * K[q + l * k];
                if (!(s > 0.0)) s = 1e-12;
                K[q + q * k] = sqrt(s);
                for (p = q + 1; p < k; p++) {
                    double t = K[p + q * k];
                    for (l = 0; l < q; l++) t -= K[p + l * k] * K[q + l * k];
                    K[p + q * k] = t / K[q + q * k];
                }
            }

            /* forward-substitute both right-hand sides in one pass */
            for (p = 0; p < k; p++) {
                double sy = yy[p], sk = kx[p];
                for (l = 0; l < p; l++) {
                    sy -= K[p + l * k] * yy[l];
                    sk -= K[p + l * k] * kx[l];
                }
                yy[p] = sy / K[p + p * k];
                kx[p] = sk / K[p + p * k];
                m1 += yy[p] * kx[p];
                q1 += kx[p] * kx[p];
            }

            mu[ii] = m1;
            if (qf) qf[ii] = q1;
        }

        free(K); free(kx); free(yy); free(id);
    }
}
