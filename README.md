# vecchia_dgp — Vecchia-approximated deep Gaussian processes in MATLAB

A from-scratch MATLAB implementation of

> A. Sauer, A. Cooper and R. B. Gramacy (2023).
> *Vecchia-approximated Deep Gaussian Processes for Computer Experiments.*
> Journal of Computational and Graphical Statistics. arXiv:2204.02904

with the same model, priors, samplers and prediction schemes as the reference
R package [**deepgp**](https://cran.r-project.org/package=deepgp) run with
`vecchia = TRUE`.

One-, two- and three-layer architectures are supported, with or without the
Vecchia approximation, so the approximation can always be checked against the
exact sampler on problems small enough to afford it.

---

## Quick start

```matlab
vdgp_setup                      % add the package to the path

f  = @(x) (x <= 0.58) .* (sin(pi*x*6) + cos(pi*x*12)) + (x > 0.58) .* (5*x - 4.9);
x  = linspace(0, 1, 25)';   y  = f(x);
xx = linspace(0, 1, 200)';  yy = f(xx);

fit  = fit_two_layer(x, y, 'nmcmc', 2000, 'vecchia', true, 'm', 10, 'true_g', 1e-6);
fit  = dgp_trim(fit, 1000, 2);          % burn-in 1000, thin by 2
pred = dgp_predict(fit, xx);            % pointwise posterior predictive

plot(xx, pred.mean); hold on
plot(xx, pred.mean + 1.96*pred.sd, ':'), plot(xx, pred.mean - 1.96*pred.sd, ':')
```

which is the MATLAB transcription of the deepgp example

```r
fit <- fit_two_layer(x, y, nmcmc = 2000, vecchia = TRUE, m = 10)
fit <- trim(fit, 1000, 2)
fit <- predict(fit, xx, cores = 1)
```

Run `demo_1d` and `demo_2d_scaling` for worked examples, and `test_vdgp` for
the verification suite.

---

## The model

For a two-layer DGP on inputs `X` (n × d) with a latent layer `W` (n × D):

```
Y | W      ~  N_n( 0 , tau2 * ( K_y(W) + g I ) )
W(:,k) | X ~  N_n( 0 , K_k(X) )                       k = 1 … D
```

The latent layer carries unit scale and no nugget for identifiability; `tau2`
is marginalised out under the reference prior `p(tau2) ∝ 1/tau2`. Three layers
insert `Z` between `X` and `W` in the same way.

Kernels use the deepgp parameterisation:

| `cov` | form |
|---|---|
| `'matern'`, `v = 0.5` | `exp(-r)` |
| `'matern'`, `v = 1.5` | `(1 + √3 r) exp(-√3 r)` |
| `'matern'`, `v = 2.5` | `(1 + √5 r + 5r²/3) exp(-√5 r)`, `r = ‖x-x'‖/θ` |
| `'exp2'` | `exp(-‖x-x'‖²/θ)` |

---

## The Vecchia approximation

The likelihood is replaced by the ordered conditional product

```
L(Y) = ∏_i  L( y_i | Y_c(i) ) ,     |c(i)| = min(m, i-1)
```

whose precision `Q = U Uᵀ` has a sparse upper-triangular Cholesky factor `U`
with at most `m` off-diagonal non-zeros per column. Column `i` is

```
b_i    = Σ(c(i),c(i))⁻¹ Σ(c(i), i)
d_i    = Σ(i,i) − Σ(i,c(i)) b_i
U(i,i)    =  1/√d_i
U(c(i),i) = −b_i/√d_i
```

Everything the sampler needs follows from `U` in O(n m) work once it is built:

| quantity | expression | file |
|---|---|---|
| log-likelihood | `½ logdet(Q) − ½‖Uᵀy‖²`, `logdet(Q) = 2 Σ log diag(U)` | `vdgp_logl.m` |
| profiled scale | `tau2 = ‖Uᵀy‖²/n` | `vdgp_logl.m` |
| prior draw | solve `Uᵀ s = z`, `z ~ N(0,I)` | `vdgp_rand_mvn.m` |
| joint prediction | block-partition `U` (observed first) | `vdgp_krig.m` |

Building `U` costs O(n m³). Two structural facts make the MATLAB version fast
without a mex file:

* rows `1 … m+1` condition on **all** their predecessors, so that leading
  block is exact and comes from a single dense Cholesky;
* every other row has exactly `m+1` entries, so all `n − m − 1` of them are
  factorised by one **batched** Cholesky, vectorised across pages
  (`vdgp_batch_chol.m`). This replaces the OpenMP loop deepgp uses in C++.

Column `i` of `U` is obtained in the numerically stable form "reverse the
conditioning set so the target is last, factorise, take the last row of the
inverse factor" — `L' \ e_{m+1}` — see `vdgp_batch_last_row_inv.m`.

### Optional MEX acceleration

The batched-array build is the fallback. If you have a C compiler, run

```matlab
vdgp_build_mex          % compiles mex/vdgp_U_entries_mex.c, OpenMP if available
```

once and `vdgp_U_entries` picks it up automatically — nothing else changes,
and the pure-MATLAB path stays available if the MEX is absent.

The flop count is identical; what changes is memory traffic. The batched
version sweeps a k×k×n array (540 MB at n = 10⁵, m = 25) once per column of
the Cholesky, so it is bandwidth-bound. The MEX keeps each 26×26 block
(5 KB) in L1 while it is factorised and back-solved, then writes only the
m+1 output entries — arithmetic intensity goes from O(1) to O(m) flops/byte.
Rows are independent, so the loop is also OpenMP-parallel, as in deepgp.

Measured on this 2-core machine under Octave, 2-D inputs, m = 25:

| n | U build, batched arrays | MEX, 1 thread | MEX, 2 threads |
|---|---|---|---|
| 1 000 | 0.055 s | 0.0075 s | 0.0045 s |
| 5 000 | 0.355 s | 0.038 s | 0.022 s |
| 20 000 | 2.330 s | 0.152 s | 0.104 s |

End-to-end, seconds per Gibbs sweep of `fit_two_layer` (setup excluded):

| n | batched | MEX | speed-up |
|---|---|---|---|
| 2 000 | 2.53 | 0.197 | 12.8× |
| 5 000 | 10.05 | 0.645 | 15.6× |

The sweep speed-up tracks the kernel speed-up because a sweep is ~30 `U`
builds and essentially nothing else: with the MEX active, `vdgp_logl` spends
96% of its time inside `vdgp_U_entries`, and the sparse assembly, `accumarray`
and triangular solves are a few percent.

Two caveats on those numbers. They are Octave, whose array operations are
slower than MATLAB's, so the MATLAB baseline is faster and the ratio there
will be smaller — expect mid single digits to ~10× single-threaded rather
than 15×. And 2 cores only buys 1.5× from OpenMP here; a real multi-core
machine will do better on the threaded column.

`test_vdgp` passes 13/13 with the MEX active, with `U` matching the
pure-MATLAB result to ~1e-9 relative.

### Ordering and conditioning sets

`vdgp_order.m` provides random ordering (the paper's default) and max-min
ordering. `vdgp_ordered_nn.m` builds the conditioning sets, exactly reproducing
`GpGp::find_ordered_nn`.

Following deepgp, the ordering and neighbour sets of a **latent** layer are
built once from its initial values and then held fixed while the layer moves —
only the coordinates entering the covariance are refreshed
(`vdgp_update_approx.m`, an O(n) operation). Set `'reapprox', k` to rebuild them
in the current warped space every `k` iterations, or call
`dgp_continue(fit, n, true)`; this is the "update the conditioning sets in the
warped space" idea of Section 4 (which the paper found gives only marginal
gains).

---

## Inference

One Gibbs sweep, in the deepgp order:

1. **nugget `g`** — Metropolis-Hastings, uniform sliding-window proposal
   `g* ~ U(l·g/u, u·g/l)` with `l = 1, u = 2`, `Gamma(α_g, β_g)` prior on
   `g − ε`. Skipped when `true_g` is supplied.
2. **outer lengthscale `θ_y`** — same MH kernel. Because the proposal density
   is `∝ 1/θ`, the Hastings ratio contributes `log θ − log θ*` to the
   acceptance ratio (`vdgp_sample_theta.m`).
3. **inner lengthscales `θ_w`** (and `θ_z`) — same MH kernel, evaluated against
   the latent layer's own Gaussian likelihood with unit scale.
4. **latent layers** — elliptical slice sampling (Murray, Adams & MacKay 2010),
   one hidden node at a time: draw `ν` from the Vecchia prior, set the slice
   threshold at `ll + log U(0,1)`, and shrink the angle bracket until the
   proposal `w cos a + ν sin a` clears it. Tuning-free, always accepts.

### Priors and defaults

Assume `x` scaled to `[0,1]^d` and `y` to zero mean / unit variance — done
automatically unless `'scale', false`.

| parameter | prior | default |
|---|---|---|
| `g` | `Gamma(1.5, 3.9)` | estimated, start `1e-3` |
| `θ` (one layer) | `Gamma(1.5, 3.9/1.5)` | start `0.1` |
| `θ_y` | `Gamma(1.5, 3.9/6)` | start `0.1` |
| `θ_w` | `Gamma(1.5, 3.9/4)` | start `0.1` |
| `θ_z` | `Gamma(1.5, 3.9/4)` | start `0.1` |
| proposal | `l = 1`, `u = 2` | |

(rate parameterisation). Override any of them:

```matlab
opts = vdgp_options('nmcmc', 5000, 'm', 30, ...
                    'alpha', struct('g', 2), 'beta', struct('theta_y', 1));
fit  = fit_two_layer(x, y, opts);
```

`vdgp_options` documents every field. These match the deepgp defaults as
published; if you are cross-checking numerically against a specific deepgp
release, confirm its `check_settings()` values and pass them explicitly.

---

## Prediction

`dgp_predict` mixes over the retained MCMC draws. For each draw the latent
layers are propagated forward by their kriging means (as deepgp does) and the
outer layer supplies `mu_t`, `s2_t`; the mixture is summarised by

```
mu = mean_t mu_t ,      s2 = mean_t( s2_t + mu_t² ) − mu²
```

Two Vecchia predictors, both from Section 3.3 of the paper:

* `'lite', true` (default) — each new location conditions on its `m` nearest
  training locations, giving pointwise means and variances in O(n_new m³).
  All the small solves are batched.
* `'lite', false` — the new locations are appended **after** the observed ones
  in the Vecchia ordering; with `U = [U11 U12; 0 U22]`,

  ```
  E[Y2|Y1] = −U22⁻ᵀ U12ᵀ Y1 ,      Cov[Y2|Y1] = tau2 · M Mᵀ ,  M = U22⁻ᵀ
  ```

  so the full predictive covariance costs only sparse triangular solves.

Predictive variances include the nugget by default (as in deepgp); set
`'pred_noise', false` in the options for the noise-free latent-function
predictive.

---

## API map: deepgp (R) → this package

| deepgp | here |
|---|---|
| `fit_one_layer(x, y, ...)` | `fit_one_layer(x, y, ...)` |
| `fit_two_layer(x, y, vecchia = TRUE, m = 10)` | `fit_two_layer(x, y, 'vecchia', true, 'm', 10)` |
| `fit_three_layer(...)` | `fit_three_layer(...)` |
| `trim(fit, burn, thin)` | `dgp_trim(fit, burn, thin)` |
| `continue(fit, n, re_approx = TRUE)` | `dgp_continue(fit, n, true)` |
| `predict(fit, xx, lite = TRUE)` | `dgp_predict(fit, xx, 'lite', true)` |
| `settings = list(l =, u =, alpha =, beta =)` | `vdgp_options('l', …, 'u', …, 'alpha', …, 'beta', …)` |
| `cov = "matern"`, `v = 2.5` | `'cov', 'matern'`, `'v', 2.5` |
| `true_g` | `'true_g'` |
| `create_approx` / `update_obs_in_approx` | `vdgp_create_approx` / `vdgp_update_approx` |
| `create_U` / `U_entries` | `vdgp_create_U` / `vdgp_U_entries` |
| `logl` | `vdgp_logl` |
| `rand_mvn_vec` | `vdgp_rand_mvn` |
| `sample_w` / `sample_z` | `vdgp_sample_w` / `vdgp_sample_z` |
| `sample_theta` / `sample_g` | `vdgp_sample_theta` / `vdgp_sample_g` |

---

## Files

```
vdgp_setup.m               add the package to the path
vdgp_options.m             all options / priors / proposal settings
vdgp_build_mex.m           compile the optional MEX acceleration
mex/vdgp_U_entries_mex.c   OpenMP C kernel for the U factor (optional)

fit_one_layer.m            MCMC for a shallow GP
fit_two_layer.m            MCMC for a two-layer DGP
fit_three_layer.m          MCMC for a three-layer DGP
dgp_trim.m                 burn-in and thinning
dgp_continue.m             extend a chain, optionally re-approximating
dgp_predict.m              posterior predictive (parallel over draws)

vdgp_create_approx.m       ordering + conditioning sets
vdgp_update_approx.m       refresh latent coordinates (O(n))
vdgp_reapprox.m            rebuild the structure in the warped space
vdgp_order.m               random / maxmin ordering
vdgp_ordered_nn.m          ordered nearest neighbours (GpGp::find_ordered_nn)
vdgp_knn.m                 k-NN with a toolbox-free fallback

vdgp_U_entries.m           the sparse factor U (the paper's core object)
vdgp_create_U.m            sparse assembly of U
vdgp_logl.m                (Vecchia) MVN log-likelihood, profiled or not
vdgp_rand_mvn.m            prior draws by sparse triangular solve
vdgp_krig.m                kriging: mean / lite / joint, Vecchia or exact

vdgp_sample_theta.m        MH for a lengthscale
vdgp_sample_g.m            MH for the nugget
vdgp_sample_w.m            ESS for the layer feeding the response
vdgp_sample_z.m            ESS for the inner layer of a 3-layer DGP

vdgp_cov.m, vdgp_sqdist.m               covariance and scaled distances
vdgp_batch_cov.m, vdgp_batch_chol.m     batched small-matrix linear algebra
vdgp_batch_fsolve.m, vdgp_batch_last_row_inv.m
vdgp_lgamma.m, vdgp_crps.m, vdgp_band.m utilities

demos/demo_1d.m            the deepgp 1-D example, 1/2/3 layers
demos/demo_2d_scaling.m    cost and accuracy vs n, Vecchia vs exact
tests/test_vdgp.m          verification suite
```

---

## Requirements

Base MATLAB is enough. Optional:

* **Statistics and Machine Learning Toolbox** — `knnsearch` is used for the
  neighbour searches when present; otherwise a chunked brute-force fallback
  runs (`vdgp_knn.m`), which is fine up to tens of thousands of points.
* **Parallel Computing Toolbox** — `dgp_predict` parallelises over MCMC draws;
  pass `'cores', N` (or set `opts.cores`) to request `N` workers. `0` runs
  serially.
* **A C compiler** — `vdgp_build_mex` compiles the `U`-factor kernel, which is
  where essentially all the MCMC time goes. Worth doing for anything past a
  few thousand points; see *Optional MEX acceleration* above.

Note that with the MEX active the one-off nearest-neighbour search
(`vdgp_ordered_nn`, ~2 s at n = 5000 under Octave's brute-force fallback)
becomes visible in the fit time. MATLAB's `knnsearch` uses a KD-tree and is
much faster, so having the Statistics Toolbox matters more once the MEX is in
play.

The code also runs unmodified under GNU Octave, which is how the results below
were produced.

---

## Verification

`test_vdgp` checks the implementation against independent exact references:

```
PASS  U*U' equals inv(Sigma) at m = n-1                    max err 1.71e-10
PASS  log|Q| from diag(U)                                  err 0.00e+00
PASS  Vecchia log-likelihood at m = n-1                    -3032.7746172972 vs -3032.7746172972
PASS  log-likelihood error decreases with m                |err| = [202.164 11.593 1.175 0.000]
PASS  profiled outer log-likelihood and tau2               tau2 135.886572 vs 135.886572
PASS  rand_mvn empirical covariance                        max err 0.018
PASS  lite prediction, m = n, matches exact                mean 4.86e-12, s2 3.06e-14
PASS  joint prediction, full m, matches exact              mean 1.75e-13, Sigma 1.47e-15
PASS  MH kernel recovers the Gamma prior                   mean 1.544 (1.538), sd 1.275 (1.256)
PASS  ESS leaves the MVN prior invariant                   mean 0.006 (0), sd 1.007 (1)
PASS  ordered nearest-neighbour sets are exact
PASS  maxmin ordering is a valid permutation
PASS  two-layer Vecchia DGP on the deepgp example          RMSE 0.0247, coverage 0.99
```

The two distributional checks are the interesting ones. With `n = 1` there are
no pairwise distances, so the likelihood is free of `θ` and of `w`; the MH
kernel must then reproduce its `Gamma` prior exactly, and the elliptical slice
sampler must leave the `N(0, Σ)` prior invariant. Both do, which validates the
Hastings correction for the sliding-window proposal and the ESS bracket logic
independently of any model fit.

### 1-D example (`demo_1d`, n = 25, m = 10, 2000 iterations)

| model | RMSE | 95% coverage | CRPS |
|---|---|---|---|
| one layer | 0.0268 | 1.000 | 0.0204 |
| two layer | 0.0263 | 0.985 | 0.0162 |
| three layer | 0.0358 | 0.995 | 0.0302 |

The two-layer DGP's advantage on this function is mostly in the *uncertainty*:
its intervals collapse over the linear regime `x > 0.58` where the stationary
GP keeps inflating them — the 20% CRPS improvement, and the visible difference
between the two top panels of `demos/demo_1d.png`.

### Scaling (`demo_2d_scaling`, 2-D, m = 25)

Seconds per MCMC iteration under Octave; the Vecchia cost grows linearly in `n`
while the exact cost grows cubically, and accuracy is unaffected:

| n | Vecchia | exact |
|---|---|---|
| 500 | 0.92 | 6.97 |
| 1000 | 1.71 | — |
| 2000 | 3.09 | — |
| 4000 | 6.80 | — |

Vecchia at n = 4000 costs about what the exact sampler costs at n = 1000.

---

## Notes and deviations

* Vecchia is implemented for both `'matern'` and `'exp2'`; deepgp restricts its
  Vecchia path to Matérn. Matérn 5/2 remains the default and mixes noticeably
  faster — `'exp2'` needs more iterations to burn in on the 1-D example.
* The lengthscale is isotropic in each layer, with one lengthscale per hidden
  node, matching deepgp's default two-layer specification. `monowarp` and
  `pmx` (prior mean at `x`) from newer deepgp releases are not implemented.
* Latent-layer draws are stored for every kept iteration; on very large `n`
  use `'store_every', k` to keep memory in check.
* `dgp_predict` returns everything on the original `y` scale.
