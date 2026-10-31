# deepImp (development version)

## Multiple imputation

* `impNNet()` gains a draw mode: `draw = "pmm"` (predictive mean matching) or
  `draw = "normal"` imputes by draws from the predictive distribution, and the
  classification heads sample from their softmax instead of taking the most
  probable class. `bootstrap = TRUE` (the default in draw mode) fits every column
  model on a bootstrap resample of the observed rows and validates on the
  out-of-bag rows. Draw mode runs a fixed number of sweeps (default 5); point mode
  (`draw = "none"`, the default) is unchanged.
* `getImputed(m = "long")` returns all completed data sets in the `mice` long
  format, and the new `as_mids()` hands them to `mice::pool()`. It warns when the
  replicates come from point mode.
* The chain records the mean and SD of the imputed cells per variable and sweep;
  `plot(imp, type = "trace")` shows them, one line per replicate.

## Network heads and training

* Every network head is now a GLM plus an MLP correction (`deepimp_arch(glm_offset
  = TRUE)`, the default): a linear model under the regression heads, a
  multinomial logit (`nnet::multinom()`) under the classification heads. The
  correction starts at zero and the GLM state competes in restore-best, so a head
  is never worse on its validation rows than its GLM. `glm_ridge` adds a small
  ridge penalty against separation and rare levels. `glm_offset = FALSE` gives the
  network of 1.1.0. This applies to `impNNetCoDa()` as well.
* The torch backend trains with its own minibatch loop on preloaded tensors, with
  early stopping and restore-best weights; luz is no longer a dependency. The
  keras backend restores the best weights too.

# deepImp 1.1.0

Major rewrite of the package for a configurable, dual-backend neural-network imputer.

## New features

* Architecture is now user-configurable via `deepimp_arch()` (number and size of hidden
  layers, dropout, activation, batch normalisation, optimiser, learning rate), replacing
  the previously hard-coded network.
* Two backends: a native `torch` backend (default; no Python required) and an optional
  `keras3` backend, selected with `backend = "torch"` / `"keras"`.
* `impNNetCoDa()` reimplemented on the shared imputation engine for rounded zeros in
  compositional data, with `correction = "truncate"` (default, Templ 2021), `"expectation"`,
  or `"none"`.
* Results are returned as a `"deepimp"` object; read the completed data with `getImputed()`
  (or `as.data.frame()`), and inspect with `print()`, `summary()`, and `plot()`.
* Reproducible runs via a `seed` argument. Reproducibility is exact when `dropout = 0`; with
  `dropout > 0` the backend's training-time randomness is not fully pinned by the seed in the
  current torch build (repeated runs are close but may not be bit-identical).
* A vignette, `vignette("deepImp")`, and a package overview page (`?deepImp`).

## Bug fixes

* `impNNetCoDa(initialize = "kNNa")` no longer aborts at high rounded-zero rates.
  The compositional-kNN initialiser (`robCompositions::impKNNa()`) can fail with
  `'probs' not in [0, 1]` when a large fraction of every part falls below the
  detection limit; the initialiser now falls back to `knn` (with a warning)
  instead of stopping the imputation.

## Deprecations and breaking changes

* `impNNet()` was redesigned. Legacy arguments `dense`, `small`, `normalize_x`, and
  `normalize_y` are soft-deprecated (a warning maps them onto the new API). The arguments
  `model_numeric`, `model_binary`, `model_nominal`, and `loss` were removed; configure the
  network through `arch` and `backend` instead.

## Notes

* With `m > 1`, `impNNet()` / `impNNetCoDa()` return several completed data sets. These are
  stochastic replicate completions, not statistically valid multiple imputation.
