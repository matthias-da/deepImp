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

## Deprecations and breaking changes

* `impNNet()` was redesigned. Legacy arguments `dense`, `small`, `normalize_x`, and
  `normalize_y` are soft-deprecated (a warning maps them onto the new API). The arguments
  `model_numeric`, `model_binary`, `model_nominal`, and `loss` were removed; configure the
  network through `arch` and `backend` instead.

## Notes

* With `m > 1`, `impNNet()` / `impNNetCoDa()` return several completed data sets. These are
  stochastic replicate completions, not statistically valid multiple imputation.
