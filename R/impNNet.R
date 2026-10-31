#' Neural-network imputation for mixed-type data
#'
#' Iterative, chained imputation of numeric, count, semi-continuous, binary and
#' nominal variables with a configurable multilayer perceptron (torch backend).
#'
#' @section Point mode and draw mode:
#' With `draw = "none"` (point mode) every missing cell gets the network's
#' conditional mean (regression heads) or its most probable class
#' (classification heads), and the chain stops once the pooled change falls
#' below `eps`. Point imputations suit prediction-type use, but replicates
#' (`m > 1`) then differ only by retraining, so they understate the imputation
#' uncertainty and must not be pooled with Rubin's rules.
#'
#' With `draw = "pmm"` or `"normal"` (draw mode) every missing cell gets a draw
#' from the predictive distribution, and by default (`bootstrap = TRUE`) every
#' fit uses a bootstrap resample of the observed rows. The classification heads
#' then sample from their softmax probabilities. Each replicate runs all
#' `iterations` sweeps. The `m` completed data sets are multiple imputations
#' that can be analysed and pooled with Rubin's rules, e.g. with
#' `mice::pool()` via [as_mids()].
#' PMM copies observed values, so it keeps count and semi-continuous variables
#' on their observed support; `"normal"` adds Gaussian noise with the residual
#' SD of the held-out (out-of-bag) rows and suits continuous variables. For
#' exactly reproducible draws use `dropout = 0`.
#'
#' @param data a data.frame (tibbles/data.tables are coerced).
#' @param arch a [deepimp_arch()] object, or `NULL` to build one from `...`.
#'   Supplying both `arch` and architecture scalars via `...` is an error.
#' @param m number of completed data sets (replicates). See "Point mode and
#'   draw mode": only draw-mode replicates are multiple imputations.
#' @param backend `"torch"` (default) or `"keras"` (optional; requires the
#'   keras3 package and a working Keras/TensorFlow backend).
#' @param vartypes `"guess"` (use [guessType()]) or a length-`ncol(data)` vector.
#' @param initialize starting values for NAs: `"knn"`, `"mean"`, or `"hotdeck"`.
#' @param iterations number of chained sweeps: the maximum in point mode
#'   (default 3), the fixed number in draw mode (default 5). `NULL` picks the
#'   default of the mode.
#' @param eps convergence tolerance on the standardised mean change (point mode
#'   only).
#' @param draw `"none"` (point imputation, default), `"pmm"` (predictive mean
#'   matching) or `"normal"` (prediction plus normal noise). Any draw mode
#'   samples the classification heads from their softmax.
#' @param bootstrap logical; fit every column model on a bootstrap resample of
#'   the observed rows, with the out-of-bag rows as the validation set. Defaults
#'   to `TRUE` in draw mode and `FALSE` in point mode.
#' @param pmm_k number of donors for `draw = "pmm"`.
#' @param normalize standardise numeric predictors.
#' @param epochs,patience,validation_split,batch_size training controls.
#'   `validation_split` is not used when `bootstrap = TRUE`.
#' @param seed optional integer seed (sets the R and backend RNGs). Exact
#'   reproducibility is guaranteed when `dropout = 0`; with `dropout > 0` the
#'   backend's training-time mask sampling is not fully pinned by the seed in the
#'   current torch build, so repeated runs are close but may not be bit-identical.
#' @param verbose print progress.
#' @param ... architecture scalar overrides forwarded to [deepimp_arch()].
#' @return a `"deepimp"` object; read the data with [getImputed()].
#' @author Matthias Templ
#' @seealso [deepimp_arch()], [getImputed()], [as_mids()], [guessType()]
#' @export
#' @examples
#' if (requireNamespace("torch", quietly = TRUE) && torch::torch_is_installed()) {
#' \donttest{
#'   data(sleep, package = "VIM")
#'   imp <- impNNet(sleep, arch = deepimp_arch_small(), epochs = 5, seed = 1)
#'   head(getImputed(imp))
#'
#'   # multiple imputation: predictive draws on bootstrap fits
#'   mi <- impNNet(sleep, arch = deepimp_arch_small(dropout = 0), m = 3,
#'                 draw = "pmm", iterations = 2, epochs = 5, seed = 1)
#'   head(getImputed(mi, m = "long"))
#' }
#' }
impNNet <- function(data, arch = NULL, m = 1L, backend = c("torch", "keras"),
                    vartypes = "guess", initialize = "knn", iterations = NULL,
                    eps = 1e-2, normalize = TRUE, epochs = 400L, patience = 40L,
                    validation_split = 0.2, batch_size = 32L, seed = NULL,
                    verbose = FALSE, draw = c("none", "pmm", "normal"),
                    bootstrap = NULL, pmm_k = 5L, ...) {
  data <- as.data.frame(data)
  backend_name <- match.arg(backend)
  draw <- match.arg(draw)
  draw_mode <- draw != "none"
  if (is.null(iterations)) iterations <- if (draw_mode) 5L else 3L
  if (is.null(bootstrap)) bootstrap <- draw_mode
  if (!is.logical(bootstrap) || length(bootstrap) != 1L || is.na(bootstrap))
    stop("`bootstrap` must be TRUE or FALSE.", call. = FALSE)
  if (!is.numeric(pmm_k) || length(pmm_k) != 1L || !is.finite(pmm_k) || pmm_k < 1 ||
      pmm_k != round(pmm_k))
    stop("`pmm_k` must be a positive integer.", call. = FALSE)

  dots <- list(...)
  dep <- .handle_deprecated_args(dots, arch, normalize)
  arch <- dep$arch; normalize <- dep$normalize; dots <- dep$dots
  if (!is.null(arch) && length(dots) > 0L) {
    stop("Supply the architecture via `arch` or via `...`, not both `arch` and `...` ",
         "(got: ", paste(names(dots), collapse = ", "), ").", call. = FALSE)
  }
  if (is.null(arch)) arch <- do.call(deepimp_arch, dots)
  if (!inherits(arch, "deepimp_arch")) {
    stop("`arch` must be a deepimp_arch object.", call. = FALSE)
  }

  if (identical(vartypes, "guess")) {
    vartypes <- guessType(data)$type
    if (verbose) message("variable types: ", paste(vartypes, collapse = ", "))
  } else if (length(vartypes) != ncol(data)) {
    stop("`vartypes` must have length ncol(data).", call. = FALSE)
  }

  control <- list(epochs = epochs, patience = patience,
                  batch_size = batch_size, verbose = verbose)

  res <- .run_chain(data, types = vartypes, arch = arch,
                    backend = .backend(backend_name), initialize = initialize,
                    iterations = iterations, m = m, eps = eps, normalize = normalize,
                    control = control, validation_split = validation_split,
                    seed = seed, verbose = verbose,
                    column_args = list(draw = draw, bootstrap = bootstrap,
                                       pmm_k = as.integer(pmm_k)),
                    early_stop = !draw_mode)

  new_deepimp(
    data = data, imputed = res$imputed, arch = arch,
    info = list(backend = backend_name, vartypes = vartypes, draw = draw,
                bootstrap = bootstrap, pmm_k = as.integer(pmm_k),
                iterations = iterations, convergence = res$convergence,
                trace = res$trace, seeds = res$seeds,
                n_missing = sum(res$na_mask))
  )
}

# Map legacy impNNet() argument names (pre-1.1.0) onto the new API. Returns a list
# with possibly-updated `arch` and `normalize`, and the non-legacy remainder of
# `dots` to flow into the normal arch-vs-... logic.
#' @noRd
.handle_deprecated_args <- function(dots, arch, normalize) {
  removed <- intersect(names(dots), c("model_numeric", "model_binary", "model_nominal", "loss"))
  if (length(removed)) {
    stop("`", removed[1], "` was removed in deepImp 1.1.0; configure the network via ",
         "`arch = deepimp_arch(...)` and `backend=`.", call. = FALSE)
  }
  if ("dense" %in% names(dots)) {
    warning("`dense` is deprecated; use `arch = deepimp_arch(hidden = ...)`. ",
            "Mapping for this call.", call. = FALSE)
    if (!is.null(arch)) {
      stop("Cannot use deprecated `dense` together with an explicit `arch`.", call. = FALSE)
    }
    arch <- deepimp_arch(hidden = dots$dense)
    dots$dense <- NULL
  }
  if ("small" %in% names(dots)) {
    warning("`small` is deprecated; use `arch = deepimp_arch_small()`. Mapping for this call.",
            call. = FALSE)
    if (isTRUE(dots$small) && is.null(arch)) arch <- deepimp_arch_small()
    dots$small <- NULL
  }
  if ("normalize_x" %in% names(dots)) {
    warning("`normalize_x` is deprecated; use `normalize`. Mapping for this call.", call. = FALSE)
    normalize <- isTRUE(dots$normalize_x)
    dots$normalize_x <- NULL
  }
  if ("normalize_y" %in% names(dots)) {
    warning("`normalize_y` is deprecated and ignored; target standardisation is now automatic.",
            call. = FALSE)
    dots$normalize_y <- NULL
  }
  list(arch = arch, normalize = normalize, dots = dots)
}
