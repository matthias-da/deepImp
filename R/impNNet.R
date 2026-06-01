#' Neural-network imputation for mixed-type data
#'
#' Iterative, chained imputation of numeric, count, semi-continuous, binary and
#' nominal variables with a configurable multilayer perceptron (torch backend).
#'
#' @param data a data.frame (tibbles/data.tables are coerced).
#' @param arch a [deepimp_arch()] object, or `NULL` to build one from `...`.
#'   Supplying both `arch` and architecture scalars via `...` is an error.
#' @param m number of stochastic replicate completions. NOTE: with `m > 1` these
#'   are stochastic replicate completions, **not** valid multiple imputation
#'   (no Rubin-rule inference).
#' @param backend `"torch"` (default). `"keras"` arrives in a later release.
#' @param vartypes `"guess"` (use [guessType()]) or a length-`ncol(data)` vector.
#' @param initialize starting values for NAs: `"knn"`, `"mean"`, or `"hotdeck"`.
#' @param iterations maximum number of chained sweeps.
#' @param eps convergence tolerance on the standardised mean change.
#' @param normalize standardise numeric predictors.
#' @param epochs,patience,validation_split,batch_size training controls.
#' @param seed optional integer seed (sets R and torch RNGs).
#' @param verbose print progress.
#' @param ... architecture scalar overrides forwarded to [deepimp_arch()].
#' @return a `"deepimp"` object; read the data with [getImputed()].
#' @author Matthias Templ
#' @seealso [deepimp_arch()], [getImputed()], [guessType()]
#' @export
#' @examples
#' \dontrun{
#' data(sleep, package = "VIM")
#' imp <- impNNet(sleep, arch = deepimp_arch_small(), epochs = 50)
#' head(getImputed(imp))
#' }
impNNet <- function(data, arch = NULL, m = 1L, backend = c("torch", "keras"),
                    vartypes = "guess", initialize = "knn", iterations = 3L,
                    eps = 1e-2, normalize = TRUE, epochs = 400L, patience = 40L,
                    validation_split = 0.2, batch_size = 32L, seed = NULL,
                    verbose = FALSE, ...) {
  data <- as.data.frame(data)
  backend_name <- match.arg(backend)

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
                    seed = seed, verbose = verbose)

  new_deepimp(
    data = data, imputed = res$imputed, arch = arch,
    info = list(backend = backend_name, vartypes = vartypes,
                convergence = res$convergence, seeds = res$seeds,
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
