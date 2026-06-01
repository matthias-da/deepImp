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
