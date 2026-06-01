#' Neural-network imputation of rounded zeros in compositional data
#'
#' Imputes rounded zeros (values below a detection limit) in compositional data
#' with a configurable neural network, following Templ (2021). Each part with
#' rounded zeros is pivoted to the front, transformed to pivot log-ratio
#' coordinates, regressed on the remaining coordinates, censored at the detection
#' limit, and back-transformed with preservation of the observed absolute values.
#'
#' @param x a data.frame of compositional parts (positive values; rounded zeros
#'   marked by `label`).
#' @param dl detection limits, a numeric vector of length `ncol(x)`. Required
#'   unless `correction = "none"`.
#' @param label the value marking a rounded zero in `x` (converted to `NA`
#'   internally). Default `0`.
#' @param coda if `TRUE` (default), work in pivot log-ratio coordinates; if
#'   `FALSE`, impute on the raw scale (ablation).
#' @param correction censoring of imputed values: `"truncate"` (default; set
#'   values above the detection limit to the limit, per Templ 2021),
#'   `"expectation"` (truncated-normal mean; opt-in, beyond Templ 2021), or
#'   `"none"` (no censoring).
#' @param initialize starting values for the rounded zeros: `"kNNa"` (default,
#'   compositional kNN), `"knn"`, `"hotdeck"`, or `"mean"`.
#' @param arch a [deepimp_arch()] object, or `NULL` to build one from `...`.
#' @param m number of stochastic replicate completions (not valid multiple
#'   imputation; see [impNNet()]).
#' @param backend `"torch"` (default).
#' @param iterations maximum number of chained sweeps.
#' @param eps convergence tolerance on the standardised mean change.
#' @param normalize standardise numeric predictors.
#' @param epochs,patience,validation_split,batch_size training controls.
#' @param seed optional integer seed (sets the R and backend RNGs). Exact
#'   reproducibility is guaranteed when `dropout = 0`; with `dropout > 0` the
#'   backend's training-time mask sampling is not fully pinned by the seed in the
#'   current torch build, so repeated runs are close but may not be bit-identical.
#' @param verbose print progress.
#' @param ... architecture scalar overrides forwarded to [deepimp_arch()].
#' @return a `"deepimp"` object; read the data with [getImputed()].
#' @author Matthias Templ
#' @seealso [impNNet()], [deepimp_arch()], [getImputed()]
#' @references Templ, M. (2021) Imputation of rounded zeros for high-dimensional
#'   compositional data.
#' @export
#' @examples
#' if (requireNamespace("torch", quietly = TRUE)) {
#' \donttest{
#'   set.seed(1)
#'   x <- data.frame(a = runif(50, 5, 10), b = runif(50, 5, 10), c = runif(50, 5, 10))
#'   x$a[1:5] <- 0
#'   imp <- impNNetCoDa(x, dl = c(1, 1, 1), label = 0,
#'                      arch = deepimp_arch_small(), epochs = 5, seed = 1)
#'   head(getImputed(imp))
#' }
#' }
impNNetCoDa <- function(x, dl = NULL, label = 0, coda = TRUE,
                        correction = c("truncate", "expectation", "none"),
                        initialize = "kNNa", arch = NULL, m = 1L,
                        backend = c("torch", "keras"),
                        iterations = 2L, eps = 1e-2, normalize = TRUE,
                        epochs = 400L, patience = 40L, validation_split = 0.2,
                        batch_size = 32L, seed = NULL, verbose = FALSE, ...) {
  x <- as.data.frame(x)
  correction <- match.arg(correction)
  backend_name <- match.arg(backend)

  if (!is.na(label)) x[x == label] <- NA

  if (correction != "none") {
    if (is.null(dl)) {
      stop("`dl` (detection limits) is required unless correction = \"none\".", call. = FALSE)
    }
    if (length(dl) != ncol(x)) stop("`dl` must have length ncol(x).", call. = FALSE)
    if (any(!is.finite(dl)) || any(dl <= 0)) stop("`dl` must be positive and finite.", call. = FALSE)
  }
  if (is.null(dl)) dl <- rep(NA_real_, ncol(x))

  dots <- list(...)
  if (!is.null(arch) && length(dots) > 0L) {
    stop("Supply the architecture via `arch` or via `...`, not both `arch` and `...` ",
         "(got: ", paste(names(dots), collapse = ", "), ").", call. = FALSE)
  }
  if (is.null(arch)) arch <- do.call(deepimp_arch, dots)
  if (!inherits(arch, "deepimp_arch")) stop("`arch` must be a deepimp_arch object.", call. = FALSE)

  control <- list(epochs = epochs, patience = patience,
                  batch_size = batch_size, verbose = verbose)

  res <- .run_chain(x, types = rep("numeric", ncol(x)), arch = arch,
                    backend = .backend(backend_name), initialize = initialize,
                    iterations = iterations, m = m, eps = eps, normalize = normalize,
                    control = control, validation_split = validation_split,
                    seed = seed, verbose = verbose,
                    column_fn = .impute_column_coda,
                    column_args = list(dl = dl, coda = coda, correction = correction))

  new_deepimp(
    data = x, imputed = res$imputed, arch = arch,
    info = list(backend = backend_name, coda = coda, dl = dl, correction = correction,
                initialize = initialize, convergence = res$convergence,
                seeds = res$seeds, n_missing = sum(res$na_mask))
  )
}
