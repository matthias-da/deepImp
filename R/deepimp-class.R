#' Construct a deepimp object
#'
#' Low-level constructor for the object returned by [impNNet()]. Most users do
#' not call this directly; use [getImputed()] to read the completed data.
#'
#' @param data data.frame with missing values (original input).
#' @param imputed list of completed data.frames (one per imputation).
#' @param arch a [deepimp_arch()] object describing the network.
#' @param info list with training metadata (backend, vartypes, convergence, ...).
#' @return an object of class `"deepimp"`.
#' @seealso [getImputed()], [impNNet()]
#' @export
new_deepimp <- function(data, imputed, arch, info = list()) {
  stopifnot(is.data.frame(data), is.list(imputed))
  structure(
    list(
      data    = data,
      imputed = imputed,
      arch    = arch,
      info    = info
    ),
    class = "deepimp"
  )
}

#' Extract imputed data from a deepimp object
#'
#' @param object a `"deepimp"` object from [impNNet()].
#' @param m which completed dataset to return: an integer index, or `"all"` for
#'   the list of all imputations. With `m > 1` the datasets are stochastic
#'   replicate completions, **not** valid multiple imputation.
#' @param ... unused.
#' @return a data.frame, or a list of data.frames when `m = "all"`.
#' @seealso [impNNet()]
#' @export
getImputed <- function(object, m = 1L, ...) {
  UseMethod("getImputed")
}

#' @export
getImputed.deepimp <- function(object, m = 1L, ...) {
  n_imp <- length(object$imputed)
  if (identical(m, "all")) {
    return(object$imputed)
  }
  m <- as.integer(m)
  if (length(m) != 1L || is.na(m) || m < 1L || m > n_imp) {
    stop(sprintf("m must be 1..%d or 'all'.", n_imp), call. = FALSE)
  }
  object$imputed[[m]]
}

#' @export
as.data.frame.deepimp <- function(x, ..., m = 1L) {
  getImputed(x, m = m)
}

#' @export
print.deepimp <- function(x, ...) {
  cat("<deepimp>\n")
  cat(sprintf("  observations : %d\n", nrow(x$data)))
  cat(sprintf("  variables    : %d\n", ncol(x$data)))
  cat(sprintf("  imputations  : %d\n", length(x$imputed)))
  if (!is.null(x$info$backend)) {
    cat(sprintf("  backend      : %s\n", x$info$backend))
  }
  if (!is.null(x$arch)) {
    cat("  architecture :\n")
    print(x$arch)
  }
  invisible(x)
}

#' @export
summary.deepimp <- function(object, ...) {
  cat("<deepimp> summary\n")
  cat(sprintf("  observations : %d\n", nrow(object$data)))
  cat(sprintf("  variables    : %d\n", ncol(object$data)))
  cat(sprintf("  imputations  : %d\n", length(object$imputed)))
  miss <- colSums(is.na(object$data))
  cat("  missing per variable:\n")
  print(miss)
  if (!is.null(object$info$loss)) {
    cat(sprintf("  final loss   : %.4f\n", utils::tail(object$info$loss, 1)))
  }
  invisible(object)
}

#' @export
plot.deepimp <- function(x, ...) {
  if (is.null(x$info$loss) || length(x$info$loss) == 0L) {
    message("No loss curve available to plot.")
    return(invisible(x))
  }
  loss <- x$info$loss
  graphics::plot(
    seq_along(loss), loss,
    type = "l",
    xlab = "epoch",
    ylab = "loss",
    main = "deepimp training loss"
  )
  invisible(x)
}
