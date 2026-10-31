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
#' @param m which completed dataset to return: an integer index, `"all"` for
#'   the list of all imputations, or `"long"` for all of them stacked in one
#'   data.frame in the `mice` long format (columns `.imp` and `.id`; `.imp = 0`
#'   holds the incomplete data). Only draw-mode replicates
#'   (`impNNet(draw = "pmm")` or `"normal"`) are multiple imputations; see
#'   [impNNet()].
#' @param ... unused.
#' @return a data.frame, or a list of data.frames when `m = "all"`.
#' @seealso [impNNet()], [as_mids()]
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
  if (identical(m, "long")) {
    n <- nrow(object$data)
    sets <- c(list(object$data), object$imputed)
    long <- do.call(rbind, lapply(seq_along(sets), function(i) {
      cbind(data.frame(.imp = i - 1L, .id = seq_len(n)), sets[[i]])
    }))
    rownames(long) <- NULL
    return(long)
  }
  m <- if (is.numeric(m)) as.integer(m) else NA_integer_
  if (length(m) != 1L || is.na(m) || m < 1L || m > n_imp) {
    stop(sprintf("m must be 1..%d, 'all' or 'long'.", n_imp), call. = FALSE)
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
  conv <- object$info$convergence
  trace <- if (is.list(conv)) conv[[1]] else conv
  if (!is.null(trace) && length(trace) > 0L) {
    cat(sprintf("  converged in : %d iteration(s); last change = %.4g\n",
                length(trace), utils::tail(trace, 1)))
  }
  invisible(object)
}

#' Plot diagnostics of a deepimp object
#'
#' `type = "convergence"` plots the pooled standardised change of the imputed
#' cells per sweep (first replicate). `type = "trace"` plots, per incomplete
#' variable, the mean and SD of the imputed cells by sweep with one line per
#' replicate (as `plot()` of a `mice::mids` object); factors enter by their
#' integer codes. Well-mixed draw-mode chains show intertwined lines without a
#' trend.
#'
#' @param x a `"deepimp"` object.
#' @param type `"convergence"` (default) or `"trace"`.
#' @param vars for `type = "trace"`: names of the variables to show (default:
#'   all incomplete variables).
#' @param ... further arguments passed to [graphics::matplot()] (trace) or
#'   [graphics::plot()] (convergence).
#' @return `x`, invisibly.
#' @export
plot.deepimp <- function(x, type = c("convergence", "trace"), vars = NULL, ...) {
  type <- match.arg(type)
  if (type == "trace") return(.plot_trace(x, vars, ...))
  conv <- x$info$convergence
  trace <- if (is.list(conv)) conv[[1]] else conv
  if (is.null(trace) || length(trace) == 0L) {
    message("No convergence trace available to plot.")
    return(invisible(x))
  }
  graphics::plot(
    seq_along(trace), trace,
    type = "b",
    xlab = "sweep",
    ylab = "mean change in imputed cells",
    main = "deepImp convergence", ...
  )
  invisible(x)
}

.plot_trace <- function(x, vars = NULL, ...) {
  tr <- x$info$trace
  if (is.null(tr) || !length(tr$mean) || all(is.na(tr$mean))) {
    message("No chain trace available to plot (fitted with deepImp < 1.2.0?).")
    return(invisible(x))
  }
  all_vars <- dimnames(tr$mean)[[1]]
  if (is.null(vars)) vars <- all_vars
  bad <- setdiff(vars, all_vars)
  if (length(bad)) {
    stop("No trace for: ", paste(bad, collapse = ", "),
         ". Choose among the incomplete variables: ", paste(all_vars, collapse = ", "),
         call. = FALSE)
  }
  op <- graphics::par(mfrow = c(length(vars), 2L), mar = c(4, 4, 2, 1))
  on.exit(graphics::par(op))
  for (v in vars) {
    for (stat in c("mean", "sd")) {
      y <- matrix(tr[[stat]][v, , ], ncol = dim(tr[[stat]])[3L])
      graphics::matplot(seq_len(nrow(y)), y, type = "l", lty = 1,
                        xlab = "sweep", ylab = stat, main = paste(v, stat), ...)
    }
  }
  invisible(x)
}

#' Convert a deepimp object to a mice `mids` object
#'
#' Hands the completed data sets to the \pkg{mice} analysis and pooling
#' machinery, e.g. `with(as_mids(imp), lm(y ~ x))` followed by `mice::pool()`.
#' Rubin's rules are only meaningful for multiple imputations, i.e. for objects
#' from [impNNet()] with `draw = "pmm"` or `"normal"` and `m > 1`. Point-mode
#' replicates (`draw = "none"`) differ only by retraining; pooling them
#' understates the between-imputation variance, so the conversion warns.
#'
#' @param object a `"deepimp"` object.
#' @return a `mids` object (requires the \pkg{mice} package).
#' @seealso [impNNet()], [getImputed()]
#' @export
#' @examples
#' if (requireNamespace("torch", quietly = TRUE) && torch::torch_is_installed() &&
#'     requireNamespace("mice", quietly = TRUE)) {
#' \donttest{
#'   data(sleep, package = "VIM")
#'   mi <- impNNet(sleep, arch = deepimp_arch_small(dropout = 0), m = 3,
#'                 draw = "pmm", iterations = 2, epochs = 5, seed = 1)
#'   fit <- with(as_mids(mi), lm(Sleep ~ log(BodyWgt) + Danger))
#'   summary(mice::pool(fit))
#' }
#' }
as_mids <- function(object) {
  if (!inherits(object, "deepimp")) stop("`object` must be a deepimp object.", call. = FALSE)
  if (!requireNamespace("mice", quietly = TRUE)) {
    stop("as_mids() needs the mice package: install.packages(\"mice\").", call. = FALSE)
  }
  draw <- object$info$draw
  if (is.null(draw) || identical(draw, "none")) {
    warning("These are point-mode replicates (draw = \"none\"), not multiple ",
            "imputations: pooling them with Rubin's rules understates the ",
            "imputation variance. Refit with impNNet(draw = \"pmm\").", call. = FALSE)
  } else if (length(object$imputed) < 2L) {
    warning("Only one imputation (m = 1): pooling needs m >= 2.", call. = FALSE)
  }
  mice::as.mids(getImputed(object, m = "long"))
}
