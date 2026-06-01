#' Neural-network imputation of rounded zeros in compositional data
#'
#' Being rewritten on the new engine (Plan 2). Use [impNNet()] for mixed-type data.
#'
#' @param x a data.frame of compositional parts.
#' @param ... currently ignored.
#' @return not yet implemented; currently raises an informative error.
#' @export
impNNetCoDa <- function(x, ...) {
  stop("impNNetCoDa() is being rewritten on the new engine and is not available ",
       "in this version. Use impNNet() for mixed-type data.", call. = FALSE)
}
