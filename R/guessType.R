#' Guess the measurement type of each variable
#'
#' Classifies each column as `"numeric"`, `"mixed"` (semi-continuous, i.e. a spike
#' of repeated values such as zeros plus a continuous part), `"binary"`,
#' `"nominal"`, or `"count"`. Used by [impNNet()] to choose the model head, and by
#' `summary()` of a `"deepimp"` object.
#'
#' @param x a data.frame, data.table, or tibble.
#' @return a list with `indices` (a type-by-variable logical matrix) and `type`
#'   (a character vector, one entry per column).
#' @author Matthias Templ
#' @export
#' @examples
#' data(sleep, package = "VIM")
#' guessType(sleep)
guessType <- function(x) {
  x <- as.data.frame(x)
  cls <- vapply(x, function(col) class(col)[1], character(1))

  is_num <- cls %in% c("numeric", "integer")
  is_fac <- cls %in% c("factor", "ordered", "character")

  check_binary <- function(col) length(unique(stats::na.omit(col))) == 2L

  check_count <- function(col) {
    v <- stats::na.omit(col)
    length(v) > 0 && all(v >= 0) && all(abs(v - round(v)) < .Machine$double.eps^0.5)
  }

  check_mixed <- function(col, spike = 0.10, distinct = 0.10) {
    v <- stats::na.omit(col)
    if (length(v) == 0) return(FALSE)
    tab <- table(v)
    any(tab >= spike * length(v)) && length(tab) >= distinct * length(v)
  }

  type <- character(ncol(x))
  for (j in seq_len(ncol(x))) {
    col <- x[[j]]
    if (is_fac[j]) {
      type[j] <- if (check_binary(col)) "binary" else "nominal"
    } else if (is_num[j]) {
      if (check_binary(col)) {
        type[j] <- "binary"
      } else if (check_mixed(col)) {
        type[j] <- "mixed"
      } else if (check_count(col)) {
        type[j] <- "count"
      } else {
        type[j] <- "numeric"
      }
    } else {
      type[j] <- "nominal"
    }
  }

  levels_all <- c("numeric", "mixed", "binary", "nominal", "count")
  indices <- matrix(
    unlist(lapply(levels_all, function(l) type == l), use.names = FALSE),
    nrow = length(levels_all), ncol = ncol(x), byrow = TRUE,
    dimnames = list(levels_all, colnames(x))
  )

  list(indices = indices, type = type)
}
