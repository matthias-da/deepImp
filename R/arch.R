#' Architecture configuration for neural-network imputation
#'
#' A backend-neutral description of the multilayer perceptron used by [impNNet()].
#' The same object is translated into a torch (and, later, keras) model.
#'
#' @param hidden integer vector of hidden-layer widths; its length is the depth.
#' @param dropout dropout rate applied after every hidden layer; a scalar in `[0, 1)`.
#' @param activation hidden-layer activation: one of `"relu"`, `"tanh"`, `"sigmoid"`,
#'   `"elu"`, `"leaky_relu"`.
#' @param batchnorm logical; apply batch normalisation after each hidden linear layer.
#' @param optimizer one of `"adam"`, `"sgd"`, `"rmsprop"`.
#' @param learning_rate positive learning rate for the optimiser.
#' @return an object of class `"deepimp_arch"`.
#' @seealso [impNNet()], [impNNetCoDa()]
#' @export
#' @examples
#' deepimp_arch(hidden = c(128, 64), dropout = 0.2)
deepimp_arch <- function(hidden = c(256, 128, 64),
                         dropout = 0.1,
                         activation = "relu",
                         batchnorm = TRUE,
                         optimizer = "adam",
                         learning_rate = 1e-3) {
  if (length(hidden) < 1L) stop("`hidden` must have at least one layer.", call. = FALSE)
  if (any(hidden <= 0)) stop("`hidden` widths must be positive.", call. = FALSE)
  if (any(abs(hidden - round(hidden)) > .Machine$double.eps^0.5))
    stop("`hidden` widths must be integer-valued.", call. = FALSE)
  if (!is.numeric(dropout) || length(dropout) != 1L || dropout < 0 || dropout >= 1)
    stop("`dropout` must be a single number in [0, 1).", call. = FALSE)
  activation_ok <- c("relu", "tanh", "sigmoid", "elu", "leaky_relu")
  if (length(activation) != 1L || !activation %in% activation_ok)
    stop("`activation` must be one of ", paste(activation_ok, collapse = ", "), ".",
         call. = FALSE)
  optimizer_ok <- c("adam", "sgd", "rmsprop")
  if (length(optimizer) != 1L || !optimizer %in% optimizer_ok)
    stop("`optimizer` must be one of ", paste(optimizer_ok, collapse = ", "), ".",
         call. = FALSE)
  if (!is.logical(batchnorm) || length(batchnorm) != 1L)
    stop("`batchnorm` must be a single logical.", call. = FALSE)
  if (!is.numeric(learning_rate) || length(learning_rate) != 1L || learning_rate <= 0)
    stop("`learning_rate` must be a positive number.", call. = FALSE)

  structure(
    list(hidden = as.integer(hidden), dropout = dropout, activation = activation,
         batchnorm = batchnorm, optimizer = optimizer, learning_rate = learning_rate),
    class = "deepimp_arch"
  )
}

#' @rdname deepimp_arch
#' @export
deepimp_arch_small <- function(dropout = 0.1, activation = "relu",
                               batchnorm = TRUE, optimizer = "adam",
                               learning_rate = 1e-3) {
  deepimp_arch(hidden = c(64, 32), dropout = dropout, activation = activation,
               batchnorm = batchnorm, optimizer = optimizer,
               learning_rate = learning_rate)
}

#' @export
print.deepimp_arch <- function(x, ...) {
  cat("<deepimp_arch>\n")
  cat("  hidden layers :", paste(x$hidden, collapse = " -> "), "\n")
  cat("  dropout       :", x$dropout, "\n")
  cat("  activation    :", x$activation, "\n")
  cat("  batchnorm     :", x$batchnorm, "\n")
  cat("  optimizer     :", x$optimizer, "(lr =", x$learning_rate, ")\n")
  invisible(x)
}
