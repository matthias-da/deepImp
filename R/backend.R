# Backend seam. The engine only ever calls build_model / fit_model / predict_model
# on a backend object created by .backend(). torch is implemented in
# backend-torch.R and keras in backend-keras.R.

.backend <- function(name = c("torch", "keras")) {
  if (missing(name)) name <- "torch"
  name <- name[1L]
  cls <- switch(name,
                torch = "deepimp_backend_torch",
                keras = "deepimp_backend_keras",
                stop("Unknown backend: ", name, call. = FALSE))
  structure(list(name = name), class = c(cls, "deepimp_backend"))
}

# Map neutral optimiser/activation names; used by the backend implementations.
.optimizer_names <- c("adam", "sgd", "rmsprop")
.activation_names <- c("relu", "tanh", "sigmoid", "elu", "leaky_relu")

build_model   <- function(backend, task, input_dim, output_dim, arch) UseMethod("build_model")
fit_model     <- function(backend, model, train_x, train_y, valid_x, valid_y, control) UseMethod("fit_model")
predict_model <- function(backend, model, newx) UseMethod("predict_model")

#' @export
build_model.default <- function(backend, task, input_dim, output_dim, arch) {
  stop("Backend '", backend$name, "' is not available. Install its packages or choose ",
       "backend = \"torch\".", call. = FALSE)
}
