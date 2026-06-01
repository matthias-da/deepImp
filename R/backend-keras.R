#' keras3 backend for deepImp
#'
#' @description
#' Internal backend implementing the deepImp model seam (build/fit/predict) with
#' keras3 and its Keras/TensorFlow backend. Optional alternative to the default
#' torch backend; not called directly by users.
#'
#' @keywords internal
#' @name backend-keras
#' @importFrom stats predict
NULL

# Map a backend-neutral activation name to a keras layer. keras takes most
# activations as a string for layer_activation, but "leaky_relu" needs its own
# layer constructor.
.keras_add_activation <- function(model, name) {
  if (identical(name, "leaky_relu")) {
    keras3::layer_activation_leaky_relu(model)
  } else {
    keras3::layer_activation(model, activation = name)
  }
}

.keras_optimizer <- function(name, lr) {
  switch(name,
    adam    = keras3::optimizer_adam(learning_rate = lr),
    sgd     = keras3::optimizer_sgd(learning_rate = lr),
    rmsprop = keras3::optimizer_rmsprop(learning_rate = lr),
    stop("Unsupported optimizer for keras backend: ", name, call. = FALSE)
  )
}

#' @export
build_model.deepimp_backend_keras <- function(backend, task, input_dim,
                                              output_dim, arch, ...) {
  if (!requireNamespace("keras3", quietly = TRUE)) {
    stop("backend 'keras' needs the keras3 package and a working Keras/TensorFlow ",
         "backend - install keras3 and run keras3::install_keras().", call. = FALSE)
  }
  model <- keras3::keras_model_sequential(input_shape = as.integer(input_dim))
  for (h in arch$hidden) {
    model <- keras3::layer_dense(model, units = as.integer(h))
    if (isTRUE(arch$batchnorm)) model <- keras3::layer_batch_normalization(model)
    model <- .keras_add_activation(model, arch$activation)
    if (arch$dropout > 0) model <- keras3::layer_dropout(model, rate = arch$dropout)
  }
  if (task == "classification") {
    model <- keras3::layer_dense(model, units = as.integer(output_dim), activation = "softmax")
    keras3::compile(model, loss = "categorical_crossentropy",
                    optimizer = .keras_optimizer(arch$optimizer, arch$learning_rate),
                    metrics = "accuracy")
  } else {
    model <- keras3::layer_dense(model, units = 1L)
    keras3::compile(model, loss = "mse",
                    optimizer = .keras_optimizer(arch$optimizer, arch$learning_rate),
                    metrics = "mae")
  }
  list(model = model, task = task, output_dim = as.integer(output_dim))
}

#' @export
fit_model.deepimp_backend_keras <- function(backend, model, train_x, train_y,
                                            valid_x, valid_y, control, ...) {
  epochs     <- if (is.null(control$epochs))     50L else control$epochs
  patience   <- if (is.null(control$patience))   10L else control$patience
  batch_size <- if (is.null(control$batch_size)) 32L else control$batch_size
  verbose    <- if (isTRUE(control$verbose)) 1L else 0L

  if (model$task == "classification") {
    # engine passes 1-based class codes; keras to_categorical is 0-based.
    ytr <- keras3::to_categorical(as.integer(train_y) - 1L, num_classes = model$output_dim)
    yva <- keras3::to_categorical(as.integer(valid_y) - 1L, num_classes = model$output_dim)
  } else {
    ytr <- as.numeric(train_y)
    yva <- as.numeric(valid_y)
  }

  keras3::fit(model$model, x = as.matrix(train_x), y = ytr,
              validation_data = list(as.matrix(valid_x), yva),
              epochs = epochs, batch_size = batch_size, verbose = verbose,
              callbacks = list(keras3::callback_early_stopping(
                monitor = "val_loss", patience = patience)))

  list(model = list(fitted = model$model, task = model$task), history = NULL)
}

#' @export
predict_model.deepimp_backend_keras <- function(backend, model, newx, ...) {
  raw <- stats::predict(model$fitted, as.matrix(newx), verbose = 0L)
  if (identical(model$task, "classification")) {
    return(as.matrix(raw))           # already softmax probabilities, n x K
  }
  as.numeric(raw)
}
