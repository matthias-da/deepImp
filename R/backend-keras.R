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
  offset <- .use_glm_offset(arch, task, output_dim)
  inp <- keras3::keras_input(shape = as.integer(input_dim))
  h <- inp
  for (u in arch$hidden) {
    h <- keras3::layer_dense(h, units = as.integer(u))
    if (isTRUE(arch$batchnorm)) h <- keras3::layer_batch_normalization(h)
    h <- .keras_add_activation(h, arch$activation)
    if (arch$dropout > 0) h <- keras3::layer_dropout(h, rate = arch$dropout)
  }
  # GLM offset: the link values arrive as a second input and are added before
  # the softmax; the last layer starts at zero, so the untrained model is the GLM.
  init <- if (offset) "zeros" else "glorot_uniform"
  out <- keras3::layer_dense(h, units = as.integer(output_dim),
                             kernel_initializer = init, bias_initializer = "zeros")
  inputs <- inp
  if (offset) {
    off <- keras3::keras_input(shape = as.integer(output_dim))
    out <- keras3::layer_add(list(out, off))
    inputs <- list(inp, off)
  }
  if (task == "classification") {
    out <- keras3::layer_activation(out, activation = "softmax")
  }
  model <- keras3::keras_model(inputs = inputs, outputs = out)
  keras3::compile(model,
                  loss = if (task == "classification") "categorical_crossentropy" else "mse",
                  optimizer = .keras_optimizer(arch$optimizer, arch$learning_rate))
  list(model = model, task = task, output_dim = as.integer(output_dim), arch = arch,
       offset = offset)
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
  xtr <- as.matrix(train_x); xva <- as.matrix(valid_x)
  glm <- NULL
  if (isTRUE(model$offset)) {
    glm <- .glm_fit(xtr, train_y, model$task, model$output_dim, model$arch$glm_ridge)
    xtr <- list(xtr, .glm_link(glm, xtr))
    xva <- list(xva, .glm_link(glm, xva))
  }
  val_loss <- function() {
    as.numeric(keras3::evaluate(model$model, x = xva, y = yva, verbose = 0L,
                                return_dict = TRUE)$loss)
  }
  # Restore-best as in the torch backend: early stopping restores the best epoch,
  # and for an offset head the untrained state (the GLM) competes as epoch 0.
  init_w <- NULL; init_loss <- Inf
  if (isTRUE(model$offset)) {
    init_w <- keras3::get_weights(model$model); init_loss <- val_loss()
  }
  epochs_used <- 0L
  if (epochs > 0L) {
    hist <- keras3::fit(model$model, x = xtr, y = ytr, validation_data = list(xva, yva),
                        epochs = as.integer(epochs), batch_size = as.integer(batch_size),
                        verbose = verbose,
                        callbacks = list(keras3::callback_early_stopping(
                          monitor = "val_loss", patience = as.integer(patience),
                          restore_best_weights = TRUE)))
    epochs_used <- length(hist$metrics$val_loss)
  }
  best <- if (epochs > 0L) val_loss() else Inf
  best_epoch <- NA_integer_
  if (!is.null(init_w) && init_loss <= best) {
    keras3::set_weights(model$model, init_w); best <- init_loss; best_epoch <- 0L
  }
  list(model = list(fitted = model$model, task = model$task, glm = glm),
       history = list(epochs = epochs_used, best_epoch = best_epoch, best_valid = best))
}

#' @export
predict_model.deepimp_backend_keras <- function(backend, model, newx, ...) {
  x <- as.matrix(newx)
  if (!is.null(model$glm)) x <- list(x, .glm_link(model$glm, x))
  raw <- stats::predict(model$fitted, x, verbose = 0L)
  if (identical(model$task, "classification")) {
    return(as.matrix(raw))           # already softmax probabilities, n x K
  }
  as.numeric(raw)
}
