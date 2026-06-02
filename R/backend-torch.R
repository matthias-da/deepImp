#' torch/luz backend for deepImp
#'
#' @description
#' Internal backend implementing the deepImp model seam (build/fit/predict) with
#' torch and luz. Not called directly by users.
#'
#' @keywords internal
#' @name backend-torch
#' @importFrom torch nn_module nn_linear nn_batch_norm1d nn_dropout
#' @importFrom torch nn_relu nn_leaky_relu nn_tanh nn_sigmoid nn_elu
#' @importFrom torch nn_sequential nn_cross_entropy_loss nn_mse_loss
#' @importFrom torch nnf_softmax torch_tensor torch_float torch_long
#' @importFrom torch optim_adam optim_sgd optim_rmsprop torch_manual_seed as_array
NULL

# Map a backend-neutral activation name to its torch module constructor.
.torch_activation <- function(name) {
  switch(name,
    relu       = torch::nn_relu,
    leaky_relu = torch::nn_leaky_relu,
    tanh       = torch::nn_tanh,
    sigmoid    = torch::nn_sigmoid,
    elu        = torch::nn_elu,
    stop("Unsupported activation for torch backend: ", name, call. = FALSE)
  )
}

# Build the parametrised MLP generator. Block order per hidden layer:
# dense -> batchnorm -> activation -> dropout. torch emits a spurious
# "object 'self' not found" warning while *constructing* the generator (a known
# false positive); we muffle only that exact message.
.make_deepimp_mlp <- function() {
  torch::nn_module(
    classname = "deepimp_mlp",
    initialize = function(input_dim, output_dim, arch, task) {
      self$task <- task
      act_fn <- .torch_activation(arch$activation)
      layers <- list()
      prev <- input_dim
      for (h in arch$hidden) {
        layers[[length(layers) + 1L]] <- torch::nn_linear(prev, h)
        if (isTRUE(arch$batchnorm)) {
          layers[[length(layers) + 1L]] <- torch::nn_batch_norm1d(h)
        }
        layers[[length(layers) + 1L]] <- act_fn()
        if (arch$dropout > 0) {
          layers[[length(layers) + 1L]] <- torch::nn_dropout(arch$dropout)
        }
        prev <- h
      }
      layers[[length(layers) + 1L]] <- torch::nn_linear(prev, output_dim)
      self$net <- do.call(torch::nn_sequential, layers)
    },
    forward = function(x) {
      self$net(x)
    }
  )
}

deepimp_mlp <- withCallingHandlers(
  .make_deepimp_mlp(),
  warning = function(w) {
    if (grepl("object 'self' not found", conditionMessage(w), fixed = TRUE)) {
      invokeRestart("muffleWarning")
    }
  }
)

#' @export
build_model.deepimp_backend_torch <- function(backend, task, input_dim,
                                              output_dim, arch, ...) {
  structure(
    list(module = deepimp_mlp, task = task, input_dim = input_dim,
         output_dim = output_dim, arch = arch),
    class = "deepimp_torch_model"
  )
}

#' @export
fit_model.deepimp_backend_torch <- function(backend, model, train_x, train_y,
                                            valid_x, valid_y, control, ...) {
  arch <- model$arch
  opt_fn <- switch(arch$optimizer,
    adam = torch::optim_adam, sgd = torch::optim_sgd, rmsprop = torch::optim_rmsprop)

  epochs     <- if (is.null(control$epochs))     50L else control$epochs
  patience   <- if (is.null(control$patience))   10L else control$patience
  batch_size <- if (is.null(control$batch_size)) 32L else control$batch_size
  verbose    <- isTRUE(control$verbose)

  if (model$task == "classification") {
    loss_fn <- torch::nn_cross_entropy_loss()      # expects 1-based class indices (R torch)
    ytrain <- torch::torch_tensor(as.integer(train_y), dtype = torch::torch_long())
    yvalid <- torch::torch_tensor(as.integer(valid_y), dtype = torch::torch_long())
  } else {
    loss_fn <- torch::nn_mse_loss()
    ytrain <- torch::torch_tensor(matrix(as.numeric(train_y), ncol = 1L),
                                  dtype = torch::torch_float())
    yvalid <- torch::torch_tensor(matrix(as.numeric(valid_y), ncol = 1L),
                                  dtype = torch::torch_float())
  }
  xtrain <- torch::torch_tensor(as.matrix(train_x), dtype = torch::torch_float())
  xvalid <- torch::torch_tensor(as.matrix(valid_x), dtype = torch::torch_float())

  # A singleton final TRAINING batch crashes nn_batch_norm1d (variance over one
  # sample). luz shares `dataloader_options` between the training and validation
  # loaders, so `drop_last = TRUE` would also drop a single-batch validation set,
  # leaving no `valid_loss` and crashing early stopping. Instead, avoid the
  # singleton by shrinking the batch size when the training set is congruent to 1
  # modulo batch_size; this never empties the validation loader.
  n_train <- nrow(as.matrix(train_x))
  if (batch_size > 2L && n_train %% batch_size == 1L) batch_size <- batch_size - 1L

  # Train on CPU. Imputation here is small tabular work that gains nothing from a
  # GPU/MPS device, MPS op-coverage is incomplete, and on Apple Silicon the MPS
  # validation pass can fail to produce the monitored `valid_loss`, which crashes
  # `luz_callback_early_stopping`. Pinning to CPU also underpins the exact-
  # reproducibility guarantee (single-threaded CPU, dropout = 0).
  fitted <- model$module |>
    luz::setup(loss = loss_fn, optimizer = opt_fn) |>
    luz::set_hparams(input_dim = model$input_dim, output_dim = model$output_dim,
                     arch = arch, task = model$task) |>
    luz::set_opt_hparams(lr = arch$learning_rate) |>
    luz::fit(
      data = list(xtrain, ytrain),
      epochs = epochs,
      valid_data = list(xvalid, yvalid),
      accelerator = luz::accelerator(cpu = TRUE),
      callbacks = list(luz::luz_callback_early_stopping(
        monitor = "valid_loss", patience = patience)),
      dataloader_options = list(batch_size = batch_size, shuffle = TRUE),
      verbose = verbose
    )

  list(model = list(fitted = fitted, task = model$task), history = NULL)
}

#' @export
predict_model.deepimp_backend_torch <- function(backend, model, newx, ...) {
  xt <- torch::torch_tensor(as.matrix(newx), dtype = torch::torch_float())
  raw <- stats::predict(model$fitted, xt)
  if (identical(model$task, "classification")) {
    probs <- torch::nnf_softmax(raw, dim = 2L)
    return(as.matrix(torch::as_array(probs)))
  }
  as.numeric(torch::as_array(raw))
}

utils::globalVariables("self")
