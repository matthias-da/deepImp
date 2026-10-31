#' torch backend for deepImp
#'
#' @description
#' Internal backend implementing the deepImp model seam (build/fit/predict) with
#' torch and a plain training loop on preloaded tensors. Not called directly by
#' users.
#'
#' @keywords internal
#' @name backend-torch
#' @importFrom torch nn_module nn_linear nn_batch_norm1d nn_dropout
#' @importFrom torch nn_relu nn_leaky_relu nn_tanh nn_sigmoid nn_elu
#' @importFrom torch nn_sequential nn_cross_entropy_loss nn_mse_loss
#' @importFrom torch nnf_softmax torch_tensor torch_float torch_long
#' @importFrom torch optim_adam optim_sgd optim_rmsprop torch_manual_seed as_array
#' @importFrom torch with_no_grad
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
# dense -> batchnorm -> activation -> dropout. With `offset = TRUE` the module
# expects the GLM link values as the last `output_dim` columns of its input,
# feeds only the first `input_dim` columns through the MLP and adds the link
# values to the MLP output; the last layer starts at zero, so the untrained
# module reproduces the GLM. torch emits a spurious "object 'self' not found"
# warning while *constructing* the generator (a known false positive); we
# muffle only that exact message.
.make_deepimp_mlp <- function() {
  torch::nn_module(
    classname = "deepimp_mlp",
    initialize = function(input_dim, output_dim, arch, task, offset = FALSE) {
      self$task <- task
      self$p <- as.integer(input_dim)
      self$offset <- isTRUE(offset)
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
      last <- torch::nn_linear(prev, output_dim)
      if (self$offset) {
        torch::with_no_grad({
          last$weight$zero_()
          last$bias$zero_()
        })
      }
      layers[[length(layers) + 1L]] <- last
      self$net <- do.call(torch::nn_sequential, layers)
    },
    forward = function(x) {
      if (self$offset) {
        self$net(x[, 1:self$p, drop = FALSE]) +
          x[, (self$p + 1L):x$size(2L), drop = FALSE]
      } else {
        self$net(x)
      }
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

# Copy a module's state so later optimiser steps cannot alter it.
.torch_snapshot <- function(module) {
  lapply(module$state_dict(), function(t) t$clone())
}

# Plain minibatch training loop on preloaded tensors with early stopping on the
# validation loss and restore-best weights. With `eval_init = TRUE` the starting
# weights compete in restore-best (best_epoch 0 = the untrained module won),
# which is what lets a GLM-offset head stay at the GLM.
.torch_train <- function(module, train_x, train_y, valid_x, valid_y, task,
                         optimizer = "adam", lr = 1e-3, epochs = 50L,
                         patience = 10L, batch_size = 32L, eval_init = FALSE,
                         verbose = FALSE) {
  cls <- identical(task, "classification")
  xt <- torch::torch_tensor(as.matrix(train_x), dtype = torch::torch_float())
  xv <- torch::torch_tensor(as.matrix(valid_x), dtype = torch::torch_float())
  if (cls) {
    loss_fn <- torch::nn_cross_entropy_loss()      # 1-based class indices (R torch)
    yt <- torch::torch_tensor(as.integer(train_y), dtype = torch::torch_long())
    yv <- torch::torch_tensor(as.integer(valid_y), dtype = torch::torch_long())
  } else {
    loss_fn <- torch::nn_mse_loss()
    yt <- torch::torch_tensor(matrix(as.numeric(train_y), ncol = 1L),
                              dtype = torch::torch_float())
    yv <- torch::torch_tensor(matrix(as.numeric(valid_y), ncol = 1L),
                              dtype = torch::torch_float())
  }
  opt_fn <- switch(optimizer,
    adam = torch::optim_adam, sgd = torch::optim_sgd, rmsprop = torch::optim_rmsprop,
    stop("Unsupported optimizer for torch backend: ", optimizer, call. = FALSE))
  opt <- opt_fn(module$parameters, lr = lr)

  valid_loss <- function() {
    module$eval()
    torch::with_no_grad(loss_fn(module(xv), yv)$item())
  }

  n <- nrow(as.matrix(train_x))
  bs <- max(1L, min(as.integer(batch_size), n))
  best <- Inf; best_state <- NULL; best_epoch <- NA_integer_; wait <- 0L; used <- 0L
  if (eval_init) {
    best <- valid_loss(); best_state <- .torch_snapshot(module); best_epoch <- 0L
  }
  for (ep in seq_len(epochs)) {
    module$train()
    perm <- sample.int(n)
    for (s in seq(1L, n, by = bs)) {
      idx <- perm[s:min(s + bs - 1L, n)]
      # A singleton batch would break nn_batch_norm1d (variance of one row).
      if (length(idx) < 2L) next
      opt$zero_grad()
      out <- module(xt[idx, , drop = FALSE])
      loss <- if (cls) loss_fn(out, yt[idx]) else loss_fn(out, yt[idx, , drop = FALSE])
      loss$backward()
      opt$step()
    }
    vl <- valid_loss()
    used <- ep
    if (is.finite(vl) && vl < best) {
      best <- vl; best_state <- .torch_snapshot(module); best_epoch <- ep; wait <- 0L
    } else {
      wait <- wait + 1L
      if (wait >= patience) break
    }
  }
  if (!is.null(best_state)) module$load_state_dict(best_state)
  module$eval()
  if (verbose) {
    message(sprintf("    %s fit: %d epoch(s), best epoch %d, valid loss %.4g",
                    task, used, best_epoch, best))
  }
  list(module = module, epochs = used, best_epoch = best_epoch, best_valid = best)
}

#' @export
build_model.deepimp_backend_torch <- function(backend, task, input_dim,
                                              output_dim, arch, ...) {
  offset <- .use_glm_offset(arch, task, output_dim)
  module <- deepimp_mlp(input_dim = input_dim, output_dim = output_dim,
                        arch = arch, task = task, offset = offset)
  structure(
    list(module = module, task = task, input_dim = input_dim,
         output_dim = output_dim, arch = arch, offset = offset),
    class = "deepimp_torch_model"
  )
}

#' @export
fit_model.deepimp_backend_torch <- function(backend, model, train_x, train_y,
                                            valid_x, valid_y, control, ...) {
  arch <- model$arch
  glm <- NULL
  if (isTRUE(model$offset)) {
    glm <- .glm_fit(train_x, train_y, model$task, model$output_dim, arch$glm_ridge)
    train_x <- cbind(as.matrix(train_x), .glm_link(glm, train_x))
    valid_x <- cbind(as.matrix(valid_x), .glm_link(glm, valid_x))
  }
  # Training runs on the CPU. Imputation here is small tabular work that gains
  # nothing from a GPU/MPS device, MPS op-coverage is incomplete, and pinning to
  # CPU underpins the exact-reproducibility guarantee (dropout = 0).
  r <- .torch_train(model$module, train_x, train_y, valid_x, valid_y, model$task,
                    optimizer = arch$optimizer, lr = arch$learning_rate,
                    epochs     = if (is.null(control$epochs))     50L else control$epochs,
                    patience   = if (is.null(control$patience))   10L else control$patience,
                    batch_size = if (is.null(control$batch_size)) 32L else control$batch_size,
                    eval_init = isTRUE(model$offset),
                    verbose = isTRUE(control$verbose))
  list(model = list(fitted = r$module, task = model$task, glm = glm),
       history = list(epochs = r$epochs, best_epoch = r$best_epoch,
                      best_valid = r$best_valid))
}

#' @export
predict_model.deepimp_backend_torch <- function(backend, model, newx, ...) {
  x <- as.matrix(newx)
  if (!is.null(model$glm)) x <- cbind(x, .glm_link(model$glm, x))
  model$fitted$eval()
  raw <- torch::with_no_grad(
    model$fitted(torch::torch_tensor(x, dtype = torch::torch_float())))
  if (identical(model$task, "classification")) {
    probs <- torch::nnf_softmax(raw, dim = 2L)
    return(as.matrix(torch::as_array(probs)))
  }
  as.numeric(torch::as_array(raw))
}

utils::globalVariables("self")
