#' @title Imputation using neural networks
#' @aliases impNNet guessType
#' @rdname impNNet
#' @description Imputation for numeric, nominal, binary and semi-continous variables 
#' using neural networks.
#' @details the scale of the variables plays a central role within the algorithm. 
#' The usual data types in R, such as factor, numeric, ..., are extended. Internally, the algorithm relies on the 
#' following types of variables: numeric (equals numeric in base R), mixed (semi-continuous variables), 
#' binary (nominal variables with two categories), nominal (like factor, but excluding binary variables), 
#' count (for count variables).
#' If vartypes equals guess, then the variable scale types are estimated. 
#' If verbose equals TRUE, the outcome of this estimation is printed to the console.
#' 
#' @param x data.frame or matrix
#' @param vartypes specify the scale of each variable as a vector. For further details see below.
#' @param initialize initialization of missing values
#' @param iterations number of iterations for the whole chain
#' @param eps if mean changes over cells is less than eps, the algorithmn will stopp
#' @param patience.val max number of epochs on which we would like to see no improvement
#' @param epochs epochs before training is stopped. Allow training to continue for up to an additional number of epochs to allow the training process to get across flat spots and to possible find some additional improvement.
#' @param dropout fraction rate of input units in each layer set to 0 to prevent overfitting. Dropout is only applied in the first layers.
#' @param dense number of layers (length of vector dense) and number of units in each layer. Currently this is fixed to the default and will
#' be made available in future versions.
#' @param optimizer the optimizer used, currently fixed to adam.
#' @param loss loss measure.
#' @param model_numeric if NULL, a predefined model for numeric variables is chosen. For the specification of a model, see 
#' \code{\link[keras]{keras_model_sequential}}.
#' @param model_binary if NULL, a predefined model for numeric variables is chosen. For the specification of a model, see 
#' \code{\link[keras]{keras_model_sequential}}.
#' @param model_nominal if NULL, a predefined model for numeric variables is chosen. For the specification of a model, see 
#' \code{\link[keras]{keras_model_sequential}}.
#' @param mi number of multiple imputations
#' @param verbose progress and additional information to be reported in the console (if TRUE)
#' @import VIM
#' @import data.table
#' @importFrom keras callback_lambda keras_model_sequential layer_dense layer_batch_normalization layer_dropout compile optimizer_adam 
#' @importFrom keras callback_early_stopping to_categorical fit %<-%
#' @importFrom magrittr %>%
#' @importFrom VIM hotdeck kNN
#' @return the imputed data set
#' @author Matthias Templ, Roberto Barbieri
#' @keywords manip
#' @examples
#' 
#' data(sleep, package = "VIM")
#' \dontrun{
#' sleep_imp <- impNNet(sleep)
#' sleep_imp
#' }
#' 
#' @export impNNet
impNNet <- function(x, vartypes = "guess", initialize = "knn", 
                    iterations = 2, eps = 1,
                    patience.val = 5, epochs = 10, 
                    dropout = 0.1, dense = seq(1000, 100, -100), 
                    optimizer = optimizer_adam(), loss = "mse",
                    model_numeric = NULL,
                    model_binary = NULL,
                    model_nominal = NULL,
                    mi = 1, verbose = TRUE){
  ## check speed 
  ptm <- proc.time()
  ## check on parameters:
  if(!(class(x) %in% c("data.frame","data.table","tibble","tbl_df"))){
    stop(cat("class or x, ", class(x), " is not supported"))
  }
  if(!(initialize %in% c("kNN","knn","mean","hotdeck"))){
    stop(cat("initialing method ", initialize, "is not supported"))
  }
  
  ## guess variable scale/type
  if(vartypes == "guess"){
    vartypes <- guessType(x)$type
    if(verbose) cat("\ntype of variables for further use\n", vartypes)
  } else {
    stopifnot(!is.vector(guessType) & !(length(vartypes) == ncol(x)))
  }
  
  ## binary vectors must be stored as 0/1
  bin_index <- vartypes == "binary"
  if(any(bin_index)){
    for(i in which(bin_index)){
      if(is.factor(x[, i])){
        st <- na.omit(levels(x[, i])) %in% c(0,1)       
      } else{
        st <- unique(na.omit(x[, i])) %in% c(0,1)       
      }

      if(!all(st)) stop("binary vectors must be with values 0 and 1")
    }
  }
  
  # for the callback, print info about status
  print_dot_callback <- callback_lambda(
    on_epoch_end = function(epoch, logs) {
      if(verbose){
        if (epoch %% 80 == 0) cat("\n   number of epochs (out of ", epochs, ") ")
        cat(".")
      }
    }
  )
  
  ### Remembering where NA' s are ###
  no.loc.mat <- is.na(x)
  
  ### initializing NA's ###
  {
    if(initialize == "knn" | initialize == "kNN"){
      x.filled <- kNN(x, imp_var = FALSE)
      x.imp <- x.filled # dataset to fill in fitted values
    }
    if(initialize == "mean"){
      m <- 1
      xmean <- x
      while(m <= ncol(xmean)){
        xmean[is.na(xmean[,m]),m] <- mean(xmean[,m], na.rm=TRUE)
        m <- m +1
      }
      # round binary and Categoric variables
      to.round <- c(grep("Bin",colnames(xmean)),
                    grep("Cat",colnames(xmean)),
                    grep("factor",colnames(xmean)),
                    grep("Count",colnames(xmean)))
      xmean[,to.round] <- round(xmean[,to.round])
      x.filled <- xmean
      x.imp <- x.filled # dataset to fill in fitted values
    }
    # if(initialize == "pmm"){
    #   x.filled <- complete(mice(data = x, method = "pmm", m = 1))
    #   x.imp <- x.filled # dataset to fill in fitted values
    # }
    if(initialize == "hotdeck"){
      x.filled <- hotdeck(x, imp_var = FALSE)      
      x.imp <- x.filled # dataset to fill in fitted values
    }
  }
  
  ### Models
  if(is.null(model_numeric)){
  build_model_numeric <- function() {
    
    # model <- keras_model_sequential() %>%
    #   layer_dense(units = 1000, activation = "relu",
    #               input_shape = dim(train_data)[2]) %>%
    #   layer_batch_normalization() %>%
    #   layer_dense(units = 900, activation = "relu") %>%
    #   layer_dropout(dropout) %>%
    #   layer_batch_normalization() %>%
    #   layer_dense(units = 800, activation = "relu") %>%
    #   layer_dropout(dropout) %>%
    #   layer_batch_normalization() %>%
    #   layer_dense(units = 700, activation = "relu") %>%
    #   layer_dropout(dropout) %>%
    #   layer_batch_normalization() %>%
    #   layer_dense(units = 600, activation = "relu") %>%
    #   layer_dropout(dropout) %>%
    #   layer_batch_normalization() %>%
    #   layer_dense(units = 500, activation = "relu") %>%
    #   layer_dropout(dropout) %>%
    #   layer_batch_normalization() %>%
    #   layer_dense(units = 400, activation = "relu") %>%
    #   layer_batch_normalization() %>%
    #   layer_dense(units = 300, activation = "relu") %>%
    #   layer_batch_normalization() %>%
    #   layer_dense(units = 200, activation = "relu") %>%
    #   layer_batch_normalization() %>%
    #   layer_dense(units = 100, activation = "relu") %>%
    #   layer_dense(units = 1,  activation = "linear") 
   
    
   model <- keras_model_sequential() %>%
     layer_dense(units = dense[1], activation = "relu",
                        input_shape = dim(train_data)[2]) %>% 
     layer_dropout(dropout) %>%
     layer_batch_normalization()%>%
     layer_dense(units = dense[2], activation = "relu") %>%
     layer_dropout(dropout) %>%
     layer_batch_normalization() %>%
     layer_dense(units = dense[3], activation = "relu") %>%
     layer_dropout(dropout) %>%
     layer_batch_normalization()  %>%  
     layer_dense(units = dense[4], activation = "relu") %>%
     layer_dropout(dropout) %>%
     layer_batch_normalization()  %>%  
     layer_dense(units = dense[5], activation = "relu") %>%
     layer_dropout(dropout) %>%
     layer_batch_normalization() %>%
     layer_dense(units = dense[6], activation = "relu") %>%
     layer_batch_normalization() %>%
     layer_dense(units = dense[7], activation = "relu") %>%
     layer_batch_normalization()  %>%  
     layer_dense(units = dense[8], activation = "relu") %>%
     layer_batch_normalization() %>%    
     layer_dense(units = dense[9], activation = "relu") %>%
     layer_batch_normalization()  %>%
     layer_dense(units = dense[10], activation = "relu") %>%
     layer_batch_normalization()  %>% 
     layer_dense(units = 1,  activation = "linear") 
   
    model %>% compile(
      loss = loss,
      optimizer = optimizer_adam(),
      metrics = list("mean_absolute_error")
    )
  }
  }
  
  if(is.null(model_binary)){
  build_model_binary <- function() {
    
    model <- keras_model_sequential() %>%
      layer_dense(units = 1000, activation = "relu",
                  input_shape = dim(train_data)[2]) %>%
      layer_dropout(dropout) %>%
      layer_batch_normalization() %>%
      layer_dense(units = 900, activation = "relu") %>%
      layer_dropout(dropout) %>%
      layer_batch_normalization() %>%
      layer_dense(units = 800, activation = "relu") %>%
      layer_dropout(dropout) %>%
      layer_batch_normalization() %>%
      layer_dense(units = 700, activation = "relu") %>%
      layer_dropout(dropout) %>%
      layer_batch_normalization() %>%
      layer_dense(units = 600, activation = "relu") %>%
      layer_dropout(dropout) %>%
      layer_batch_normalization() %>%
      layer_dense(units = 500, activation = "relu") %>%
      layer_dropout(dropout) %>%
      layer_batch_normalization() %>%
      layer_dense(units = 400, activation = "relu") %>%
      layer_batch_normalization() %>%
      layer_dense(units = 300, activation = "relu") %>%
      layer_batch_normalization() %>%
      layer_dense(units = 200, activation = "relu") %>%
      layer_batch_normalization() %>%
      layer_dense(units = 100, activation = "relu") %>%
      layer_dense(units = 2,  activation = "softmax")
    
    model %>% compile(
      loss = "categorical_crossentropy",
      optimizer = optimizer_adam(),
      metrics = list("accuracy")
    )
  }
  }
  
  if(is.null(model_nominal)){
  build_model_nominal <- function() {
    
    model <- keras_model_sequential() %>%
      layer_dense(units = 1000, activation = "relu",
                  input_shape = dim(train_data)[2]) %>%
      layer_dropout(dropout) %>%
      layer_batch_normalization() %>%
      layer_dense(units = 900, activation = "relu") %>%
      layer_dropout(dropout) %>%
      layer_batch_normalization() %>%
      layer_dense(units = 800, activation = "relu") %>%
      layer_dropout(dropout) %>%
      layer_batch_normalization() %>%
      layer_dense(units = 700, activation = "relu") %>%
      layer_dropout(dropout) %>%
      layer_batch_normalization() %>%
      layer_dense(units = 600, activation = "relu") %>%
      layer_dropout(dropout) %>%
      layer_batch_normalization() %>%
      layer_dense(units = 500, activation = "relu") %>%
      layer_dropout(dropout) %>%
      layer_batch_normalization() %>%
      layer_dense(units = 400, activation = "relu") %>%
      layer_batch_normalization() %>%
      layer_dense(units = 300, activation = "relu") %>%
      layer_batch_normalization() %>%
      layer_dense(units = 200, activation = "relu") %>%
      layer_batch_normalization() %>%
      layer_dense(units = 100, activation = "relu") %>%
      #layer_dense(units = ncol(x.data$train$y),  activation = "softmax")
      layer_dense(units = length(levels(train_labels)),  activation = "softmax")
  #    layer_dense(units = length(levels(train_labels)),  activation = "softmax")
   #   layer_dense(units = ncol(train_labels),  activation = "softmax")
    
    model %>% compile(
      loss = "categorical_crossentropy",
      optimizer = optimizer_adam(),
      metrics = list("accuracy")
    )
  }
  }
  
  # setting limits for "runtime"
  early_stop <- callback_early_stopping(monitor = "val_loss", patience = patience.val)
  
  x.imp.start <- x.filled
  ### Loop over mi
  results <- list()
  for(mis in 1:mi){
  if(verbose & mi > 1){
    cat("\n\n---")
    cat("\nmultiple imputation", mis, "\n---")
  }
  x.imp <- x.filled <- x.imp.start
  ### Loop over iterations
  changes <- numeric(iterations)
  for(iter in 1:iterations){
    x.imp.old <- x.imp
    if(verbose) cat("\n\niteration (", iter, "/", iterations,")\n")
  ### Loop over columns ###
  {
    for(j in 1:ncol(x)){
      if(sum(is.na(x[, j])) > 0){
        if(verbose) cat("\n  imputing variable ", j,"/",ncol(x))

          train_data <- x.filled[!no.loc.mat[, j], -j]
          train_labels <- x.filled[!no.loc.mat[,j], j]
          test_data <- x.filled[no.loc.mat[,j], -j]
          test_labels <- x.filled[no.loc.mat[,j], j] 
          
          # Normalize training data (only non-factors)
          facs <- vartypes[-j] %in% c("factor","nominal","binary")
          trainscale <- scale(train_data[, !facs])
          train_data[, !facs] <- trainscale
          
          # Use means and standard deviations from training set to normalize test set
          col_means_train <- attr(trainscale, "scaled:center") 
          col_stddevs_train <- attr(trainscale, "scaled:scale")
          test_data[, !facs] <- scale(test_data[, !facs], 
                                      center = col_means_train, 
                                      scale = col_stddevs_train)
          
          # build design matrix for factors
          if(any(facs)){
            vars <- (1:(ncol(train_data)))[facs]
            nc <- max(c(unique(train_data[,vars[1]])),
                      c(unique(test_data[,vars[1]]))) #+ 1
            td <- data.frame("labels" = train_data[,vars[1]])
            td2 <- data.frame("labels" = test_data[,vars[1]])
#            binmat_train <- to_categorical(train_data[,vars[1]], num_classes = nc)
#            binmat_test <- to_categorical(test_data[,vars[1]], num_classes = nc)   
            binmat_train <- model.matrix(~labels-1, data = td)
            binmat_test <- model.matrix(~labels-1, data = td2)
            if(sum(facs) > 1){
              for(i in (1:ncol(train_data))[facs]){
                nc <- max(c(length(unique(train_data[,i])),
                            length(unique(test_data[,i])))) + 1
#                binmat_train <- cbind(binmat_train, 
#                                      to_categorical(train_data[,i], num_classes = nc))
#                binmat_test <- cbind(binmat_test, 
#                                     to_categorical(test_data[,i], num_classes = nc))
                tdi <- data.frame("labels" = train_data[,i])
                tdi2 <- data.frame("labels" = test_data[,i])
                binmat_train <- cbind(binmat_train, 
                                      model.matrix(~labels-1, data = tdi))
                binmat_test <- cbind(binmat_test, 
                                      model.matrix(~labels-1, data = tdi2))               
              }
            }
            train_data <- train_data[, !facs]
            browser()
            train_data <- cbind(train_data, binmat_train)
            test_data <- test_data[, !facs]
            test_data <- cbind(test_data, binmat_test)            
          }
        
         if(vartypes[j] %in% c("numeric","count")){   
           # setting limits for "runtime"
           early_stop <- callback_early_stopping(monitor = "val_loss", patience = patience.val)
           if(exists("model")){
             rm(model)
           }
           model <- build_model_numeric()

          history <- model %>% fit(
            as.matrix(train_data),
            train_labels,
            epochs = epochs,
            validation_split = 0.2,
            verbose = 0,
            callbacks = list(early_stop, print_dot_callback)
          )
          
          test_predictions <- model %>% predict(as.matrix(test_data))

          x.imp[(no.loc.mat[,j]),j] <- test_predictions
          if(vartypes[j] == "count"){
            x.imp[(no.loc.mat[,j]),j] <- round(x.imp[(no.loc.mat[,j]),j])
          }

          x.filled <- x.imp
        }
 
          
        if(vartypes[j] %in% c("nominal","binary")){   
         # if(vartypes[j] == "binary") browser()
            
            # setting limits for "runtime"
            early_stop <- callback_early_stopping(monitor = "val_loss", patience = patience.val)
            
            # building actual model
            if(exists("model")){
              rm(model)}
            if(vartypes[j] == "nominal"){
              model <- build_model_nominal()
              # fitting on model
              #nc <- max(c(ncol(to_categorical(train_labels)), 
              #            ncol(to_categorical(train_labels)))) 
              #  to_categorical(train_labels, num_classes = nc)
              tm <- data.frame("labels" = train_labels)
              history <- model %>% fit(
                as.matrix(train_data),
                #to_categorical(train_labels, num_classes = nc),
                model.matrix(~ labels-1, data = tm),
                epochs = epochs,
                validation_split = 0.2,
                verbose = 0,
                callbacks = list(early_stop, print_dot_callback)
              )
            } else {
              model <- build_model_binary()  
              # fitting on model
              nc <- 2 
              history <- model %>% fit(
                as.matrix(train_data),
                to_categorical(train_labels, num_classes = nc),
                epochs = epochs,
                validation_split = 0.2,
                verbose = 0,
                callbacks = list(early_stop, print_dot_callback)
              )
            }

            
            test_predictions <- model %>% predict(as.matrix(test_data))
            if(vartypes[j] == "binary"){
              x.imp[(no.loc.mat[,j]),j] <- round(test_predictions)[,2]
            }
            if(vartypes[j] == "nominal"){
              bestlev <- apply(test_predictions,1,which.max)
              bestlev <- levels(train_labels)[bestlev]
              x.imp[(no.loc.mat[,j]),j] <- bestlev
#              x.imp[(no.loc.mat[,j]),j] <- apply(test_predictions,1,which.max)
            }
            x.filled <- x.imp
          }
    
      } 
     } # end check if na in variable
    } # end loop over all columns
    
    changes[iter] <- improvement(x.imp.old, x.imp, no.loc.mat)

    if(verbose) cat("\nmean of relative differences to previous iteration: ", changes[iter])
    if(changes[iter] < eps & iter < iterations){
      if(verbose) cat("\n\nconverged after", iter,"iterations (< eps), \n skipping remaining iterations\n")
      break()
    }
   } # end loop over iter
   results[[mis]] <- x.imp
  } # end loop over mi
  if(mi == 1){
    results <- data.frame(results[[1]])
  }
  p <- proc.time() - ptm
  attr(p[1], which = "proctime")
  return(results)
  
}
improvement <- function(x, y, na.orig){
  facs <- unlist(lapply(x, class)) %in% c("factor","nominal","binary")
  s_num <- sum(abs(x[, !facs] - y[, !facs])) 
  if(any(facs)){
    s_fac <- sum(x[, facs] == y[, facs])
  } else {
    s_fac <- 0
  }
  change <- (s_num + s_fac) / sum(na.orig)
  return(change)
}
