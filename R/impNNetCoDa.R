#' @title Imputation using neural networks
#' @rdname impNNetCoda
#' @description Imputation of rounded zeros in compositional data 
#' using neural networks.
#' @details tbd
#' @param x data.frame or matrix
#' @param initialize initialization of missing values
#' @param dl detection limit as vector of length ncol(x)
#' @param label label of the rounded zeros in x (default: NA)
#' @param iterations number of iterations for the whole chain
#' @param eps if mean changes over cells is less than eps, the algorithmn will stopp
#' @param patience.val max number of epochs on which we would like to see no improvement
#' @param epochs epochs before training is stopped. Allow training to continue for up to an additional number of epochs to allow the training process to get across flat spots and to possible find some additional improvement.
#' @param dropout fraction rate of input units in each layer set to 0 to prevent overfitting. Dropout is only applied in the first layers.
#' @param dense number of layers (length of vector dense) and number of units in each layer. Currently this is fixed to the default and will
#' be made available in future versions.
#' @param optimizer the optimizer used, currently fixed to adam.
#' @param loss loss measure.
#' @param correction currently not implemented but foreseen to be included
#' @param verbose progress and additional information to be reported in the console (if TRUE)
#' @importFrom keras callback_lambda keras_model_sequential layer_dense layer_batch_normalization layer_dropout compile optimizer_adam 
#' @importFrom keras callback_early_stopping to_categorical fit %<-%
#' @importFrom robCompositions impKNNa
#' @importFrom magrittr %>%
# importFrom robCompositions impKNNa pivotCoord pivotCoordInv adjustImputed
# importFrom robCompositions expenditures
#' @return the imputed data set
#' @author Matthias Templ, Roberto Barbieri
#' @keywords manip
#' @export
#' @examples
#' 
#' expenditures <- data.frame(
#' "housing" = c(640,1800,2085, 616, 875, 770, 990, 414,
#' 1394,1285,1102,1717,1549, 838, 845,1130,1765,1195,2180,1017),
#' "foodstuffs" = c(328, 484, 445, 331, 368, 364, 415, 305, 
#' 440, 374, 469, 452, 454, 386, 386, 394, 466, 443, 521, 410),
#' "alcohol" = c(147, 515, 725, 126, 191, 196, 284,  94, 393, 
#' 363, 243, 452, 424, 155, 211, 271, 524, 329, 553, 225),
#' "other" = c(169,2291,8373, 117, 290, 242, 588,  68,1161, 
#' 785, 496,1977,1345, 208, 317, 490,2133, 974,2781, 419),
#' "services" = c(196, 912,1732, 149, 275, 236, 420, 112, 
#' 636, 487, 388, 832, 676, 222, 280, 386, 822, 523,1010, 345)
#' )
#' qu <- apply(expenditures, 2, quantile, probs = 0.1)
#' expenditures[expenditures[, 1] < qu[1], 1] <- NA
#' expenditures[expenditures[, 2] < qu[2], 2] <- NA
#' expenditures[expenditures[, 3] < qu[3], 3] <- NA
#' expenditures[expenditures[, 4] < qu[4], 4] <- NA
#' 
#' \dontrun{
#' library(keras)
#' library(tensorflow)
#' expenditures_imp <- impNNetCoDa(expenditures)
#' expenditures_imp
#' }
#' 
impNNetCoDa <- function(x, initialize = "kNNa", 
                    coda = TRUE,
                    dl = NULL, 
                    label = NA,
                    iterations = 2, 
                    eps = 1,
                    patience.val = 5, 
                    epochs = 10, 
                    dropout = 0.1, 
                    dense = seq(1000, 100, 100), 
                    optimizer = optimizer_adam(), 
                    loss = "mse",
                    correction = FALSE, 
                    verbose = TRUE){
  ## check on parameters:
  if(!(class(x) %in% c("data.frame","data.table","tibble","tbl_df"))){
    stop(cat("class or x, ", class(x), " is not supported"))
  }
  if(!(initialize %in% c("kNN","kNNa","kNNA","impKNNa","knna","knn","mean","hotdeck","em"))){
    stop(cat("initialing method ", initialize, "is not supported"))
  }
  
  # for the callback, print info about status
  print_dot_callback <- callback_lambda(
    on_epoch_end = function(epoch, logs) {
      if(verbose){
        if (epoch %% 80 == 0) cat("\n number of epochs (out of ", epochs, ") ")
        cat(".")
      }
    }
  )
  
  ### Remembering where NA's are ###
  if(!is.na(label)){
    x[x == label] <- NA
  }
  no.loc.mat <- is.na(x)
  
  ### initializing NA's ###
  {
    if(initialize == "knn" | initialize == "kNN"){
      x.filled <- kNN(x, imp_var = FALSE)
      x_imp <- x.filled # dataset to fill in fitted values
    }
    if(initialize == "knna" | initialize == "kNNa"){
      x.filled <- robCompositions::impKNNa(x)$xImp
      x_imp <- x.filled # dataset to fill in fitted values
    }
    if(initialize == "em"){
      suppressWarnings(x.imp <- tryCatch(lrEM(x, 
                                              dl=qu, 
                                              label=0), 
                                         error = function(e) m))
      
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
      x_imp <- x.filled # dataset to fill in fitted values
    }
    # if(initialize == "pmm"){
    #   x.filled <- complete(mice(data = x, method = "pmm", m = 1))
    #   x.imp <- x.filled # dataset to fill in fitted values
    # }
    if(initialize == "hotdeck"){
      x.filled <- hotdeck(x, imp_var = FALSE)      
      x_imp <- x.filled # dataset to fill in fitted values
    }
  }
  
  ### Models
  build_model_numeric <- function() {
    
    model <- keras_model_sequential() %>%
      layer_dense(units = 1000, activation = "relu",
                  input_shape = dim(train_data)[2]) %>%
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
      layer_dense(units = 1,  activation = "linear")
    
    model %>% compile(
      loss = loss,
      optimizer = optimizer_adam(),
      metrics = list("mean_absolute_error")
    )
  }
  
  n <- nrow(x)
  x_orig <- x
  
  # setting limits for "runtime"
  early_stop <- callback_early_stopping(monitor = "val_loss", patience = patience.val)
  
  ### Loop over iterations (outer loop) ###
  
  changes <- numeric(iterations)
  for(iter in 1:iterations){
    x.imp.old <- x_imp
    if(verbose) cat("\n\niteration (", iter, "/", iterations,")\n")
    
  ### Loop over columns (inner loop) ###
  {
    for(j in 1:ncol(x)){
      if(sum(is.na(x[, j])) > 0){
        if(verbose) cat("\nimputing variable ", j,"/",ncol(x))

        # Apply pivot coord trans
        x_neworder <- cbind(x_imp[, j], x_imp[, -j])
        x_neworder[x_neworder < 2 * .Machine$double.eps] <- 2 * 
          .Machine$double.eps
        if(coda){
          x_ilr <- robCompositions::pivotCoord(x_neworder)
          phi_ilr <- robCompositions::pivotCoord(cbind(rep(dl[j], n), x_neworder[, -1, drop = FALSE]))[, 1]
          if (any(is.na(phi_ilr))){
            stop("a phi_ilr value is NA, please check your data.")
          } 
        } else {
          x_ilr <- x_neworder
          phi_ilr <- rep(dl[j], n)
        }
        train_data <- as.matrix(x_ilr[!no.loc.mat[, j], -1, drop = FALSE])
        train_labels <- as.matrix(x_ilr[!no.loc.mat[, j], 1, drop = FALSE])
        test_data <- as.matrix(x_ilr[no.loc.mat[, j], -1, drop = FALSE])
        
        early_stop <- callback_early_stopping(monitor = "val_loss", patience = patience.val)
        
        if(exists("model")){
          rm(model)
        }
# 
#         train_data <- x.filled[!no.loc.mat[, j], -j]
#           train_labels <- x.filled[!no.loc.mat[,j], j]
#           test_data <- x.filled[no.loc.mat[,j], -j]
#           test_labels <- x.filled[no.loc.mat[,j], j] 

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
          
          # if (correction == "density") {
          #   den <- density(ex[w[, i]])
          #   ##PF# distr <- sROC::kCDF(ex[w, i])
          #   ##distr <- sROC::kCDF(ex[w[, i]])
          # }
          if (any(is.na(test_predictions)) || any(test_predictions == "Inf")) 
            stop("Problems in ilr because of infinite or NA estimates")
          if(correction == "simple"){
            x_ilr[(no.loc.mat[,j]), 1] <- ifelse(test_predictions > phi_ilr[no.loc.mat[, j]], phi_ilr[no.loc.mat[, j]] - 0.001, test_predictions)
          } else{
            s <- sd(x_ilr[, 1])
            num <- dnorm(phi_ilr[no.loc.mat[, j]] - test_predictions / s)
            denum <- pnorm(phi_ilr[no.loc.mat[, j]] - test_predictions / s) 
            lambda <-  num / denum
            x_ilr[(no.loc.mat[,j]), 1] <- 
              test_predictions - s * lambda
          }
          if(coda){
          x_ilr_back <- robCompositions::pivotCoordInv(x_ilr)
          x_ilr_back <- robCompositions::adjustImputed(x_ilr_back, x_neworder, 
                                                       cbind(no.loc.mat[, j], 
                                                             no.loc.mat[, -j]))
          x_imp[(no.loc.mat[,j]), j] <- x_ilr_back[(no.loc.mat[,j]), 1]
          } else{
            x_imp[(no.loc.mat[,j]), j] <- x_ilr[(no.loc.mat[,j]), 1]
          }
 #         x_imp <- setToDL(x_imp, dl, no.loc.mat)
       #   print(colnames(x_imp))
       #   print(checkToDL(x_imp, dl, no.loc.mat, j))
          
          x.filled <- x_imp
        }

      } 
     } # end loop over all columns

    changes[iter] <- improvementCoDa(x.imp.old, x_imp, no.loc.mat)

    if(verbose) cat("\nmean of relative differences to previous iteration: ", changes[iter])
    if(changes[iter] < eps & iter < iterations){
      if(verbose) cat("\n\nconverged after", iter,"iterations (< eps), \n skipping remaining iterations\n")
      break()
    }
   } # end loop over iter
  
  return(x_imp)
  
}

improvementCoDa <- function(x, y, na.orig){
  change <- sum(abs(x - y)) / sum(na.orig)
  return(change)
}

setToDL <- function(x, dl, indexNA){
  check <- logical(ncol(x))
  su <- 0
  for(i in 1:ncol(x)){
    critvals <- x[indexNA[,i],i] > dl[i]
    x[critvales, i] <- dl[i]
  }
  return(x)
}

checkToDL <- function(x, dl, indexNA, j){
  check <- logical(ncol(x))
  critvals <- x[indexNA[,j],j] > dl[j]
  su <- sum(critvals)
  return(ifelse(su == 0, "ok", "values above dl"))
}
