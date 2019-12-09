#' @title Guessing the variable type
#' @rdname guessType
#' @description guess the type of the variables in a data set, e.g. numeric, nominal, binary and semi-continous.
#' @details the scale of the variables plays a central role for imputing mixed type of variables. 
#' The usual data types in R, such as factor, numeric, ..., are extended. Internally, the algorithm relies on the 
#' following types of variables: numeric (equals numeric in base R), mixed (semi-continuous variables), 
#' binary (nominal variables with two categories), nominal (like factor, but excluding binary variables), 
#' count (for count variables).
#' 
#' @param x a data.frame, data.table or tibble 
#' @return a list of two elements with information on the guessed variable types
#' @author Matthias Templ
#' @keywords manip
#' @export
#' @examples
#' 
#' data(sleep, package = "VIM")
#' guessType(sleep)
guessType <- function(x){
  type <- as.character(unlist(lapply(x, function(x) class(x)[1])))
  indexNumeric <- type == "numeric" | type =="integer"
  indexNominal <- type == "factor"
  indexOrdered <- type == "ordered"
  indexMixed <- rep(FALSE, length(type))
  indexBinary <- rep(FALSE, length(type))
  
  if(any(type == "character")){
    indexNominal <- type == "character" | indexNominal
  }
  
  checkMixed <- function(x, constant = 0.2){
    if(any(table(x) > constant * length(x)) & length(table(x)) > length(x) * 1/10){
      return(TRUE)
    } else {
      return(FALSE)
    }
  }
  
  checkNominal <- function(x, constant = 0.2){
    if(length(table(x)) < length(x) * 1/10 & !any(x[!is.na(x)] %% 1 > .Machine$double.eps)){
      return(TRUE)
    } else {
      return(FALSE)
    }
  }
  
  checkBinary <- function(x){
    if(length(table(x)) == 2){
      return(TRUE)
    } else {
      return(FALSE)
    }
  }
  
  checkMixedConst <- function(x, constant = 0.2){
    return(as.numeric(names(which(table(x) > constant * length(x)))))
  }
  
  # if numeric
  #   is it binary?
  #   is it nominal?
  #   is it mixed?
  # if nominal
  #   is it binary?
  
  if(any(indexNumeric)){ 
    for(i in which(indexNumeric)){
      if(checkBinary(x[, i])){
        indexBinary[i] <- TRUE
        indexNumeric[i] <- FALSE    
      }
      if(checkNominal(x[, i]) & !checkBinary(x[, i])){
        indexNominal[i] <- TRUE  
        indexNumeric[i] <- FALSE   
      }
      if(checkMixed(x[, i] & !checkBinary(x[, i]) & !checkNominal(x[, i]))){
        indexMixed[i] <- TRUE  
        indexNumeric[i] <- FALSE           
      }
    }
  }
  
  if(any(indexNominal)){
    for(i in which(indexNominal)){   
      if(checkBinary(x[, i])){
        indexBinary[i] <- TRUE
        indexNominal[i] <- FALSE    
      }      
    }
  }
  
  m <- rbind(indexBinary, indexNominal, indexOrdered, indexNumeric, indexMixed)
  colnames(m) <- colnames(x)
  type <- numeric(ncol(x))
  for(i in 1:ncol(x)){
    type[i] <- names(which(m[, i]))
  }
  type <- tolower(substr(type, 6, nchar(type)))
  res <- list("indices" = m,
              "type" = type)
  return(res)
}