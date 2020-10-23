add2 <- function(x,y) {
  x+ y
}

above10 <- function(x) {
  condition <- x > 10
  x[condition]  
}

above <- function(x,n=10) {
  condition <- x > n
  x[condition]
}

column_mean <- function(x, removeNA = TRUE) {
  nc <- ncol(x)
  means <- numeric(nc) # numeric emtpy vector with nc columns
  for(i in 1:nc){
    means[i] <- mean(x[,i], na.rm = removeNA)
  }
  means
}

# it is possible to pass ... as an argument to specify that those arguments will 
# be passed to another function
# 
# special_plot <- function(x,y, type="l", ...) {
#     plot(x,y, type = type, ...)
# }

