## The following functions are meant to save time-consuming computation (in this
## case, the computation of matrix's inverse). In order to do that, a special 
## object will cache the result of the computation, so that if the matrix 
## has not changed it is not necessary to compute the inverse another time and 
## it only need to retrieve the cached result.

# creates a matrix object that can cache its inverse

makeCacheMatrix <- function(x = matrix()) {
  
  i <- NULL
  
  set <- function(y) {
    x <<- y
    i <<- NULL
  }
  
  get <- function() x
  
  set_inverse <- function(inverse) i <<- inverse
  
  get_inverse <- function() i
  
  list(set = set,
       get = get,
       set_inverse = set_inverse,
       get_inverse = get_inverse)
}


# computes the inverse of a special matrix: if the inverse has already been
# calculated (and the matrix has not changed), then it retrieves the inverse
# from the cache

cacheSolve <- function(x, ...) {
  
  i <- x$get_inverse()
  
  if (!is.null(i)) {
    return(i)
  }
  
  data <- x$get()
  
  i <- solve(data, ...)
  
  x$set_inverse(i)
  
  i
}