pollutantmean <- function(directory, pollutant, id = 1:332){
  
  sum <- 0
  count <- 0
  
  for (index in id){
    
    filename <- formatC(index, width = 3, flag = "0")
    
    path <- paste(getwd(), directory, paste(filename, 'csv', sep='.'), sep = '/')
    
    data <- read.csv(path, sep=',')
    
    if(!all(is.na(data[, pollutant]))){
      
      sum <- sum + sum(data[, pollutant], na.rm = TRUE)
      
      count <- count + sum(!is.na(data[, pollutant]))
    }
  }
  
  sum / count
}

complete <- function(directory, id = 1:332) {
  
  df <- data.frame(id=integer(0), nobs=integer(0))
  
  for (index in id){
    
    filename <- formatC(index, width = 3, flag = "0")
    
    path <- paste(getwd(), directory, paste(filename, 'csv', sep='.'), sep = '/')
    
    data <- read.csv(path, sep=',')
    
    
    df[nrow(df) +1, ] <- c(index, sum(complete.cases(data)))
    
  }
  
  df
}

corr <- function(directory, threshold = 0){
  
  correlations = c()
  
  for (file in list.files(paste(getwd(),directory,sep = '/'))) {
    
    data <- read.csv(paste(getwd(), directory, file, sep = '/'))
    
    data <- data[complete.cases(data), ]
    
    if(nrow(data) > threshold) {
      
      correlations <- c(correlations, cor(data['sulfate'], data['nitrate']))
    }
  }
  
  if(length(correlations) == 0){
    return(numeric(length = 0))
  }

  correlations
}


