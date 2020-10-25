# Data Science Specialization

# Ranking hospitals in all states

rankall <- function(outcome, num = "best") {
  
  source('rankhospital.R')

  outcome_data <- read.csv('data/outcome-of-care-measures.csv', colClasses = 'character')
  
  states <- unique(outcome_data$State)
  
  result <- data.frame(hospital = character(), state = character())
  
  for (state in states) {
    
    hname <- rankhospital(state, outcome, num)
    
    row <- data.frame('hospital' = hname, 'state' = state)
    
    result <- rbind(result, row)
  }
  
  result <- result[order(result$state), ]
  
  return(result)
}


