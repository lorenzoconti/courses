# Data Science Specialization

# Finding the best hospital in a state

rm(list = ls())

## Reads the outcome-of-care-measures.csv file and returns a character vector 
## with the name of the hospital that has the best 30-day mortality for the
## specified outcome in that state.

best <- function(state, outcome) {
  
  
  outcome_data <- read.csv('data/outcome-of-care-measures.csv', colClasses = 'character')
  
  outcome_data <- outcome_data[outcome_data$State == state, ]
  
  outcomes <- c('heart attack', 'heart failure', 'pneumonia')
  
  if (nrow(outcome_data) == 0 || !is.element(outcome, outcomes)) {
    stop('invalid outcome')
  }
  
  switch (outcome,
          
    'heart attack' = {
      outcome_data$Hospital.30.Day.Death..Mortality..Rates.from.Heart.Attack <- 
        as.numeric(outcome_data$Hospital.30.Day.Death..Mortality..Rates.from.Heart.Attack)
      
      filter <- !is.na(outcome_data$Hospital.30.Day.Death..Mortality..Rates.from.Heart.Attack)
      
      outcome_data <- outcome_data[filter, ]
      
      outcome_data <- outcome_data[order(outcome_data$Hospital.30.Day.Death..Mortality..Rates.from.Heart.Attack,
                                         outcome_data$Hospital.Name), ]
      
    }, 
    
    'heart failure' = {
      outcome_data$Hospital.30.Day.Death..Mortality..Rates.from.Heart.Failure <- 
        as.numeric(outcome_data$Hospital.30.Day.Death..Mortality..Rates.from.Heart.Failure)
      
      filter <- !is.na(outcome_data$Hospital.30.Day.Death..Mortality..Rates.from.Heart.Failure)
      
      outcome_data <- outcome_data[filter, ]
      
      outcome_data <- outcome_data[order(outcome_data$Hospital.30.Day.Death..Mortality..Rates.from.Heart.Failure,
                                         outcome_data$Hospital.Name), ]
      
      
    }, 
    
    'pneumonia' = {
      outcome_data$Hospital.30.Day.Death..Mortality..Rates.from.Pneumonia <- 
        as.numeric(outcome_data$Hospital.30.Day.Death..Mortality..Rates.from.Pneumonia)
      
      filter <- !is.na(outcome_data$Hospital.30.Day.Death..Mortality..Rates.from.Pneumonia)
      
      outcome_data <- outcome_data[filter, ]
      
      outcome_data <- outcome_data[order(outcome_data$Hospital.30.Day.Death..Mortality..Rates.from.Pneumonia,
                                         outcome_data$Hospital.Name), ]
      

      
    }
  )
  
  
  return(outcome_data$Hospital.Name[1])
}


