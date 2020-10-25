# Data Science Specialization

# Ranking hospitals by outcome in a state

## Reads the outcome-of-care-measures.csv file and returns a character vector 
## with the name of the required hospital: it can be the best, the worst or the 
## one specified with the num parameter.

rankhospital <- function(state, outcome, num = "best") {

  outcome_data <- read.csv('data/outcome-of-care-measures.csv', colClasses = 'character')
  
  outcome_data <- outcome_data[outcome_data$State == state, ]
  
  outcomes <- c('heart attack', 'heart failure', 'pneumonia')
  
  if (nrow(outcome_data) == 0 || !is.element(outcome, outcomes)) {
    stop('invalid outcome')
  }
  
  switch (outcome,
          
          'heart attack' = {
            
            suppressWarnings(
                outcome_data$Hospital.30.Day.Death..Mortality..Rates.from.Heart.Attack <- 
                as.numeric(outcome_data$Hospital.30.Day.Death..Mortality..Rates.from.Heart.Attack)
              )
            
            filter <- !is.na(outcome_data$Hospital.30.Day.Death..Mortality..Rates.from.Heart.Attack)
            
            outcome_data <- outcome_data[filter, ]
            
            outcome_data <- outcome_data[order(outcome_data$Hospital.30.Day.Death..Mortality..Rates.from.Heart.Attack,
                                               outcome_data$Hospital.Name), ]
            
          }, 
          
          'heart failure' = {
            
            suppressWarnings(
                outcome_data$Hospital.30.Day.Death..Mortality..Rates.from.Heart.Failure <- 
                as.numeric(outcome_data$Hospital.30.Day.Death..Mortality..Rates.from.Heart.Failure)
            )
            
            filter <- !is.na(outcome_data$Hospital.30.Day.Death..Mortality..Rates.from.Heart.Failure)
            
            outcome_data <- outcome_data[filter, ]
            
            outcome_data <- outcome_data[order(outcome_data$Hospital.30.Day.Death..Mortality..Rates.from.Heart.Failure,
                                               outcome_data$Hospital.Name), ]
            
          }, 
          
          'pneumonia' = {
            
            suppressWarnings(
                outcome_data$Hospital.30.Day.Death..Mortality..Rates.from.Pneumonia <- 
                as.numeric(outcome_data$Hospital.30.Day.Death..Mortality..Rates.from.Pneumonia)
            )
            
            filter <- !is.na(outcome_data$Hospital.30.Day.Death..Mortality..Rates.from.Pneumonia)
            
            outcome_data <- outcome_data[filter, ]
            
            outcome_data <- outcome_data[order(outcome_data$Hospital.30.Day.Death..Mortality..Rates.from.Pneumonia,
                                               outcome_data$Hospital.Name), ]
            
          }
  )
  
  if(num == 'best') return(head(outcome_data$Hospital.Name, 1))
  
  if(num == 'worst') return(tail(outcome_data$Hospital.Name, 1))
  
  if(is.numeric(num) && num <= nrow(outcome_data)) return(outcome_data$Hospital.Name[num])
  
  return(NA)
  
}
