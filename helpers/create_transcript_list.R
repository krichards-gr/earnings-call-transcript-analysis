# renv::install(c('tcltk', 'dplyr'))

library(dplyr)
library(tcltk)

create_transcript_list <- function() {
  top_directory <- tcltk::tk_choose.dir(default = "C:/Users/KRosh/OneDrive - Bradley Media Holdings, Inc/Content",
caption = "Select the top-level folder that conatins all transcripts to process")
  
  transcript_list <- list.files(top_directory, pattern = '\\.pdf$',
  full.names = TRUE,
  recursive = TRUE,
  ignore.case = TRUE)

  print(paste0('Ingested ', length(transcript_list), ' filepaths'))

  return(transcript_list)
}

list_of_earnings_calls <- create_transcript_list()
