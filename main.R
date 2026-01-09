#!/usr/bin/env -S Rscript --vanilla --verbose --slave=FALSE
.libPaths("C:/Users/KRosh/renv/library/windows/R-4.5/x86_64-w64-mingw32")

# Force interactive mode
options(pboptions = list(type = "txt", char = "=", txt.width = 50))

# Or explicitly set interactive
options(interactive = TRUE)


# renv::activate()
# renv::install(c('devtools', 'tidytext', 'tcltk', "tidyverse", "pdftools", "quanteda", "pbapply", "spacyr", "parallel", "here", "quanteda", "DBI", "RSQLite"))
# devtools::install_github("chris31415926535/tidyvader")

# renv::snapshot()
renv::restore()

library(tidytext)
library(tidyverse)
library(pdftools)
library(quanteda)
library(pbapply)
library(spacyr)
library(parallel)
library(here)
library(tcltk)
library(RSQLite)
library(DBI)
library(furrr)


### Features to Add ###----
# Pull company name and call name from transcript
# Add entity parsing for each sentence/segment & create entity map
# Break data into two tables (statements & call info)
# Create standing visualizations
#---

write_to_sqlite <- function(data, db_path = "data.db", append = FALSE) {
  # Connect to database
  con <- dbConnect(RSQLite::SQLite(), db_path)
  
  # Ensure connection closes even if error occurs
  on.exit(dbDisconnect(con))
  
  table_name <- readline("Enter table name: ")

  # Write data to table
  dbWriteTable(con, table_name, data, append = append, overwrite = !append)
  
  # Return success message
  print(paste0("Successfully wrote ", nrow(data), " rows to table:", table_name))
} # Function to interactively write to db

read_from_sqlite <- function(table_name) {
  con <- DBI::dbConnect(RSQLite::SQLite(), tk_choose.files(caption = "Select database file..."))
  on.exit(DBI::dbDisconnect(con))

  table_name <- select.list(DBI::dbListTables(con))

  from_sqlite <- dbReadTable(con, table_name)
} # Function to interactively import from db

### Operations ----

# Grab list of filepaths interactively --- Add feature to build sector/industry info interactively
source('helpers/create_transcript_list.R')

# Interactively select & ingest lists of key terms to search for and their associated topics, to label entries with
source('helpers/term_topic_ingest.R')

# Extract metadata from transcript text and file paths
source('helpers/get_metadata.R')

# Extract and format text from earnings calls
source('helpers/parse_transcript_text.R')

# Read in data to pick up where I left off
# results_final_from_sqlite <- read_from_sqlite()
# results_final_sentences_from_sqlite <- read_from_sqlite()

# Build features (currently sentiment & key term counts)
source('helpers/build_features.R')

# Clean up environment to prepare for visualization & analysis work
print("cleaning up environment")
rm(list = setdiff(ls(), c("results_final", "results_final_tidy", "results_final_sentences", "results_final_sentences_tidy", 'write_to_sqlite', 'read_from_sqlite')))
gc()

results_final <- results_final %>% 
  mutate(call_date = as.character(call_date))


#### Temp

# term_lookup <- read.csv('inputs/generated_regex.csv')

# results_joined <- results_final %>% 
#   pivot_longer(16:245) %>% 
#   left_join(term_lookup, join_by(name == term), relationship = 'many-to-many') %>% 
#   mutate(name = Key.term) %>% 
#   select()

#### Temp

## Write to file
print("writing to disk")
# Write to database file -- will create if doesn't exist
write_to_sqlite(results_final, "outputs/q4_2025_f100_analysis.db", append = FALSE)
write_to_sqlite(results_final_sentences, "outputs/q4_2025_f100_analysis.db", append = FALSE)
write_to_sqlite(results_final_tidy, "outputs/q4_2025_f100_analysis.db", append = FALSE)
write_to_sqlite(results_final_sentences_tidy, "outputs/q4_2025_f100_analysis.db", append = FALSE)
