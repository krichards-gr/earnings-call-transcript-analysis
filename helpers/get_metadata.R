# renv::install(c('stringr', 'dplyr', 'tibble', 'tcltk'))

library(stringr)
library(dplyr)
library(tibble)
library(tcltk)
library(purrr)


get_earnings_date <- function(file) {
  pattern <- "(?i)\\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)?(?:,\\s*)?(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\\s*\\d{1,2}(?:st|nd|rd|th)?,?\\s*\\d{4}(?:\\s*\\d{1,2}:\\d{2}\\s*[APM]{2})?\\b|\\b\\d{1,2}-[A-Za-z]{3}-\\d{4}\\b"
  # print(file)
  first_page <- pdf_text(file)[1]
  # print(first_page)
  date_text <- first_page %>%
    stringr::str_split("\n") %>%
    unlist() %>%
    stringr::str_extract(pattern) %>%
    na.omit()
    
  # Remove ordinal suffix and ensure it's a clean character string
  clean_date_text <- date_text %>%
    stringr::str_remove_all("(?<=\\d)(st|nd|rd|th)") %>%
    first() %>% 
    as.character()
  
  date <- parse_date_time(clean_date_text, orders = c("B d, Y HM", "mdy HM", "ymd HM", "d-b-Y", "b d, Y HM", "B d, Y"))
    
  return(date)
} # Function to extract dates from earnings calls

setup_paths <- function(sample_file) {
  parts <- unlist(strsplit(sample_file, "/"))
  parts <- parts[parts != ""] # remove empty strings

  # Show numbered options
  numbered_parts <- paste(1:length(parts), parts, sep = ": ")

  sector_choice <- tk_select.list(
    numbered_parts,
    title = "Pick SECTOR position"
  )
  industry_choice <- tk_select.list(
    numbered_parts,
    title = "Pick INDUSTRY position"
  )

  # Extract the position numbers
  sector_pos <- as.numeric(sub(":.*", "", sector_choice))
  industry_pos <- as.numeric(sub(":.*", "", industry_choice))

  list(sector_pos = sector_pos, industry_pos = industry_pos)
}# Function to interactively identify and save the relative location of sector and industry information in earnings call transcript filepaths

get_metadata <- function(filename, path_config) {
  
  name_chunk <- basename(filename) %>% stringr::str_remove('.pdf$') %>% stringr::str_squish()
  
  company <- stringr::str_extract(name_chunk, "^(.*?)(?= - )")
  quarter <- stringr::str_extract(name_chunk, "[Qq]{1}[0-9]{1}")
  year <- stringr::str_extract(name_chunk, "[0-9]{4}")
  
  # Extract by position
  parts <- unlist(strsplit(filename, "/")) # Breaks filepath into component parts
  parts <- parts[parts != ""] # Removes empty parts (if present)
  
  sector <- parts[path_config$sector_pos] # Selects the part associated with the saved position number
  industry <- parts[path_config$industry_pos] # Selects the part associated with the saved position number
  
  tibble::tibble(company, sector, industry, quarter, year)
} # Function to extract sector, industry, year, company name, and quarter from file paths


print("Extracting dates from all transcripts...")
# Extract dates from list of files, clean, & format into dataframe
dates_df <- purrr::map(
  list_of_earnings_calls,
  get_earnings_date,
  .progress = TRUE
) %>%
  dplyr::tibble() %>%
  dplyr::mutate(filename = list_of_earnings_calls) %>%
  tidyr::unnest(1:2) %>%
  dplyr::rename(call_date = ".") %>%
  dplyr::mutate(call_date = as.Date(call_date))

print('Identifying locations of sector & industry information in file paths...')
# Interactively select sector & industry information locations in file path
config <- setup_paths(list_of_earnings_calls[1])

# Apply to our data
print('Parsing metadata from file paths & building dataframe...')
all_data <- purrr::map_dfr(
  list_of_earnings_calls,
  ~ get_metadata(.x, path_config = config),
  .progress = TRUE
) %>%
  cbind(dates_df)