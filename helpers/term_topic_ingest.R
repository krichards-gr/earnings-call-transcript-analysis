# renv::install(c('tcltk', 'ggvis', 'here'))

library(tcltk)
# library(ggvis)
library(here)

ingest_terms_topics <- function() {
  # ... (File selection and column selection logic remains the same) ...
  # Assuming the required libraries (dplyr, tibble, etc.) are loaded.
  
  # Create filter to only allow csv files to be selected
  file_filter <- matrix(c("CSV Files", ".csv"), 1, 2, byrow = TRUE)

  # Select the file you want to ingest
  print("please select the file that contains your key terms & topic pairings")
  # tk_choose.files is a Tcl/Tk function, assuming it works fine
  term_topic_file <- read.csv(
    tk_choose.files(
      default = paste0(here::here(), "/inputs"),
      caption = "Select the csv file containing your key terms",
      multi = FALSE,
      filters = file_filter
    ),
    header = TRUE
  )

  col_options <- colnames(term_topic_file)

  # Column selection logic
  terms_col <- select.list(
    col_options,
    title = "Key Term Column?",
    multiple = FALSE
  )
  topics_col <- select.list(
    col_options,
    title = "Topics Column?",
    multiple = FALSE
  )

  ## --- CORRECTED LOGIC BELOW --- ##

  # Standardize all relevant columns to lowercase
  term_topic_data <- term_topic_file %>%
    dplyr::select(!!rlang::sym(terms_col), !!rlang::sym(topics_col)) %>%
    dplyr::mutate(
      dplyr::across(
        c(!!rlang::sym(terms_col), !!rlang::sym(topics_col)), 
        tolower
      )
    )

  # 1. Ingest list of terms to search for (as a single vector)
  #    This part is fine, but using select() then deframe() is cleaner
  key_terms <- term_topic_data %>%
    dplyr::pull(!!rlang::sym(terms_col)) %>% # Get the single vector of all terms
    unique()

  # 2. For topic labeling (The Fix!)
  #    Group the key terms by their associated topic
  named_terms <- term_topic_data %>%
    # Use split() to create a list of key terms grouped by the topic column
    split(x = .[[terms_col]], f = .[[topics_col]]) %>%
    # Remove any empty/NA groups that might result from the split
    purrr::discard(purrr::is_empty)
  
  # named_terms now looks like: 
  # list("politics" = c("trump", "rfk", "administration"), "finance" = c("stock", "dividend"))

  return(list(key_terms = key_terms, named_terms = named_terms))
}
# ingest_terms_topics <- function() {
#   # Create filter to only allow csv files to be selected
#   file_filter <- matrix(c("CSV Files", ".csv"), 1, 2, byrow = TRUE)

#   # Select the file you want to ingest
#   print("please select the file that contains your key terms & topic pairings")
#   term_topic_file <- read.csv(
#     tk_choose.files(
#       default = paste0(here(), "/inputs"),
#       caption = "Select the csv file containing your key terms",
#       multi = FALSE,
#       filters = file_filter
#     ),
#     header = TRUE
#   )

#   col_options <- colnames(term_topic_file)

#   terms_col <- select.list(
#     col_options,
#     title = "Key Term Column?",
#     multiple = FALSE
#   )

#   print(paste0('You selected: ', terms_col))

#   topics_col <- select.list(
#     col_options,
#     title = "Topics Column?",
#     multiple = FALSE
#   )

#   print(paste0('You selected: ', topics_col))

#   # Ingest list of terms to search for -- for key term labeling
#   key_terms <- term_topic_file %>%
#     dplyr::select(terms_col) %>%
#     tibble::deframe() %>%
#     tolower()

#   # For topic labeling
#   named_terms <- term_topic_file %>%
#     tibble::deframe() %>%
#     tolower()

#   return(list(key_terms = key_terms, named_terms = named_terms))
# }

# Usage:
inputs_file <- ingest_terms_topics()
key_terms <- inputs_file$key_terms
named_terms <- inputs_file$named_terms
rm(inputs_file)
