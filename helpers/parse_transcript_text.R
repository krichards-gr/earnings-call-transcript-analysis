# renv::install(c('dplyr', 'stringr'))

library(dplyr)
library(stringr)
library(purrr)
library(furrr)

get_text_segments <- function(transcript, key_terms) {
  # Read and process transcript
  transcript_text_raw <- pdf_text(transcript)

  transcript_text_clean <- transcript_text_raw %>%
    str_remove_all(regex(
      '\\sabout:srcdoc\\spage\\s[0-9]+\\sof\\s[0-9]+',
      dotall = TRUE,
      ignore_case = TRUE
    )) %>%
    str_remove_all(regex(
      '\\d{1,2}/\\d{1,2}/\\d{2},\\s*\\d{1,2}:\\d{2}\\s*[ap]m ',
      dotall = TRUE,
      ignore_case = TRUE
    )) %>%
    str_remove(regex("analysts", , dotall = TRUE, ignore_case = TRUE)) %>%
    str_remove(regex("executives", , dotall = TRUE, ignore_case = TRUE)) %>%
    unlist() %>%
    str_squish() %>%
    paste(collapse = " ")

  if (is.null(transcript_text_clean) || length(transcript_text_clean) == 0) {
    warning(paste("Transcript", transcript, "could not be read or is empty."))
    return(tibble(
      filename = transrcipt,
      Speaker = character(),
      Text = character(),
      key_terms_mentioned = logical()
    ))
  }

  names_roles <- .grab_participants(transcript_text_raw)

  # Handle empty title case
  if (length(names_roles) == 0) {
    warning(paste(
      "No speakers found in",
      transcript,
      ". Returning empty tibble."
    ))
    return(tibble(
      filename = transcript,
      Speaker = character(),
      Text = character(),
      key_terms_mentioned = logical()
    ))
  }

  # Create regex pattern to match any instance of a name/role combo
  title_reg <- paste(names_roles, collapse = "|")

  # Split text at each place that the title_reg pattern matches something
  split_text <- str_split(
    transcript_text_clean,
    regex(title_reg, dotall = TRUE, ignore_case = TRUE)
  ) %>%
    unlist()

  # Extract all the matches that the title_reg pattern detects (all of our name/role combos)
  delimiters <- str_extract_all(
    transcript_text_clean,
    regex(title_reg, dotall = TRUE, ignore_case = TRUE)
  ) %>%
    unlist() %>%
    c(NA, .)

  # Construct tibble with full file path in every row
  df <- tibble(
    filename = transcript,
    Speaker = delimiters,
    Text = split_text
  )

  # Ensure text is not empty before applying `terms_mentioned`
  df <- df %>%
    dplyr::filter(Text != "") %>%
    dplyr::mutate(Text = stringr::str_squish(str_replace_all(Text, 'tari"', "tariff"))) %>% # Can I remove this?
    dplyr::rowwise() %>%
    dplyr::mutate(
      key_terms_mentioned = ifelse(
        Text == "",
        NA,
        tag_terms(Text, key_terms)
      ),
        topics_mentioned = ifelse(
          Text == "",
          NA,
          label_topics(Text, named_terms)
        )
      )

  return(df)
} # Function to grab and combine text segments with speaker/role/company information

.grab_participants <- function(transcript) {
  names <- str_extract(
    transcript,
    regex(
      "(?<=Event Participants).*?(?=(Operator\\s+Operator|\\n[A-Z][a-z]+\\s[A-Z][a-z]+\\s+Executive))",
      dotall = TRUE,
      ignore_case = TRUE
    )
  ) %>%
    str_replace_all(
      regex("[:digit:]|executives|analysts|attendees", ignore_case = TRUE),
      "\\\n"
    ) %>%
    str_split(",|(\\n{2,})") %>%
    unlist() %>%
    str_squish()

  names <- names[names != "" & !is.na(names)]

  role_pattern <- "\\s(executive|analyst)"

  names_roles <- names %>%
    str_c(role_pattern) %>%
    append("Operator Operator")

  return(names_roles)
} # Parses call participants from the first page of the transcript to build delimiters dynamically

label_topics <- function(text, named_terms) {
  matched_names <- c() # Store only the names from the named list

  # Fix: iterate over indices, not names
  for (i in seq_along(named_terms)) {
    name <- names(named_terms)[i] # Get the name
    term_pattern <- paste(named_terms[[i]], collapse = '|') # Get the term pattern

    if (grepl(term_pattern, text, ignore.case = TRUE, perl = TRUE)) {
      matched_names <- append(matched_names, name)
    }
  }

  matched_names <- unique(matched_names) # Remove duplicates

  return(ifelse(
    length(matched_names) > 0,
    paste(matched_names, collapse = " | "),
    NA
  ))
} # Function to identify (using key terms) and label all topic categories mentioned in each text segment

tag_terms <- function(text, key_terms) {
  matches <- c() # Store matched company names

  for (term in key_terms) {
    pattern <- term

    if (grepl(pattern, text, ignore.case = TRUE, perl = TRUE)) {
      matches <- append(matches, term)
    }
  }

  matches <- unique(matches) # Remove duplicates

  return(ifelse(length(matches) > 0, paste(matches, collapse = " | "), NA))

  print(matches)
} # Function to check for and return the terms mentioned, delimited by pipe ("|")

print("extracting text content from all transcripts")
plan(multisession, workers = 6) # Set up parallel workers

# Extract text from each transcript, identify and pull out participant names & roles, and label topics and terms mentioned in each segment
results <- furrr::future_map_dfr(
  list_of_earnings_calls,
  function(file) {
    get_text_segments(file, key_terms)
  },
  .progress = TRUE
)

## Clean up and add metadata to collected and categorized earnings calls # can move this all to the parsing step/script?
print("Transforming Data")
results_transformed <- results %>%
  left_join(all_data, join_by(filename)) %>%
  mutate(Speaker = str_squish(Speaker),
         Role = case_when(
           str_detect(Speaker, regex("Analyst", ignore_case = TRUE)) ~ "Analyst",
           str_detect(Speaker, regex("Executive", ignore_case = TRUE)) ~ "Executive",
           str_detect(Speaker, regex("Operator", ignore_case = TRUE)) ~ "Operator")) %>% 
  dplyr::mutate(Name = str_trim(str_remove(Speaker, regex("Analyst$|Executive$|Operator$", ignore_case = TRUE))))

# Add row ID to support sequntial "categorization" for Q&A pairings
results_transformed$ID <- 1:nrow(results_transformed)

# Create question/answer pairings
print("Creating question answer pairings")
results_final <- results_transformed %>% 
  relocate(ID, .before = company) %>% 
    mutate(Interaction_Type = case_when(
    # Role == "Analyst" & str_detect(Text, "(\\?|curious|question|wanted to ask|i'm.*\\swondering)") ~ "Question",
    Role == "Analyst" ~ "Question",
    Role == "Executive" & (coalesce(lag(Role), "None") != "Operator") ~ "Answer",
    Role == "Operator" ~ "Admin",
    TRUE ~ NA_character_)) %>% 
  group_by(Interaction_Type) %>%
  mutate(QA_group = case_when(
    Interaction_Type == "Question" ~ row_number(),
    Interaction_Type == "Answer" ~ NA_real_,
    Interaction_Type == "Admin" ~ 0
  )) %>%
  ungroup() %>% 
  fill(QA_group, .direction = "down") %>%
  dplyr::mutate(QA_group = as.integer(QA_group)) %>% 
  mutate(Length = as.integer(nchar(Text))) %>% 
  mutate(Text = stringr::str_squish(str_replace_all(Text, regex('tari"', dotall = TRUE, ignore_case = TRUE), "tariff"))) %>% 
  mutate(company = company) %>%
  mutate(n_words = as.integer(quanteda::ntoken(quanteda::tokens(Text, what = "word", remove_punct = TRUE)))) %>% 
  select(-filename, -Speaker, -Length) %>% 
  relocate(ID, .before = company) %>% 
  mutate(Interaction_Type = ifelse(Interaction_Type == "Answer" & QA_group == 0, "Admin", Interaction_Type)) %>% 
  mutate(Name = stringr::str_squish(ifelse(grepl("unknown", Name, ignore.case = TRUE), "Unknown", Name))) %>% 
  relocate(Name, .before = Role) %>% 
  tidyr::replace_na(list(Name = 'Admin', Role = 'Admin', Interaction_Type = 'Admin', QA_group = 0)) %>% 
  dplyr::relocate(ID:Interaction_Type, .before = Text)


# print("Extracting sentences")
# results_final_sentences <- results_final %>%
#   select(-n_words, -topics_mentioned, -key_terms_mentioned) %>%
#   tidytext::unnest_sentences(sentences, Text, to_lower = FALSE) %>%
#   mutate(
#     n_words = quanteda::ntoken(quanteda::tokens(
#       sentences,
#       what = "word",
#       remove_punct = TRUE
#     ))
#   ) %>%
#   rowwise() %>%
#   mutate(
#     key_terms_mentioned = tag_terms(sentences, key_terms),
#     sentences = stringr::str_squish(sentences),
#     Name = str_squish(Name)
#   ) %>%
#   mutate(
#     topics_mentioned = label_topics(sentences, named_terms)
#   ) %>%
#   relocate(QA_group, .before = n_words) %>%
#   relocate(QA_group:n_words, .after = topics_mentioned)
