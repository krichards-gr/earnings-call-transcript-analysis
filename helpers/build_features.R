# renv::install(c('stringr', 'tidyr', 'dplyr', 'devtools'))
# devtools::install_github("chris31415926535/tidyvader")

library(stringr)
library(tidyr)
library(dplyr)
library(tidyvader)

# Add term count columns
print("Adding key term count columns...")
for (term in key_terms) {
  results_final[[term]] <- stringr::str_count(results_final$Text, regex(term, dotall = TRUE, ignore_case = TRUE))
}

# for (term in key_terms) {
#   results_final_sentences[[term]] <- stringr::str_count(results_final_sentences$sentences, regex(term, dotall = TRUE, ignore_case = TRUE))
# }

# # Generate sentiment labels at the sentence level
# print('Adding sentiment labels...')
# sentiment_data <- results_final_sentences %>%
#   tidyvader::vader(sentences) %>%
#   dplyr::mutate(
#     sentiment_category = case_when(
#       compound >= 0.05 ~ 'Positive',
#       compound <= -0.05 ~ 'Negative',
#       .default = 'Neutral'
#     )
#   ) %>% 
#   dplyr::mutate(sentiment_category = as.factor(sentiment_category)) %>% 
#   dplyr::select(ID, compound, sentiment_category)


# # Bind sentiment labels to output dataframes
# results_final_sentences_sent <- sentiment_data %>% 
#   dplyr::select(sentiment_category) %>% 
#   dplyr::bind_cols(results_final_sentences, .) %>% 
#   dplyr::relocate(sentiment_category, .after = sentences)

# results_final_sent <- sentiment_data %>%
#   dplyr::group_by(ID) %>% # First summarise sentiment scores by segment ID
#   dplyr::summarise(
#     segment_sent = sum(compound)
#   ) %>%
#   dplyr::mutate( # Then create new column with categories and join
#     sentiment_category = case_when(
#       segment_sent >= 0.05 ~ 'Positive',
#       segment_sent <= -0.05 ~ 'Negative',
#       .default = 'Neutral'
#     )
#   ) %>%
#   dplyr::select(-segment_sent) %>% 
#   dplyr::left_join(results_final, ., join_by(ID)) %>% 
#   dplyr::relocate(sentiment_category, .after = Text)


# # Tidy data for easy analysis
# print("Tidying data...")
# results_final_tidy <- results_final_sent %>% 
#   pivot_longer(17:ncol(.)) %>% 
#   rename(Term = name, Term_Count = value)

# results_final_sentences_tidy <- results_final_sentences_sent %>% 
#   pivot_longer(17:ncol(.)) %>% 
#   rename(Term = name, Term_Count = value)