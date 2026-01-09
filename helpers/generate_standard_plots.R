# renv::install(c('ggplot2', 'dplyr'))

library(ggplot2)
library(dplyr)



# #######
# # Summarise sentiment by speaker type
# speaker_sent <- single_ec_sentiments %>% 
#   group_by(Role, sentiment_category) %>% 
#   dplyr::summarise(sent_count = n())

# # Bring in my custom gravity ggplot theme
# source("C:/Users/KRosh/To_Backup/Tools/gravity_theme_ggplot/gravity_ggplot_theme.R")

# install.packages('ggplot2')
# library(ggplot2)

# speaker_plot <- ggplot2::ggplot(speaker_sent, aes(Role, sent_count, fill = sentiment_category)) +
#   geom_col() +
#   theme_gravity() +
#   scale_color_manual(gravity_colors)

# speaker_plot


# # Bar plot
# ggplot2::ggplot(segment_sent, aes(sector, overall_sent_count, fill = sector)) +
#   geom_col() +
#   theme_gravity() +
#   scale_fill_manual(values = gravity_colors) +
#   facet_wrap(~ sentiment_category)


# all_speaker_sent <- all_ec_sentiments %>% 
#   group_by(Role, sentiment_category) %>% 
#   dplyr::summarise(sent_count = n())

# # Bar plot
# ggplot2::ggplot(all_speaker_sent, aes(Role, sent_count, fill = sentiment_category)) +
#   geom_col() +
#   theme_gravity() +
#   scale_color_manual(gravity_colors)

# # Line chart
# all_speaker_line <- all_ec_sentiments %>% 
#   dplyr::filter(Role != 'Admin') %>% 
#   dplyr::group_by(Role, sentiment_category, call_date) %>% 
#   dplyr::summarise(per_sent = n(), .groups = 'drop') %>% 
#   dplyr::mutate(call_date = as.Date(call_date)) %>% 
#   dplyr::group_by(Role, call_date) %>% 
#   dplyr::mutate(per_sent = per_sent / sum(per_sent) * 100) %>% 
#   dplyr::ungroup()


# ggplot2::ggplot(all_speaker_line, aes(call_date, per_sent, colour = sentiment_category)) +
#   geom_line() +
#   theme_gravity() +
#   facet_wrap(~ Role) +
#   scale_color_manual(values = c(Neutral = "gold1", Positive = "green4", Negative = "firebrick3"))


# ### sector breakdown
# # Line chart
# sector_per_sent <- all_ec_sentiments %>% 
#   dplyr::filter(Role != 'Admin') %>% 
#   dplyr::group_by(sector, Role, sentiment_category, call_date) %>% 
#   dplyr::summarise(per_sent = n(), .groups = 'drop') %>% 
#   dplyr::mutate(call_date = as.Date(call_date)) %>% 
#   dplyr::group_by(sector, Role, call_date) %>% 
#   dplyr::mutate(per_sent = per_sent / sum(per_sent) * 100) %>% 
#   dplyr::ungroup()

# ggplot2::ggplot(sector_per_sent, aes(call_date, per_sent, color = sector)) +
#   geom_smooth(se = FALSE) +
#   theme_gravity() +
#   facet_wrap(~ sentiment_category) +
#   scale_color_manual(values = gravity_colors) +
#   ylim(0, 100)
