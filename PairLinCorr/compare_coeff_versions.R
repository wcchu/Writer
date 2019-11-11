suppressPackageStartupMessages(library(tidyverse))

d <-
  rbind(
    read.csv("item_coefficients_1.csv", stringsAsFactors = F) %>% mutate(version = 'v1'),
    read.csv("item_coefficients_2.csv", stringsAsFactors = F) %>% mutate(version = 'v2')
  )

## reshape table: old and new coefficients are two columns
t <-
  d %>%
  gather(key = "order", value = "coefficient", c0, c1) %>%
  spread(key = version, value = coefficient) %>%
  mutate(dif = ifelse(is.na(v2), 0.0, v2) - ifelse(is.na(v1), 0.0, v1),
         existence = ifelse(!is.na(v1) & !is.na(v2), "existing", "only v1 or v2"))

## general difference between old and new
p_hist <-
  ggplot(t) +
  geom_histogram(data = t %>% filter(!is.na(v1)),
                 aes(x = v1), binwidth = 0.025, alpha = 0.5, fill = 'black') +
  geom_histogram(data = t %>% filter(!is.na(v2)),
                 aes(x = v2), binwidth = 0.025, alpha = 0.5, fill = 'red') +
  facet_grid(. ~ order, scales = "free") +
  labs(title = 'Distribution of coeffs: black - v1, red - v2',
       x = 'Coefficient value', y = 'Count')
ggsave(p_hist, filename = 'p_hist.png', height = 10, width = 20, units = 'cm')

p_diff <-
  ggplot(t) +
  geom_histogram(aes(x = dif), binwidth = 0.025) +
  facet_grid(. ~ order, scales = "free") +
  labs(title = 'Distribution of offset differences between v1 and v2',
       x = 'Offset difference (cm)', y = 'Count')
ggsave(p_diff, filename = 'p_diff.png', height = 10, width = 20, units = 'cm')

## existing, old-only, new-only
p_hist_existence <-
  ggplot(t) +
  geom_histogram(data = t %>% filter(!is.na(v1)),
                 aes(x = v1), binwidth = 0.025, alpha = 1, fill = 'black') +
  geom_histogram(data = t %>% filter(!is.na(v2)),
                 aes(x = v2), binwidth = 0.025, alpha = 0.5, fill = 'red') +
  facet_grid(existence ~ order, scales = "free") +
  labs(title = 'Distribution of offsets: black - v1, red - v2',
       x = 'Offset (cm)', y = 'Count')
ggsave(p_hist_existence, filename = 'p_hist_existence.png', height = 10, width = 15, units = 'cm')

p_diff_existence <-
  ggplot(t) +
  geom_histogram(aes(x = dif), binwidth = 0.025) +
  facet_grid(existence ~ order, scales = "free") +
  labs(title = 'Distribution of offset differences between v1 and v2',
       x = 'Offset difference (cm)', y = 'Count')
ggsave(p_diff_existence, filename = 'p_diff_existence.png', height = 10, width = 15, units = 'cm')
