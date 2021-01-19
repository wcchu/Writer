suppressPackageStartupMessages(library(tidyverse))

d <- read.csv("trump_tweets.csv", stringsAsFactors = F)
t <- d$text

# join tweets to one string with new lines
t <- paste(t, collapse = "\n")

# remove emojis
t <- gsub("[^\x01-\x7F]", "", t)

# remove indication of an account is speaking, such as "@my_name:"
t <- str_remove_all(t, "@\\w*:")

# change double white spaces to single white spaces
t <- gsub("  ", " ", t)

# change double new lines to single new lines
t <- gsub("\n\n", "\n", t)

# output
write(t, "trump.txt")
