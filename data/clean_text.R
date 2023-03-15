suppressPackageStartupMessages(library(readr))
suppressPackageStartupMessages(library(tidyverse))

t <- "[_something_]"

args = commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("Enter 2 arguments (input_file, output_file")
}

t <- read_file(args[1])

# remove technical characters in playwrights such as "[", "]", and "_"
t <- gsub("\\[|\\]|_", "", t)

# change double white spaces to single white spaces
t <- gsub("  ", " ", t)

# combine \r\n to \n
t <- gsub("\r\n", "\n", t)

# change double new lines to single new lines
for (i in c(1:5)) {
    t <- gsub("\n\n", "\n", t)
}

# output
write(t, args[2])
