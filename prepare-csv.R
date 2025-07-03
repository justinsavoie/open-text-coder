setwd("~/Documents/personal_repos/ai_creative/open-text-coder/")

library(tidyverse)
library(haven)

df <- read_stata("~/Downloads/2021 Canadian Election Study v2.0.dta")

set.seed(232)

df %>%
  select(cps21_ResponseId,cps21_imp_iss) %>%
  sample_n(400) %>% 
  write_csv("data-cps21.csv")

#zz <- read.csv("~/Desktop/ACAN_ELECTION_FREENEXT_WHY_merged.csv")
#write.csv(zz[1:240,],
#          "~/Documents/personal_repos/ai_creative/coding-open-text-local/survey_classification/ACAN_ELECTION_FREENEXT_WHY_merged.csv")
