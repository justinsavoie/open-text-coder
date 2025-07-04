setwd("~/Documents/personal_repos/ai_creative/open-text-coder/data/")

library(tidyverse)
library(haven)

df <- read_stata("~/Downloads/2021 Canadian Election Study v2.0.dta")

set.seed(232)

df %>%
  select(cps21_ResponseId,cps21_imp_iss) %>%
  sample_n(400) %>% 
  write_csv("data-cps21-GOLD-raw.csv")

set.seed(111)

ids_in_gold <- read.csv("data-cps21-GOLD-raw.csv") %>% pull(cps21_ResponseId)

df %>%
  filter(!(cps21_ResponseId %in% ids_in_gold)) %>%
  select(cps21_ResponseId,cps21_imp_iss) %>%
  sample_n(400) %>% 
  write_csv("data-cps21.csv")

df %>%
  select(cps21_ResponseId,cps21_imp_iss,Q_Language) %>%
  filter(complete.cases(.)) %>% 
  write_csv("data-cps21-FULL.csv")

df6 <- read_rds("~/Documents/IIUData/tides/Wave6/tides_w6_clean.rds")

df6 %>%
  select(UniqueID, ACAN_ELECTION_FREENEXT_WHY_NEITHER, ACAN_ELECTION_FREENEXT_WHY_DISAGREE) %>%
  mutate(open=ifelse(ACAN_ELECTION_FREENEXT_WHY_NEITHER=="", 
                                 ACAN_ELECTION_FREENEXT_WHY_DISAGREE, 
                                 ACAN_ELECTION_FREENEXT_WHY_NEITHER)) %>%
  select(UniqueID, open) %>%
  filter(open != "") %>%
  write_csv("data-z.csv")
