#' Developability calculations and filters
#' 
#' Runs developability filters on in-silico created sequences.
#' Created on February 2019
#' 
#' @author Derek Mason
#' 
#' @examples 
#' R developability-filteres4.R

# Load libraries
library(Biostrings)
library(dplyr)
library(ggplot2)
library(readr)
library(stringr)
library(stringdist)
library(tibble)
library(tidyr)

# Source custom functions
source("helpers.R")

# Create all global variables used in this script
create_global_variables()

# Create all folders if they do not exist yet
create_all_dirs()


# ----------------------
# Prepare classification
# data sets
# ----------------------

# Load in the .csv output files from Python containing the
# amino acid sequences and prediction values
CNN_df <- read_csv("data/CNN_H3_all.csv") %>%
  select(AASeq, Pred)

# Filter out sequences with a prediction values less than 0.70
consensus <- filter(CNN_df, Pred > 0.70)

# Pad amino acid sequences with 5' CSR residues and 3' YW residues
consensus$AASeq <- paste("CSR", consensus$AASeq, "YW", sep = "")

# Calculate the net charge of the amino acid sequence and add the net charge of the entire VH sequence minus CDRH3
consensus <- consensus %>%
  mutate(VHNetCharge = sapply(AASeq, function(x) net_charge(x)) +  net_charge(Her_VH_mCDR3),
         FabNetCharge = VHNetCharge + net_charge(Her_VK),
         FvCSP = VHNetCharge * net_charge(Her_VK))

# Calculate the hydrophobicity index
consensus <- consensus %>%
  mutate(CDR3_HI = sapply(AASeq, function(x) HIndex(x)),
         HISum = CDR3_HI + HIndex(Her_CDRL1) + HIndex(Her_CDRL3))

# Calculate the Levenshtein distance from the wild-type CDRH3 to CDRH3 variants
consensus$LD <- sapply(consensus$AASeq, function(x) stringdist(x, Her_CDRH3, method = "lv"))

# Pad CDR3 sequences with +/-10 amino acids for use with CamSol and NetMHCIIpan
consensus <- consensus %>%
  mutate(paddedAA = paste(substr(Her_VH, gregexpr("C[ASTV][KRST]", Her_VH)[[1]] - 10,
                                 gregexpr("C[ASTV][KRST]", Her_VH)[[1]] - 1),
                          AASeq, substr(Her_VH, gregexpr("WGQG", Her_VH)[[1]] + 1,
                                        gregexpr("WGQG", Her_VH)[[1]] + 10), sep = ""))


# ----------------------
# Prepare data of
# binding sequences
# ----------------------

# Load binding sequences
AgPos_df <- read_csv("data/mHER_H3_AgPos.csv") %>%
  select(-X1)

# Remove duplicated sequences
AgPos_df <- AgPos_df %>%
  distinct(AASeq, .keep_all = TRUE)

# Pad amino acid sequences with 5' CSR residues and 3' YW residues
AgPos_df$AASeq <- paste("CSR", AgPos_df$AASeq, "YW", sep = "")

# Calculate the Fab net charge, FvCSP, and L1+L3+H3 hydrophobicity index sum
AgPos_df <- AgPos_df %>%
  mutate(VHNetCharge = sapply(AASeq, function(x) net_charge(x)) + net_charge(Her_VH_mCDR3),
         FabNetCharge = VHNetCharge + net_charge(Her_VK),
         FvCSP = VHNetCharge*net_charge(Her_VK))

# Calculate the hydrophobicity index
AgPos_df <- AgPos_df %>%
  mutate(CDR3_HI = sapply(AASeq, function(x) HIndex(x)),
         HISum = CDR3_HI + HIndex(Her_CDRL1) + HIndex(Her_CDRL3))

# Calculate the Levenshtein distance from the wild-type CDRH3 to CDRH3 variants
AgPos_df$LD <- sapply(AgPos_df$AASeq, function(x) stringdist(x, Her_CDRH3, method = "lv"))

# Pad CDR3 sequences with +/-10 amino acids for use with CamSol and NetMHCIIpan
AgPos_df <- AgPos_df %>%
  mutate(paddedAA = paste(substr(Her_VH, gregexpr("C[ASTV][KRST]", Her_VH)[[1]] - 10,
                                 gregexpr("C[ASTV][KRST]", Her_VH)[[1]] - 1),
                          AASeq, substr(Her_VH, gregexpr("WGQG", Her_VH)[[1]] + 1,
                                        gregexpr("WGQG", Her_VH)[[1]] + 10), sep = ""))


# ----------------------
# Generate figures:
# 6a and 6b
# ----------------------

# Create dataframe for plotting
Fv_data <- consensus %>%
  select(AASeq, FabNetCharge, FvCSP, HISum, LD) %>%
  mutate(data = "Predicted")
tmp <- AgPos_df %>%
  select(AASeq, FabNetCharge, FvCSP, HISum, LD) %>%
  mutate(data = "Experimental")
Fv_data <- rbind(Fv_data, tmp)

# Levenshtein distance
ggplot(data = Fv_data, aes(LD, fill = data, color = data)) +
  plot_geom_histogram(binwidth = 1, title = "Levenshtein Distance",
                      x_lab = "LD", lim = c(1,1e7))
ggsave("figures/final/VH_Edit-Dist.pdf", width = 5.08, height = 3.8)

# FvCSP distribution
ggplot(data = Fv_data, aes(FvCSP, fill = data, color = data)) +
  plot_geom_histogram(binwidth = 2, title = "FvCSP Distribution",
                      x_lab = "FvCSP", lim = c(1,1e7))
ggsave("figures/final/VH_FvCSP.pdf", width = 5.08, height = 3.8)


# ----------------------
# Generate figure:
# 6c
# ----------------------

# Filter out sequences with a FvCSP less than trastuzumab
consensus_filt_trast <- filter(consensus, FvCSP > 6.61)
AgPos_filt_trast <- filter(AgPos_df, FvCSP > 6.61)

# Create dataframe for plotting
Fv_data2 <- consensus_filt_trast %>%
  select(AASeq, FabNetCharge, FvCSP, HISum, LD) %>%
  mutate(data = "Predicted")
tmp <- AgPos_filt_trast %>%
  select(AASeq, FabNetCharge, FvCSP, HISum, LD) %>%
  mutate(data = "Experimental")
Fv_data2 <- rbind(Fv_data2, tmp)

# Fv charge distribution
ggplot(data = Fv_data2, aes(FabNetCharge, fill = data, color = data)) +
  plot_geom_histogram(binwidth = 0.5, title = "Fv Charge Distribution",
                      x_lab = "Fv Charge", lim = c(1,1e7))
ggsave("figures/final/VH_FvCharge.pdf", width = 5.08, height = 3.8)


# ----------------------
# Prepare and write FASTA
# files for CamSol
# ----------------------

# Filter out sequences with a Fv net charge greater than 6.2
consensus_filt_fv <- filter(consensus_filt_trast, FabNetCharge < 6.2)
AgPos_filt_fv <- filter(AgPos_filt_trast, FabNetCharge < 6.2)

# Filter out sequences with an HI Sum that is negative or greater than 4
consensus_filt_hi <- filter(consensus_filt_fv, (HISum > 0) & (HISum < 4))
AgPos_filt_hi <- filter(AgPos_filt_fv, (HISum > 0) & (HISum < 4))

# Save AgPos_filt_hi as FASTA directly
# A warning is printed if the file contains more than 5000 sequences
if (dim(AgPos_filt_hi)[1] > 5000) {
  warning("AgPos_filt_hi contains more than 5000 sequences.")
}
VH_fq <- AAStringSet(AgPos_filt_hi$paddedAA)
names(VH_fq) <- AgPos_filt_hi$AASeq
writeXStringSet(VH_fq, "data/fasta/AgPos_VH_CamSol.fasta")

# IMPORTANT INFO:
# This FASTA file needs to be run on the CamSol web server!

# Load the CamSol result for AgPos sequences
Ag_CamSol_df <- read_tsv("data/camsol/AgCamSol.txt") %>%
  select(Name, CamSol = `protein variant score`)

# Add CamSol results to the previous dataframe
len_before = dim(AgPos_filt_hi)[1]
AgPos_filt_hi <- inner_join(AgPos_filt_hi, Ag_CamSol_df,
                            by = c("AASeq" = "Name"))
len_after = dim(AgPos_filt_hi)[1]

# Assert that length of dataframe has not changed
stopifnot("Data lost/added in CamSol score (A)." = (len_before == len_after))

# Load the CamSol input file from Pietro Sormanni
CamSol_df <- read_csv("data/camsol/VH_CamSol.csv") %>%
  select(AASeq, CamSol)

# Add CamSol results to the previous dataframe
len_before = dim(consensus_filt_hi)[1]
consensus_filt_hi <- inner_join(consensus_filt_hi, CamSol_df, by = "AASeq")
len_after = dim(consensus_filt_hi)[1]

# Assert that length of dataframe has not changed
stopifnot("Data lost/added in CamSol score (B)." = (len_before == len_after))


# ----------------------
# Generate figure:
# 6d
# ----------------------

# Create dataframe for plotting
CamSol_data <- consensus_filt_hi %>%
  select(AASeq, CamSol) %>%
  mutate(data = "Predicted")
tmp <- AgPos_filt_hi %>%
  select(AASeq, CamSol) %>%
  mutate(data = "Experimental")
CamSol_data <- rbind(CamSol_data, tmp)

# CamSol score distribution
ggplot(data = CamSol_data, aes(CamSol, fill = data, color = data)) +
  plot_geom_histogram(binwidth = 0.2, title = "CamSol Score Distribution",
                      x_lab = "CamSol Score", lim = c(1,NaN))
ggsave("figures/final/VH_CamSol.pdf", width = 5.08, height = 3.8)


# ----------------------
# Prepare and write FASTA
# files for NetMHCIIpan
# ----------------------

# Filter out sequences with a CamSol score greater than 0.5
consensus_filt_cs <- filter(consensus_filt_hi, CamSol > 0.5)
AgPos_filt_cs <- filter(AgPos_filt_hi, CamSol > 0.5)

# Write separate FASTA files for use in NetMHCIIpan
VH_fq <- AAStringSet(consensus_filt_cs$paddedAA)
names(VH_fq) <- consensus_filt_cs$AASeq
writeXStringSet(VH_fq, "data/fasta/VH_NetMHCII.fasta")

VH_fq2 <- AAStringSet(AgPos_filt_cs$paddedAA)
names(VH_fq2) <- AgPos_filt_cs$AASeq
writeXStringSet(VH_fq2, "data/fasta/Ag_NetMHCII.fasta")

# Run NetMHC code with Bash
# system("./run_netMHCIIpan.sh")

# Read in NetMHC output
# Only select ID and Rank information
VH_netMHC_df <- read_tsv("data/netMHC/VH_NetMHCIIpan.txt", skip = 1) %>%
  select(ID, starts_with("Rank"))

Ag_netMHC_df <- read_tsv("data/netMHC/Ag_NetMHCIIpan.txt", skip = 1) %>%
  select(ID, starts_with("Rank"))

# Assert that all sequences are present
stopifnot("Data lost/added in NetMHCII score (A)." =
          (dim(consensus_filt_cs)[1] == length(unique(VH_netMHC_df$ID))))
stopifnot("Data lost/added in NetMHCII score (B)." =
            (dim(AgPos_filt_cs)[1] == length(unique(Ag_netMHC_df$ID))))

# Save unique IDs
VH_netMHC_unique <- unique(VH_netMHC_df$ID)
Ag_netMHC_unique <- unique(Ag_netMHC_df$ID)


# ----------------------
# Generate figure:
# 6e
# ----------------------

# Determine the minimum rank across all peptide strings
consensus_filt_cs$minNetMHC <- sapply(
  VH_netMHC_unique, function(x) min(filter(VH_netMHC_df, ID == x)[, -1])
)
AgPos_filt_cs$minNetMHC <- sapply(
  Ag_netMHC_unique, function(x) min(filter(Ag_netMHC_df, ID == x)[, -1])
)

# Calculate number of peptides that have a rank lower than 10
consensus_filt_cs$NminNetMHC <- sapply(
  VH_netMHC_unique, function(x) sum(filter(VH_netMHC_df, ID == x)[, -1] < 10)
)
AgPos_filt_cs$NminNetMHC <- sapply(
  Ag_netMHC_unique, function(x) sum(filter(Ag_netMHC_df, ID == x)[, -1] < 10)
)

# Calculate the average rank for the remaining sequences
consensus_filt_cs$avgNetMHC <- sapply(
  VH_netMHC_unique, function(x) mean(as.matrix(filter(VH_netMHC_df, ID == x)[, -1]))
)
AgPos_filt_cs$avgNetMHC <- sapply(
  Ag_netMHC_unique, function(x) mean(as.matrix(filter(Ag_netMHC_df, ID == x)[, -1]))
)

# Create dataframe for plotting
netMHC_data <- consensus_filt_cs %>%
  select(AASeq, minNetMHC) %>%
  mutate(data = "Predicted")
tmp <- AgPos_filt_cs %>%
  select(AASeq, minNetMHC) %>%
  mutate(data = "Experimental")
netMHC_data <- rbind(netMHC_data, tmp)

# NetMHC score distribution
ggplot(data = netMHC_data, aes(minNetMHC, fill = data, color = data)) +
  plot_geom_histogram(binwidth = NULL, title = "Minimum NetMHC Distribution",
                      x_lab = "Minimum NetMHC Score", lim = c(1,NaN))
ggsave("figures/final/VH_MinNetMHC.pdf", width = 5.08, height = 3.8)


# ----------------------
# Generate figure:
# 6f
# ----------------------

# Filter out sequences with a minNetMHC score greater than 5.5
consensus_filt_minMHC <- filter(consensus_filt_cs, minNetMHC > 5.5)
AgPos_filt_minMHC <- filter(AgPos_filt_cs, minNetMHC > 5.5)

# Create dataframe for plotting
netMHC_data2 <- consensus_filt_minMHC %>%
  select(AASeq, NminNetMHC) %>%
  mutate(data = "Predicted")
tmp <- AgPos_filt_minMHC %>%
  select(AASeq, NminNetMHC) %>%
  mutate(data = "Experimental")
netMHC_data2 <- rbind(netMHC_data2, tmp)

# NminNetMHC score distribution
ggplot(data = netMHC_data2, aes(NminNetMHC, fill = data, color = data)) +
  plot_geom_histogram(binwidth = NULL, title = "15-mers with % Rank < 10",
                      x_lab = "No. 15-mers", lim = c(1,NaN))
ggsave("figures/final/VH_NMinNetMHC.pdf", width = 5.08, height = 3.8)


# ----------------------
# Generate figure:
# 6g
# ----------------------

# Filter out sequences with NminNetMHC smaller than 2
consensus_filt_NminMHC <- filter(consensus_filt_minMHC, NminNetMHC <= 2)
AgPos_filt_NminMHC <- filter(AgPos_filt_minMHC, NminNetMHC <= 2)

# Create dataframe for plotting
netMHC_data3 <- consensus_filt_NminMHC %>%
  select(AASeq, avgNetMHC) %>%
  mutate(data = "Predicted")
tmp <- AgPos_filt_NminMHC %>%
  select(AASeq, avgNetMHC) %>%
  mutate(data = "Experimental")
netMHC_data3 <- rbind(netMHC_data3, tmp)

# NminNetMHC score distribution
ggplot(data = netMHC_data3, aes(avgNetMHC, fill = data, color = data)) +
  plot_geom_histogram(binwidth = NULL, title = "Average NetMHC Distribution",
                      x_lab = "Average NetMHC Score", lim = c(1, NaN))
ggsave("figures/final/VH_AvgNetMHC.pdf", width = 5.08, height = 3.8)


# ----------------------
# Generate figure:
# 6h
# ----------------------

# Filter out sequences with avgNetMHC greater than 60.56
consensus_filt_avgMHC <- filter(consensus_filt_minMHC, avgNetMHC > 60.56)
AgPos_filt_avgMHC <- filter(AgPos_filt_NminMHC, avgNetMHC > 60.56)

# Create dataframe for plotting
Consensus_overall <- consensus_filt_avgMHC %>%
  select(AASeq, FvCSP, CamSol, minNetMHC, avgNetMHC, NminNetMHC) %>%
  mutate(data = "DL-predicted\nBinders")
AgPos_overall <- AgPos_filt_avgMHC %>%
  select(AASeq, FvCSP, CamSol, minNetMHC, avgNetMHC, NminNetMHC) %>%
  mutate(data = "Experimental\nBinders")
Overall <- rbind(Consensus_overall, AgPos_overall)

# Calculate overall developability score
Overall <- Overall %>%
  mutate(FvCSP = (FvCSP - min(FvCSP))/(max(FvCSP - min(FvCSP))),
         CamSol = (CamSol - min(CamSol))/(max(CamSol - min(CamSol))),
         CamSol = CamSol * 2,
         avgNetMHC = (avgNetMHC - min(avgNetMHC))/(max(avgNetMHC - min(avgNetMHC))),
         Sum = rowSums(select(., .dots = c(FvCSP, CamSol, NminNetMHC))),
         Mean = rowMeans(select(., .dots = c(FvCSP, CamSol, NminNetMHC)))
  )

# Make separate dataframe for Predicted and Experimental
Consensus_overall2 <- filter(Overall, data == "DL-predicted\nBinders")
Ag_overall2 <- filter(Overall, data == "Experimental\nBinders")

# Generate combined violin/scatter plot to view
# the distribution of the developability score
ggplot(data = Consensus_overall2, aes(x = data, y = Mean, fill = data)) +
  geom_violin(show.legend = FALSE) +
  geom_point(data = Ag_overall2, aes(x = data, y = Mean, fill = data),
             position = position_jitterdodge(jitter.width = 0.37),
             show.legend = FALSE) +
  labs(title = "Overall Developability Score") +
  labs(x = "", y = "Developability Score") +
  scale_color_grey() + scale_fill_grey() +
  theme_bw(base_size = 18)
ggsave("figures/final/VH_overall_combined.pdf", width = 5.08, height = 3.8)


# Calculate developability score for DL-predicted binders
Consensus_overall2 <- Consensus_overall2 %>%
  mutate(FvCSP = (FvCSP - min(FvCSP))/(max(FvCSP - min(FvCSP))),
         CamSol = (CamSol - min(CamSol))/(max(CamSol - min(CamSol))),
         CamSol = CamSol * 2,
         avgNetMHC = (avgNetMHC - min(avgNetMHC))/(max(avgNetMHC - min(avgNetMHC))),
         Sum = rowSums(select(., .dots = c(FvCSP, CamSol, NminNetMHC))),
         Mean = rowMeans(select(., .dots = c(FvCSP, CamSol, NminNetMHC)))
  )

# Save CSV file
Consensus_overall2 %>%
  write_csv("data/Pred_all.csv")


# ----------------------
# Generate txt file:
# 6i
# ----------------------

# Set parameters used for printing txt file
out_file <- "figures/final/n_remain_seqs.txt"
border <- paste(rep("-", 55), collapse = "")
ws <- paste(rep(" ", 20), collapse = "")
len_filt1 <- format(dim(CNN_df)[1], big.mark=",")
len_filt2 <- format(dim(consensus)[1], big.mark=",")
len_filt3 <- format(dim(consensus_filt_hi)[1], big.mark=",")
len_filt4 <- format(dim(consensus_filt_cs)[1], big.mark=",")
len_filt5 <- format(dim(consensus_filt_avgMHC)[1], big.mark=",")

# Write txt file
write(border, out_file)
write("Filtering Parameters \t \t No. Predicted Binders",
      out_file, append=TRUE)
write(border, out_file, append=TRUE)
write(paste0("CNN P(binder) > 0.50", ws, len_filt1),
      out_file, append=TRUE)
write(border, out_file, append=TRUE)
write(paste0("CNN P(binder) > 0.70", ws, len_filt2),
      out_file, append=TRUE)
write(border, out_file, append=TRUE)
write("FvCSP > 6.61", out_file, append=TRUE)
write(paste0("Fv charge < 6.2 \t \t", ws, len_filt3),
      out_file, append=TRUE)
write("HI Sum > 0, < 4", out_file, append=TRUE)
write(border, out_file, append=TRUE)
write(paste0("Solubility score > 0.5", ws, len_filt4),
      out_file, append=TRUE)
write(border, out_file, append=TRUE)
write("Minimum % Rank > 5.5", out_file, append=TRUE)
write(paste0("No. 15-mers w/ % Rank < 10 <= 2 \t \t \t", len_filt5),
      out_file, append=TRUE)
write("Average % Rank > 60.6", out_file, append=TRUE)
write(border, out_file, append=TRUE)
