create_global_variables <- function(){
  # This function creates and saves variables in the global environment.
  # This allows all other functions to access them.
  
  # Herceptin (trastuzumab) VH sequence information
  Her_VH <<- paste0("EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEW",
                    "VARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVY",
                    "YCSRWGGDGFYAMDYWGQGTLVTVSS")
  Her_VH_mCDR3 <<- paste0("EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQA",
                          "PGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAY",
                          "LQMNSLRAEDTAVYYGQGTLVTVSS")
  Her_CDRH3 <<- "CSRWGGDGFYAMDYW"
  
  # Herceptin (trastuzumab) VK sequence information
  Her_VK <<- paste0("DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLI",
                    "YSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPP",
                    "TFGQGTKVEIK")
  Her_CDRL1 <<- "QDVNTA"
  Her_CDRL3 <<- "QQHYTTPPT"
  
  # pH for net charge calculation (Sharma et. al)
  pH <<- 5.5
  
  # The following code reads in the pKa values taken from:
  # http://homepage.smc.edu/kline_peggy/Organic/Amino_Acid_pKa.pdf
  pKas <<- read_csv("data/pKas.csv", col_names = FALSE) %>%
    column_to_rownames(var = "X1")
  
  # The following code reads in the Eisenberg hydrophobicity scales taken from:
  # Eisenberg et al. 1984
  Eisenberg <<- read_csv("data/Eisenberg.csv", col_names = FALSE) %>%
    column_to_rownames(var = "X1")
  
  # Identify the positions of hydrophobic and hydrophilic residues
  phobic <<- c(1,2,5,8,10,13,18,19,20)
  philic <<- c(3,4,6,7,9,11,12,14,15,16,17)
}


create_all_dirs <- function(){
  # This function creates all directories that are used in the script.
  # If a directory already exists, the warning error is suppressed.

  dir.create("figures", showWarnings = FALSE)
  dir.create("figures/final", showWarnings = FALSE)
  dir.create("data", showWarnings = FALSE)
  dir.create("data/fasta", showWarnings = FALSE)
  dir.create("data/camsol", showWarnings = FALSE)
  dir.create("data/netMHC", showWarnings = FALSE)
}


net_charge <- function(aa_seq){
  # The following function calculates the net charge of an input amino acid sequence
  # at a given pH.
  # aa_seq: an amino acid sequence

  # Create integer list of amino acid counts
  AA_counts <- sapply(rownames(pKas), function(x) str_count(aa_seq, x))
  
  # TODO: What does this calculate?
  dod <- 1/(10^(pKas - pH) + 1)
  dod[, 1] <- -1*dod[, 1]
  dod[, 2] <- 1 - dod[, 2]
  dod[c(2,3,4,20), 3] <- -1*dod[c(2,3,4,20), 3]
  dod[c(7,9,15), 3] <- 1 - dod[c(7,9,15), 3]
  
  # Multiply integer list with WHATEVER DOD IS, and sum up to create net charge
  net_charge <- sum(AA_counts*rowSums(dod, na.rm = TRUE))
  
  return(net_charge)
}


HIndex <- function(aa_seq){
  # The following function calculates the hydrophobicity index of an input amino acid sequence.
  # aa_seq: an amino acid sequence

  # Create integer list of amino acid counts
  AA_counts <- sapply(rownames(Eisenberg), function(x) str_count(aa_seq, x))

  # Calculate the hydrophobicity index
  HIndex <- -(sum(AA_counts[phobic]*Eisenberg[phobic, 1]) /
              sum(AA_counts[philic]*Eisenberg[philic, 1]))
  
  return(HIndex)
}


plot_geom_histogram <- function(binwidth, title, x_lab, lim){
  # Create a theme to be used for all ggplot visualizations
  # binwidth: the width of the bins in the histogram
  # title: the plot title
  # x_lab: the labels for the x-axis
  # lim: the limits to be used for the y-axis
  
  list(
    geom_histogram(binwidth = binwidth, position = "dodge",
                   show.legend = FALSE),
    labs(title = title, x = x_lab, y = "Count"),
    scale_y_log10(limits = lim),
    scale_color_grey(),
    scale_fill_grey(),
    theme_bw(base_size = 18)
  )
}
