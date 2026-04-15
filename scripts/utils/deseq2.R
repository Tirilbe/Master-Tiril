# Script to perform variance stabilizing transformation (VST) on RNA-seq count data using DESeq2 in R. 
# This is a common step in RNA-seq data analysis to stabilize the variance across the range of mean values.
# Result saved in processed data folders for GTEx and TCGA datasets.

# Set CRAN repository and library path
options(repos = c(CRAN = "https://cran.rstudio.com/"))
user_lib <- "C:/Users/tiril/R/library"
if (!dir.exists(user_lib)) dir.create(user_lib, recursive = TRUE)
.libPaths(c(user_lib, .libPaths()))

# Install packages if needed
if (!require("BiocManager", quietly = TRUE)) {
    install.packages("BiocManager", lib = user_lib)
}
if (!require("DESeq2", quietly = TRUE)) {
    BiocManager::install("DESeq2", lib = user_lib)
}

# Load libraries
library(DESeq2)

write_matrix_with_gene_id <- function(matrix_data, output_path) {
    output_df <- data.frame(
        gene_id = rownames(matrix_data),
        matrix_data,
        check.names = FALSE
    )

    write.csv(output_df, output_path, row.names = FALSE, quote = FALSE)
}

# Read in count data
gtex <- read.csv(
    "../../../data/processed/gtex/gtex_counts_harm.csv",
    row.names = 1,
    check.names = FALSE
)


tcga <- read.csv(
    "../../../data/processed/tcga/tcga_counts_harm.csv",
    row.names = 1,
    check.names = FALSE
)

# Create metadata of samples for DESeq2
gtex_coldata <- data.frame(condition = rep("GTEx", ncol(gtex)))
rownames(gtex_coldata) <- colnames(gtex)

tcga_coldata <- data.frame(condition = rep("TCGA", ncol(tcga)))
rownames(tcga_coldata) <- colnames(tcga)


# Create DESeqDataSet objects
dds_gtex <- DESeqDataSetFromMatrix(
    countData = round(gtex),
    colData = gtex_coldata,
    design = ~1
)


dds_tcga <- DESeqDataSetFromMatrix(
    countData = round(tcga),
    colData = tcga_coldata,
    design = ~1
)



# Estimate size factors (Library size normalization) and perform variance stabilizing transformation
dds_gtex <- estimateSizeFactors(dds_gtex)
dds_tcga <- estimateSizeFactors(dds_tcga)


vst_gtex <- vst(dds_gtex, blind = TRUE) # Removes mean-variance relationship
vst_tcga <- vst(dds_tcga, blind = TRUE) # blind = true, transformation is used without group knowledge


# Transformed matrix
gtex_vst <- assay(vst_gtex)
tcga_vst <- assay(vst_tcga)

# Writes to file
write_matrix_with_gene_id(gtex_vst, "../../../data/processed/gtex/gtex_vst.csv")
write_matrix_with_gene_id(tcga_vst, "../../../data/processed/tcga/tcga_vst.csv")

