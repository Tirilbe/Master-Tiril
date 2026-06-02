# Compute wTO from RhoAndVar edge lists using fast vectorized implementation.
# Soft thresholding is applied as adj = |rho|^BETA before wTO is calculated.
#
# Run select_wTO_beta.r first if you want to choose BETA from scale-free
# topology diagnostics. That pre-step tests beta 1:10 without computing wTO.
#
# Input:
#   - results/Corr/RhoAndVar_proteome_normal_FINAL_FINAL.txt
#   - results/Corr/RhoAndVar_proteome_cancer_FINAL_FINAL.txt
#
# Output:
#   - results/wTO/wTO_proteome_normal_edgelist_<BETA>.txt
#   - results/wTO/wTO_proteome_cancer_edgelist_<BETA>.txt

library(data.table)

cmd_args <- commandArgs(FALSE)
file_arg <- cmd_args[grepl("^--file=", cmd_args)]
if (length(file_arg) > 0) {
    script_path <- normalizePath(sub("^--file=", "", file_arg[1]), winslash = "/", mustWork = FALSE)
    script_dir <- dirname(script_path)
} else {
    script_dir <- getwd()
}

root_dir <- normalizePath(file.path(script_dir, "..", ".."), winslash = "/", mustWork = FALSE)
results_dir <- file.path(root_dir, "results")
corr_dir <- file.path(results_dir, "Corr")
wto_dir <- file.path(results_dir, "wTO")
dir.create(wto_dir, showWarnings = FALSE, recursive = TRUE)

# Default beta. Override from the command line with:
#   Rscript wTO_calculation.r 12
BETA <- 1
args <- commandArgs(trailingOnly = TRUE)
if (length(args) >= 1) {
    BETA <- as.integer(args[1])
    if (is.na(BETA) || BETA < 1) {
        stop("Beta must be a positive integer, for example: Rscript wTO_calculation.r 6")
    }
}
BETA_SUFFIX <- paste0("_", BETA)

# ---------------------------------------------------------------------------
# Fast vectorized wTO computation
# Soft thresholding: adj = |rho|^beta
# Signs restored after computation
# ---------------------------------------------------------------------------

compute_wTO_fast <- function(adj) {
    cat("    Computing weighted connectivity K...\n")
    K <- rowSums(adj)

    cat("    Computing shared neighbourhood L = adj %*% adj...\n")
    L <- adj %*% adj

    cat("    Computing wTO matrix...\n")
    n <- nrow(adj)
    Ki <- matrix(K, nrow = n, ncol = n, byrow = FALSE)
    Kj <- matrix(K, nrow = n, ncol = n, byrow = TRUE)

    denom <- pmin(Ki, Kj) + 1 - adj
    wto <- (L + adj) / denom
    diag(wto) <- 1

    return(wto)
}

# ---------------------------------------------------------------------------
# Memory-efficient edge list writer
# ---------------------------------------------------------------------------

write_edgelist <- function(mat, out_file) {
    genes <- rownames(mat)
    n <- nrow(mat)
    cat(sprintf("  Writing %d genes to %s...\n", n, out_file))

    con <- file(out_file, "w")
    for (i in 1:n) {
        if (i %% 500 == 0) {
            cat(sprintf("  Progress: %.1f%%\r", 100 * i / n))
        }
        idx <- which(seq_len(n) != i)
        lines <- paste(genes[i], genes[idx], mat[i, idx], 0, sep = "\t")
        writeLines(lines, con)
    }
    close(con)
    cat("\n  Done writing edge list.\n")
}

# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

process_condition <- function(rhovar_file, out_edgelist, beta = 1) {
    cat(sprintf("\nProcessing: %s\n", rhovar_file))
    cat(sprintf("  Using beta: %d\n", beta))

    # Load edge list
    cat("  Loading edge list...\n")
    dt <- fread(
        rhovar_file,
        sep = "\t",
        header = FALSE,
        col.names = c("Gene1", "Gene2", "Rho", "Var"),
        colClasses = c("character", "character", "numeric", "numeric")
    )
    dt <- dt[Gene1 != Gene2]
    cat(sprintf("  Gene pairs loaded: %s\n", format(nrow(dt), big.mark = ",")))

    # Convert to matrix
    cat("  Converting to adjacency matrix...\n")
    wide <- dcast(dt, Gene1 ~ Gene2, value.var = "Rho", fill = 0)
    genes <- wide$Gene1
    mat <- as.matrix(wide[, -1])
    rownames(mat) <- genes
    rm(dt, wide); gc()

    # Ensure square symmetric matrix
    all_genes <- union(rownames(mat), colnames(mat))
    mat_sq <- matrix(
        0,
        nrow = length(all_genes),
        ncol = length(all_genes),
        dimnames = list(all_genes, all_genes)
    )
    mat_sq[rownames(mat), colnames(mat)] <- mat
    lower <- lower.tri(mat_sq)
    mat_sq[lower] <- t(mat_sq)[lower]
    diag(mat_sq) <- 1
    cat(sprintf("  Matrix dimensions: %d x %d\n", nrow(mat_sq), ncol(mat_sq)))
    rm(mat); gc()

    # Save sign matrix before taking absolute values
    sign_mat <- sign(mat_sq)

    # Use absolute correlations as adjacency with soft thresholding
    adj <- abs(mat_sq)^beta
    diag(adj) <- 0
    rm(mat_sq); gc()

    # Compute wTO
    cat("  Computing wTO...\n")
    wto_mat <- compute_wTO_fast(adj)
    rm(adj); gc()

    # Restore signs from original correlations
    cat("  Adding signs from original correlations...\n")
    wto_signed <- wto_mat * sign_mat
    diag(wto_signed) <- 1
    rm(wto_mat, sign_mat); gc()

    # Write edge list
    write_edgelist(wto_signed, out_edgelist)
    rm(wto_signed); gc()
}

# ---------------------------------------------------------------------------
# Process both conditions
# ---------------------------------------------------------------------------

process_condition(
    rhovar_file = file.path(corr_dir, "RhoAndVar_proteome_normal_FINAL_FINAL.txt"),
    out_edgelist = file.path(wto_dir, paste0("wTO_proteome_normal_edgelist_1_1", BETA_SUFFIX, ".txt")),
    beta = BETA
)

process_condition(
    rhovar_file = file.path(corr_dir, "RhoAndVar_proteome_cancer_FINAL_FINAL.txt"),
    out_edgelist = file.path(wto_dir, paste0("wTO_proteome_cancer_edgelist_1_1", BETA_SUFFIX, ".txt")),
    beta = BETA
)

cat(sprintf(
    "\nDone. Use wTO_proteome_normal_edgelist_1_1%s.txt and wTO_proteome_cancer_edgelist_1_1%s.txt in FindCSD.py\n",
    BETA_SUFFIX,
    BETA_SUFFIX
))
