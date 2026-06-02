// Simplified version of FindCorrAndVar.cpp
// Computes only full-sample Spearman correlations — no subsampling.
// Variance is set to 0 since it is not used in the modified CSD pipeline.
//
// This is significantly faster than the original since the O(n^2 * subsamples)
// subsampling loop is removed. Only the O(n^2 * N) correlation computation remains.
//
// Output format is identical to original RhoAndVar.txt:
// Gene1 \t Gene2 \t Rho \t Var(=0)
 
#include <stdio.h>
#include <cmath>
#include <iostream>
#include <list>
#include <fstream>
#include <stdlib.h>
#include <algorithm>
#include <numeric>
#include <vector>
 
using namespace std;
 
 
// ---------------------------------------------------------------------------
// Parameters — update for each dataset
// ---------------------------------------------------------------------------
 
const char* expDataFile = "../../data/processed/rna_cancer_FINAL_FINAL_no_header.txt";
const char* outFile     = "../../results/Corr/RhoAndVar_rna_cancer_FINAL_FINAL.txt";
const int   sampleSize  = 105;   // number of samples (columns)
const int   numberOfGenes = 8547; // number of genes (rows)
 
 
// ---------------------------------------------------------------------------
// Spearman correlation
// ---------------------------------------------------------------------------
 
vector<double> rank_values(const vector<double>& values) {
    int n = values.size();
    vector<int> order(n);
    iota(order.begin(), order.end(), 0);

    sort(order.begin(), order.end(), [&](int left, int right) {
        if (values[left] == values[right]) {
            return left < right;
        }
        return values[left] < values[right];
    });

    vector<double> ranks(n, 0.0);
    int start = 0;
    while (start < n) {
        int end = start + 1;
        while (end < n && values[order[end]] == values[order[start]]) {
            end++;
        }

        double average_rank = (start + 1 + end) / 2.0;
        for (int index = start; index < end; index++) {
            ranks[order[index]] = average_rank;
        }
        start = end;
    }

    return ranks;
}

double pearson_on_ranks(const vector<double>& ranks1, const vector<double>& ranks2) {
    int n = ranks1.size();
    double avg = (n + 1.0) / 2.0;
    double num = 0.0, den1 = 0.0, den2 = 0.0;

    for (int i = 0; i < n; i++) {
        double centered1 = ranks1[i] - avg;
        double centered2 = ranks2[i] - avg;
        num += centered1 * centered2;
        den1 += centered1 * centered1;
        den2 += centered2 * centered2;
    }

    den1 = sqrt(den1);
    den2 = sqrt(den2);
    if (den1 == 0.0 || den2 == 0.0) return 0.0;
    return num / (den1 * den2);
}
 
// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
 
int main() {
 
    // Heap allocation
    vector<vector<double>> exprByGene(numberOfGenes, vector<double>(sampleSize, 0.0));
    vector<string> geneName(numberOfGenes);

    cerr << "Starting FindCorrOnly...\n";
 
    // Load expression data
    ifstream inStream(expDataFile);
    if (!inStream.is_open()) {
        cerr << "Error opening: " << expDataFile << "\n";
        return 1;
    }
    cerr << "Loading expression data...\n";
 
    int i = 0;
    while (i < numberOfGenes && (inStream >> geneName[i])) {
        for (int k = 0; k < sampleSize; k++) {
            inStream >> exprByGene[i][k];
        }
        i++;
    }
    inStream.close();
    cerr << "Loaded " << i << " genes.\n";

    cerr << "Precomputing ranks...\n";
    vector<vector<double>> ranksByGene(numberOfGenes, vector<double>(sampleSize, 0.0));
    for (int gene = 0; gene < numberOfGenes; gene++) {
        ranksByGene[gene] = rank_values(exprByGene[gene]);
    }
    exprByGene.clear();
    exprByGene.shrink_to_fit();
 
    // Store correlations for upper triangle
    // then write both directions
    ofstream outStream(outFile);
    if (!outStream.is_open()) {
        cerr << "Error opening output: " << outFile << "\n";
        return 1;
    }
 
    int progress_step = max(1, numberOfGenes / 20);
 
    // Store upper triangle results
    vector<vector<double>> rhoMatrix(
        numberOfGenes, vector<double>(numberOfGenes, 0.0)
    );
 
    // Compute upper triangle only
    for (int g1 = 0; g1 < numberOfGenes; g1++) {
        if (g1 % progress_step == 0) {
            cerr << "Progress: " << (100 * g1 / numberOfGenes) << "%\n";
        }
 
        for (int g2 = g1 + 1; g2 < numberOfGenes; g2++) {
            double rho = pearson_on_ranks(ranksByGene[g1], ranksByGene[g2]);
            if (isnan(rho)) rho = 0.0;
            rhoMatrix[g1][g2] = rho;
            rhoMatrix[g2][g1] = rho;
        }
    }
 
    cerr << "Writing output...\n";
 
    // Write all pairs including self-pairs (required by FindCSD.py)
    for (int g1 = 0; g1 < numberOfGenes; g1++) {
        for (int g2 = 0; g2 < numberOfGenes; g2++) {
            outStream << geneName[g1] << "\t"
                      << geneName[g2] << "\t"
                      << rhoMatrix[g1][g2] << "\t"
                      << 0.0 << "\n";
        }
    }
 
    outStream.close();
    cerr << "Done. Written to " << outFile << "\n";
    return 0;
}
