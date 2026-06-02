#Takes input from 4 files, 'UseableCValues.txt', 'UseableSValues.txt', 'UseableDValues.txt', 'AllValues.txt' (output from findCSD.py) and generates 4 networks - one for each of the C/S/D-type interactions, as well as an aggregate network containing all 3

#Outputs selected pairs, along with metric type and value, to SelectedSNodes.txt, SelectedCNodes.txt, SelectedDNodes.txt, and CSDSelection.txt


##############################################
#Parameters to vary are selSize and noSels

#selSize indicates approximate proportion of selected nodes, being equal to the 1/(desired p-value). Increasing selSize selects fewer edges. 
#noSels may be increased as desireable, at the expense of longer running time. May also be decreased, at the expense of accuracy


import math
import numpy
import random
from pathlib import Path


def print_progress(step_name, current=None, total=None):
    if current is None or total is None:
        print(f"[CreateNetwork] {step_name}")
        return

    percent_complete = (current / total) * 100
    print(f"[CreateNetwork] {step_name}: {current}/{total} ({percent_complete:.1f}%)")

selSize = 20000
noSels = 10000

ROOT_DIR = Path(__file__).resolve().parents[2]
NETWORKS_DIR = ROOT_DIR / "results" / "networks"
CSD_DIR = ROOT_DIR / "results" / "CSD"
CSD_DIR.mkdir(parents=True, exist_ok=True)

cValueFile = NETWORKS_DIR / 'UseableCValues_proteome_WTO_1.txt'
sValueFile = NETWORKS_DIR / 'UseableSValues_proteome_WTO_1.txt'
dValueFile = NETWORKS_DIR / 'UseableDValues_proteome_WTO_1.txt'
progress_interval = max(1, noSels // 10)

print_progress('Leser C-verdier')
f = open(cValueFile)

valueList = []

for line in f:
    valueList.append(float(line))


cutoffAtSel = []
print_progress('Beregner C-cutoff')
for i in range(noSels):

    selection = [valueList[i] for i in random.sample(range(0, len(valueList)), selSize)]
    cutoffAtSel.append(max(selection))
    if (i + 1) % progress_interval == 0 or i == noSels - 1:
        print_progress('Beregner C-cutoff', i + 1, noSels)

cCutoff = numpy.mean(cutoffAtSel)
print_progress(f'C-cutoff ferdig: {cCutoff}')

f.close()



print_progress('Leser S-verdier')
f = open(sValueFile)

valueList = []

for line in f:
    valueList.append(float(line))


cutoffAtSel = []
print_progress('Beregner S-cutoff')
for i in range(noSels):

    selection = [valueList[i] for i in random.sample(range(0, len(valueList)), selSize)]
    cutoffAtSel.append(max(selection))
    if (i + 1) % progress_interval == 0 or i == noSels - 1:
        print_progress('Beregner S-cutoff', i + 1, noSels)

sCutoff = numpy.mean(cutoffAtSel)
print_progress(f'S-cutoff ferdig: {sCutoff}')
f.close()


print_progress('Leser D-verdier')
f = open(dValueFile)

valueList = []

for line in f:
    valueList.append(float(line))


cutoffAtSel = []
print_progress('Beregner D-cutoff')
for i in range(noSels):

    selection = [valueList[i] for i in random.sample(range(0, len(valueList)), selSize)]
    cutoffAtSel.append(max(selection))
    if (i + 1) % progress_interval == 0 or i == noSels - 1:
        print_progress('Beregner D-cutoff', i + 1, noSels)

dCutoff = numpy.mean(cutoffAtSel)
print_progress(f'D-cutoff ferdig: {dCutoff}')
f.close()

print_progress('Aapner output-filer')
cNetF = open(CSD_DIR / 'CNetwork_proteome_WTO_1_20.txt', 'w')
sNetF = open(CSD_DIR / 'SNetwork_proteome_WTO_1_20.txt', 'w')
dNetF = open(CSD_DIR / 'DNetwork_proteome_WTO_1_20.txt', 'w')
csdNetF = open(CSD_DIR / 'CSDSelection_proteome_WTO_1_20.txt', 'w')
selected_edges = {'C': set(), 'S': set(), 'D': set()}

print_progress('Teller linjer i AllValues-filen')
with open(NETWORKS_DIR / 'AllValues_proteome_WTO_1.txt') as count_file:
    total_lines = sum(1 for _ in count_file) - 1

f = open(NETWORKS_DIR / 'AllValues_proteome_WTO_1.txt')

f.readline()

all_values_progress_interval = max(1, total_lines // 10)
print_progress('Bygger nettverk fra AllValues')

for line_number, line in enumerate(f, start=1):

    splitLine = [value.strip() for value in line.rstrip().split('\t')]
    gene_a = splitLine[0]
    gene_b = splitLine[1]
    edge_key = tuple(sorted((gene_a, gene_b)))
    if float(splitLine[4]) > cCutoff and edge_key not in selected_edges['C']:
        selected_edges['C'].add(edge_key)
        print(str(gene_a)+'\t'+str(gene_b)+'\t'+str(splitLine[4])+'\t'+'C', file=cNetF)
        print(str(gene_a)+'\t'+str(gene_b)+'\t'+str(splitLine[4])+'\t'+'C', file=csdNetF)

    if float(splitLine[5]) > sCutoff and edge_key not in selected_edges['S']:
        selected_edges['S'].add(edge_key)
        print(str(gene_a)+'\t'+str(gene_b)+'\t'+str(splitLine[5])+'\t'+'S', file=sNetF)
        print(str(gene_a)+'\t'+str(gene_b)+'\t'+str(splitLine[5])+'\t'+'S', file=csdNetF)
    if float(splitLine[6]) > dCutoff and edge_key not in selected_edges['D']:
        selected_edges['D'].add(edge_key)
        print(str(gene_a)+'\t'+str(gene_b)+'\t'+str(splitLine[6])+'\t'+'D', file=dNetF)
        print(str(gene_a)+'\t'+str(gene_b)+'\t'+str(splitLine[6])+'\t'+'D', file=csdNetF)

    if line_number % all_values_progress_interval == 0 or line_number == total_lines:
        print_progress('Bygger nettverk fra AllValues', line_number, total_lines)



cNetF.close()
sNetF.close()
dNetF.close()
csdNetF.close()
f.close()
print_progress('CreateNetwork ferdig')
