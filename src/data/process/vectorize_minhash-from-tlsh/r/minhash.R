#!/usr/bin/env Rscript

.libPaths( c( "./local_r_libraries" , .libPaths() ) )

library(blink)
library(plyr)
library(tlsh)

args <- commandArgs(trailingOnly=TRUE)
dataset_path <- args[1]
shingle_size <- strtoi(args[2])
number_of_permutations <- strtoi(args[3])
seed <- strtoi(args[4])
output_path <- args[5]

array_of_subsets = c("train", "test", "val")

set.seed(seed)

if (file.exists(output_path)) {
  #Delete file if it exists
  file.remove(output_path)
}

full_dataset <- read.csv(dataset_path)
dataset_without_ids = full_dataset[, 1:(ncol(full_dataset)-2) ]

# entity_ids[i] = entity_ids[j] iff i and j are a match.
entity_ids <- full_dataset[, ncol(full_dataset)-1]
record_ids <- full_dataset[, ncol(full_dataset)]

print(paste("record_ids=", length(record_ids)))

#print(paste("dataset_without_ids=", dataset_without_ids))
shingled_records <- apply(dataset_without_ids,1,shingles,k=shingle_size)
minhashed_records <- minhash_v2(shingled_records,p=number_of_permutations)

# transpose
minhashed_records = t(minhashed_records)

out = cbind(minhashed_records, entity_ids)
out = cbind(out, record_ids)

write.table(out, output_path, sep=",",row.names=FALSE, col.names=FALSE, append=TRUE)
