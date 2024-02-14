#!/usr/bin/env Rscript

.libPaths( c( "./local_r_libraries" , .libPaths() ) )

library(blink)
library(plyr)
library(klsh)

args <- commandArgs(trailingOnly=TRUE)
dataset_name <- args[1]
number_of_blocks <- strtoi(args[2])
number_of_random_projections <- strtoi(args[3])
shingle_size <- strtoi(args[4])
seed <- strtoi(args[5])
output_dir <- args[6]


# 51 set.seed(1234)
# 52 klsh.blocks <- klsh(data.500, p=100, num.blocks=5, k=2)

array_of_subsets = c("test", "val")

#subset = "test"
for (subset in array_of_subsets) {
    set.seed(seed)

    dataset_path <- paste("research/data/processed/", dataset_name,"/",subset,"-textual.csv", sep="")
    full_dataset <- read.csv(dataset_path)
    dataset_without_ids = full_dataset[, 1:(ncol(full_dataset)-2) ]

    # entity_ids[i] = entity_ids[j] iff i and j are a match.
    entity_ids <- full_dataset[, ncol(full_dataset)-1]
    record_ids <- full_dataset[, ncol(full_dataset)]
    blocks <- klsh(dataset_without_ids, p=number_of_random_projections, num.blocks=number_of_blocks, k=shingle_size)

    str_clusters <- "["
    for (a in blocks) {
        str_clusters = paste(str_clusters, "[", sep="")
        for (x in a) {
            str_clusters = paste(str_clusters, '"', record_ids[x],'"', ",", sep="")
        }
        str_clusters = substring(str_clusters, 1, nchar(str_clusters)-1)
        str_clusters = paste(str_clusters, "],", sep="")

    }
    str_clusters = substring(str_clusters, 1, nchar(str_clusters)-1)
    str_clusters <- paste(str_clusters, "]", sep="")
    fileConn<-file(paste(output_dir,"/",subset,"-transitive_prediction.json", sep=""))
    writeLines(c(str_clusters), fileConn)
    close(fileConn)

    set.seed(seed)
}
