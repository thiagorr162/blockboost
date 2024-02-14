#!/bin/bash

# Creates directory for the $USER calling the script
#
# Example usage: 
#   src/setup/setup_directories.sh 


BASE_DIR=${PWD##*/}

if [[ $BASE_DIR != "blockboost" ]]; then
    echo "All scripts must be run from base directory 'blockboost'"
    exit
fi


# Create src subdirectories
SRC_DIR=research
declare -a subdirs=("data" "models" "eval")
for subdir in "${subdirs[@]}"
do
   cmd="mkdir -p $SRC_DIR/$subdir"
   echo "$cmd"; $cmd
done

